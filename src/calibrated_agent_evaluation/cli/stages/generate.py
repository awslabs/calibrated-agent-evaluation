# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""``generate``: synthesize frame-tagged inputs, optionally route through a real agent.

Reuses the ``calibrated_agent_evaluation.synth`` engine (``generate`` / ``generate_and_run``) and
the frame assembly. The stage supplies the concrete injections those layers left open:

  * the **renderer**; an offline deterministic template by default, or the concrete
    ``BedrockRenderer`` when the ``[bedrock]`` extra + a ``--model`` id are configured;
  * with ``--run``, a **task builder** + **executor** to route inputs through a real agent
    the Inspect builder for ``--executor inspect``, the Strands builder for ``strands``, and for
    ``agentcore`` the builder for an agent hosted on Amazon Bedrock AgentCore Runtime;
  * with ``--run --multi-turn``, the **persona** each simulated conversation is held with, from
    the input's taxonomy cell or a ``--persona`` file;
  * an **embedder** for the dedup/diversity guards; the concrete ``BedrockEmbedder`` when the
    ``[bedrock]`` extra is present, else none (dedup is skipped offline).

The frame is read from ``--frame`` (what ``calibrated-agent-evaluation define --out`` writes), built
from a ``--taxonomy`` file, or built from the stored traces' ``taxonomy_cell`` values. The
synthesized inputs are written as JSONL with ``--out``; with ``--run`` the agent's traces are
persisted.

With ``--executor agentcore``, every run that can call AWS (any cassette mode but replay) first
checks the billed-call budget variable and that a span reader is installed, then that the trace
store can take the traces, then prints what it will call and goes ahead only with ``--yes``. The
run manifest (``agentcore_run_record.json`` and ``.csv``, one record per run) goes under
``<root>/records/agentcore`` or ``--log-dir``.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING, Any

from calibrated_agent_evaluation.cli.agentcore_route import (
    billed_call_budget,
    cassette_mode_from_env,
    check_trace_store,
    extra_missing_error,
    manifest_dir,
    manifest_report,
    print_plan,
    refuse_agentcore_flags_without_executor,
    require_agentcore_extra,
    require_agentcore_options,
    target_fields,
)
from calibrated_agent_evaluation.cli.dispatch import EXIT_OK, StageOutcome
from calibrated_agent_evaluation.cli.errors import CliError, UsageError
from calibrated_agent_evaluation.execute.cassette import MODE_ENV_VAR

if TYPE_CHECKING:
    from calibrated_agent_evaluation.agentcore import (
        AgentCoreRuntimeExecutor,
        AwsClients,
        RunSpanReader,
        RuntimeTarget,
    )
    from calibrated_agent_evaluation.synth.conversation import ConversationTurn, SimulatedUser
    from calibrated_agent_evaluation.synth.conversation_cassette import UserCassette
    from calibrated_agent_evaluation.synth.types import SynthInput

__all__ = ["TemplateRenderer", "generate_handler"]


class TemplateRenderer:
    """Deterministic offline tuple→NL renderer: the zero-dep default.

    Satisfies the ``Renderer`` protocol (``render(cell) -> str``). Produces a stable,
    non-empty NL string from a taxonomy cell without any network call, so
    ``calibrated-agent-evaluation generate`` runs offline on a laptop. A real caller upgrades to the
    ``BedrockRenderer`` by installing ``[bedrock]`` and passing ``--model``.
    """

    def render(self, cell: dict[str, str]) -> str:
        """Describe a taxonomy cell as one customer-request sentence."""
        parts = ", ".join(f"{k}={v}" for k, v in sorted(cell.items()))
        return f"A customer request exercising {parts}."


def generate_handler(args: argparse.Namespace) -> StageOutcome:
    """Synthesize inputs (and optionally route them through a real agent)."""
    from calibrated_agent_evaluation.cli.config import require_extras

    if getattr(args, "multi_turn", False) and not getattr(args, "run", False):
        raise UsageError(
            "--multi-turn is a way of routing inputs through your agent, so it needs --run "
            "--executor strands; without --run there is nobody to converse with"
        )
    # First, so `generate` without --run refuses them too rather than ignoring them.
    refuse_agentcore_flags_without_executor(args, remedy=_AGENTCORE_REMEDY, sent_by="--run")

    # The frame/synth engine lives in optional extras ([sampling]->allpairspy, [stats]->scipy),
    # NOT in the default install: probe first so a bare install gets a typed hint, not a raw
    # ModuleNotFoundError traceback.
    require_extras("allpairspy", "scipy")
    from calibrated_agent_evaluation.cli.backends import open_backend
    from calibrated_agent_evaluation.synth import generate

    with open_backend(args) as backend:
        frame, source = _frame_for(args, backend)
        renderer = _resolve_renderer(args)
        embedder = _resolve_embedder(args)

        if getattr(args, "run", False):
            return _generate_and_run(args, frame, source, renderer, embedder, backend)

    inputs = generate(frame, renderer, embedder=embedder)
    model_note = "LLM renderer" if getattr(args, "model", None) else "offline template renderer"
    out = getattr(args, "out", None)
    if out:
        _write_inputs(Path(out), inputs)
        kept = f"; wrote them to {out} (JSONL, one input per line)"
    else:
        kept = "; not kept (pass --out FILE to write them as JSONL)"
    return StageOutcome(
        stage="generate",
        exit_code=EXIT_OK,
        summary=(
            f"synthesized {len(inputs)} frame-tagged input(s) from a {len(frame.rows)}-row "
            f"{frame.frame} frame ({source}) via the {model_note}{kept}"
        ),
    )


def _frame_for(args: argparse.Namespace, backend: Any) -> tuple[Any, str]:
    """The frame to synthesize from, and where it came from (for the summary line)."""
    from calibrated_agent_evaluation.cli.scope import read_scope
    from calibrated_agent_evaluation.cli.taxonomy import (
        frame_from_file,
        grid_from_file,
        grid_from_traces,
    )
    from calibrated_agent_evaluation.sampling import discovery_frame

    frame_file = getattr(args, "frame_file", None)
    taxonomy = getattr(args, "taxonomy", None)
    if frame_file and taxonomy:
        raise UsageError("pass --frame or --taxonomy, not both: each one fixes the frame")
    if frame_file:
        return frame_from_file(frame_file), f"frame file {frame_file}"
    if taxonomy:
        return discovery_frame(grid_from_file(taxonomy)), f"taxonomy file {taxonomy}"
    scan = read_scope(backend, args)
    grid, padded = grid_from_traces(scan.traces)
    note = "taxonomy of the stored traces"
    if scan.skipped:
        # Named in full by the stages that score traces; here the count says the cells are partial.
        note += f"; {len(scan.skipped)} trace(s) Calibrated Agent Evaluation did not write skipped"
    if padded:
        note += f"; placeholder second value for {', '.join(padded)}"
    return discovery_frame(grid), note


def _write_inputs(path: Path, inputs: list[Any]) -> None:
    """One JSON object per input: ``text``, ``cell``, ``frame``, ``weight`` (and ``turns``)."""
    lines = []
    for item in inputs:
        doc: dict[str, Any] = {
            "text": item.text,
            "cell": dict(item.cell),
            "frame": item.frame,
            "weight": item.weight,
        }
        if item.turns is not None:
            doc["turns"] = item.turns
        lines.append(json.dumps(doc, sort_keys=True))
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("".join(f"{line}\n" for line in lines), encoding="utf-8")
    except OSError as exc:
        raise CliError(f"cannot write the inputs to {path}: {exc.strerror or exc}") from exc


def _generate_and_run(
    args: argparse.Namespace, frame: Any, source: str, renderer: Any, embedder: Any, backend: Any
) -> StageOutcome:
    """Route synthesized inputs through the selected executor + matching task builder."""
    from calibrated_agent_evaluation.synth import generate_and_run

    executor_flag = getattr(args, "executor", None)
    if executor_flag not in ("inspect", "strands", "agentcore"):
        raise UsageError(
            "calibrated-agent-evaluation generate --run needs --executor "
            "inspect|strands|agentcore (an executor to "
            "route through)"
        )
    multi_turn = bool(getattr(args, "multi_turn", False))
    if multi_turn and executor_flag == "inspect":
        raise UsageError(
            "--multi-turn routes each turn through the agent; pass --executor strands or "
            "agentcore (the inspect route evaluates one input per sample, so it has no turns)"
        )
    max_turns = getattr(args, "max_turns", 6)
    max_turns = 6 if max_turns is None else int(max_turns)
    if multi_turn and max_turns < 1:
        raise UsageError(f"--max-turns needs at least one turn, got {max_turns}")
    if executor_flag == "agentcore":
        return _generate_and_run_agentcore(args, frame, source, renderer, embedder, backend)
    cassette = getattr(args, "cassette", None)
    users = _user_cassette(cassette, cassette_mode_from_env()) if multi_turn else None
    agent = _resolve_agent(getattr(args, "agent", None))
    executor, task_builder = _build_executor_and_builder(
        executor_flag,
        agent,
        model=getattr(args, "agent_model", None),
        cassette=cassette,
    )
    conversation = (
        _conversation_kwargs(args, task_builder, max_turns, users=users) if multi_turn else {}
    )
    # The executor owns a temp log directory; the block releases it on success and on failure.
    with executor:
        traces = generate_and_run(
            frame,
            renderer,
            executor,
            task_builder,
            embedder=embedder,
            multi_turn=multi_turn,
            max_turns=max_turns,
            **conversation,
        )
    if users is not None:
        # Only once every conversation finished: a partial recording would replay and diverge.
        users.save()
    for trace in traces:
        backend.put_trace(trace)
    if multi_turn:
        persona = getattr(args, "persona", None)
        whose = f"the persona in {persona}" if persona else "a persona per taxonomy cell"
        summary = (
            f"held {len(traces)} simulated conversation(s) of up to {max_turns} agent turn(s) "
            f"with {whose} against the {executor_flag} executor "
            f"({source}): {len(traces)} {frame.frame}-frame, taxonomy-attributable "
            "conversation trace(s) persisted"
        )
    else:
        summary = (
            f"routed {len(frame.rows)} synthesized input(s) ({source}) through the "
            f"{executor_flag} executor: {len(traces)} {frame.frame}-frame, "
            "taxonomy-attributable trace(s) persisted"
        )
    return StageOutcome(stage="generate", exit_code=EXIT_OK, summary=summary)


def _conversation_kwargs(
    args: argparse.Namespace,
    task_builder: Any,
    max_turns: int = 6,
    *,
    end: Callable[[str], object] | None = None,
    conversation_task: Callable[[SynthInput, list[dict[str, Any]], ConversationTurn], Any]
    | None = None,
    users: UserCassette | None = None,
) -> dict[str, Any]:
    """The multi-turn injections: how a turn becomes a task, the persona and the simulated user.

    ``conversation_task`` (a ``(input, history, turn)`` task builder) and ``end`` (called with a
    conversation's id when it ends) plug in an executor that holds one session per conversation.
    With ``users``, each conversation's simulated user is replayed from or recorded into that
    recording, keyed by the conversation's id; a replayed one never builds the real simulator. The
    caller saves ``users`` once the run has finished.
    """
    from calibrated_agent_evaluation.cli.config import probe_extra
    from calibrated_agent_evaluation.cli.conversation_route import (
        require_simulator_extras,
        simulator_from,
        user_factory,
    )
    from calibrated_agent_evaluation.synth.conversation import ConversationHooks

    simulator = simulator_from(args)
    probe = probe_extra("strands_evals")
    if not probe.available:
        raise CliError(f"--multi-turn needs the synth extra: {probe.install_hint}")
    require_simulator_extras(simulator)
    from calibrated_agent_evaluation.synth.borrow.simulator import (
        actor_profile_from_cell,
        actor_profile_from_dict,
    )

    persona_file = getattr(args, "persona", None)
    if persona_file:
        profile = actor_profile_from_dict(_read_persona(Path(persona_file)))
        profile_for = lambda _item: profile  # noqa: E731 - one persona for every input
    else:
        profile_for = lambda item: actor_profile_from_cell(item.cell)  # noqa: E731
    model = getattr(args, "simulator_model", None)
    # The strands actor gets the run's --max-turns too, so its own limit is the run's.
    build_user = user_factory(simulator, max_turns=max_turns, model=model)

    def user_for(item: SynthInput, profile: Any) -> SimulatedUser:
        return build_user(item.text, profile, "")

    def user_for_conversation(
        item: SynthInput, profile: Any, conversation_id: str
    ) -> SimulatedUser:
        if users is None:
            return user_for(item, profile)
        return users.user_for(conversation_id, lambda: user_for(item, profile))

    hooks = ConversationHooks(
        turn_task=conversation_task,
        end=end,
        user_for=user_for,
        user_for_conversation=user_for_conversation,
        simulator=simulator,
    )
    kwargs: dict[str, Any] = {"actor_profile_for": profile_for, "conversation_hooks": hooks}
    if conversation_task is None:
        kwargs["turn_task"] = task_builder.build_conversation_turn
    return kwargs


def _user_cassette(cassette: str | None, mode: Any) -> UserCassette | None:
    """The simulated-user recording that goes with ``--cassette``, or ``None`` without one."""
    if not cassette:
        return None
    from calibrated_agent_evaluation.synth.conversation_cassette import (
        UserCassette,
        actor_cassette_path,
    )

    return UserCassette(actor_cassette_path(cassette), mode)


def _read_persona(path: Path) -> dict[str, Any]:
    """Read a ``--persona`` JSON file; a missing, unreadable or non-object file is a typed error."""
    try:
        loaded = json.loads(path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise CliError(f"cannot read the persona file {path}: {exc.strerror or exc}") from exc
    except json.JSONDecodeError as exc:
        raise CliError(f"the persona file {path} is not valid JSON: {exc}") from exc
    if not isinstance(loaded, dict):
        raise CliError(
            f"the persona file {path} must hold a JSON object with traits, context and actor_goal"
        )
    return loaded


def _build_executor_and_builder(
    executor_flag: str,
    agent: Callable[..., Any],
    *,
    model: str | None = None,
    cassette: str | None = None,
) -> tuple[Any, Any]:
    """Construct the executor + its matching task builder, with typed install-hints.

    ``agent`` is the caller-supplied agent under test (resolved from ``--agent module:callable``):
    an Inspect solver for the Inspect route, a Strands topology builder for the Strands route. The
    builder needs it: the synthesized inputs are only the inputs, not the agent. Both executors
    are told ``frame="discovery"``, the frame every synthesized input carries; their default is
    production-representative, which the synthesis frame-propagation check refuses.
    """
    from calibrated_agent_evaluation.cli.config import probe_extra

    cassette_path = Path(cassette) if cassette else None
    if executor_flag == "inspect":
        probe = probe_extra("inspect_ai")
        if not probe.available:
            raise CliError(f"--executor inspect needs the inspect extra: {probe.install_hint}")
        if not model:
            raise UsageError(
                "--executor inspect needs --agent-model, the Inspect model the agent runs on "
                "(e.g. bedrock/<model-id>, or cassette/<name> with --cassette)"
            )
        from calibrated_agent_evaluation.execute import InspectExecutor
        from calibrated_agent_evaluation.execute.task_builders import InspectTaskBuilder

        executor = InspectExecutor(model=model, cassette_path=cassette_path, frame="discovery")
        return executor, InspectTaskBuilder(solver=agent)

    probe = probe_extra("strands")
    if not probe.available:
        raise CliError(f"--executor strands needs the strands extra: {probe.install_hint}")
    from calibrated_agent_evaluation.execute import StrandsExecutor
    from calibrated_agent_evaluation.execute.task_builders import StrandsTaskBuilder

    kwargs: dict[str, Any] = {"cassette_path": cassette_path, "frame": "discovery"}
    if model:
        kwargs["model_id"] = model
    return StrandsExecutor(**kwargs), StrandsTaskBuilder(build_topology=agent)


# --- --executor agentcore -------------------------------------------------------------------

_OFFLINE_HINT = (
    f"offline runs need a cassette in replay mode: set {MODE_ENV_VAR}=replay and pass "
    "--cassette PATH"
)

# How `generate` sends its inputs to an AgentCore Runtime agent, for a flag given without it.
_AGENTCORE_REMEDY = (
    "pass --run --executor agentcore to send the inputs to an AgentCore Runtime agent"
)

# The calibrated_agent_evaluation.agentcore names this route uses, each checked for the [agentcore]
# extra up front.
_AGENTCORE_NAMES = (
    "RuntimeTarget",
    "AgentCoreRuntimeExecutor",
    "AgentCoreTaskBuilder",
    "AwsClients",
)


def _generate_and_run_agentcore(
    args: argparse.Namespace, frame: Any, source: str, renderer: Any, embedder: Any, backend: Any
) -> StageOutcome:
    """Route each synthesized input through an agent hosted on Amazon Bedrock AgentCore Runtime.

    Everything that can be refused without AWS is refused before the executor or any AWS client
    is built. A replay calls no AWS, so it needs no plan, ``--yes``, span reader or clients. Any
    other run checks the budget variable, the span reader and the trace store, prints its plan and
    goes ahead only with ``--yes``. It then builds one set of AWS clients carrying the billed-call
    budget for the executor and the span reader both, so the reader's billed reads count too.
    """
    from calibrated_agent_evaluation.execute.cassette import Mode

    _refuse_agent_flags(args)
    require_agentcore_options(args)
    mode = cassette_mode_from_env()
    cassette = getattr(args, "cassette", None)
    if mode is Mode.REPLAY and not cassette:
        raise CliError(
            f"replay needs a cassette: {MODE_ENV_VAR} is replay, but no --cassette was given, so "
            f"nothing was run. Pass --cassette PATH, or set {MODE_ENV_VAR} to auto or record to "
            "call AWS"
        )
    _require_discovery_rows(frame, source)
    require_agentcore_extra(_AGENTCORE_NAMES)
    import calibrated_agent_evaluation.agentcore as agentcore
    from calibrated_agent_evaluation.agentcore.run_manifest import manifest_path

    target = agentcore.RuntimeTarget(**target_fields(args))
    log_dir = manifest_dir(args)
    # The run manifest's JSON file, the path printed: the executor adds one record per run.
    manifest = manifest_path(log_dir)
    options: dict[str, Any] = {
        "log_dir": log_dir,
        # The frame every synthesized input carries; the frame-propagation check refuses others.
        "frame": "discovery",
    }
    if cassette:
        # The executor refuses a mode passed without a cassette, so a run with no cassette passes
        # neither and runs every input live.
        options.update(cassette_path=Path(cassette), cassette_mode=mode)
    read_timeout = getattr(args, "read_timeout", None)
    if read_timeout is not None:
        options["read_timeout_seconds"] = read_timeout
    multi_turn = bool(getattr(args, "multi_turn", False))
    max_turns = int(getattr(args, "max_turns", 6) or 6)
    conversation = (args, max_turns) if multi_turn else None
    if conversation is not None:
        # Refused before any AWS call: the simulated user's extra is known up front.
        from calibrated_agent_evaluation.cli.conversation_route import (
            require_simulator_extras,
            simulator_from,
        )

        require_simulator_extras(simulator_from(args))
    route = (frame, source, renderer, embedder, backend, target, manifest)
    # Opened before anything runs: a replay with no simulated-user recording is refused here.
    users = _user_cassette(cassette, mode) if multi_turn else None
    if mode is Mode.REPLAY:
        return _route_through_agentcore(
            agentcore.AgentCoreRuntimeExecutor(
                target, profile=getattr(args, "profile", None), **options
            ),
            *route,
            conversation=conversation,
            users=users,
        )

    # Known without any call, so refused before the store check, the plan and --yes: a run that
    # would be refused anyway is not asked to go ahead.
    budget = billed_call_budget()
    reader_class = _span_reader_class()
    # Before the plan and before --yes: a trace store that can't take the traces would lose every
    # billed run, since they are stored only once the last input has run.
    check_trace_store(backend)
    print_plan(
        "generate",
        target,
        manifest,
        count=len(frame.rows) * (max_turns if multi_turn else 1),
        noun="agent turn" if multi_turn else "input",
        reads_spans=True,
        cassette=cassette,
    )
    if not getattr(args, "yes", False):
        raise CliError(
            "nothing was sent: this run can make the billed AWS calls listed above, so it needs "
            f"--yes. Pass --yes to go ahead; {_OFFLINE_HINT}"
        )
    clients = agentcore.AwsClients.from_profile(
        region=target.region, profile=getattr(args, "profile", None), handlers=[budget]
    )
    # Opened around the executor, whose exit stops any session a run left open through them. The
    # executor gets these clients and no profile: the profile already chose them.
    with clients:
        executor = agentcore.AgentCoreRuntimeExecutor(
            target, span_reader=reader_class(clients), clients=clients, **options
        )
        return _route_through_agentcore(executor, *route, conversation=conversation, users=users)


def _route_through_agentcore(
    executor: AgentCoreRuntimeExecutor,
    frame: Any,
    source: str,
    renderer: Any,
    embedder: Any,
    backend: Any,
    target: RuntimeTarget,
    manifest: Path,
    *,
    conversation: tuple[argparse.Namespace, int] | None = None,
    users: UserCassette | None = None,
) -> StageOutcome:
    """Route every synthesized input through ``executor``, then store the traces it returned.

    With ``conversation`` (the args and ``--max-turns``), each input is a simulated conversation
    whose turns share one runtime session, stopped when the conversation ends. With ``users``, the
    simulated user is replayed or recorded there, and the recording is saved once every
    conversation has finished.
    """
    from calibrated_agent_evaluation.agentcore import AgentCoreTaskBuilder
    from calibrated_agent_evaluation.synth import generate_and_run

    builder = AgentCoreTaskBuilder()
    turns: dict[str, Any] = {}
    if conversation is not None:
        conv_args, max_turns = conversation
        turns = {
            "multi_turn": True,
            "max_turns": max_turns,
            **_conversation_kwargs(
                conv_args,
                builder,
                max_turns,
                end=executor.end_conversation,
                conversation_task=builder.build_conversation_turn,
                users=users,
            ),
        }
    try:
        # Leaving the block stops any session left open and releases the executor's clients.
        with executor:
            traces = generate_and_run(
                frame, renderer, executor, builder, embedder=embedder, **turns
            )
            if users is not None:
                users.save()
            # Stored before the block is left, so a failure to clean up after every billed run
            # still keeps their traces; inside the try, a store that fails leaves their records.
            for trace in traces:
                backend.put_trace(trace)
    except BaseException:
        # The run's error reaches the user rewrapped, without the manifest's path or the notes
        # added to it, so the manifest is named here, with any record it could not take.
        if executor.run_records:
            print(
                f"calibrated-agent-evaluation generate: {manifest_report(executor, manifest)}",
                file=sys.stderr,
                flush=True,
            )
        raise
    if conversation is None:
        verb, what, kept = "routed", "synthesized input(s)", "trace(s)"
    else:
        verb = "held"
        what = "simulated conversation(s) from synthesized input(s)"
        kept = "conversation trace(s)"
    return StageOutcome(
        stage="generate",
        exit_code=EXIT_OK,
        summary=(
            f"{verb} {len(frame.rows)} {what} ({source}) through the agentcore executor (runtime "
            f"{target.runtime_arn}): {len(traces)} {frame.frame}-frame, taxonomy-attributable "
            f"{kept} persisted; the run manifest, one record per run, is {manifest}"
        ),
    )


def _refuse_agent_flags(args: argparse.Namespace) -> None:
    for dest, flag in (("agent", "--agent"), ("agent_model", "--agent-model")):
        if getattr(args, dest, None) is not None:
            raise UsageError(
                f"{flag} doesn't apply to --executor agentcore: the agent under test is the "
                f"runtime --runtime-arn names, already deployed with its model. Drop {flag}"
            )


def _require_discovery_rows(frame: Any, source: str) -> None:
    # Refused up front: the frame-propagation check would only refuse them after every input's
    # billed run, since the executor stamps each trace with the discovery frame.
    others = sorted({row.frame for row in frame.rows if row.frame != "discovery"})
    if others:
        raise CliError(
            f"{source} holds {', '.join(others)} rows, and --executor agentcore runs discovery "
            "inputs only, so nothing was run. Pass a discovery frame with --frame FILE "
            "(`calibrated-agent-evaluation "
            "define --out FILE` writes one)"
        )


def _span_reader_class() -> Callable[[AwsClients], RunSpanReader]:
    import calibrated_agent_evaluation.agentcore as agentcore
    from calibrated_agent_evaluation.agentcore import AgentCoreExtraMissing

    try:
        # The per-run span reader ships in the agentcore extra; the check stays so an install that
        # cannot load it is refused before any AWS call.
        reader = getattr(agentcore, "AgentCoreSpanReader", None)
    except AgentCoreExtraMissing as exc:
        raise extra_missing_error(exc) from exc
    if reader is None:
        raise CliError(
            "live runs need the AgentCore span reader "
            "(calibrated_agent_evaluation.agentcore.AgentCoreSpanReader) to "
            "read each run's spans, and this install has none, so nothing was sent; "
            f"{_OFFLINE_HINT}"
        )
    return reader


def _resolve_renderer(args: argparse.Namespace):
    """Return the concrete renderer: BedrockRenderer if [bedrock]+--model, else offline template."""
    model = getattr(args, "model", None)
    if not model:
        return TemplateRenderer()
    from calibrated_agent_evaluation.cli.config import probe_extra

    probe = probe_extra("boto3")
    if not probe.available:
        raise CliError(f"--model needs the bedrock extra;{probe.install_hint}")
    from calibrated_agent_evaluation.execute.bedrock_renderer import BedrockRenderer

    return BedrockRenderer(model=model)


def _resolve_embedder(args: argparse.Namespace):
    """Return the concrete embedder if [bedrock] is present, else None (dedup skipped offline)."""
    from calibrated_agent_evaluation.cli.config import probe_extra

    if not getattr(args, "model", None):
        # No model configured → offline path; skip dedup rather than force a network embedder.
        return None
    if not probe_extra("boto3").available:
        return None
    from calibrated_agent_evaluation.execute.bedrock_embedder import BedrockEmbedder

    return BedrockEmbedder()


def _resolve_agent(spec: str | None) -> Callable[..., Any]:
    """Import the agent-under-test callable from a ``module:callable`` spec, cwd first.

    Required with ``--run``; the agent is the caller's, not the library's. A missing/malformed spec,
    or one that doesn't resolve, is a typed ``CliError`` (never a raw ImportError/AttributeError).
    """
    if not spec:
        raise UsageError(
            "calibrated-agent-evaluation generate --run needs --agent module:callable (the agent "
            "under test is yours; "
            "e.g. --agent myagents.orderbot:build)"
        )
    from calibrated_agent_evaluation.cli.agentspec import load_agent

    return load_agent(spec)
