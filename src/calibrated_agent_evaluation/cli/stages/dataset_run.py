# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""``dataset run SLICE``: one simulated conversation per scenario of a simulated slice.

Each scenario's simulated user (``--simulator strands`` or ``agentcore``) opens with the scenario's
input and pursues its goal; the agent under test (``--executor strands`` with ``--agent``, or
``--executor agentcore`` with the runtime flags) answers each turn. Every conversation is stored as
one trace that keeps every turn's spans, the transcript, the goal, the assertions and why it
stopped, and names the slice's published dataset version when it has one.

Nothing is stored unless every conversation finishes. A run that can call AWS or a model prints its
plan and goes ahead only with ``--yes``; with ``CALIBRATED_AGENT_EVALUATION_CASSETTE_MODE=replay``
and ``--cassette`` the whole run, simulated user included, replays offline.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any

from calibrated_agent_evaluation.cli.dispatch import EXIT_OK, StageOutcome
from calibrated_agent_evaluation.cli.errors import CliError, UsageError
from calibrated_agent_evaluation.synth.conversation import Simulator

__all__ = ["run_slice"]

# The calibrated_agent_evaluation.agentcore names the AgentCore route uses, checked for the
# [agentcore] extra up front.
_AGENTCORE_NAMES = ("AgentCoreRuntimeExecutor", "AgentCoreTaskBuilder", "AwsClients")
# Flags only --executor agentcore reads (dataset's own --profile and --yes serve every route).
_RUNTIME_FLAGS = {
    "runtime_arn": "--runtime-arn",
    "payload_format": "--payload-format",
    "qualifier": "--qualifier",
    "config_bundle_arn": "--config-bundle-arn",
    "config_bundle_version": "--config-bundle-version",
    "read_timeout": "--read-timeout",
    "log_dir": "--log-dir",
}


def run_slice(args: argparse.Namespace) -> StageOutcome:
    """Run every scenario of the simulated slice ``args.slice`` and store one trace per scenario."""
    from calibrated_agent_evaluation.cli.agentcore_route import (
        cassette_mode_from_env,
        check_trace_store,
    )
    from calibrated_agent_evaluation.cli.backends import open_backend
    from calibrated_agent_evaluation.cli.conversation_route import (
        require_simulator_extras,
        simulator_from,
    )
    from calibrated_agent_evaluation.execute.cassette import Mode
    from calibrated_agent_evaluation.scenarios.slicefile import read_slice
    from calibrated_agent_evaluation.scenarios.types import SimulatedScenario

    slice_ = read_slice(Path(args.slice))
    if slice_.form != "simulated":
        raise CliError(
            f"`calibrated-agent-evaluation dataset run` runs simulated slices, and {args.slice} "
            f"is a {slice_.form} "
            "slice; nothing was run"
        )
    executor_flag = getattr(args, "executor", None) or "strands"
    _check_executor_flags(args, executor_flag)
    judge_file = _judge_file(args)
    simulator = simulator_from(args)
    require_simulator_extras(simulator)
    mode = cassette_mode_from_env()
    cassette = getattr(args, "cassette", None)
    if mode is Mode.REPLAY and not cassette:
        raise CliError(
            "replay needs a cassette: CALIBRATED_AGENT_EVALUATION_CASSETTE_MODE is replay, but no "
            "--cassette was "
            "given, so nothing was run"
        )
    replay = mode is Mode.REPLAY
    if replay and judge_file is not None:
        raise UsageError(
            "--judge can't run in replay: judging is a live, billed Evaluate call per "
            "conversation, and replay makes no AWS call. Replay without --judge, or judge a live "
            "or recording run"
        )
    scenarios = [s for s in slice_.scenarios if isinstance(s, SimulatedScenario)]
    most_turns = sum(s.max_turns for s in scenarios)
    # Every billed call the run can make: each scenario's last possible turn on a runtime, and
    # one Evaluate per conversation when judging.
    most_billed = (most_turns if executor_flag == "agentcore" else 0) + (
        len(scenarios) if judge_file is not None else 0
    )

    with open_backend(args) as backend:
        # Before any call: a store that can't take the traces would lose every conversation.
        check_trace_store(backend)
        describe = (
            f"the AgentCore Runtime agent {getattr(args, 'runtime_arn', None)}"
            if executor_flag == "agentcore"
            else f"the strands agent {args.agent}"
        )
        if not replay and not getattr(args, "yes", False):
            if most_billed:
                # Refused before the plan: a run that would be refused anyway is not planned.
                _check_budget(most_billed)
            _print_plan(len(scenarios), most_turns, describe, simulator, judge_file)
            raise CliError(
                "nothing has run: this run calls the agent and the simulated user's model, so it "
                "needs --yes. Pass --yes to go ahead, or replay a cassette offline with "
                "CALIBRATED_AGENT_EVALUATION_CASSETTE_MODE=replay and --cassette PATH"
            )
        budget = _check_budget(most_billed) if most_billed and not replay else None
        if executor_flag == "agentcore":
            executor, task_for = _agentcore_executor(args, mode, cassette, budget)
        else:
            executor, task_for = _strands_executor(args, cassette)
        traces, lines = _converse_all(
            args, scenarios, slice_, executor, task_for, simulator, mode, cassette
        )
        for trace in traces:
            backend.put_trace(trace)
        judged = (
            []
            if judge_file is None
            else _judge(args, judge_file, traces, budget, backend, executor_flag)
        )

    summary = [
        f"Ran {len(traces)} simulated conversation(s) from slice {args.slice} against "
        f"{describe} (simulated user: {simulator}):",
        *lines,
        f"Stored {len(traces)} conversation trace(s).",
        *judged,
        "Next: open the conversations with `calibrated-agent-evaluation label --serve` to review "
        "them.",
    ]
    return StageOutcome(stage="dataset", exit_code=EXIT_OK, summary="\n".join(summary))


def _check_executor_flags(args: argparse.Namespace, executor_flag: str) -> None:
    """Refuse the flags the chosen executor would ignore, before anything else runs."""
    if executor_flag == "agentcore":
        for dest, flag in (("agent", "--agent"), ("agent_model", "--agent-model")):
            if getattr(args, dest, None) is not None:
                raise UsageError(
                    f"{flag} doesn't apply to --executor agentcore: the agent under test is the "
                    f"runtime --runtime-arn names, already deployed with its model. Drop {flag}"
                )
        from calibrated_agent_evaluation.cli.agentcore_route import (
            require_agentcore_extra,
            require_agentcore_options,
        )

        require_agentcore_options(args)
        require_agentcore_extra(_AGENTCORE_NAMES)
        return
    stray = [flag for dest, flag in _RUNTIME_FLAGS.items() if getattr(args, dest, None)]
    if stray:
        raise UsageError(
            f"{', '.join(stray)} {'is' if len(stray) == 1 else 'are'} read only with --executor "
            "agentcore; pass --executor agentcore to run the scenarios against an AgentCore "
            "Runtime agent, or drop them"
        )
    if not getattr(args, "agent", None):
        raise UsageError(
            "`calibrated-agent-evaluation dataset run --executor strands` needs --agent "
            "module:callable (the agent "
            "under test is yours; e.g. --agent corpus.agents.orderbot_strands:build)"
        )


def _strands_executor(args: argparse.Namespace, cassette: str | None) -> tuple[Any, Any]:
    from calibrated_agent_evaluation.cli.stages.generate import (
        _build_executor_and_builder,
        _resolve_agent,
    )

    agent = _resolve_agent(args.agent)
    executor, builder = _build_executor_and_builder(
        "strands", agent, model=getattr(args, "agent_model", None), cassette=cassette
    )

    def task_for(item: Any, history: list[dict[str, Any]], _turn: Any) -> Any:
        return builder.build_conversation_turn(item, history)

    return executor, task_for


def _agentcore_executor(
    args: argparse.Namespace, mode: Any, cassette: str | None, budget: Any
) -> tuple[Any, Any]:
    """The Runtime executor, built after every check that needs no AWS call has passed."""
    import calibrated_agent_evaluation.agentcore as agentcore
    from calibrated_agent_evaluation.cli.agentcore_route import manifest_dir, target_fields
    from calibrated_agent_evaluation.execute.cassette import Mode

    replay = mode is Mode.REPLAY
    target = agentcore.RuntimeTarget(**target_fields(args))
    options: dict[str, Any] = {"log_dir": manifest_dir(args), "frame": "discovery"}
    if cassette:
        # Without a cassette the executor gets no mode, and every turn runs live.
        options.update(cassette_path=Path(cassette), cassette_mode=mode)
    read_timeout = getattr(args, "read_timeout", None)
    if read_timeout is not None:
        options["read_timeout_seconds"] = read_timeout
    builder = agentcore.AgentCoreTaskBuilder()
    if replay:
        executor = agentcore.AgentCoreRuntimeExecutor(
            target, profile=getattr(args, "profile", None), **options
        )
        return executor, builder.build_conversation_turn
    from calibrated_agent_evaluation.cli.stages.generate import _span_reader_class

    reader_class = _span_reader_class()
    clients = agentcore.AwsClients.from_profile(
        region=target.region, profile=getattr(args, "profile", None), handlers=[budget]
    )
    executor = agentcore.AgentCoreRuntimeExecutor(
        target, span_reader=reader_class(clients), clients=clients, **options
    )
    return _ClosingWith(executor, clients), builder.build_conversation_turn


def _check_budget(most_billed: int) -> Any:
    """The billed-call budget, refused when it can't cover every billed call the run can make.

    Raises:
        BilledCallBudgetExceeded: The budget allows fewer calls than the run can make.
    """
    from calibrated_agent_evaluation.agentcore.errors import BilledCallBudgetExceeded
    from calibrated_agent_evaluation.cli.agentcore_route import billed_call_budget

    budget = billed_call_budget()
    if budget.remaining is not None and budget.remaining < most_billed:
        raise BilledCallBudgetExceeded(
            f"the run can make up to {most_billed} billed AgentCore call(s) (InvokeAgentRuntime "
            "for each possible turn, and one Evaluate per conversation when judging), and the "
            f"billed-call budget allows {budget.remaining}; nothing was sent. Raise "
            "CALIBRATED_AGENT_EVALUATION_AGENTCORE_MAX_BILLED_CALLS or run fewer scenarios"
        )
    return budget


def _judge_file(args: argparse.Namespace) -> Any:
    """The ``--judge`` file, read and checked with no AWS call, or ``None`` without ``--judge``.

    Raises:
        CliError: The file is not an ``agentcore`` judge file of level ``session``, or the
            ``[agentcore]`` extra is missing.
    """
    path = getattr(args, "judge", None)
    if not path:
        return None
    from calibrated_agent_evaluation.cli.agentcore_route import require_agentcore_extra

    require_agentcore_extra(("AwsClients",))
    from calibrated_agent_evaluation.agentcore.errors import AgentCoreError
    from calibrated_agent_evaluation.agentcore.judge import load_agentcore_judge_file

    try:
        judge_file = load_agentcore_judge_file(path)
    except AgentCoreError as exc:
        raise CliError(f"--judge {path}: {exc}") from exc
    if judge_file.level != "session":
        raise CliError(
            f"--judge {path} is a {judge_file.level} judge; `dataset run` judges whole "
            'conversations, so it needs a judge file with "level": "session" (a SESSION '
            "evaluator such as Builtin.GoalSuccessRate)"
        )
    return judge_file


def _judge(
    args: argparse.Namespace,
    judge_file: Any,
    traces: list[Any],
    budget: Any,
    backend: Any,
    executor_flag: str,
) -> list[str]:
    """Judge every stored conversation; store the verdicts once every result has mapped."""
    from calibrated_agent_evaluation._model import replace
    from calibrated_agent_evaluation.agentcore import AwsClients
    from calibrated_agent_evaluation.agentcore.evaluate import run_agentcore_judge
    from calibrated_agent_evaluation.agentcore.judge import build_agentcore_judge

    region = (
        args.runtime_arn.split(":")[3]
        if executor_flag == "agentcore" and getattr(args, "runtime_arn", None)
        else getattr(args, "region", None)
    )
    handlers = [budget] if budget is not None else []
    estimate_text: list[str] = []

    def confirm(estimate: Any) -> bool:
        # --yes was given before anything ran; the estimate is shown, not asked about again.
        estimate_text.append(estimate.describe())
        return True

    with AwsClients.from_profile(
        region=region, profile=getattr(args, "profile", None), handlers=handlers
    ) as clients:
        judge = build_agentcore_judge(judge_file, clients=clients)
        labels = run_agentcore_judge(judge, traces, clients=clients, confirm=confirm)
    verdicts = [replace(label, failure_mode=judge_file.failure_mode) for label in labels]
    for label in verdicts:
        backend.put_score(label)
    by_trace = {label.trace_id: label for label in verdicts}
    lines = [
        f"Judge {judge_file.judge_id} ({judge_file.evaluator_id}, session): "
        + " ".join(estimate_text)
    ]
    for trace in traces:
        label = by_trace.get(trace.trace_id)
        if label is None:
            raise CliError(
                f"the judge returned no verdict for conversation {trace.trace_id!r}; its other "
                "verdicts are stored"
            )
        lines.append(f"  {trace.input['scenario_id']}   {label.outcome}")
    bare = [t.input["scenario_id"] for t in traces if not t.input.get("assertions")]
    if bare:
        lines.append(
            f"{len(bare)} of {len(traces)} scenario(s) had no assertions and were judged on their "
            f"goal alone: {', '.join(bare)}."
        )
    lines.append(f"Stored {len(verdicts)} advisory llm-judge label(s).")
    return lines


class _ClosingWith:
    """The executor, closed before the clients it was given (which the CLI built and owns)."""

    def __init__(self, executor: Any, clients: Any) -> None:
        self.executor = executor
        self._clients = clients

    def __enter__(self) -> Any:
        self._clients.__enter__()
        try:
            return self.executor.__enter__()
        except BaseException:
            # The executor never opened, so only the clients are closed.
            self._clients.__exit__(None, None, None)
            raise

    def __exit__(self, *exc: Any) -> None:
        try:
            self.executor.__exit__(*exc)
        finally:
            self._clients.__exit__(*exc)


def _print_plan(
    scenarios: int, most_turns: int, describe: str, simulator: str, judge_file: Any
) -> None:
    judging = (
        ""
        if judge_file is None
        else f" Then up to {scenarios} billed Evaluate call(s), one per conversation, with "
        f"judge {judge_file.judge_id} ({judge_file.evaluator_id}); the cost estimate is printed "
        "when the judging starts."
    )
    print(
        f"calibrated-agent-evaluation dataset run: plan (nothing has run): {scenarios} "
        f"scenario(s), up to {most_turns} "
        f"agent turn(s) on {describe}, and up to {max(most_turns - scenarios, 0)} simulated-user "
        f"model call(s) through the {simulator} simulator. A strands agent replays the whole "
        f"history on each turn, so its model tokens grow with the turn count.{judging}",
        file=sys.stderr,
        flush=True,
    )


def _converse_all(
    args: argparse.Namespace,
    scenarios: list[Any],
    slice_: Any,
    executor: Any,
    task_for: Callable[[Any, list[dict[str, Any]], Any], Any],
    simulator: Simulator,
    mode: Any,
    cassette: str | None,
) -> tuple[list[Any], list[str]]:
    """Hold every conversation, then return their traces and one summary line each."""
    from calibrated_agent_evaluation.cli.conversation_route import user_factory
    from calibrated_agent_evaluation.synth.borrow.simulator import actor_profile_from_dict
    from calibrated_agent_evaluation.synth.conversation import (
        assemble_conversation_trace,
        run_simulated_conversation,
    )
    from calibrated_agent_evaluation.synth.conversation_cassette import (
        UserCassette,
        actor_cassette_path,
    )
    from calibrated_agent_evaluation.synth.types import SynthInput

    published = slice_.published
    dataset = (
        None
        if published is None
        else {"dataset_id": published.dataset_id, "version": published.version}
    )
    users = None if cassette is None else UserCassette(actor_cassette_path(cassette), mode)
    traces: list[Any] = []
    lines: list[str] = []
    with executor as runner:
        end = getattr(runner, "end_conversation", None)
        for scenario in scenarios:
            opening = _opening(scenario.input)
            profile = actor_profile_from_dict(
                {
                    "traits": dict(scenario.actor_profile.traits),
                    "context": scenario.actor_profile.context,
                    "actor_goal": scenario.actor_profile.goal,
                }
            )
            build_user = user_factory(
                simulator,
                max_turns=scenario.max_turns,
                model=getattr(args, "simulator_model", None),
            )

            def build(
                o: str = opening, p: Any = profile, s: Any = scenario, b: Any = build_user
            ) -> Any:
                return b(o, p, s.scenario_description)

            user = build() if users is None else users.user_for(scenario.scenario_id, build)
            frame = scenario.calibrated_agent_evaluation.frame or "discovery"
            item = SynthInput(
                text=opening,
                cell=dict(scenario.calibrated_agent_evaluation.taxonomy_cell),
                frame=frame,
            )

            def agent_turn(turn: Any, history: list[dict[str, Any]], i: Any = item) -> Any:
                produced = runner.run(task_for(i, history, turn))
                if len(produced) != 1:
                    raise CliError(
                        f"the agent's turn produced {len(produced)} traces, and a conversation "
                        "turn must produce exactly one"
                    )
                return produced[0]

            sid = scenario.scenario_id
            run = run_simulated_conversation(
                user,
                agent_turn,
                conversation_id=sid,
                opening=opening,
                max_turns=scenario.max_turns,
                on_end=None if end is None else (lambda c=sid: end(c)),
            )
            traces.append(
                assemble_conversation_trace(
                    run,
                    scenario_id=sid,
                    goal=scenario.actor_profile.goal,
                    assertions=scenario.assertions,
                    simulator=simulator,
                    frame=frame,
                    taxonomy_cell=scenario.calibrated_agent_evaluation.taxonomy_cell,
                    dataset=dataset,
                )
            )
            turns = traces[-1].input["turns"]
            lines.append(f"  {sid}   {turns} agent turn(s)   {run.stop_reason}")
    if users is not None:
        users.save()
    return traces, lines


def _opening(value: str | dict[str, Any]) -> str:
    """A scenario's input as the opening message: its text, or its JSON object as compact text."""
    if isinstance(value, str):
        return value
    return json.dumps(value, sort_keys=True, ensure_ascii=False)
