# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""``calibrated-agent-evaluation probe``: black-box probe an endpoint you don't own (Stage 0).

Probe is the one pipeline stage whose input is an **external endpoint**, not the trace store. It
runs the real probe engine (``calibrated_agent_evaluation.probe.probe``) against that target: it
discovers a lower-bound capability map, captures seed ``probed`` Traces, and persists them into the
backend so the downstream error-analysis stages (``define``/``label``/``cluster``/``judge``) can
operate on them; the front of the pipeline, live.

``--battery`` runs published safety benchmarks against the same endpoint instead of the seed
prompts: each benchmark sample is one trial, the benchmark's own scorer decides whether the target
broke, and the result is a one-sided 95% upper bound on the break rate (a black-box target has no
calibration set, so there is no honest point estimate). The findings are written as a ``probe``
record. A benchmark that scores an agent inside a tool environment it sets up is refused with the
reason, because a text-only probe cannot stand that environment up.

The flags only ``--executor agentcore`` reads (``--runtime-arn``, ``--payload-format`` and the
rest) are refused on the URL probe, naming ``--executor agentcore``, before the interactive setup,
the battery or anything else runs; they would otherwise be ignored.

With ``--executor agentcore`` the prompts go to an agent hosted on Amazon Bedrock AgentCore
Runtime, named by ``--runtime-arn``, through
``calibrated_agent_evaluation.agentcore.RuntimeProbeAdapter`` and
``calibrated_agent_evaluation.probe.probe_with``. Calls are signed with SigV4 from ``--profile``,
each prompt is one run in a session of its own, and no spans are read, so each reply is stored as a
``probed``, outcome-only trace. Before any AWS request, and without ever asking on a terminal, the
route refuses the URL probe's endpoint, auth, timeout and battery flags, a missing runtime ARN or
payload format, an empty ``--prompt``, a missing ``--authorized``,
``CALIBRATED_AGENT_EVALUATION_CASSETTE_MODE=replay`` (a probe keeps no cassette), a missing
``[agentcore]`` extra, an empty configuration bundle flag, an invalid billed-call budget variable
and a trace store that can't take the traces. It then prints what it will call and goes ahead only
with ``--yes``, under the billed-call budget. Each prompt's run record goes to the run manifest
under ``<root>/records/agentcore`` or ``--log-dir``. The first prompt that fails stops the probe,
and none of the replies are stored.

Two entry paths for the URL probe, one probe core:

* **flag / one-shot**: an ``--endpoint`` is supplied (scriptable); auth by env-var name.
* **interactive**: no ``--endpoint`` **and** stdin is a TTY → prompt for endpoint, auth type, the
  env var holding the credential (never the secret), an authorization confirmation, and seeds.

There is no corpus-reporter fallback (that was the original  gap). No ``--endpoint`` in
a non-interactive context (CI/piped) is a typed ``CliError``; never a hang.

Reuses the shipped ``calibrated_agent_evaluation.probe`` surface (never re-implementing it) and
imports it lazily so importing ``calibrated_agent_evaluation.cli`` never pulls the probe package at
module load. Auth secrets are always sourced from named environment variables (never a flag value,
never pasted at a prompt); the engine resolves them at send time and redacts them from every
artifact. A probe failure is a typed ``ProbeError`` the CLI boundary turns into one diagnostic line
+ exit 1, never a traceback.
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any

from calibrated_agent_evaluation.cli.agentcore_route import (
    billed_call_budget,
    cassette_mode_from_env,
    check_trace_store,
    flag_given,
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

if TYPE_CHECKING:
    from calibrated_agent_evaluation.agentcore import RuntimeProbeAdapter
    from calibrated_agent_evaluation.probe import ProbeReport

# How `probe` sends its prompts to an AgentCore Runtime agent, for a flag given without it.
_AGENTCORE_REMEDY = "pass --executor agentcore to send the prompts to an AgentCore Runtime agent"

# The URL probe's wait for each request when --timeout is not given.
_URL_TIMEOUT_SECONDS = 30.0


def probe_handler(args: argparse.Namespace) -> StageOutcome:
    """Run the black-box probe: flag/one-shot when ``--endpoint`` is given, else interactive.

    With ``--executor agentcore`` the prompts go to an AgentCore Runtime agent instead, through
    :func:`_probe_through_agentcore`; that route is never interactive.
    """
    # First, before the interactive setup, the battery or any request: each of these flags would
    # otherwise be ignored.
    refuse_agentcore_flags_without_executor(args, remedy=_AGENTCORE_REMEDY)
    if getattr(args, "executor", None) == "agentcore":
        return _probe_through_agentcore(args)
    if getattr(args, "policy_cases", None) is not None:
        raise UsageError(
            "--policy-cases applies only to --executor agentcore: the policy battery scores each "
            "case from an AgentCore Gateway's Policy decisions. Pass --executor agentcore and "
            "--runtime-arn, or drop --policy-cases"
        )

    from calibrated_agent_evaluation.cli.backends import open_backend
    from calibrated_agent_evaluation.probe import probe
    from calibrated_agent_evaluation.probe.authorize import require_authorized

    endpoint = getattr(args, "endpoint", None)
    if endpoint:
        config, seeds, authorized = _config_from_flags(args)
    else:
        config, seeds, authorized = _config_from_interactive()

    if getattr(args, "battery", None) is not None:
        return _run_battery(args, config, authorized)

    # Refuse before the status line, so an unauthorized run never reads as if it probed.
    require_authorized(config, authorized)

    # Status line before the blocking network call so the CLI doesn't look frozen (a reasoning
    # agent can take tens of seconds). Goes to stderr so stdout stays the machine-readable summary.
    n_seeds = len(seeds) if seeds is not None else "default"
    print(f"Probing {config.endpoint} with {n_seeds} prompt(s)…", file=sys.stderr, flush=True)

    # probe() refuses without authorized=True (AuthorizationError), routes ws/wss to the WS adapter
    # (ProbeMissingExtraError if [probe-ws] absent), and raises EndpointUnreachableError on a dead
    # target; all ProbeError subclasses, caught by the CLI boundary as exit-1 diagnostics.
    if seeds is None:
        report = probe(config, authorized=authorized)
    else:
        report = probe(config, authorized=authorized, seeds=seeds)

    # Persist the probed traces so the downstream stages can read them (probe CREATES traces).
    with open_backend(args) as backend:
        for trace in report.traces:
            backend.put_trace(trace)

    tools = list(report.capability_map.observed_tools)
    return StageOutcome(
        stage="probe",
        exit_code=EXIT_OK,
        summary=(
            f"probed {config.endpoint} -> {len(report.traces)} probed trace(s) stored; "
            f"observed tools (lower bound): {', '.join(tools) or 'none observed'}\n"
            f"{_limits_lines(report.capability_map.framing)}"
        ),
    )


_LIMIT_SENTENCES = {
    "tool_inventory_is_lower_bound": (
        "the tools seen are a lower bound: the agent may have tools its replies never showed"
    ),
    "refusal_is_not_inability": "a refusal is not proof the agent cannot do something",
    "verdict_bounded_by_judge_ceiling": (
        "a verdict on a reply is only as good as the judge that made it"
    ),
}


def _limits_lines(framing: Any) -> str:
    """The limits that apply to every probe result, one plain sentence per line."""
    flags = framing.model_dump()
    lines = [f"  - {text}" for name, text in _LIMIT_SENTENCES.items() if flags.get(name)]
    return "limits:\n" + "\n".join(lines)


# --- --executor agentcore -------------------------------------------------------------------

# The URL probe's flags, by parsed name, that an AgentCore probe has no use for.
_URL_PROBE_FLAGS: Mapping[str, str] = {
    "endpoint": "--endpoint",
    "auth_bearer_env": "--auth-bearer-env",
    "auth_api_key_env": "--auth-api-key-env",
    "auth_api_key_header": "--auth-api-key-header",
    "auth_header_env": "--auth-header-env",
    "auth_subprotocol_env": "--auth-subprotocol-env",
    "timeout": "--timeout",
}

# The battery's flags, by parsed name: the battery runs only through a URL probe's endpoint.
_BATTERY_FLAGS: Mapping[str, str] = {
    "battery": "--battery",
    "trials": "--trials",
    "allow_use_restricted": "--allow-use-restricted",
    "grader_model": "--grader-model",
}

_SIGV4_REASON = (
    "the runtime ARN names the target, calls are signed with SigV4 from --profile, and OAuth-only "
    "runtimes are not supported yet"
)

# The calibrated_agent_evaluation.agentcore names the AgentCore route uses, each checked for the
# [agentcore] extra.
_AGENTCORE_NAMES = ("RuntimeTarget", "AwsClients", "RuntimeProbeAdapter")


def _probe_through_agentcore(args: argparse.Namespace) -> StageOutcome:
    """Send the probe's prompts to an AgentCore Runtime agent, and store the replies as traces.

    Every refusal that needs no AWS call comes before any AWS request, and none of them asks
    anything on a terminal. The trace store is built and checked before the plan is printed, so
    a store that can't take the traces costs no billed call. Each prompt is one billed
    invocation, sent only with ``--yes``; the adapter's read-only preflight then refuses an
    OAuth-only runtime before the first one. The replies are stored only once every prompt has
    answered, and the first prompt that fails raises its own typed error unchanged. They are
    stored before the adapter and the clients are closed, so a failure to close them is raised
    with every reply already kept.
    """
    # Free of any AWS SDK, so the authorization refusal can name the target before the extra
    # check, by the label the adapter fingerprints.
    from calibrated_agent_evaluation.agentcore.endpoint_label import (
        DEFAULT_QUALIFIER,
        endpoint_label,
    )
    from calibrated_agent_evaluation.cli.backends import open_backend
    from calibrated_agent_evaluation.execute.cassette import MODE_ENV_VAR, Mode
    from calibrated_agent_evaluation.probe import DEFAULT_SEEDS, probe_with
    from calibrated_agent_evaluation.probe.authorize import require_authorized

    _refuse_url_probe_flags(args)
    if getattr(args, "policy_cases", None) is not None:
        from calibrated_agent_evaluation.cli.stages.probe_policy import (
            policy_battery_through_agentcore,
        )

        return policy_battery_through_agentcore(args)
    require_agentcore_options(args)
    prompts: list[str] = getattr(args, "prompts", None) or []
    _refuse_empty_prompts(prompts)
    authorized = bool(getattr(args, "authorized", False))
    qualifier = getattr(args, "qualifier", None) or DEFAULT_QUALIFIER
    require_authorized(endpoint_label(args.runtime_arn, qualifier), authorized)
    # Only replay is refused: auto and record have nothing to do, since a probe keeps no cassette.
    if cassette_mode_from_env() is Mode.REPLAY:
        raise CliError(
            f"{MODE_ENV_VAR} is replay, but probe has no replay; nothing was sent. An AgentCore "
            "probe always calls the runtime, and goes ahead only with --yes: unset "
            f"{MODE_ENV_VAR}, or set it to auto or record"
        )
    require_agentcore_extra(_AGENTCORE_NAMES)
    import calibrated_agent_evaluation.agentcore as agentcore
    from calibrated_agent_evaluation.agentcore.run_manifest import manifest_path

    target = agentcore.RuntimeTarget(**target_fields(args))
    log_dir = manifest_dir(args)
    # The run manifest's JSON file, the path printed: the adapter adds one record per prompt.
    manifest = manifest_path(log_dir)
    seeds: tuple[str, ...] = tuple(prompts) if prompts else DEFAULT_SEEDS
    adapter_options: dict[str, Any] = {}
    read_timeout = getattr(args, "read_timeout", None)
    if read_timeout is not None:
        # As given: the adapter refuses one that is not a positive, finite number of seconds.
        adapter_options["read_timeout_seconds"] = read_timeout

    # Known without any call, so refused before the store check, the plan and --yes: a probe that
    # would be refused anyway is not asked to go ahead.
    budget = billed_call_budget()
    # Held open until the traces are stored, and checked before the plan: the replies are stored
    # only once every prompt has answered, so a store that can't take them would lose every
    # billed call.
    with open_backend(args) as backend:
        check_trace_store(backend)
        print_plan("probe", target, manifest, count=len(seeds), noun="prompt", reads_spans=False)
        if not getattr(args, "yes", False):
            raise CliError(
                "nothing was sent: this probe makes the billed AWS calls listed above, so it "
                "needs --yes. Pass --yes to go ahead"
            )
        adapter: RuntimeProbeAdapter | None = None
        try:
            # The adapter's exit stops any session a prompt left open, through these clients.
            with agentcore.AwsClients.from_profile(
                region=target.region, profile=getattr(args, "profile", None), handlers=[budget]
            ) as clients:
                adapter = agentcore.RuntimeProbeAdapter(
                    target, clients=clients, log_dir=log_dir, **adapter_options
                )
                with adapter:
                    report = probe_with(
                        adapter,
                        # The label the adapter's trace IDs carry the fingerprint of.
                        target=adapter.target_label,
                        authorized=authorized,
                        seeds=seeds,
                        payload_format=target.payload_format,
                    )
                    # Stored before the adapter and the clients are closed. Every prompt's run
                    # has ended, and closing only retries a stop that failed and releases
                    # connections, so a failure there, raised as the blocks are left, finds every
                    # reply kept. Inside the try: a store that fails after every billed prompt
                    # still names the manifest that keeps their records.
                    for trace in report.traces:
                        backend.put_trace(trace)
        except BaseException:
            # The error reaches the user on its own line, so the manifest is named here, with any
            # record it could not take, once a prompt's run has started.
            if adapter is not None and adapter.run_records:
                print(
                    f"calibrated-agent-evaluation probe: {manifest_report(adapter, manifest)}",
                    file=sys.stderr,
                    flush=True,
                )
            raise
    return _agentcore_outcome(adapter.target_label, report, manifest_report(adapter, manifest))


def _agentcore_outcome(label: str, report: ProbeReport, manifest_line: str) -> StageOutcome:
    """The summary of an AgentCore probe whose every prompt answered and whose traces are stored.

    ``label`` is the probed target, ``<runtime ARN>@<qualifier>``.
    """
    tools = list(report.capability_map.observed_tools)
    return StageOutcome(
        stage="probe",
        exit_code=EXIT_OK,
        summary=(
            f"probed {label} through AgentCore Runtime -> "
            f"{len(report.traces)} probed trace(s) stored; observed tools (lower bound): "
            f"{', '.join(tools) or 'none observed'}\n"
            f"payload format: {report.detected_format} (as --payload-format named it, not "
            "detected)\n"
            f"{manifest_line}\n"
            f"{_limits_lines(report.capability_map.framing)}"
        ),
    )


def _refuse_empty_prompts(prompts: Sequence[str]) -> None:
    """Refuse an empty ``--prompt`` before the plan, so no prompt is billed for a probe that fails.

    The adapter refuses an empty prompt too, but only when its turn comes, once every prompt
    before it has been sent and billed; the probe would then store none of their replies.
    """
    empty = sum(1 for prompt in prompts if not prompt)
    if not empty:
        return
    which = (
        "--prompt is empty"
        if len(prompts) == 1
        else f"{empty} of the {len(prompts)} --prompt values {'is' if empty == 1 else 'are'} empty"
    )
    raise CliError(
        f"{which}, so nothing was sent: each prompt is a billed invocation, and an empty one gives "
        'the agent nothing to answer. Give each --prompt the text to send; --prompt "$VAR" is '
        "empty when VAR is unset"
    )


def _refuse_url_probe_flags(args: argparse.Namespace) -> None:
    """Refuse each URL probe flag an AgentCore probe would otherwise ignore, by name."""
    for dest, flag in _URL_PROBE_FLAGS.items():
        if flag_given(getattr(args, dest, None)):
            replaced = "; --read-timeout sets the invocation timeout" if dest == "timeout" else ""
            raise UsageError(
                f"{flag} doesn't apply to --executor agentcore: {_SIGV4_REASON}{replaced}. "
                f"Drop {flag}"
            )
    for dest, flag in _BATTERY_FLAGS.items():
        if flag_given(getattr(args, dest, None)):
            raise UsageError(
                f"{flag} doesn't apply to --executor agentcore: the benchmark battery runs only "
                "against a URL probe's --endpoint, and an AgentCore probe sends only its prompts. "
                f"Drop {flag}, or run the battery with --endpoint and no --executor"
            )


def _run_battery(args: argparse.Namespace, config: Any, authorized: bool) -> StageOutcome:
    """Run the benchmark battery against the endpoint and write the findings as a record."""
    import tempfile

    from calibrated_agent_evaluation.cli.config import probe_extra
    from calibrated_agent_evaluation.cli.errors import CliError, UsageError
    from calibrated_agent_evaluation.probe import run_battery
    from calibrated_agent_evaluation.probe.benchmarks import (
        TEXT_ONLY_BATTERY,
        requires_tool_environment,
        tool_environment_refusal,
    )

    requested = tuple(getattr(args, "battery", None) or ()) or TEXT_ONLY_BATTERY
    trials = getattr(args, "trials", None)
    if trials is None:
        raise UsageError(
            "calibrated-agent-evaluation probe --battery needs --trials K: one benchmark sample "
            "is one trial, and the "
            "upper bound on the break rate is only as tight as the trials you ran "
            "(0 breaks in 59 trials bounds it at 0.05)"
        )
    if trials < 1:
        raise UsageError(f"--trials needs at least one trial, got {trials}")
    use_restricted = bool(getattr(args, "allow_use_restricted", False))
    for name in requested:
        if requires_tool_environment(name):
            if not use_restricted and name == "agentharm":
                raise CliError(
                    "agentharm's dataset carries a use restriction; pass --allow-use-restricted "
                    "if your use is covered by it"
                )
            raise CliError(str(tool_environment_refusal(name)))
    for extra in ("inspect_ai", "inspect_evals"):
        found = probe_extra(extra)
        if not found.available:
            raise CliError(f"--battery needs the probing extra: {found.install_hint}")

    send = _endpoint_sender(config)
    grader_model = getattr(args, "grader_model", None)
    with tempfile.TemporaryDirectory(prefix="calibrated_agent_evaluation_battery_") as tmp:
        findings = run_battery(
            config,
            authorized=authorized,
            requested=requested,
            allow_use_restricted=use_restricted,
            k=trials,
            run_trials=_battery_runner(send, Path(tmp), grader_model),
        )
    record_id = persist_findings(args, config.endpoint, findings)
    return StageOutcome(
        stage="probe",
        exit_code=EXIT_OK,
        summary="\n".join(
            [
                f"probed {config.endpoint} with {len(findings)} benchmark(s), "
                f"{trials} trial(s) each (one sample = one trial); recorded {record_id}",
                *[_finding_line(finding) for finding in findings],
                *[f"  note: {note}" for finding in findings for note in finding.substitutions],
                "  the bound is the benchmark scorer's own verdict, uncorrected: there is no "
                "calibrated point rate for an endpoint you do not own",
            ]
        ),
        detail="\n".join(_finding_detail(finding) for finding in findings),
    )


def _finding_line(finding: Any) -> str:
    """One plain line per benchmark: what broke, and the bound on how often it would."""
    payload = finding.payload
    verdict = {
        "true": "broke",
        "false": "no break, and the bound clears the tolerated rate",
        "not_ruled_out": "no break observed, but too few trials to bound the rate that low",
    }[payload["vulnerable"]]
    bound = float(payload["break_rate_ub95"])
    return (
        f"  {payload['benchmark']}: {payload['breaks']} of {payload['k']} trial(s) broke; "
        f"{verdict}; break rate at most {bound:.1%} (95% upper bound)"
    )


def _finding_detail(finding: Any) -> str:
    """The statistics behind one finding, for ``--explain``."""
    payload = finding.payload
    return (
        f"  {payload['benchmark']} ({payload['layer']}): breaks={payload['breaks']}, "
        f"k={payload['k']}, 95% CI {payload['break_rate_ci95']}, "
        f"one-sided 95% upper bound {payload['break_rate_ub95']}, tolerated "
        f"{payload['max_break_rate']}, endpoint {finding.endpoint_fingerprint}"
    )


def _battery_runner(send: Any, log_dir: Path, grader_model: str | None) -> Any:
    """The battery's trial seam: run K samples of one benchmark through the endpoint."""
    from calibrated_agent_evaluation.cli.errors import CliError
    from calibrated_agent_evaluation.execute.errors import ExecutionError

    def _run(identifier: str, k: int) -> list[bool]:
        from calibrated_agent_evaluation.execute.probe_bridge import benchmark_trials

        try:
            return benchmark_trials(
                identifier,
                send,
                trials=k,
                log_dir=log_dir / identifier,
                grader_model=grader_model,
            )
        except ExecutionError as exc:
            raise CliError(str(exc)) from exc

    return _run


def _endpoint_sender(config: Any) -> Any:
    """A ``send(prompt) -> reply`` over the probe endpoint adapter."""
    from calibrated_agent_evaluation.probe import EndpointAdapter

    adapter = EndpointAdapter(config)

    def _send(prompt: str) -> str:
        return adapter.send(prompt).text

    return _send


def persist_findings(args: argparse.Namespace, endpoint: str, findings: tuple[Any, ...]) -> str:
    """Write a battery's findings as a ``probe`` record under the local root; return its ID."""
    import secrets
    from datetime import UTC, datetime

    from calibrated_agent_evaluation._provenance import capture_provenance
    from calibrated_agent_evaluation.cli.backends import local_root
    from calibrated_agent_evaluation.cli.records import append_records
    from calibrated_agent_evaluation.probe.fingerprint import endpoint_fingerprint
    from calibrated_agent_evaluation.probe.record import ProbeBatteryRecord

    stamp = datetime.now(tz=UTC).strftime("%Y%m%dT%H%M%SZ")
    probe_id = f"probe-{stamp}-{secrets.token_hex(3)}"
    record = ProbeBatteryRecord(
        probe_id=probe_id,
        endpoint_fingerprint=endpoint_fingerprint(endpoint),
        findings=tuple(findings),
        provenance=capture_provenance(),
    )
    append_records(local_root(args), "probe", [record])
    return probe_id


def _config_from_flags(args: argparse.Namespace):
    """Build (config, seeds, authorized) from the one-shot flags.

    ``--timeout`` parses to ``None`` when it is not given, and the URL probe then waits 30 seconds
    for each request.
    """
    from calibrated_agent_evaluation.probe import ConnectionConfig

    timeout = getattr(args, "timeout", None)
    config = ConnectionConfig(
        endpoint=args.endpoint,
        auth=_auth_from_args(args),
        timeout_seconds=_URL_TIMEOUT_SECONDS if timeout is None else timeout,
    )
    prompts = getattr(args, "prompts", None)
    seeds = tuple(prompts) if prompts else None
    return config, seeds, bool(getattr(args, "authorized", False))


def _config_from_interactive():
    """Prompt for the connection surface; refuse fast on a non-TTY.

    Best-practice secret handling: the credential is **referenced by environment-variable name**,
    never pasted into the terminal (a long token exceeds the terminal's canonical line limit and
    would freeze the paste, and pasting a secret is poor practice regardless). The engine resolves
    the value from the environment at send time and redacts it from every artifact. Empty required
    fields re-prompt rather than abort; Ctrl-D/Ctrl-C aborts cleanly with a typed error.
    """
    from calibrated_agent_evaluation.probe import ConnectionConfig

    if not sys.stdin.isatty():
        raise UsageError(
            "calibrated-agent-evaluation probe needs an --endpoint (and --authorized) when run "
            "non-interactively; "
            "e.g. calibrated-agent-evaluation probe --endpoint <url> --authorized "
            "--auth-bearer-env <ENV>"
        )

    try:
        endpoint = _prompt_nonempty("Endpoint URL")
        auth = _prompt_auth_interactive()
        confirm = input("Authorized to probe this endpoint? [y/N]: ").strip().lower()
        authorized = confirm in ("y", "yes")

        print("Enter seed prompts, one per line (blank line to start probing):")
        prompts: list[str] = []
        while True:
            line = input("> ").strip()
            if not line:
                break
            prompts.append(line)
    except (EOFError, KeyboardInterrupt):
        raise CliError("interactive probe setup aborted; nothing sent.") from None

    config = ConnectionConfig(endpoint=endpoint, auth=auth)
    seeds = tuple(prompts) if prompts else None
    return config, seeds, authorized


def _prompt_auth_interactive():
    """Prompt for the auth type and the env var holding the secret (never the secret itself)."""
    from calibrated_agent_evaluation.probe import ApiKeyAuth, BearerAuth, HeaderAuth

    print("Auth type; how the endpoint authenticates you:")
    print("  bearer   Authorization: Bearer <token>   (most common)")
    print("  api-key  a key in a named header")
    print("  header   a JSON object of raw headers")
    print("  none     no authentication")
    print("Your secret is read from an environment variable; never pasted here, logged, or stored.")
    choice = input("Auth type [bearer/api-key/header/none] (default bearer): ").strip().lower()
    choice = choice or "bearer"

    if choice == "none":
        return None
    if choice == "bearer":
        env = _prompt_nonempty("Env var holding the bearer token (set it first: export TOKEN=...)")
        return BearerAuth(token_env=env)
    if choice == "api-key":
        header = _prompt_nonempty("Header name for the API key (e.g. x-api-key)")
        env = _prompt_nonempty("Env var holding the API key")
        return ApiKeyAuth(header=header, key_env=env)
    if choice == "header":
        env = _prompt_nonempty("Env var holding a JSON object of header name -> value")
        return HeaderAuth(headers_env=env)
    raise CliError(f"unknown auth type {choice!r}; choose bearer, api-key, header, or none.")


def _prompt_nonempty(label: str, *, max_tries: int = 5) -> str:
    """Prompt for a required value, re-prompting on empty rather than aborting."""
    for _ in range(max_tries):
        value = input(f"{label}: ").strip()
        if value:
            return value
        print(f"{label} is required.")
    raise CliError(f"no {label.lower()} entered after {max_tries} attempts; aborting.")


def _auth_from_args(args: argparse.Namespace):
    """Build the probe ``Auth`` from the CLI flags, or ``None`` for an unauthenticated endpoint.

    At most one auth variant may be selected; combining them is a typed ``CliError`` rather than a
    silently-ignored flag.
    """
    from calibrated_agent_evaluation.probe import (
        ApiKeyAuth,
        BearerAuth,
        HeaderAuth,
        SubprotocolAuth,
    )

    bearer = getattr(args, "auth_bearer_env", None)
    api_key = getattr(args, "auth_api_key_env", None)
    api_key_header = getattr(args, "auth_api_key_header", None)
    header = getattr(args, "auth_header_env", None)
    subprotocol = getattr(args, "auth_subprotocol_env", None)

    selected = [x for x in (bearer, api_key, header, subprotocol) if x is not None]
    if len(selected) > 1:
        raise UsageError(
            "choose at most one auth variant (--auth-bearer-env / --auth-api-key-env / "
            "--auth-header-env / --auth-subprotocol-env)"
        )

    if bearer is not None:
        return BearerAuth(token_env=bearer)
    if api_key is not None:
        if not api_key_header:
            raise UsageError("--auth-api-key-env requires --auth-api-key-header (the header name)")
        return ApiKeyAuth(header=api_key_header, key_env=api_key)
    if header is not None:
        return HeaderAuth(headers_env=header)
    if subprotocol is not None:
        return SubprotocolAuth(token_env=subprotocol)
    return None
