# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""``probe --executor agentcore --policy-cases PATH``: score cases from Policy decisions.

Each case's prompt is sent once per trial to the AgentCore Runtime agent, through the same SigV4
probe adapter as ``calibrated-agent-evaluation probe --executor agentcore``. After the last trial
the Gateway's Policy decisions on the agent's tool calls are read from ``aws/spans`` by each trial's
trace ID, and each case is scored from them with Clopper-Pearson bounds. The findings are written as
a ``probe`` record, then each trial's probed trace and, for each trial in which the engine denied a
call, one ``policy-engine`` FAIL label go to the trace store ``--backend`` selects.

Every refusal that needs no AWS call comes first, then the trace store check, the plan and
``--yes``. The decision store's read check runs before the first billed trial.
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from typing import TYPE_CHECKING, Any

from calibrated_agent_evaluation.cli.agentcore_route import (
    billed_call_budget,
    cassette_mode_from_env,
    check_trace_store,
    manifest_dir,
    manifest_report,
    require_agentcore_extra,
    require_agentcore_options,
    target_fields,
)
from calibrated_agent_evaluation.cli.dispatch import EXIT_OK, StageOutcome
from calibrated_agent_evaluation.cli.errors import CliError, UsageError

if TYPE_CHECKING:
    from calibrated_agent_evaluation.probe.record import ProbeFinding

_POLICY_NAMES = (
    "RuntimeTarget",
    "AwsClients",
    "RuntimeProbeAdapter",
    "AgentCorePolicyDecisionSource",
    "agent_reported_error_trial",
)


def policy_battery_through_agentcore(args: argparse.Namespace) -> StageOutcome:
    """Run the policy case file against the runtime, and write the findings as a probe record."""
    from calibrated_agent_evaluation.agentcore.endpoint_label import (
        DEFAULT_QUALIFIER,
        endpoint_label,
    )
    from calibrated_agent_evaluation.cli.backends import open_backend
    from calibrated_agent_evaluation.execute.cassette import MODE_ENV_VAR, Mode
    from calibrated_agent_evaluation.probe.authorize import require_authorized
    from calibrated_agent_evaluation.probe.policy import load_policy_cases, policy_labels
    from calibrated_agent_evaluation.probe.policy_battery import run_policy_battery

    path = args.policy_cases
    if getattr(args, "prompts", None):
        raise UsageError(
            "--prompt doesn't apply with --policy-cases: the case file supplies every prompt, one "
            f"per case and trial. Drop --prompt, or move the prompt into a case of {path}"
        )
    require_agentcore_options(args)
    authorized = bool(getattr(args, "authorized", False))
    qualifier = getattr(args, "qualifier", None) or DEFAULT_QUALIFIER
    require_authorized(endpoint_label(args.runtime_arn, qualifier), authorized)
    cases = load_policy_cases(path)
    if cassette_mode_from_env() is Mode.REPLAY:
        raise CliError(
            f"{MODE_ENV_VAR} is replay, but a policy battery has no replay; nothing was sent. "
            f"Unset {MODE_ENV_VAR}, or set it to auto or record"
        )
    require_agentcore_extra(_POLICY_NAMES)
    import calibrated_agent_evaluation.agentcore as agentcore
    from calibrated_agent_evaluation.agentcore.policy_spans import settle_deadline_for
    from calibrated_agent_evaluation.agentcore.run_manifest import manifest_path

    target = agentcore.RuntimeTarget(**target_fields(args))
    log_dir = manifest_dir(args)
    manifest = manifest_path(log_dir)
    budget = billed_call_budget()
    adapter_options: dict[str, Any] = {}
    read_timeout = getattr(args, "read_timeout", None)
    if read_timeout is not None:
        adapter_options["read_timeout_seconds"] = read_timeout
    trials = sum(case.trials for case in cases.cases)
    settle_bound = settle_deadline_for(trials)
    with open_backend(args) as backend:
        # Checked before the plan: the traces and labels are stored only after the last trial,
        # so a store that can't take them would lose every billed call.
        check_trace_store(backend)
        store = "Langfuse" if getattr(args, "backend", None) == "langfuse" else "local root"
        _print_plan(
            target, manifest, cases=len(cases.cases), trials=trials, store=store, wait=settle_bound
        )
        if not getattr(args, "yes", False):
            raise CliError(
                "nothing was sent: this battery makes the billed AWS calls listed above, so it "
                "needs --yes. Pass --yes to go ahead"
            )
        adapter = None
        try:
            with agentcore.AwsClients.from_profile(
                region=target.region, profile=getattr(args, "profile", None), handlers=[budget]
            ) as clients:
                adapter = agentcore.RuntimeProbeAdapter(
                    target, clients=clients, log_dir=log_dir, **adapter_options
                )
                with adapter:
                    live = adapter
                    source = agentcore.AgentCorePolicyDecisionSource(
                        clients,
                        run_records=lambda: live.run_records,
                        settle_deadline_seconds=settle_bound,
                    )
                    result = run_policy_battery(
                        adapter,
                        target=adapter.target_label,
                        authorized=authorized,
                        cases=cases,
                        decisions=source,
                        agent_error=agentcore.agent_reported_error_trial,
                    )
        except BaseException:
            if adapter is not None and adapter.run_records:
                print(
                    f"calibrated-agent-evaluation probe: {manifest_report(adapter, manifest)}",
                    file=sys.stderr,
                    flush=True,
                )
            raise
        from calibrated_agent_evaluation.cli.stages.probe import persist_findings

        try:
            record_id = persist_findings(args, adapter.target_label, result.findings)
        except BaseException:
            # The billed trials already ran: say where their run records are, as the other two
            # failure points do.
            print(
                f"calibrated-agent-evaluation probe: {manifest_report(adapter, manifest)}",
                file=sys.stderr,
                flush=True,
            )
            raise
        labels = policy_labels(result.findings)
        try:
            # Traces first: no label is written on a trace the store does not hold.
            for trace in result.traces:
                backend.put_trace(trace)
            for label in labels:
                backend.put_score(label)
        except BaseException:
            print(
                f"calibrated-agent-evaluation probe: probe record {record_id} is stored; "
                f"{manifest_report(adapter, manifest)}",
                file=sys.stderr,
                flush=True,
            )
            raise
    lines = policy_lines(
        adapter.target_label, result.findings, explain=bool(getattr(args, "explain", False))
    )
    lines.append(
        f"  policy-engine labels: {len(labels)} probed trace(s) labelled FAIL for a denied call "
        "(never gold: a human label is needed before any calibration uses one)"
    )
    lines.append(manifest_report(adapter, manifest))
    lines.append(f"probe record {record_id}")
    return StageOutcome(stage="probe", exit_code=EXIT_OK, summary="\n".join(lines))


def _print_plan(
    target: Any, manifest: Any, *, cases: int, trials: int, store: str, wait: float
) -> None:
    from calibrated_agent_evaluation.agentcore.policy_spans import (
        POLL_INTERVAL_SECONDS,
        SETTLE_BASE_SECONDS,
        SETTLE_PER_TRIAL_SECONDS,
    )

    bundle = (
        f", configuration bundle {target.config_bundle_arn} at version "
        f"{target.config_bundle_version}"
        if target.config_bundle_arn is not None
        else ""
    )
    print(
        f"calibrated-agent-evaluation probe: this run can call AWS. It runs {cases} policy "
        f"case(s), {trials} trial(s), "
        "against the AgentCore "
        f"runtime {target.runtime_arn} at qualifier {target.qualifier}{bundle}: each trial is one "
        "billed invocation, retried with the same session when it is busy or throttled, after "
        "read-only calls that check the runtime. Then read-only Logs Insights and FilterLogEvents "
        "reads of aws/spans fetch each trial's Policy decisions. After the last trial they are "
        f"read every {POLL_INTERVAL_SECONDS:g} s until, for each trial, the span index holds every "
        f"span the log store has on two reads in a row, for at most {wait:g} s "
        f"({SETTLE_BASE_SECONDS:g} s plus {SETTLE_PER_TRIAL_SECONDS:g} s per trial); a trial whose "
        "spans are still arriving then refuses the battery. The findings go to the probe record. "
        "Trace store: "
        f"{store}; each trial's reply, or the error code of a trial whose agent reported an error, "
        "is stored there as a probed trace, and each trace with a denied call gets one "
        "policy-engine label. Each run's record goes to the run manifest "
        f"{manifest}; the first trial "
        "that fails for any reason but an error the agent reported stops the battery, and no "
        "probe record is written.",
        file=sys.stderr,
        flush=True,
    )


def _pct(bound: str) -> str:
    return f"{float(bound) * 100:.1f}%"


_VERDICT = {
    "observed_allow": "vulnerable",
    "bound_clears": "not vulnerable, and the bound clears the tolerated rate",
    "too_few_trials": "none ran, but too few trials to bound the rate that low",
}


def policy_lines(
    label: str, findings: Sequence[ProbeFinding], *, explain: bool = False
) -> list[str]:
    """The lines a policy battery prints: header, framing, then each case's lines."""
    payloads = [f.payload for f in findings]
    trials = sum(int(p["k"]) for p in payloads)
    lines = [
        f"probed {label} with {len(payloads)} policy case(s), {trials} trial(s)",
        "  verdicts come from the Policy engine's recorded decisions compared with the case file; "
        "no judge scored these trials",
        '  each "at most" is the one-sided 95% upper bound on that rate over the case\'s scored '
        "trials (trials with at least one Policy decision)",
    ]
    for finding, p in zip(findings, payloads, strict=True):
        errors = [e["error_code"] for e in p["trials"] if e.get("error_code")]
        error_note = (
            f"; {len(errors)} of {p['k']} trial(s) ended in an error the agent reported "
            f"({', '.join(sorted(set(errors)))}), and their decisions are counted like any other "
            "trial's"
            if errors
            else ""
        )
        if p["verdict_reason"] == "no_decision":
            lines.append(
                f"  {p['case_id']} (no decision): none of {p['k']} trial(s) held a Policy decision "
                "on a tool call, so no call was scored; not ruled out (the agent called no tool, "
                "the engine hid the tool from tools/list, or trace context did not reach the "
                f"Gateway){error_note}"
            )
            continue
        b = p["bounds"]
        c = p["counts"]
        mode = "log-only" if p["mode"] == "log_only" else p["mode"]
        verdict = _VERDICT.get(p["verdict_reason"])
        if p["verdict_reason"] == "unmatched":
            verdict = "no decision named a tool this case lists; the decisions named " + ", ".join(
                p["unlabelled_tools"]
            )
        tail = ""
        if p["unscored"]:
            tail += (
                f"; {p['unscored']} of {p['k']} trial(s) held no Policy decision on a tool call "
                "and are not scored"
            )
        if c["unlabelled_decisions"]:
            tail += f"; {c['unlabelled_decisions']} unlabelled decision(s) not counted"
        lines.append(
            f"  {p['case_id']} ({mode}): {c['observed_allow']} of {p['scored']} scored trial(s) "
            f"ran a call the case expects denied, at most {_pct(b['observed_allow']['ub95'])}; "
            f"{verdict}{tail}{error_note}"
        )
        ran = " (log-only: the action actually ran)" if p["mode"] == "log_only" else ""
        lines.append(
            f"    of the {p['scored']} scored trial(s): {c['gap']} policy gap, at most "
            f"{_pct(b['gap']['ub95'])}; {c['denied']} denied by the policy{ran}, at most "
            f"{_pct(b['denied']['ub95'])}; {c['over_deny']} over-deny, at most "
            f"{_pct(b['over_deny']['ub95'])}"
        )
        for policy, flip in p["flips"].items():
            lower = (
                " (lower bounds: the engine reported an incomplete log-only evaluation)"
                if flip["lower_bound"]
                else ""
            )
            lines.append(
                f"    log-only policy {policy} would flip a decision in {flip['trials']} of "
                f"{p['scored']} scored trial(s), at most {_pct(flip['bounds']['trials']['ub95'])}, "
                f"and close a gap in {flip['closes_gap']}, at most "
                f"{_pct(flip['bounds']['closes_gap']['ub95'])}{lower}"
            )
        if explain:
            detail = ", ".join(
                f"{name} {c[name]} ci95 {b[name]['ci95']} ub95 {b[name]['ub95']}"
                for name in ("observed_allow", "gap", "denied", "over_deny")
            )
            lines.append(
                f"    explain: {detail}; tolerated rate {p['max_break_rate']}; endpoint "
                f"{finding.endpoint_fingerprint}"
            )
    return lines
