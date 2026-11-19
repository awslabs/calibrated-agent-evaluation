# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""``policy-engine`` labels: one FAIL per trial whose agent attempted a call the engine denied.

The recorded batteries come from the spans a real Gateway wrote (``test_probe_policy_battery``);
the hand-written findings cover what the recordings don't hold (a log-only deny, two denies in one
trial). The label kind is never gold and no verdict source or clustering reads it.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, cast

import pytest

pytest.importorskip("boto3")

from calibrated_agent_evaluation.agentcore.policy_spans import agent_reported_error_trial
from calibrated_agent_evaluation.agentcore.runtime_errors import AgentReportedError
from calibrated_agent_evaluation.backends.langfuse import LangfuseBackend, LangfuseLike
from calibrated_agent_evaluation.backends.local import LocalBackend
from calibrated_agent_evaluation.calibrate.errors import UnratifiedGoldError
from calibrated_agent_evaluation.calibrate.pairing import pair_observations
from calibrated_agent_evaluation.calibrate.record import LabelPair
from calibrated_agent_evaluation.cli.pairing import SOURCE_KINDS, verdicts_for
from calibrated_agent_evaluation.dumpload.io import dump, load
from calibrated_agent_evaluation.judges.errors import SpanGoldError
from calibrated_agent_evaluation.judges.levels import assert_span_gold_human
from calibrated_agent_evaluation.probe.policy import (
    GatewayMode,
    PolicyCase,
    PolicyDecision,
    TrialRef,
    case_finding,
    policy_labels,
)
from calibrated_agent_evaluation.schema.types import Label
from test_backends_langfuse import StubLangfuseClient
from test_probe_policy_battery import _FEEDS, _Adapter, _run

_RAW = Path(__file__).resolve().parents[1] / "fixtures" / "raw" / "agentcore" / "policy"
_ORACLE = json.loads(
    (
        Path(__file__).resolve().parents[1] / "fixtures/expected/probe/policy_decisions.json"
    ).read_text()
)


def _denied_traces(battery: str) -> set[str]:
    """The recorded traces the run's own probe log says the engine denied a call in."""
    traces = [t for ids in _FEEDS[battery].values() for t in ids]
    return {f"probe-fp-{t}" for t in traces if any(d["decision"] == "deny" for d in _ORACLE[t])}


def _labels(battery: str) -> tuple[Label, ...]:
    return policy_labels(_run(battery).findings)


def test_each_trial_with_a_denied_call_gets_one_fail_label() -> None:
    labels = _labels("phase2")
    assert {lb.trace_id for lb in labels} == _denied_traces("phase2")
    assert len(labels) == 6
    assert {(lb.kind, lb.outcome, lb.span_id) for lb in labels} == {("policy-engine", "FAIL", None)}


def test_the_recorded_batteries_label_the_traces_their_probe_log_denied() -> None:
    assert {lb.trace_id for lb in _labels("phase3")} == _denied_traces("phase3")
    assert len(_denied_traces("phase3")) == 3


def test_a_trial_without_a_deny_gets_no_label() -> None:
    labelled = {lb.trace_id for lb in _labels("phase2")}
    small = {f"probe-fp-{t}" for t in _FEEDS["phase2"]["read-small"]}
    assert labelled.isdisjoint(small)


def test_an_over_deny_flags_its_trace_and_says_the_case_expected_allow() -> None:
    bulk = f"probe-fp-{_FEEDS['phase2']['read-bulk'][0]}"
    [label] = [lb for lb in _labels("phase2") if lb.trace_id == bulk]
    assert "the case file expected this call allowed" in label.critique
    assert "calibrated-agent-evaluation-f040-rows___read_rows under enforce" in label.critique
    assert label.critique.endswith("it is not a human verdict.")


def test_a_label_names_the_engine_that_denied_the_call() -> None:
    label = _labels("phase2")[0]
    assert label.rater_id is not None
    assert label.rater_id.startswith("arn:aws:bedrock-agentcore:")


def test_a_case_with_no_scored_trial_gets_no_label() -> None:
    labelled = {lb.trace_id for lb in _labels("silent")}
    silent_delete = {f"probe-fp-{t}" for t in _FEEDS["silent"]["delete"]}
    assert labelled.isdisjoint(silent_delete)


def test_no_pass_label_is_ever_written() -> None:
    for battery in ("phase2", "phase3", "silent", "partial"):
        assert {lb.outcome for lb in _labels(battery)} <= {"FAIL"}


def test_an_agent_error_trial_with_a_deny_gets_one_label_naming_its_error_code() -> None:
    feed = _FEEDS["phase2"]
    failed = f"probe-fp-{feed['delete'][1]}"
    error = AgentReportedError("x", error_code="ToolDenied", trace_id=failed)
    result = _run(
        "phase2",
        adapter=_Adapter(feed, fail={("delete", 2): error}),
        agent_error=agent_reported_error_trial,
    )
    [label] = [lb for lb in policy_labels(result.findings) if lb.trace_id == failed]
    assert "The agent's reply reported an error (ToolDenied)" in label.critique
    [trace] = [t for t in result.traces if t.trace_id == failed]
    assert trace.output == {"error_code": "ToolDenied"}


# --- hand-written: what the recordings don't hold ----------------------------------------------


def _finding(*decisions: PolicyDecision) -> Any:
    case = PolicyCase(
        case_id="c", prompt="p", trials=1, expect={"t___read": "allow", "t___delete": "deny"}
    )
    trial = TrialRef(
        case_id="c",
        trial=1,
        probe_trace_id="probe-fp-1",
        started_at="s",
        ended_at="e",
        w3c_trace_id="1" * 32,
    )
    return case_finding(case, [trial], {"probe-fp-1": list(decisions)}, endpoint_fingerprint="fp")


def _deny(tool: str, span: str, mode: GatewayMode = "enforce") -> PolicyDecision:
    return PolicyDecision(
        trace_id="1" * 32,
        span_id=span,
        tool=tool,
        decision="deny",
        mode=mode,
        determining_policies=("forbid-1",),
    )


def test_a_log_only_denial_label_says_the_action_actually_ran() -> None:
    [label] = policy_labels([_finding(_deny("t___delete", "s1", "log_only"))])
    assert "t___delete under log-only" in label.critique
    assert "under log-only mode the action actually ran" in label.critique


def test_a_trial_with_two_denied_calls_gets_one_label_naming_both_spans() -> None:
    [label] = policy_labels([_finding(_deny("t___delete", "s1"), _deny("t___read", "s2"))])
    assert "denied 2 tool call(s)" in label.critique
    assert "span s1" in label.critique
    assert "span s2" in label.critique


# --- the kind is stored like any label, and nothing reads it as gold or a verdict ---------------


def _policy_label() -> Label:
    return _labels("phase2")[0]


def test_a_policy_engine_label_round_trips_through_the_local_and_langfuse_backends(
    tmp_path: Path,
) -> None:
    result = _run("phase2")
    label = _policy_label()
    [trace] = [t for t in result.traces if t.trace_id == label.trace_id]
    for backend in (
        LocalBackend(tmp_path / "store"),
        LangfuseBackend(cast("LangfuseLike", StubLangfuseClient())),
    ):
        backend.put_trace(trace)
        backend.put_score(label)
        assert backend.get_scores(label.trace_id) == [label]


def test_a_policy_engine_label_survives_dump_and_load(tmp_path: Path) -> None:
    result = _run("phase2")
    label = _policy_label()
    source = LocalBackend(tmp_path / "a")
    source.put_trace(next(t for t in result.traces if t.trace_id == label.trace_id))
    source.put_score(label)
    dump(source, tmp_path / "dump")
    target = LocalBackend(tmp_path / "b")
    load(target, tmp_path / "dump")
    assert target.get_scores(label.trace_id) == [label]


def test_policy_engine_gold_is_refused() -> None:
    label = _policy_label()
    with pytest.raises(UnratifiedGoldError):
        pair_observations([LabelPair(gold=label, verdict=label)])


def test_span_gold_of_kind_policy_engine_is_refused() -> None:
    span_label = _policy_label().model_copy(update={"span_id": "s1"})
    with pytest.raises(SpanGoldError):
        assert_span_gold_human([span_label])


def test_no_run_source_selects_policy_engine_labels() -> None:
    assert all("policy-engine" not in kinds for kinds in SOURCE_KINDS.values())
    # The label names the very mode asked for, so only its kind can keep it out of a verdict.
    label = _policy_label().model_copy(update={"failure_mode": "any-mode"})
    for source in SOURCE_KINDS:
        assert verdicts_for([label], "any-mode", source)[0] is None


def test_clustering_counts_human_fail_labels_and_never_policy_engine_ones(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    from calibrated_agent_evaluation.cli import main
    from calibrated_agent_evaluation.cli.dispatch import EXIT_DOMAIN_ERROR, EXIT_OK

    monkeypatch.delenv("CALIBRATED_AGENT_EVALUATION_LOCAL_ROOT", raising=False)
    result = _run("phase2")
    labels = policy_labels(result.findings)
    root = tmp_path / "store"
    with LocalBackend(root=root) as backend:
        for trace in result.traces:
            backend.put_trace(trace)
        for label in labels:
            backend.put_score(label)
    capsys.readouterr()

    assert main(["--root", str(root), "cluster"]) == EXIT_DOMAIN_ERROR
    assert "no human FAIL labels" in capsys.readouterr().err

    human = labels[0].model_copy(
        update={"kind": "human", "failure_mode": "read_too_many_rows", "rater_id": "owner"}
    )
    with LocalBackend(root=root) as backend:
        backend.put_score(human)

    assert main(["--root", str(root), "cluster"]) == EXIT_OK
    out = capsys.readouterr().out
    assert "1 human FAIL label(s)" in out
    assert "read_too_many_rows (1)" in out
