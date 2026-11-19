# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""The policy battery: trials through an adapter, decisions from a source, and the battery checks.

The recorded batteries feed each case the spans a real Gateway wrote for that case's calls in the
spike, through a fake adapter and a fixture-backed source that runs the shipped extractor.
"""

from __future__ import annotations

import gzip
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import pytest

pytest.importorskip("boto3")

from calibrated_agent_evaluation.agentcore.policy_spans import (
    agent_reported_error_trial,
    extract_decisions,
)
from calibrated_agent_evaluation.agentcore.runtime_errors import (
    AgentReportedError,
    RuntimeInvocationError,
)
from calibrated_agent_evaluation.probe import (
    AuthorizationError,
    PolicyCaseError,
    PolicyDecisionsMissing,
    ProbeError,
    ProbeResponse,
)
from calibrated_agent_evaluation.probe.policy import (
    PolicyCaseFile,
    TrialDecisions,
    TrialRef,
    load_policy_cases,
)
from calibrated_agent_evaluation.probe.policy_battery import run_policy_battery

_RAW = Path(__file__).resolve().parents[1] / "fixtures" / "raw" / "agentcore" / "policy"
_FEEDS = json.loads((_RAW / "feeds.json").read_text())
_CASES = load_policy_cases(_RAW / "cases.json")
_TARGET = "arn:aws:bedrock-agentcore:us-west-2:111122223333:runtime/agent-AbCdEf1234@DEFAULT"


class _Adapter:
    """Answers each case's prompt with the next trace its feed gives, and names it."""

    def __init__(
        self,
        feed: Mapping[str, Sequence[str]],
        cases: PolicyCaseFile = _CASES,
        fail: Mapping[tuple[str, int], Exception] | None = None,
    ) -> None:
        self._by_prompt = {c.prompt: (c.case_id, list(feed[c.case_id])) for c in cases.cases}
        self._fail = fail or {}
        self._sent: dict[str, int] = {}
        self.sent: list[str] = []

    def send(self, prompt: str) -> ProbeResponse:
        case_id, traces = self._by_prompt[prompt]
        n = self._sent.get(case_id, 0)
        self._sent[case_id] = n + 1
        self.sent.append(prompt)
        failure = self._fail.get((case_id, n + 1))
        if failure is not None:
            raise failure
        return ProbeResponse(status=200, text="done", trace_id=f"probe-fp-{traces[n]}")


class _Source:
    """Answers from the recorded spans of each trial's trace, through the shipped extractor."""

    def __init__(
        self,
        *,
        readable: Exception | None = None,
        events: list[str] | None = None,
        rewrite: Any = None,
    ) -> None:
        self._readable = readable
        self.events = events if events is not None else []
        self._rewrite = rewrite

    def check_readable(self) -> None:
        self.events.append("check")
        if self._readable is not None:
            raise self._readable

    def read_decisions(self, trials: Sequence[TrialRef]) -> Mapping[str, TrialDecisions]:
        answers = {}
        for trial in trials:
            w3c = trial.probe_trace_id.rsplit("-", 1)[-1]
            spans = json.loads(gzip.decompress((_RAW / f"{w3c}.spans.json.gz").read_bytes()))
            answers[trial.probe_trace_id] = TrialDecisions(
                w3c_trace_id=w3c, decisions=extract_decisions(spans)
            )
        return self._rewrite(answers) if self._rewrite else answers


def _run(feed: str | Mapping[str, Sequence[str]], cases: PolicyCaseFile = _CASES, **kw: Any):
    adapter = kw.pop("adapter", None) or _Adapter(
        _FEEDS[feed] if isinstance(feed, str) else feed, cases
    )
    return run_policy_battery(
        adapter,
        target=_TARGET,
        authorized=True,
        cases=cases,
        decisions=kw.pop("decisions", None) or _Source(),
        **kw,
    )


def _payloads(result: Any) -> dict[str, dict[str, Any]]:
    return {f.payload["case_id"]: f.payload for f in result.findings}


def test_each_case_prompt_is_sent_once_per_trial_and_joined_by_trace_id() -> None:
    adapter = _Adapter(_FEEDS["phase2"])
    result = _run("phase2", adapter=adapter)
    assert len(adapter.sent) == 9
    assert [t.trial for t in result.trials] == [1, 2, 3] * 3
    for trial in result.trials:
        assert trial.probe_trace_id == f"probe-fp-{trial.w3c_trace_id}"


def test_the_recorded_enforce_battery_classes_each_case_as_the_spike_recorded() -> None:
    payloads = _payloads(_run("phase2"))
    assert payloads["read-small"]["counts"] == {
        "observed_allow": 0,
        "gap": 0,
        "denied": 0,
        "over_deny": 0,
        "unlabelled_decisions": 0,
    }
    assert payloads["read-bulk"]["counts"]["over_deny"] == 3
    assert payloads["delete"]["counts"]["denied"] == 3
    assert payloads["delete"]["counts"]["observed_allow"] == 0
    assert {p["mode"] for p in payloads.values()} == {"enforce"}
    assert payloads["delete"]["vulnerable"] == "not_ruled_out"
    assert payloads["delete"]["verdict_reason"] == "too_few_trials"
    assert all(entry["denied"] for entry in payloads["delete"]["trials"])


def test_the_recorded_log_only_policy_battery_names_the_forbid_policy_as_closing_the_gap() -> None:
    delete = _payloads(_run("phase3"))["delete"]
    assert delete["counts"]["gap"] == 3
    assert delete["vulnerable"] == "true"
    assert delete["verdict_reason"] == "observed_allow"
    [(policy, flip)] = delete["flips"].items()
    assert policy.startswith("calibrated_agent_evaluation_f040_forbid_delete")
    assert (flip["trials"], flip["closes_gap"]) == (3, 3)
    assert flip["bounds"]["trials"] == {"ci95": "[0.2924, 1.0000]", "ub95": "1.0000"}


@pytest.mark.parametrize("battery", ["phase0", "phase1a", "phase1b"])
def test_a_battery_with_no_decisions_at_all_is_refused_and_returns_nothing(battery: str) -> None:
    with pytest.raises(PolicyDecisionsMissing) as refused:
        _run(battery)
    message = str(refused.value)
    for cause in ("trace context", "no Policy engine", "called no tool", "tools/list", "log-only"):
        assert cause in message


def test_a_silent_case_that_is_not_a_control_is_reported_and_not_refused() -> None:
    delete = _payloads(_run("silent"))["delete"]
    assert (delete["mode"], delete["scored"], delete["bounds"]) == ("none", 0, None)
    assert (delete["vulnerable"], delete["verdict_reason"]) == ("not_ruled_out", "no_decision")


def test_a_partly_silent_case_scores_its_trials_with_a_decision_and_tallies_the_rest() -> None:
    delete = _payloads(_run("partial"))["delete"]
    assert (delete["scored"], delete["unscored"]) == (2, 1)
    assert delete["counts"]["denied"] == 2


def test_a_silent_control_case_is_refused_even_when_other_cases_have_decisions() -> None:
    feed = dict(_FEEDS["phase2"], **{"read-small": _FEEDS["phase0"]["read-small"]})
    with pytest.raises(PolicyDecisionsMissing, match="control case 'read-small'") as refused:
        _run(feed)
    assert "a decision in at least one of its trials" in str(refused.value)


def test_a_control_case_with_a_decision_in_one_trial_only_is_scored_not_refused() -> None:
    silent = _FEEDS["phase0"]["read-small"]
    feed = dict(
        _FEEDS["phase2"],
        **{"read-small": [silent[0], silent[1], _FEEDS["phase2"]["read-small"][2]]},
    )
    payload = _payloads(_run(feed))["read-small"]
    assert (payload["scored"], payload["unscored"]) == (1, 2)


def test_a_case_file_without_a_control_states_the_control_rule() -> None:
    raw = json.loads((_RAW / "cases.json").read_text())
    for case in raw["cases"]:
        case["control"] = False
    with pytest.raises(PolicyCaseError, match="a decision in at least one of its trials"):
        PolicyCaseFile.model_validate(raw)


def test_decisions_that_name_no_listed_tool_refuse_the_battery_naming_the_tools_seen() -> None:
    bare = load_policy_cases(_RAW / "cases_bare_names.json")
    with pytest.raises(PolicyCaseError) as refused:
        _run("phase2", cases=bare)
    assert "calibrated-agent-evaluation-f040-rows___delete_rows" in str(refused.value)
    assert "delete_rows" in str(refused.value)


def test_one_case_matching_nothing_beside_a_matching_case_is_not_refused() -> None:
    data = json.loads((_RAW / "cases.json").read_text())
    data["cases"][1]["expect"] = {"some___other_tool": "allow"}
    payloads = _payloads(_run("phase2", cases=PolicyCaseFile.model_validate(data)))
    assert payloads["read-bulk"]["verdict_reason"] == "unmatched"


def test_the_planted_agent_trial_reads_the_call_decision_and_skips_the_tool_list() -> None:
    bulk = _payloads(_run("planted-agent-trial"))["read-bulk"]
    assert [len(entry["decisions"]) for entry in bulk["trials"]] == [1, 1, 1]


def test_an_unauthorized_battery_sends_nothing() -> None:
    adapter = _Adapter(_FEEDS["phase2"])
    source = _Source()
    with pytest.raises(AuthorizationError):
        run_policy_battery(
            adapter, target=_TARGET, authorized=False, cases=_CASES, decisions=source
        )
    assert adapter.sent == []
    assert source.events == []


def test_the_read_check_runs_before_the_first_prompt_and_a_failed_one_sends_nothing() -> None:
    adapter = _Adapter(_FEEDS["phase2"])
    with pytest.raises(ProbeError, match="cannot read"):
        _run("phase2", adapter=adapter, decisions=_Source(readable=ProbeError("cannot read")))
    assert adapter.sent == []


def test_a_reply_without_a_trace_id_is_refused() -> None:
    class _Unnamed(_Adapter):
        def send(self, prompt: str) -> ProbeResponse:
            super().send(prompt)
            return ProbeResponse(status=200, text="done")

    with pytest.raises(ProbeError, match="carries no trace ID"):
        _run("phase2", adapter=_Unnamed(_FEEDS["phase2"]))


def test_a_source_answer_that_misstates_the_trials_is_refused() -> None:
    def drop_one(answers: dict[str, TrialDecisions]) -> dict[str, TrialDecisions]:
        return dict(list(answers.items())[1:])

    def add_one(answers: dict[str, TrialDecisions]) -> dict[str, TrialDecisions]:
        return {**answers, "probe-fp-unknown": next(iter(answers.values()))}

    def foreign(answers: dict[str, TrialDecisions]) -> dict[str, TrialDecisions]:
        keys = list(answers)
        moved = answers[keys[-1]]
        return {
            **answers,
            keys[0]: answers[keys[0]].model_copy(update={"decisions": moved.decisions}),
        }

    for rewrite, problem in (
        (drop_one, "no answer"),
        (add_one, "which no trial sent"),
        (foreign, "not its own"),
    ):
        with pytest.raises(ProbeError, match=problem):
            _run("phase2", decisions=_Source(rewrite=rewrite))


def test_an_agent_reported_error_is_a_trial_with_its_decisions_classified() -> None:
    feed = _FEEDS["phase2"]
    failed = feed["delete"][1]
    error = AgentReportedError(
        "the agent reported an error", error_code="ToolDenied", trace_id=f"probe-fp-{failed}"
    )
    adapter = _Adapter(feed, fail={("delete", 2): error})
    result = _run("phase2", adapter=adapter, agent_error=agent_reported_error_trial)
    delete = _payloads(result)["delete"]
    entry = delete["trials"][1]
    assert (entry["error_code"], entry["scored"], entry["denied"]) == ("ToolDenied", True, True)
    assert delete["counts"]["denied"] == 3


def test_any_other_trial_error_stops_the_battery() -> None:
    adapter = _Adapter(_FEEDS["phase2"], fail={("delete", 1): RuntimeInvocationError("budget")})
    with pytest.raises(RuntimeInvocationError, match="budget"):
        _run("phase2", adapter=adapter, agent_error=agent_reported_error_trial)


def test_an_agent_reported_error_stops_the_battery_without_a_classifier() -> None:
    error = AgentReportedError("x", error_code="e", trace_id="probe-fp-x")
    adapter = _Adapter(_FEEDS["phase2"], fail={("read-small", 1): error})
    with pytest.raises(AgentReportedError):
        _run("phase2", adapter=adapter)


def test_each_trial_gets_one_outcome_only_probed_trace_under_its_probe_trace_id() -> None:
    result = _run("phase2")
    assert [t.trace_id for t in result.traces] == [t.probe_trace_id for t in result.trials]
    prompts = {c.case_id: c.prompt for c in _CASES.cases}
    for trace, trial in zip(result.traces, result.trials, strict=True):
        assert (trace.ownership, trace.spans, trace.frame) == ("probed", [], "discovery")
        assert trace.input == {"prompt": prompts[trial.case_id]}
        assert trace.output == {"text": "done"}


def test_an_agent_error_trial_trace_holds_its_error_code_and_no_reply_text() -> None:
    feed = _FEEDS["phase2"]
    failed = f"probe-fp-{feed['delete'][1]}"
    error = AgentReportedError(
        "the agent reported an error; its text: secret rows",
        error_code="ToolDenied",
        trace_id=failed,
    )
    result = _run(
        "phase2",
        adapter=_Adapter(feed, fail={("delete", 2): error}),
        agent_error=agent_reported_error_trial,
    )
    [trace] = [t for t in result.traces if t.trace_id == failed]
    assert trace.output == {"error_code": "ToolDenied"}
    assert "secret" not in trace.model_dump_json()


def test_a_battery_result_refuses_traces_that_do_not_match_its_trials() -> None:
    from calibrated_agent_evaluation.probe.policy_battery import PolicyBatteryResult

    result = _run("phase2")
    for traces in (result.traces[:-1], (result.traces[1], result.traces[0], *result.traces[2:])):
        with pytest.raises(ProbeError, match="one per trial"):
            PolicyBatteryResult(findings=result.findings, trials=result.trials, traces=traces)


def test_two_trials_under_one_probe_trace_id_are_refused_rather_than_counted_twice() -> None:
    feed = {case: list(traces) for case, traces in _FEEDS["phase2"].items()}
    first = next(iter(feed))
    feed[first][1] = feed[first][0]
    with pytest.raises(ProbeError, match="same probe trace ID"):
        _run(feed)


def test_a_source_answer_that_skipped_validation_cannot_set_a_bad_w3c_trace_id() -> None:
    def unvalidated(answers: Mapping[str, TrialDecisions]) -> dict[str, TrialDecisions]:
        return {
            key: TrialDecisions.model_construct(w3c_trace_id="NOT-A-TRACE", decisions=())
            for key in answers
        }

    with pytest.raises(ProbeError, match="w3c_trace_id"):
        _run("phase2", decisions=_Source(rewrite=unvalidated))


def test_the_result_holds_one_trace_per_trial_an_agent_error_trial_included() -> None:
    feed = _FEEDS["phase2"]
    failed = f"probe-fp-{feed['delete'][1]}"
    error = AgentReportedError("reported", error_code="ToolDenied", trace_id=failed)
    result = _run(
        "phase2",
        adapter=_Adapter(feed, fail={("delete", 2): error}),
        agent_error=agent_reported_error_trial,
    )
    assert [t.trace_id for t in result.traces] == [t.probe_trace_id for t in result.trials]
    assert len(result.traces) == 9
    [errored] = [t for t in result.traces if t.output == {"error_code": "ToolDenied"}]
    assert errored.trace_id == failed
    assert errored.spans == []


def test_an_agent_error_trial_without_a_decision_is_stored_unscored_and_unlabelled() -> None:
    from calibrated_agent_evaluation.probe.policy import policy_labels

    silent = _FEEDS["phase0"]["delete"][0]
    feed = {**_FEEDS["phase2"], "delete": [silent, *_FEEDS["phase2"]["delete"][1:]]}
    failed = f"probe-fp-{silent}"
    error = AgentReportedError("reported", error_code="ToolDenied", trace_id=failed)
    result = _run(
        feed,
        adapter=_Adapter(feed, fail={("delete", 1): error}),
        agent_error=agent_reported_error_trial,
    )
    entry = _payloads(result)["delete"]["trials"][0]
    assert (entry["error_code"], entry["scored"], entry["denied"]) == ("ToolDenied", False, False)
    assert [t.output for t in result.traces if t.trace_id == failed] == [
        {"error_code": "ToolDenied"}
    ]
    assert failed not in {label.trace_id for label in policy_labels(result.findings)}


def test_reloaded_decisions_equal_the_oracle_with_their_trace_and_span_ids(tmp_path: Path) -> None:
    from calibrated_agent_evaluation._provenance import capture_provenance
    from calibrated_agent_evaluation.probe import ProbeBatteryRecord, dump_probes, load_probes
    from calibrated_agent_evaluation.probe.policy import decisions_from_finding

    oracle = json.loads(
        (_RAW.parents[2] / "expected" / "probe" / "policy_decisions.json").read_text()
    )
    result = _run("phase3")
    record = ProbeBatteryRecord(
        probe_id="probe-1",
        endpoint_fingerprint="fp",
        findings=result.findings,
        provenance=capture_provenance(),
    )
    dump_probes([record], tmp_path, "json")
    [loaded] = load_probes(tmp_path, "json")
    for finding, counted in zip(loaded.findings, result.findings, strict=True):
        rebuilt = decisions_from_finding(finding)
        assert rebuilt == decisions_from_finding(counted)
        by_trace: dict[str, list[dict[str, Any]]] = {}
        for d in rebuilt:
            by_trace.setdefault(d.trace_id, []).append(
                {
                    "tool": d.tool,
                    "decision": d.decision,
                    "mode": d.mode,
                    "determining_policies": list(d.determining_policies),
                    "log_only_flipping_policies": list(d.log_only_flipping_policies),
                }
            )
        traces = {e["w3c_trace_id"] for e in finding.payload["trials"]}
        assert {t: by_trace.get(t, []) for t in traces} == {t: oracle[t] for t in traces}
        assert all(d.span_id for d in rebuilt)
        assert len({d.span_id for d in rebuilt}) == len(rebuilt)
