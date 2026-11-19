# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Gateway Policy decisions read from recorded spans, and the decision source over CW Logs."""

from __future__ import annotations

import copy
import gzip
import json
import time
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import pytest

pytest.importorskip("boto3")

from calibrated_agent_evaluation.agentcore import AgentCoreSettlingLag, PolicySpanMalformed
from calibrated_agent_evaluation.agentcore.errors import AgentCoreError, AgentCoreTruncatedResult
from calibrated_agent_evaluation.agentcore.policy_spans import (
    AgentCorePolicyDecisionSource,
    agent_reported_error_trial,
    extract_decisions,
)
from calibrated_agent_evaluation.agentcore.runtime_errors import (
    AgentReportedError,
    RuntimeInvocationError,
)
from calibrated_agent_evaluation.probe.errors import ProbeError
from calibrated_agent_evaluation.probe.policy import TrialRef

_ROOT = Path(__file__).resolve().parents[1]
_RAW = _ROOT / "fixtures" / "raw" / "agentcore" / "policy"
_ORACLE = json.loads(
    (_ROOT / "fixtures" / "expected" / "probe" / "policy_decisions.json").read_text()
)
_FEEDS = json.loads((_RAW / "feeds.json").read_text())
_PLANTED = json.loads((_RAW / "planted.json").read_text())


def _spans(trace_id: str) -> list[dict[str, Any]]:
    return json.loads(gzip.decompress((_RAW / f"{trace_id}.spans.json.gz").read_bytes()))


def _traces(*batteries: str) -> list[str]:
    return [t for b in batteries for ids in _FEEDS[b].values() for t in ids]


def _summary(trace_id: str) -> list[dict[str, Any]]:
    return [
        {
            "tool": d.tool,
            "decision": d.decision,
            "mode": d.mode,
            "determining_policies": list(d.determining_policies),
            "log_only_flipping_policies": list(d.log_only_flipping_policies),
        }
        for d in extract_decisions(_spans(trace_id))
    ]


# --- the extractor over the recorded spans ---------------------------------------------------


@pytest.mark.parametrize("trace_id", _traces("phase2", "phase3"))
def test_recorded_enforce_traces_yield_the_oracle_decisions(trace_id: str) -> None:
    assert _summary(trace_id) == _ORACLE[trace_id]


def test_every_decision_keeps_its_trace_span_engine_and_reason() -> None:
    delete = _FEEDS["phase2"]["delete"][0]
    [decision] = extract_decisions(_spans(delete))
    assert decision.trace_id == delete
    assert decision.span_id
    assert decision.engine is not None
    assert decision.engine.startswith("arn:aws:bedrock-agentcore:")
    assert decision.reason is not None
    assert "denied" in decision.reason


@pytest.mark.parametrize("trace_id", _traces("phase0"))
def test_a_trace_with_no_engine_attached_yields_no_decisions(trace_id: str) -> None:
    assert extract_decisions(_spans(trace_id)) == ()


@pytest.mark.parametrize("trace_id", _traces("phase1a", "phase1b"))
def test_a_log_only_gateway_records_no_per_call_decision(trace_id: str) -> None:
    # Recorded live: a log-only gateway decides asynchronously and writes no per-call span.
    spans = _spans(trace_id)
    markers = {
        s["attributes"].get("aws.agentcore.policy.authorization_decision")
        for s in spans
        if s["name"] == "AgentCore.Gateway.InvokeTool"
    }
    assert markers == {"LOG_ONLY_ASYNC"}
    assert extract_decisions(spans) == ()


def test_recorded_tool_list_traces_yield_no_decisions() -> None:
    tool_lists = [
        t
        for t, decisions in _ORACLE.items()
        if decisions == [] and t not in _traces("phase0", "phase1a", "phase1b")
    ]
    assert tool_lists
    for trace_id in tool_lists:
        assert extract_decisions(_spans(trace_id)) == ()


def test_a_span_the_store_holds_more_than_once_is_read_once() -> None:
    delete = _FEEDS["phase2"]["delete"][0]
    spans = _spans(delete)
    copies = [s for s in spans if s["name"] == "AgentCore.Policy.AuthorizeAction"]
    assert len(copies) > 1
    assert len(extract_decisions(spans)) == 1


def test_spans_without_a_decision_attribute_are_skipped() -> None:
    spans = [
        s
        for s in _spans(_FEEDS["phase2"]["delete"][0])
        if s["name"].startswith("AgentCore.Gateway.Inbound")
    ]
    assert spans
    assert extract_decisions(spans) == ()


def test_a_tool_list_authorization_in_a_trial_trace_is_skipped_not_refused() -> None:
    planted = _PLANTED["planted-agent-trial"]
    assert _summary(planted) == _ORACLE[planted]


def test_a_decision_span_without_its_tool_is_refused_by_name() -> None:
    with pytest.raises(PolicySpanMalformed, match="tool.name"):
        extract_decisions(_spans(_PLANTED["planted-no-tool"]))


def test_a_decision_span_without_a_call_selector_is_refused_by_name() -> None:
    with pytest.raises(PolicySpanMalformed, match="neither a per-call"):
        extract_decisions(_spans(_PLANTED["planted-no-selector"]))


@pytest.mark.parametrize(
    ("key", "value"),
    [
        ("aws.agentcore.policy.authorization_decision", "MAYBE"),
        ("aws.agentcore.gateway.policy.mode", "AUDIT"),
    ],
)
def test_unknown_decision_or_mode_values_are_refused(key: str, value: str) -> None:
    spans = copy.deepcopy(_spans(_FEEDS["phase2"]["delete"][0]))
    for span in spans:
        if span["name"] == "AgentCore.Policy.AuthorizeAction":
            span["attributes"][key] = value
    with pytest.raises(PolicySpanMalformed, match=value):
        extract_decisions(spans)


def test_a_list_attribute_in_an_unknown_encoding_is_refused() -> None:
    spans = copy.deepcopy(_spans(_FEEDS["phase2"]["delete"][0]))
    for span in spans:
        if span["name"] == "AgentCore.Policy.AuthorizeAction":
            span["attributes"]["aws.agentcore.policy.determining_policies"] = "forbid,permit"
    with pytest.raises(PolicySpanMalformed, match="not a list of strings"):
        extract_decisions(spans)


def test_an_agent_reported_error_becomes_its_trial_and_other_errors_do_not() -> None:
    reported = AgentReportedError(
        "the agent reported an error", error_code="ToolDenied", trace_id="probe-fp-abc"
    )
    trial = agent_reported_error_trial(reported)
    assert trial is not None
    assert (trial.probe_trace_id, trial.error_code) == ("probe-fp-abc", "ToolDenied")
    assert agent_reported_error_trial(RuntimeInvocationError("budget refused")) is None
    assert agent_reported_error_trial(AgentReportedError("x", error_code="e")) is None


# --- the decision source ---------------------------------------------------------------------


@pytest.fixture
def fake_clock(monkeypatch: pytest.MonkeyPatch) -> None:
    clock = {"t": 0.0}
    monkeypatch.setattr(time, "monotonic", lambda: clock["t"])
    monkeypatch.setattr(time, "sleep", lambda s: clock.__setitem__("t", clock["t"] + max(s, 0.5)))


def _insights(records: list[dict[str, Any]]) -> dict[str, Any]:
    rows = [
        [{"field": "@message", "value": json.dumps(r)}, {"field": "@ptr", "value": f"p{i}"}]
        for i, r in enumerate(records)
    ]
    return {
        "results": rows,
        "statistics": {"recordsMatched": float(len(rows))},
        "status": "Complete",
    }


class _Logs:
    """Logs Insights serving each trace's spans poll by poll while the log store holds them all."""

    def __init__(
        self, index: dict[str, list[list[dict[str, Any]]]], store: dict[str, list[dict[str, Any]]]
    ):
        self._index = index
        self._store = store
        self._served: dict[str, int] = {}
        self._results: dict[str, list[dict[str, Any]]] = {}
        self.queries: list[dict[str, Any]] = []

    def client(self, service: str) -> _Logs:
        assert service == "logs"
        return self

    def start_query(self, **params: Any) -> dict[str, str]:
        self.queries.append(params)
        trace = next((t for t in self._index if t in params["queryString"]), None)
        if trace is None:
            records: list[dict[str, Any]] = []
        else:
            n = self._served.get(trace, 0)
            reads = self._index[trace]
            records = reads[min(n, len(reads) - 1)]
            self._served[trace] = n + 1
        qid = f"q{len(self._results)}"
        self._results[qid] = records
        return {"queryId": qid}

    def get_query_results(self, *, queryId: str) -> dict[str, Any]:
        return _insights(self._results[queryId])

    def filter_log_events(self, **params: Any) -> dict[str, Any]:
        trace = params["filterPattern"].split('"')[1]
        return {
            "events": [
                {"eventId": f"e{i}", "message": json.dumps(r)}
                for i, r in enumerate(self._store.get(trace, []))
            ]
        }


def _run(trace: str, n: int) -> tuple[TrialRef, Any]:
    probe_id = f"probe-fp-{trace}"
    record = SimpleNamespace(
        trace_id=probe_id,
        w3c_trace_id=trace,
        started_at="2026-10-30T20:20:00+00:00",
        ended_at="2026-10-30T20:20:05+00:00",
        runtime_arn="arn:aws:bedrock-agentcore:us-west-2:111122223333:runtime/agent-AbCdEf1234",
    )
    trial = TrialRef(
        case_id="delete",
        trial=n,
        probe_trace_id=probe_id,
        started_at=record.started_at,
        ended_at=record.ended_at,
    )
    return trial, record


@pytest.mark.usefixtures("fake_clock")
def test_the_source_waits_until_the_index_holds_what_the_store_holds() -> None:
    trace = _FEEDS["phase2"]["delete"][0]
    whole = _spans(trace)
    partial = [s for s in whole if s["name"] != "AgentCore.Policy.AuthorizeAction"]
    logs = _Logs({trace: [partial, partial, whole]}, {trace: whole})
    trial, record = _run(trace, 1)
    source = AgentCorePolicyDecisionSource(cast(Any, logs), run_records=lambda: [record])

    answer = source.read_decisions([trial])

    [decision] = answer[trial.probe_trace_id].decisions
    assert (decision.tool, decision.decision) == (
        "calibrated-agent-evaluation-f040-rows___delete_rows",
        "deny",
    )
    assert answer[trial.probe_trace_id].w3c_trace_id == trace
    assert all(f"traceId = '{trace}'" in q["queryString"] for q in logs.queries)


@pytest.mark.usefixtures("fake_clock")
def test_a_trial_with_no_spans_maps_to_no_decisions() -> None:
    trace = "0" * 31 + "1"
    trial, record = _run(trace, 1)
    source = AgentCorePolicyDecisionSource(cast(Any, _Logs({}, {})), run_records=lambda: [record])
    assert source.read_decisions([trial])[trial.probe_trace_id].decisions == ()


@pytest.mark.usefixtures("fake_clock")
def test_spans_still_arriving_at_the_deadline_are_refused() -> None:
    trace = _FEEDS["phase2"]["delete"][0]
    whole = _spans(trace)
    logs = _Logs({trace: [whole[:2]]}, {trace: whole})
    trial, record = _run(trace, 1)
    source = AgentCorePolicyDecisionSource(
        cast(Any, logs), run_records=lambda: [record], settle_deadline_seconds=30
    )
    with pytest.raises(AgentCoreSettlingLag, match="still arriving"):
        source.read_decisions([trial])


def test_the_wait_runs_on_the_sleep_and_clock_it_is_given() -> None:
    now = {"t": 0.0}
    slept: list[float] = []

    def sleep(seconds: float) -> None:
        slept.append(seconds)
        now["t"] += seconds

    trace = _FEEDS["phase2"]["delete"][0]
    whole = _spans(trace)
    trial, record = _run(trace, 1)
    source = AgentCorePolicyDecisionSource(
        cast(Any, _Logs({trace: [whole[:2]]}, {trace: whole})),
        run_records=lambda: [record],
        settle_deadline_seconds=30,
        sleep=sleep,
        monotonic=lambda: now["t"],
    )

    with pytest.raises(AgentCoreSettlingLag, match="still arriving"):
        source.read_decisions([trial])
    assert slept
    assert set(slept) == {10.0}


def test_a_trial_with_no_run_record_is_refused() -> None:
    trial, _ = _run("0" * 32, 1)
    source = AgentCorePolicyDecisionSource(cast(Any, _Logs({}, {})), run_records=lambda: [])
    with pytest.raises(ProbeError, match="no run record"):
        source.read_decisions([trial])


@pytest.mark.usefixtures("fake_clock")
def test_the_source_reads_the_run_records_when_it_reads_not_when_it_is_built() -> None:
    records: list[Any] = []
    source = AgentCorePolicyDecisionSource(cast(Any, _Logs({}, {})), run_records=lambda: records)
    trial, record = _run("0" * 31 + "2", 1)
    records.append(record)
    assert source.read_decisions([trial])[trial.probe_trace_id].w3c_trace_id == record.w3c_trace_id


def test_the_read_check_runs_one_query_over_the_last_second() -> None:
    logs = _Logs({}, {})
    now = datetime(2026, 11, 2, 12, 0, 0, tzinfo=UTC)
    AgentCorePolicyDecisionSource(
        cast(Any, logs), run_records=list, clock=lambda: now
    ).check_readable()
    [query] = logs.queries
    assert query["logGroupName"] == "aws/spans"
    assert query["endTime"] - query["startTime"] == 1


def test_a_truncated_read_check_counts_as_readable() -> None:
    class _Truncated(_Logs):
        def get_query_results(self, *, queryId: str) -> dict[str, Any]:
            result = super().get_query_results(queryId=queryId)
            result["statistics"]["recordsMatched"] = 20_000.0
            return result

    source = AgentCorePolicyDecisionSource(cast(Any, _Truncated({}, {})), run_records=list)
    source.check_readable()


def test_access_denied_on_the_read_check_propagates_naming_the_log_group() -> None:
    import boto3
    from botocore.stub import Stubber

    from calibrated_agent_evaluation.agentcore.session import AwsClients

    clients = AwsClients(
        boto3.Session(
            region_name="us-west-2", aws_access_key_id="testing", aws_secret_access_key="testing"
        )
    )
    stub = Stubber(clients.client("logs"))
    stub.add_client_error("start_query", "AccessDeniedException", "not allowed")
    with stub, pytest.raises(AgentCoreError, match="aws/spans"):
        AgentCorePolicyDecisionSource(clients, run_records=list).check_readable()


@pytest.mark.usefixtures("fake_clock")
def test_a_truncated_logs_insights_read_propagates_unchanged() -> None:
    class _Truncated(_Logs):
        def get_query_results(self, *, queryId: str) -> dict[str, Any]:
            result = super().get_query_results(queryId=queryId)
            result["statistics"]["recordsMatched"] = 20_000.0
            return result

    trace = _FEEDS["phase2"]["delete"][0]
    whole = _spans(trace)
    trial, record = _run(trace, 1)
    source = AgentCorePolicyDecisionSource(
        cast(Any, _Truncated({trace: [whole]}, {trace: whole})), run_records=lambda: [record]
    )
    with pytest.raises(AgentCoreTruncatedResult):
        source.read_decisions([trial])


@pytest.mark.usefixtures("fake_clock")
def test_throttling_on_get_query_results_propagates_as_an_agentcore_error() -> None:
    import boto3
    from botocore.stub import ANY, Stubber

    from calibrated_agent_evaluation.agentcore.session import AwsClients

    clients = AwsClients(
        boto3.Session(
            region_name="us-west-2", aws_access_key_id="testing", aws_secret_access_key="testing"
        )
    )
    trial, record = _run(_FEEDS["phase2"]["delete"][0], 1)
    stub = Stubber(clients.client("logs"))
    stub.add_response(
        "start_query",
        {"queryId": "q0"},
        {"logGroupName": "aws/spans", "startTime": ANY, "endTime": ANY, "queryString": ANY},
    )
    stub.add_client_error("get_query_results", "ThrottlingException", "Rate exceeded")
    source = AgentCorePolicyDecisionSource(clients, run_records=lambda: [record])
    with stub, pytest.raises(AgentCoreError, match="aws/spans.*ThrottlingException"):
        source.read_decisions([trial])
    stub.assert_no_pending_responses()


def test_unusable_poll_settings_are_refused() -> None:
    with pytest.raises(AgentCoreError, match="poll_interval_seconds"):
        AgentCorePolicyDecisionSource(
            cast(Any, _Logs({}, {})), run_records=list, poll_interval_seconds=0
        )


def test_a_settle_bound_shorter_than_one_poll_is_refused() -> None:
    with pytest.raises(AgentCoreError, match="at least that long"):
        AgentCorePolicyDecisionSource(
            cast(Any, _Logs({}, {})),
            run_records=list,
            settle_deadline_seconds=5,
            poll_interval_seconds=10,
        )


@pytest.mark.usefixtures("fake_clock")
def test_a_run_record_whose_times_are_not_iso_8601_is_refused_by_trial() -> None:
    trial, record = _run("0" * 31 + "2", 1)
    record.started_at = "yesterday"
    source = AgentCorePolicyDecisionSource(cast(Any, _Logs({}, {})), run_records=lambda: [record])
    with pytest.raises(ProbeError, match="not ISO-8601"):
        source.read_decisions([trial])


def test_a_run_record_whose_w3c_trace_id_is_malformed_is_refused_by_trial_before_any_query() -> (
    None
):
    trial, record = _run("0" * 31 + "3", 1)
    record.w3c_trace_id = "not-a-trace-id"
    logs = _Logs({}, {})
    source = AgentCorePolicyDecisionSource(cast(Any, logs), run_records=lambda: [record])
    with pytest.raises(ProbeError, match=f"trial {trial.probe_trace_id}.*not 32 hex characters"):
        source.read_decisions([trial])
    assert logs.queries == []


# --- the settling wait: per trial, with a confirming read, bounded by the battery's size --------


@pytest.fixture
def clock(monkeypatch: pytest.MonkeyPatch) -> dict[str, float]:
    state = {"t": 0.0}
    monkeypatch.setattr(time, "monotonic", lambda: state["t"])
    monkeypatch.setattr(time, "sleep", lambda s: state.__setitem__("t", state["t"] + max(s, 0.5)))
    return state


class _TimedLogs(_Logs):
    """Each query costs ``cost`` seconds of the fake clock; queries are counted per trace."""

    def __init__(self, *args: Any, clock: dict[str, float], cost: float) -> None:
        super().__init__(*args)
        self._clock = clock
        self._cost = cost
        self.per_trace: dict[str, int] = {}

    def start_query(self, **params: Any) -> dict[str, str]:
        self._clock["t"] += self._cost
        trace = params["queryString"].split("'")[1] if "'" in params["queryString"] else ""
        self.per_trace[trace] = self.per_trace.get(trace, 0) + 1
        return super().start_query(**params)

    def filter_log_events(self, **params: Any) -> dict[str, Any]:
        self._clock["t"] += self._cost
        return super().filter_log_events(**params)


def _runs(n: int) -> tuple[list[TrialRef], list[Any]]:
    pairs = [_run(f"{i:032x}", i) for i in range(1, n + 1)]
    return [t for t, _ in pairs], [r for _, r in pairs]


def _settled_spans(trace: str) -> list[dict[str, Any]]:
    return [{**s, "traceId": trace} for s in _spans(_FEEDS["phase2"]["delete"][0])]


class _LateLogs(_Logs):
    """The index and the log store hold nothing for the first ``empty`` reads, then every span."""

    def __init__(self, trace: str, spans: list[dict[str, Any]], *, empty: int) -> None:
        super().__init__({trace: [[]] * empty + [spans]}, {trace: spans})
        self._empty = empty
        self.store_reads = 0

    def filter_log_events(self, **params: Any) -> dict[str, Any]:
        self.store_reads += 1
        if self.store_reads <= self._empty:
            return {"events": []}
        return super().filter_log_events(**params)


def test_an_empty_trial_waits_for_the_bound_and_then_holds_no_decision(
    clock: dict[str, float],
) -> None:
    trial, record = _run("0" * 31 + "3", 1)
    logs = _TimedLogs({}, {}, clock=clock, cost=0.0)
    source = AgentCorePolicyDecisionSource(
        cast(Any, logs), run_records=lambda: [record], settle_deadline_seconds=30
    )

    answer = source.read_decisions([trial])

    assert answer[trial.probe_trace_id].decisions == ()
    assert clock["t"] >= 30
    assert len(logs.queries) > 2


@pytest.mark.usefixtures("fake_clock")
def test_spans_that_reach_the_store_after_two_empty_reads_are_still_read() -> None:
    trace = _FEEDS["phase2"]["delete"][0]
    logs = _LateLogs(trace, _spans(trace), empty=3)
    trial, record = _run(trace, 1)
    source = AgentCorePolicyDecisionSource(
        cast(Any, logs), run_records=lambda: [record], settle_deadline_seconds=60
    )

    [decision] = source.read_decisions([trial])[trial.probe_trace_id].decisions

    assert (decision.tool, decision.decision) == (
        "calibrated-agent-evaluation-f040-rows___delete_rows",
        "deny",
    )


def test_a_large_battery_whose_first_poll_outlasts_300_seconds_still_settles(
    clock: dict[str, float],
) -> None:
    trials, records = _runs(130)
    spans = {r.w3c_trace_id: _settled_spans(r.w3c_trace_id) for r in records}
    logs = _TimedLogs({k: [v] for k, v in spans.items()}, spans, clock=clock, cost=2.5)
    source = AgentCorePolicyDecisionSource(cast(Any, logs), run_records=lambda: records)

    answers = source.read_decisions(trials)

    assert len(answers) == 130
    assert clock["t"] > 300


def test_a_trial_that_matched_on_its_last_read_before_the_bound_gets_its_confirming_read(
    clock: dict[str, float],
) -> None:
    trials, records = _runs(20)
    spans = {r.w3c_trace_id: _settled_spans(r.w3c_trace_id) for r in records}
    logs = _TimedLogs({k: [v] for k, v in spans.items()}, spans, clock=clock, cost=2.5)
    source = AgentCorePolicyDecisionSource(
        cast(Any, logs), run_records=lambda: records, settle_deadline_seconds=30
    )

    assert len(source.read_decisions(trials)) == 20


def test_a_settled_trial_is_not_read_again(clock: dict[str, float]) -> None:
    trials, records = _runs(2)
    fast, slow = (r.w3c_trace_id for r in records)
    whole = {t: _settled_spans(t) for t in (fast, slow)}
    behind = [s for s in whole[slow] if s["name"] != "AgentCore.Policy.AuthorizeAction"]
    logs = _TimedLogs(
        {fast: [whole[fast]], slow: [behind, behind, behind, whole[slow]]},
        whole,
        clock=clock,
        cost=1.0,
    )
    source = AgentCorePolicyDecisionSource(cast(Any, logs), run_records=lambda: records)

    source.read_decisions(trials)

    assert logs.per_trace[fast] == 2
    assert logs.per_trace[slow] == 5


def test_a_trial_whose_confirming_read_changed_is_refused(clock: dict[str, float]) -> None:
    trial, record = _run("0" * 31 + "7", 1)
    whole = list({s["spanId"]: s for s in _settled_spans(record.w3c_trace_id)}.values())

    class _Growing(_TimedLogs):
        def __init__(self) -> None:
            super().__init__({}, {}, clock=clock, cost=40.0)
            self.reads = 0

        def _now(self) -> list[dict[str, Any]]:
            return whole[: 2 + self.reads]

        def start_query(self, **params: Any) -> dict[str, str]:
            self._index = {record.w3c_trace_id: [self._now()]}
            self._served = {}
            return super().start_query(**params)

        def filter_log_events(self, **params: Any) -> dict[str, Any]:
            result = {
                "events": [
                    {"eventId": f"e{i}", "message": json.dumps(r)}
                    for i, r in enumerate(self._now())
                ]
            }
            self.reads += 1
            self._clock["t"] += self._cost
            return result

    source = AgentCorePolicyDecisionSource(
        cast(Any, _Growing()), run_records=lambda: [record], settle_deadline_seconds=30
    )
    with pytest.raises(AgentCoreSettlingLag, match="still arriving"):
        source.read_decisions([trial])


def test_the_default_bound_grows_with_the_number_of_trials() -> None:
    from calibrated_agent_evaluation.agentcore.policy_spans import settle_deadline_for

    assert settle_deadline_for(0) == 300
    assert settle_deadline_for(130) == 300 + 10 * 130


def test_a_span_record_that_is_not_a_json_object_is_refused(clock: dict[str, float]) -> None:
    trial, record = _run("0" * 31 + "8", 1)

    class _ListRecord(_Logs):
        def get_query_results(self, *, queryId: str) -> dict[str, Any]:
            return {
                "results": [
                    [{"field": "@message", "value": "[1, 2]"}, {"field": "@ptr", "value": "p0"}]
                ],
                "statistics": {"recordsMatched": 1.0},
                "status": "Complete",
            }

    source = AgentCorePolicyDecisionSource(
        cast(Any, _ListRecord({}, {})), run_records=lambda: [record]
    )
    with pytest.raises(AgentCoreError, match="not a JSON object"):
        source.read_decisions([trial])


# --- a Gateway call span is skipped only when its decision was made asynchronously -------------


def _phase2_delete() -> list[dict[str, Any]]:
    return copy.deepcopy(_spans(_FEEDS["phase2"]["delete"][0]))


def test_a_gateway_call_span_without_its_per_call_decision_is_refused() -> None:
    spans = [s for s in _phase2_delete() if s["name"] != "AgentCore.Policy.AuthorizeAction"]
    with pytest.raises(PolicySpanMalformed, match="no per-call"):
        extract_decisions(spans)


def test_a_gateway_call_span_with_an_unknown_decision_value_is_refused() -> None:
    spans = _phase2_delete()
    for span in spans:
        if span["name"] == "AgentCore.Gateway.InvokeTool":
            span["attributes"]["aws.agentcore.policy.authorization_decision"] = "MAYBE"
    with pytest.raises(PolicySpanMalformed, match="MAYBE"):
        extract_decisions(spans)


@pytest.mark.parametrize(
    ("key", "value"),
    [
        ("aws.agentcore.policy.log_only_eval_incomplete", "yes"),
        ("aws.agentcore.gateway.policy.arn", 5),
        ("aws.agentcore.policy.authorization_reason", ["denied"]),
    ],
    ids=["incomplete-not-bool", "engine-not-string", "reason-not-string"],
)
def test_a_decision_attribute_of_the_wrong_type_is_refused_not_coerced(
    key: str, value: Any
) -> None:
    spans = _phase2_delete()
    for span in spans:
        if span["name"] == "AgentCore.Policy.AuthorizeAction":
            span["attributes"][key] = value
    with pytest.raises(PolicySpanMalformed, match=key.rsplit(".", 1)[-1]):
        extract_decisions(spans)


def test_the_recorded_span_fixtures_name_only_placeholder_accounts_and_resource_ids() -> None:
    import re

    texts = {p.name: gzip.decompress(p.read_bytes()).decode("utf-8") for p in _RAW.glob("*.gz")}
    assert len(texts) >= 50
    for name, text in texts.items():
        accounts = set(re.findall(r"arn:aws:[a-z0-9-]+:[a-z0-9-]*:(\d{12}):", text))
        accounts |= set(re.findall(r'"aws\.account\.id":\s*"(\d+)"', text))
        assert accounts <= {"111122223333"}, (name, accounts)
        for kind, suffix in re.findall(
            r"(gateway|policy-engine|policy)/[A-Za-z0-9_-]*-([A-Za-z0-9_]{10})\b", text
        ):
            assert suffix in {"abcdef1234", "abcdef5678"}, (name, kind, suffix)
