# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""``calibrated-agent-evaluation probe --executor agentcore --policy-cases``: refusals before any
AWS call; lines.

The printed lines are checked on the findings of recorded batteries (the spans a real Gateway
wrote, through the shipped extractor); the refusals run the real CLI with every AWS client
creation failing the test.
"""

from __future__ import annotations

import gzip
import json
import re
import time
from pathlib import Path
from typing import Any

import pytest

pytest.importorskip("boto3")

import boto3.session

from _agentcore_runtime_fakes import (
    REGION,
    RUNTIME_ARN,
    StubbedAwsClients,
    add_invocation_response,
    add_preflight_responses,
    add_stop_response,
    sse_body,
)
from calibrated_agent_evaluation.agentcore import AwsClients
from calibrated_agent_evaluation.agentcore.session import DEFAULT_CLIENT_CONFIG
from calibrated_agent_evaluation.cli import main
from calibrated_agent_evaluation.cli.dispatch import EXIT_DOMAIN_ERROR, EXIT_OK, EXIT_USAGE
from calibrated_agent_evaluation.cli.stages.probe_policy import policy_lines
from calibrated_agent_evaluation.probe import load_probes
from test_probe_policy_battery import _run

_RAW = Path(__file__).resolve().parents[1] / "fixtures" / "raw" / "agentcore" / "policy"
_ARN = "arn:aws:bedrock-agentcore:us-west-2:111122223333:runtime/agent-AbCdEf1234"
_LABEL = f"{_ARN}@DEFAULT"


@pytest.fixture
def no_aws(monkeypatch: pytest.MonkeyPatch) -> None:
    def refuse(*_a: object, **_k: object) -> None:
        raise AssertionError("an AWS client was built before the refusal")

    monkeypatch.setattr(boto3.session.Session, "client", refuse)


def _policy(tmp_path: Path, *extra: str, cases: Path = _RAW / "cases.json") -> int:
    return main(
        [
            "--root",
            str(tmp_path),
            "probe",
            "--executor",
            "agentcore",
            "--runtime-arn",
            _ARN,
            "--payload-format",
            "agentcore_sse",
            "--policy-cases",
            str(cases),
            *extra,
        ]
    )


def test_policy_cases_with_prompt_exits_1_naming_both_flags_before_any_aws_request(
    tmp_path: Path, no_aws: None, capsys: pytest.CaptureFixture[str]
) -> None:
    assert _policy(tmp_path, "--prompt", "hi") == EXIT_USAGE
    err = capsys.readouterr().err
    assert "--prompt doesn't apply with --policy-cases" in err
    assert "Traceback" not in err


def test_policy_cases_without_the_agentcore_executor_exits_1_naming_the_flag(
    tmp_path: Path, no_aws: None, capsys: pytest.CaptureFixture[str]
) -> None:
    rc = main(
        [
            "--root",
            str(tmp_path),
            "probe",
            "--endpoint",
            "http://127.0.0.1:9/x",
            "--authorized",
            "--policy-cases",
            str(_RAW / "cases.json"),
        ]
    )
    assert rc == EXIT_USAGE
    assert "--policy-cases applies only to --executor agentcore" in capsys.readouterr().err


def test_policy_cases_without_authorized_is_refused_before_the_plan(
    tmp_path: Path, no_aws: None, capsys: pytest.CaptureFixture[str]
) -> None:
    assert _policy(tmp_path, "--yes") == EXIT_DOMAIN_ERROR
    err = capsys.readouterr().err
    assert "requires confirmed authorization" in err
    assert "this run can call AWS" not in err


def test_a_bad_case_file_exits_1_before_any_aws_request(
    tmp_path: Path, no_aws: None, capsys: pytest.CaptureFixture[str]
) -> None:
    bad = tmp_path / "cases.json"
    bad.write_text(json.dumps({"schema_version": "1", "cases": []}), encoding="utf-8")
    assert _policy(tmp_path, "--authorized", "--yes", cases=bad) == EXIT_DOMAIN_ERROR
    assert str(bad) in capsys.readouterr().err


def test_without_yes_the_plan_is_printed_and_nothing_is_sent(
    tmp_path: Path, no_aws: None, capsys: pytest.CaptureFixture[str]
) -> None:
    assert _policy(tmp_path, "--authorized") == EXIT_DOMAIN_ERROR
    err = capsys.readouterr().err
    assert "It runs 3 policy case(s), 9 trial(s)" in err
    assert "Trace store: local root" in err
    assert "nothing was sent" in err


def test_the_agentcore_probe_without_policy_cases_is_unchanged(
    tmp_path: Path,
    no_aws: None,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    import calibrated_agent_evaluation.cli.stages.probe_policy as probe_policy

    def not_this_route(_args: object) -> None:
        raise AssertionError("the policy battery ran without --policy-cases")

    monkeypatch.setattr(probe_policy, "policy_battery_through_agentcore", not_this_route)
    rc = main(
        [
            "--root",
            str(tmp_path),
            "probe",
            "--executor",
            "agentcore",
            "--runtime-arn",
            _ARN,
            "--payload-format",
            "agentcore_sse",
            "--authorized",
            "--prompt",
            "What can you do?",
        ]
    )
    err = capsys.readouterr().err
    assert rc == EXIT_DOMAIN_ERROR
    assert "this probe makes the billed AWS calls listed above, so it needs --yes" in err
    assert "policy case" not in err
    assert "--policy-cases" not in err


def test_the_plan_states_the_wait_and_its_bound(
    tmp_path: Path, no_aws: None, capsys: pytest.CaptureFixture[str]
) -> None:
    assert _policy(tmp_path, "--authorized") == EXIT_DOMAIN_ERROR
    err = " ".join(capsys.readouterr().err.split())
    assert (
        "After the last trial they are read every 10 s until, for each trial, the span index "
        "holds every span the log store has on two reads in a row, for at most 390 s (300 s plus "
        "10 s per trial); a trial whose spans are still arriving then refuses the battery"
    ) in err


def test_each_case_prints_its_case_and_counts_lines_with_their_upper_bounds() -> None:
    lines = policy_lines(_LABEL, _run("phase2").findings)
    assert lines[0] == f"probed {_LABEL} with 3 policy case(s), 9 trial(s)"
    text = "\n".join(lines)
    assert "no judge scored these trials" in text
    assert (
        "  delete (enforce): 0 of 3 scored trial(s) ran a call the case expects denied, at most "
        "63.2%; none ran, but too few trials to bound the rate that low"
    ) in lines
    assert "3 denied by the policy, at most 100.0%" in text
    assert "3 over-deny, at most 100.0%" in text


def test_the_flips_line_prints_each_flip_count_with_its_upper_bound() -> None:
    text = "\n".join(policy_lines(_LABEL, _run("phase3").findings))
    assert (
        "would flip a decision in 3 of 3 scored trial(s), at most 100.0%, and close a gap in 3"
        in text
    )
    assert "delete (enforce): 3 of 3 scored trial(s) ran a call the case expects denied" in text
    assert "; vulnerable" in text


def test_a_case_with_no_scored_trial_prints_its_no_decision_line() -> None:
    lines = policy_lines(_LABEL, _run("silent").findings)
    [line] = [x for x in lines if x.startswith("  delete")]
    assert line.startswith("  delete (no decision): none of 3 trial(s) held a Policy decision")
    assert "the engine hid the tool from tools/list" in line
    assert "not vulnerable" not in line


def test_a_case_with_unscored_trials_says_how_many() -> None:
    text = "\n".join(policy_lines(_LABEL, _run("partial").findings))
    assert "1 of 3 trial(s) held no Policy decision on a tool call and are not scored" in text


def test_explain_adds_each_count_with_its_interval() -> None:
    text = "\n".join(policy_lines(_LABEL, _run("phase2").findings, explain=True))
    assert "explain: observed_allow 0 ci95 [0.0000, 0.7076] ub95 0.6316" in text


# --- a whole battery through the CLI, with every AWS call stubbed ------------------------------

_FEEDS = json.loads((_RAW / "feeds.json").read_text())


class _RecordedLogs:
    """CloudWatch Logs answering each run's trace ID with the next recorded trace of the feed.

    The runtime's runs get fresh W3C trace IDs, so each new ID is given, in the order the source
    first asks for it, the next recorded Gateway trace of the battery, re-keyed to that ID. The
    index and the store answer alike.
    """

    def __init__(self, feed: list[str]) -> None:
        self._feed = list(feed)
        self._assigned: dict[str, list[dict[str, Any]]] = {}
        self._results: dict[str, list[dict[str, Any]]] = {}
        self.read_checks = 0

    def _spans(self, trace_id: str) -> list[dict[str, Any]]:
        if trace_id not in self._assigned:
            recorded = json.loads(
                gzip.decompress((_RAW / f"{self._feed.pop(0)}.spans.json.gz").read_bytes())
            )
            self._assigned[trace_id] = [{**s, "traceId": trace_id} for s in recorded]
        return self._assigned[trace_id]

    def start_query(self, **params: Any) -> dict[str, str]:
        found = re.search(r"traceId = '([0-9a-f]{32})'", params["queryString"])
        if found is None:
            self.read_checks += 1
            records: list[dict[str, Any]] = []
        else:
            records = self._spans(found.group(1))
        query_id = f"q{len(self._results)}"
        self._results[query_id] = records
        return {"queryId": query_id}

    def get_query_results(self, *, queryId: str) -> dict[str, Any]:
        rows = [
            [{"field": "@message", "value": json.dumps(r)}, {"field": "@ptr", "value": f"p{i}"}]
            for i, r in enumerate(self._results[queryId])
        ]
        return {
            "results": rows,
            "statistics": {"recordsMatched": float(len(rows))},
            "status": "Complete",
        }

    def filter_log_events(self, **params: Any) -> dict[str, Any]:
        trace_id = params["filterPattern"].split('"')[1]
        return {
            "events": [
                {"eventId": f"e{i}", "message": json.dumps(r)}
                for i, r in enumerate(self._spans(trace_id))
            ]
        }


class _PolicyClients(StubbedAwsClients):
    def __init__(self, logs: _RecordedLogs, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._logs = logs

    def client(self, service_name: str, *, region: str | None = None) -> Any:
        if service_name == "logs":
            return self._logs
        return super().client(service_name, region=region)


def _stub_aws(
    monkeypatch: pytest.MonkeyPatch,
    battery: str,
    *,
    error_at: int | None = None,
    logs_class: type[_RecordedLogs] = _RecordedLogs,
) -> _RecordedLogs:
    feed = [t for case in ("read-small", "read-bulk", "delete") for t in _FEEDS[battery][case]]
    logs = logs_class(feed)

    def from_profile(
        *,
        region: str | None = None,
        profile: str | None = None,
        handlers: Any = (),
        config: Any = DEFAULT_CLIENT_CONFIG,
    ) -> _PolicyClients:
        clients = _PolicyClients(logs, region=region or REGION, handlers=handlers, config=config)
        add_preflight_responses(clients)
        for index, _ in enumerate(feed):
            reply = (
                sse_body({"error": "the call was denied", "error_type": "ToolDenied"})
                if index == error_at
                else sse_body({"response": "done"})
            )
            add_invocation_response(clients, reply)
            add_stop_response(clients)
        return clients

    clock = {"t": 0.0}
    monkeypatch.setattr(time, "monotonic", lambda: clock["t"])
    monkeypatch.setattr(time, "sleep", lambda s: clock.__setitem__("t", clock["t"] + max(s, 0.5)))
    monkeypatch.setattr(AwsClients, "from_profile", from_profile)
    return logs


def _battery_cli(root: Path) -> int:
    return main(
        [
            "--root",
            str(root),
            "probe",
            "--executor",
            "agentcore",
            "--runtime-arn",
            RUNTIME_ARN,
            "--payload-format",
            "agentcore_sse",
            "--policy-cases",
            str(_RAW / "cases.json"),
            "--authorized",
            "--yes",
        ]
    )


def test_policy_cases_run_a_battery_over_recorded_spans_and_write_one_probe_record(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    logs = _stub_aws(monkeypatch, "phase3")

    assert _battery_cli(tmp_path) == EXIT_OK

    out = capsys.readouterr()
    assert "It runs 3 policy case(s), 9 trial(s)" in out.err
    assert logs.read_checks == 1
    assert "delete (enforce): 3 of 3 scored trial(s) ran a call the case expects denied" in out.out
    assert "would flip a decision in 3 of 3 scored trial(s)" in out.out
    record_id = re.search(r"probe record (probe-\S+)", out.out)
    assert record_id is not None
    [record] = load_probes(tmp_path / "records" / "probe", "json")
    assert record.probe_id == record_id.group(1)
    payloads = {f.payload["case_id"]: f.payload for f in record.findings}
    assert payloads["delete"]["vulnerable"] == "true"
    assert payloads["read-bulk"]["counts"]["over_deny"] == 3
    manifest = json.loads(
        (tmp_path / "records" / "agentcore" / "agentcore_run_record.json").read_text()
    )
    by_probe = {r["trace_id"]: r["w3c_trace_id"] for r in manifest}
    for payload in payloads.values():
        for entry in payload["trials"]:
            assert entry["w3c_trace_id"] == by_probe[entry["probe_trace_id"]]


def test_a_battery_against_a_log_only_gateway_exits_1_naming_the_cause(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    _stub_aws(monkeypatch, "phase1b")
    assert _battery_cli(tmp_path) == EXIT_DOMAIN_ERROR
    err = capsys.readouterr().err
    assert "PolicyDecisionsMissing" in err
    assert "log-only mode" in err
    assert not (tmp_path / "records" / "probe").exists()


def _labels_line(out: str) -> str:
    return next(line for line in out.splitlines() if "policy-engine labels:" in line)


def test_probed_traces_and_labels_are_stored_after_the_record(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    from calibrated_agent_evaluation.backends.local import LocalBackend

    _stub_aws(monkeypatch, "phase2")
    assert _battery_cli(tmp_path) == EXIT_OK
    out = capsys.readouterr().out
    assert "6 probed trace(s) labelled FAIL for a denied call" in _labels_line(out)
    [record] = load_probes(tmp_path / "records" / "probe", "json")
    store = LocalBackend(tmp_path)
    trials = [e for f in record.findings for e in f.payload["trials"]]
    assert len(trials) == 9
    for entry in trials:
        trace = store.get_trace(entry["probe_trace_id"])
        assert trace.ownership == "probed"
        scores = store.get_scores(entry["probe_trace_id"])
        assert [s.kind for s in scores] == (["policy-engine"] if entry["denied"] else [])


def test_the_same_battery_stores_the_same_labels_in_langfuse(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    from typing import cast

    import calibrated_agent_evaluation.cli.backends as cli_backends
    from calibrated_agent_evaluation.backends.langfuse import LangfuseBackend, LangfuseLike
    from test_backends_langfuse import StubLangfuseClient

    client = StubLangfuseClient()
    real = cli_backends.backend_from_flag
    monkeypatch.setattr(
        cli_backends,
        "backend_from_flag",
        lambda flag, **kw: (
            LangfuseBackend(cast("LangfuseLike", client))
            if flag == "langfuse"
            else real(flag, **kw)
        ),
    )
    monkeypatch.setenv("LANGFUSE_PUBLIC_KEY", "pk-lf-calibrated-agent-evaluation-test")
    _stub_aws(monkeypatch, "phase2")
    rc = main(
        [
            "--root",
            str(tmp_path),
            "--backend",
            "langfuse",
            "probe",
            "--executor",
            "agentcore",
            "--runtime-arn",
            RUNTIME_ARN,
            "--payload-format",
            "agentcore_sse",
            "--policy-cases",
            str(_RAW / "cases.json"),
            "--authorized",
            "--yes",
        ]
    )
    assert rc == EXIT_OK
    assert "6 probed trace(s) labelled FAIL" in _labels_line(capsys.readouterr().out)
    backend = LangfuseBackend(cast("LangfuseLike", client))
    [record] = load_probes(tmp_path / "records" / "probe", "json")
    denied = [
        e["probe_trace_id"] for f in record.findings for e in f.payload["trials"] if e["denied"]
    ]
    assert all([s.kind for s in backend.get_scores(t)] == ["policy-engine"] for t in denied)


def test_an_unreachable_trace_store_is_refused_before_the_plan_and_any_runtime_call(
    tmp_path: Path,
    no_aws: None,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    import calibrated_agent_evaluation.cli.backends as cli_backends

    class _Unreachable:
        def check_reachable(self) -> None:
            from calibrated_agent_evaluation.schema.errors import BackendError

            raise BackendError("Langfuse reachability check failed: connection refused")

        def close(self) -> None:
            return None

    real = cli_backends.backend_from_flag
    monkeypatch.setattr(
        cli_backends,
        "backend_from_flag",
        lambda flag, **kw: _Unreachable() if flag == "langfuse" else real(flag, **kw),
    )
    monkeypatch.setenv("LANGFUSE_PUBLIC_KEY", "pk-lf-calibrated-agent-evaluation-test")
    monkeypatch.setenv("LANGFUSE_SECRET_KEY", "sk-lf-calibrated-agent-evaluation-test")
    rc = main(
        [
            "--root",
            str(tmp_path),
            "--backend",
            "langfuse",
            "probe",
            "--executor",
            "agentcore",
            "--runtime-arn",
            _ARN,
            "--payload-format",
            "agentcore_sse",
            "--policy-cases",
            str(_RAW / "cases.json"),
            "--authorized",
            "--yes",
        ]
    )
    err = capsys.readouterr().err
    assert rc == EXIT_DOMAIN_ERROR
    assert "reachability check failed" in err
    assert "this run can call AWS" not in err


def test_an_agent_error_trial_with_a_deny_has_its_trace_and_label_stored(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    from calibrated_agent_evaluation.backends.local import LocalBackend

    _stub_aws(monkeypatch, "phase2", error_at=7)
    assert _battery_cli(tmp_path) == EXIT_OK
    out = capsys.readouterr().out
    assert "ended in an error the agent reported (ToolDenied)" in out
    [record] = load_probes(tmp_path / "records" / "probe", "json")
    [entry] = [e for f in record.findings for e in f.payload["trials"] if e["error_code"]]
    assert (entry["error_code"], entry["denied"]) == ("ToolDenied", True)
    store = LocalBackend(tmp_path)
    assert store.get_trace(entry["probe_trace_id"]).output == {"error_code": "ToolDenied"}
    [label] = store.get_scores(entry["probe_trace_id"])
    assert "reported an error (ToolDenied)" in label.critique


@pytest.mark.parametrize("failing", ["put_score", "put_trace"])
def test_a_failed_label_write_names_the_stored_record_and_the_manifest(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    failing: str,
) -> None:
    from calibrated_agent_evaluation.backends.local import LocalBackend
    from calibrated_agent_evaluation.schema.errors import BackendError

    calls = {"put_trace": 0}
    real_put_trace = LocalBackend.put_trace

    def put_trace(self: LocalBackend, trace: Any) -> None:
        calls["put_trace"] += 1
        if failing == "put_trace" and calls["put_trace"] == 3:
            raise BackendError("the trace store refused the write")
        real_put_trace(self, trace)

    def put_score(self: LocalBackend, label: Any) -> None:
        raise BackendError("the trace store refused the write")

    monkeypatch.setattr(LocalBackend, "put_trace", put_trace)
    if failing == "put_score":
        monkeypatch.setattr(LocalBackend, "put_score", put_score)
    _stub_aws(monkeypatch, "phase2")

    assert _battery_cli(tmp_path) == EXIT_DOMAIN_ERROR

    err = capsys.readouterr().err
    [record] = load_probes(tmp_path / "records" / "probe", "json")
    assert f"probe record {record.probe_id} is stored" in err
    assert "agentcore_run_record.json" in err
    assert "BackendError" in err
    store = LocalBackend(tmp_path)
    trials = [e["probe_trace_id"] for f in record.findings for e in f.payload["trials"]]
    assert all(store.get_scores(t) == [] for t in trials)


def test_a_failed_probe_record_write_names_the_manifest_and_stores_no_trace(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    from calibrated_agent_evaluation.backends.local import LocalBackend
    from calibrated_agent_evaluation.cli.stages import probe as probe_stage
    from calibrated_agent_evaluation.probe.errors import ProbeError

    def refuse(*args: Any, **kwargs: Any) -> str:
        raise ProbeError("the probe record store refused the write")

    monkeypatch.setattr(probe_stage, "persist_findings", refuse)
    _stub_aws(monkeypatch, "phase2")

    assert _battery_cli(tmp_path) == EXIT_DOMAIN_ERROR

    err = capsys.readouterr().err
    assert "agentcore_run_record.json" in err
    assert "refused the write" in err
    assert LocalBackend(tmp_path).query() == []


def test_access_denied_on_the_read_check_exits_1_before_any_runtime_call(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    from botocore.exceptions import ClientError

    class _Denied(_RecordedLogs):
        def start_query(self, **params: Any) -> dict[str, str]:
            raise ClientError(
                {"Error": {"Code": "AccessDeniedException", "Message": "not allowed"}},
                "StartQuery",
            )

    _stub_aws(monkeypatch, "phase2", logs_class=_Denied)
    assert _battery_cli(tmp_path) == EXIT_DOMAIN_ERROR
    err = capsys.readouterr().err
    assert "aws/spans" in err
    assert "AccessDeniedException" in err
    assert not (tmp_path / "records" / "agentcore").exists()
    assert not (tmp_path / "records" / "probe").exists()


def test_a_refused_battery_stores_no_trace_and_no_label(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    from calibrated_agent_evaluation.backends.local import LocalBackend

    _stub_aws(monkeypatch, "phase1b")
    assert _battery_cli(tmp_path) == EXIT_DOMAIN_ERROR
    assert "PolicyDecisionsMissing" in capsys.readouterr().err
    store = LocalBackend(tmp_path)
    assert store.all_traces() == []
    assert store.all_scores() == []


def test_a_billed_call_budget_refusal_stops_the_battery_and_stores_no_record(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    from calibrated_agent_evaluation.backends.local import LocalBackend

    monkeypatch.setenv("CALIBRATED_AGENT_EVALUATION_AGENTCORE_MAX_BILLED_CALLS", "4")
    logs = _stub_aws(monkeypatch, "phase2")
    assert _battery_cli(tmp_path) == EXIT_DOMAIN_ERROR
    err = capsys.readouterr().err
    assert "CALIBRATED_AGENT_EVALUATION_AGENTCORE_MAX_BILLED_CALLS" in err
    assert not (tmp_path / "records" / "probe").exists()
    assert LocalBackend(tmp_path).all_traces() == []
    assert logs.read_checks == 1
