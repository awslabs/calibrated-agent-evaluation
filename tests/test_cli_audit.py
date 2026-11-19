# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""``calibrated-agent-evaluation audit abtest`` and ``arms`` end to end, with no AWS call reaching
the wire."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, ClassVar, cast

import pytest

pytest.importorskip("boto3")

from botocore.stub import Stubber

from _arms_fixture import MAIN, Spec, build, otlp_id
from calibrated_agent_evaluation.agentcore import optimization
from calibrated_agent_evaluation.agentcore.sources import RawTrace
from calibrated_agent_evaluation.cli import main
from calibrated_agent_evaluation.cli.dispatch import EXIT_DOMAIN_ERROR, EXIT_OK, EXIT_USAGE

_RAW = Path(__file__).resolve().parents[1] / "fixtures" / "raw" / "agentcore" / "abtest"
_BUNDLE_ID = "fixture-bundle-AbCdEf1234"
_TEST_ID = "fixture-abtest-stopped"
_WINDOW = ["--since", "2026-10-28T19:00:00+00:00", "--until", "2026-10-28T20:00:00+00:00"]


def _fixture(name: str) -> dict[str, Any]:
    return json.loads((_RAW / name).read_text(encoding="utf-8"))


@pytest.fixture
def aws(monkeypatch: pytest.MonkeyPatch) -> Any:
    """The audit's real clients, with the AgentCore reads stubbed; no request leaves the process."""
    for key, value in {
        "AWS_ACCESS_KEY_ID": "testing",
        "AWS_SECRET_ACCESS_KEY": "testing",
        "AWS_DEFAULT_REGION": "us-west-2",
        "CALIBRATED_AGENT_EVALUATION_AGENTCORE_SETTLE_SECONDS": "0",
    }.items():
        monkeypatch.setenv(key, value)
    monkeypatch.delenv("AWS_PROFILE", raising=False)
    built: list[Any] = []
    real = optimization.audit_clients

    def clients(region: str | None = None, profile: str | None = None) -> Any:
        made = real(region or "us-west-2", profile)
        built.append(made)
        return made

    monkeypatch.setattr(optimization, "audit_clients", clients)
    return built


def _stub_abtest(clients: Any, test: str = "abtest_stopped.json") -> list[Stubber]:
    data = Stubber(clients.client("bedrock-agentcore"))
    control = Stubber(clients.client("bedrock-agentcore-control"))
    doc = _fixture(test)
    data.add_response("get_ab_test", doc, {"abTestId": doc["abTestId"]})
    for name in ("bundle_version_control.json", "bundle_version_treatment.json"):
        version = _fixture(name)
        control.add_response(
            "get_configuration_bundle_version",
            version,
            {"bundleId": _BUNDLE_ID, "versionId": version["versionId"]},
        )
    data.activate()
    control.activate()
    return [data, control]


def _run_abtest(
    root: Path, monkeypatch: pytest.MonkeyPatch, test: str = "abtest_stopped.json"
) -> int:
    stubs: list[Stubber] = []
    real = optimization.audit_clients

    def stubbed(region: str | None = None, profile: str | None = None) -> Any:
        made = real(region, profile)
        stubs.extend(_stub_abtest(made, test))
        return made

    monkeypatch.setattr(optimization, "audit_clients", stubbed)
    doc = _fixture(test)
    try:
        return main(["--root", str(root), "audit", "abtest", "--ab-test-id", doc["abTestId"]])
    finally:
        for stub in stubs:
            stub.deactivate()
        # The next read stubs the real clients again, not this read's stubbed ones.
        monkeypatch.setattr(optimization, "audit_clients", real)


def _claims(root: Path) -> list[dict[str, Any]]:
    path = root / "records" / "agentcore-abtest" / "agentcore_abtest_record.json"
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else []


def test_audit_abtest_stores_the_claim(
    tmp_path: Path, aws: Any, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    assert _run_abtest(tmp_path, monkeypatch) == EXIT_OK

    out = capsys.readouterr().out
    assert "AgentCore reported (vendor claim, not evidence)" in out
    [claim] = _claims(tmp_path)
    assert claim["ab_test_id"] == _TEST_ID
    assert "F036-CANARY-PROMPT" not in json.dumps(claim)


def test_audit_abtest_on_a_missing_test_exits_one(
    tmp_path: Path, aws: Any, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    real = optimization.audit_clients
    stubs: list[Stubber] = []

    def missing(region: str | None = None, profile: str | None = None) -> Any:
        made = real(region, profile)
        stub = Stubber(made.client("bedrock-agentcore"))
        stub.add_client_error("get_ab_test", service_error_code="ResourceNotFoundException")
        stub.activate()
        stubs.append(stub)
        return made

    monkeypatch.setattr(optimization, "audit_clients", missing)
    code = main(["--root", str(tmp_path), "audit", "abtest", "--ab-test-id", "nope"])
    for stub in stubs:
        stub.deactivate()

    assert code == EXIT_DOMAIN_ERROR
    err = capsys.readouterr().err
    assert "GetABTest" in err
    assert "nope" in err
    assert _claims(tmp_path) == []


@pytest.mark.parametrize("test", ["abtest_variant_both.json", "abtest_variant_neither.json"])
def test_a_test_whose_variant_names_both_or_neither_writes_nothing(
    tmp_path: Path, aws: Any, monkeypatch: pytest.MonkeyPatch, test: str
) -> None:
    real = optimization.audit_clients
    stubs: list[Stubber] = []

    def refusing(region: str | None = None, profile: str | None = None) -> Any:
        made = real(region, profile)
        stub = Stubber(made.client("bedrock-agentcore"))
        doc = _fixture(test)
        stub.add_response("get_ab_test", doc, {"abTestId": doc["abTestId"]})
        stub.activate()
        stubs.append(stub)
        return made

    monkeypatch.setattr(optimization, "audit_clients", refusing)
    code = main(
        ["--root", str(tmp_path), "audit", "abtest", "--ab-test-id", _fixture(test)["abTestId"]]
    )
    for stub in stubs:
        stub.deactivate()

    assert code == EXIT_DOMAIN_ERROR
    assert _claims(tmp_path) == []


# --- arms -------------------------------------------------------------------------------------


class _FakeReader:
    """Stands in for the ingest's reader classes: yields fixture raw traces, makes no AWS call."""

    made: ClassVar[list[str]] = []
    traces: ClassVar[list[RawTrace]] = []

    def __init__(self, clients: Any, config: Any) -> None:
        type(self).made.append(type(self).__name__)

    def read_traces(self) -> Any:
        yield from type(self).traces

    def read_scores(self) -> Any:
        return iter(())


class _FakeOmni(_FakeReader):
    pass


class _FakeInsights(_FakeReader):
    pass


def _raw(spec: Spec) -> RawTrace:
    spans = build(spec)["resourceSpans"][0]["scopeSpans"][0]["spans"]
    sessions = tuple(
        dict.fromkeys(
            s["attributes"]["session.id"] for s in spans if "session.id" in s["attributes"]
        )
    )
    content = [
        {
            "spanId": s["spanId"],
            "eventName": "gen_ai.user.message",
            "body": {"content": [{"text": "q"}]},
        }
        for s in spans
        if s["name"].startswith("agent.")
    ]
    return RawTrace(
        trace_id=spec.trace_id,
        span_records=spans,
        log_records=content,
        session_ids=sessions,
        # As the ingest's Logs Insights reader sets it when a node span's row comes first: the
        # trace keeps its own id when re-keyed.
        runtime_session_id=spec.trace_id,
    )


@pytest.fixture
def readers(monkeypatch: pytest.MonkeyPatch) -> type[_FakeReader]:
    from calibrated_agent_evaluation.agentcore import readers as f029

    _FakeReader.made = []
    _FakeReader.traces = [_raw(s) for s in MAIN]
    monkeypatch.setattr(f029, "OmniReader", _FakeOmni)
    monkeypatch.setattr(f029, "LogsInsightsReader", _FakeInsights)
    return _FakeReader


def _arms(root: Path, *extra: str) -> int:
    return main(
        [
            "--root",
            str(root),
            "audit",
            "arms",
            "--ab-test-id",
            _TEST_ID,
            "--runtime",
            "fixture-runtime",
            *_WINDOW,
            *extra,
        ]
    )


def _assignments(root: Path) -> list[dict[str, Any]]:
    path = root / "records" / "agentcore-arms" / "agentcore_arms_record.json"
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else []


def test_audit_arms_reads_through_the_agentcore_source(
    tmp_path: Path,
    aws: Any,
    monkeypatch: pytest.MonkeyPatch,
    readers: type[_FakeReader],
    capsys: pytest.CaptureFixture[str],
) -> None:
    assert _run_abtest(tmp_path, monkeypatch) == EXIT_OK
    monkeypatch.setattr(
        optimization, "audit_clients", lambda region=None, profile=None: _NoClients()
    )

    assert _arms(tmp_path, "--rekey-sessions") == EXIT_OK
    assert readers.made == ["_FakeInsights"]
    stored = {a["trace_id"]: a for a in _assignments(tmp_path)}
    assert stored[otlp_id("c-1")]["status"] == "assigned"
    assert stored[otlp_id("c-1")]["variant"] == "C"
    assert stored[otlp_id("c-1")]["rekey_sessions"] is True
    assert stored[otlp_id("c-1")]["omni"] is False
    assert stored[otlp_id("split-c")]["status"] == "refused"
    out = capsys.readouterr().out
    assert "arms read arms-" in out

    assert _arms(tmp_path, "--rekey-sessions", "--omni") == EXIT_OK
    assert readers.made == ["_FakeInsights", "_FakeOmni"]
    assert len({a["read_id"] for a in _assignments(tmp_path)}) == 2


def test_the_reader_arms_wraps_is_the_one_agentcore_source_picks() -> None:
    from calibrated_agent_evaluation.agentcore.readers import (
        LogsInsightsReader,
        OmniReader,
        agentcore_source,
    )
    from calibrated_agent_evaluation.agentcore.sources import AgentCoreSourceConfig, TimeWindow

    window = TimeWindow(start=_WINDOW[1], end=_WINDOW[3])
    for omni, expected in ((True, OmniReader), (False, LogsInsightsReader)):
        source = agentcore_source(
            cast("Any", _NoClients()), AgentCoreSourceConfig(runtime="r", window=window, omni=omni)
        )
        assert type(source._reader) is expected  # pins the ingest's choice


class _NoClients:
    """Clients that are never asked for a client: the fake readers make no AWS call."""

    def __enter__(self) -> _NoClients:
        return self

    def __exit__(self, *exc: object) -> None:
        return None

    def client(self, name: str) -> Any:
        raise AssertionError(f"no AWS client should be built ({name})")


def test_audit_arms_without_a_runtime_exits_one_before_any_aws_call(
    tmp_path: Path, aws: Any, capsys: pytest.CaptureFixture[str]
) -> None:
    code = main(["--root", str(tmp_path), "audit", "arms", "--ab-test-id", _TEST_ID, *_WINDOW])

    assert code == EXIT_USAGE
    assert "--runtime" in capsys.readouterr().err
    assert aws == []


def test_audit_arms_needs_a_stored_claim_record_and_one_named_when_several(
    tmp_path: Path,
    aws: Any,
    monkeypatch: pytest.MonkeyPatch,
    readers: type[_FakeReader],
    capsys: pytest.CaptureFixture[str],
) -> None:
    assert _arms(tmp_path) == EXIT_DOMAIN_ERROR
    assert "calibrated-agent-evaluation audit abtest" in capsys.readouterr().err

    assert _run_abtest(tmp_path, monkeypatch) == EXIT_OK
    assert _run_abtest(tmp_path, monkeypatch) == EXIT_OK
    capsys.readouterr()
    ids = [c["record_id"] for c in _claims(tmp_path)]
    assert len(ids) == 2
    monkeypatch.setattr(
        optimization, "audit_clients", lambda region=None, profile=None: _NoClients()
    )

    assert _arms(tmp_path, "--rekey-sessions") == EXIT_DOMAIN_ERROR
    err = capsys.readouterr().err
    assert all(i in err for i in ids)
    assert "--claim-record" in err
    assert readers.made == []
    assert _arms(tmp_path, "--rekey-sessions", "--claim-record", ids[0]) == EXIT_OK


def test_every_arms_read_is_kept_and_a_changed_reread_is_refused(
    tmp_path: Path,
    aws: Any,
    monkeypatch: pytest.MonkeyPatch,
    readers: type[_FakeReader],
    capsys: pytest.CaptureFixture[str],
) -> None:
    assert _run_abtest(tmp_path, monkeypatch) == EXIT_OK
    monkeypatch.setattr(
        optimization, "audit_clients", lambda region=None, profile=None: _NoClients()
    )
    assert _arms(tmp_path, "--rekey-sessions") == EXIT_OK
    stored = tmp_path / "records" / "agentcore-arms" / "agentcore_arms_record.json"
    before = stored.read_bytes()
    capsys.readouterr()

    assert _arms(tmp_path, "--rekey-sessions") == EXIT_OK
    assert "unchanged" in capsys.readouterr().out
    assert stored.read_bytes() == before

    readers.traces = [_raw(s) for s in MAIN if s.label != "t-2"]
    assert _arms(tmp_path, "--rekey-sessions") == EXIT_DOMAIN_ERROR
    err = capsys.readouterr().err
    assert otlp_id("t-2") in err
    assert stored.read_bytes() == before


def test_assigned_ids_the_store_lacks_are_counted_beside_the_read_path(
    tmp_path: Path,
    aws: Any,
    monkeypatch: pytest.MonkeyPatch,
    readers: type[_FakeReader],
    capsys: pytest.CaptureFixture[str],
) -> None:
    from calibrated_agent_evaluation.backends.local import LocalBackend
    from calibrated_agent_evaluation.schema.serde import trace_from_dict

    fixture = Path(__file__).resolve().parents[1] / "fixtures" / "normalized" / "CLM-4471.json"
    backend = LocalBackend(tmp_path)
    backend.put_trace(trace_from_dict(json.loads(fixture.read_text(encoding="utf-8"))))
    backend.close()
    assert _run_abtest(tmp_path, monkeypatch) == EXIT_OK
    monkeypatch.setattr(
        optimization, "audit_clients", lambda region=None, profile=None: _NoClients()
    )
    capsys.readouterr()

    assert _arms(tmp_path, "--rekey-sessions") == EXIT_OK

    out = capsys.readouterr().out
    assert "9 assigned trace id(s) not in this root's trace store" in out
    assert "read path Logs Insights, re-key sessions on" in out


@pytest.mark.parametrize("omni", [False, True], ids=["logs-insights", "omni"])
def test_audit_arms_reads_through_the_agentcore_source_over_its_real_readers(
    tmp_path: Path, aws: Any, monkeypatch: pytest.MonkeyPatch, omni: bool
) -> None:
    from _arms_fixture import content_records, span_records, stub_insights, stub_omni

    assert _run_abtest(tmp_path, monkeypatch) == EXIT_OK
    real = optimization.audit_clients
    stubs: list[Stubber] = []

    def stubbed(region: str | None = None, profile: str | None = None) -> Any:
        made = real(region, profile)
        stub = Stubber(made.client("cloudwatchomni" if omni else "logs"))
        (stub_omni if omni else stub_insights)(stub, span_records(MAIN), content_records(MAIN))
        stub.activate()
        stubs.append(stub)
        return made

    monkeypatch.setattr(optimization, "audit_clients", stubbed)
    flags = ["--rekey-sessions", *(["--omni"] if omni else [])]
    code = _arms(tmp_path, *flags)
    for stub in stubs:
        stub.assert_no_pending_responses()
        stub.deactivate()

    assert code == EXIT_OK
    stored = {a["trace_id"]: a for a in _assignments(tmp_path)}
    assert stored[otlp_id("c-1")]["variant"] == "C"
    assert stored[otlp_id("t-1")]["variant"] == "T1"
    assert stored[otlp_id("split-c")]["status"] == "refused"
    assert stored[otlp_id("c-nosession")]["runtime_session_id"] is None
    assert {a["omni"] for a in stored.values()} == {omni}
