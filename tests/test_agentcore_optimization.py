# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""The A/B audit's AWS reads over the real service models, through botocore's ``Stubber``."""

from __future__ import annotations

import contextlib
import copy
import json
from datetime import datetime
from pathlib import Path
from typing import Any

import pytest

pytest.importorskip("boto3")

from botocore.exceptions import EndpointConnectionError
from botocore.stub import Stubber

from calibrated_agent_evaluation.agentcore.audit_types import dump_abtest_claims
from calibrated_agent_evaluation.agentcore.errors import AgentCoreError, ReadOnlyViolation
from calibrated_agent_evaluation.agentcore.optimization import (
    AUDIT_READ_OPERATIONS,
    INGEST_READ_OPERATIONS,
    audit_clients,
    read_ab_test,
    read_bundle_version,
)
from calibrated_agent_evaluation.agentcore.session import AwsClients

_RAW = Path(__file__).resolve().parents[1] / "fixtures" / "raw" / "agentcore" / "abtest"
_GOLDEN = Path(__file__).resolve().parents[1] / "fixtures" / "expected" / "audit"
_CANARY = "F036-CANARY-PROMPT"
_BUNDLE_ID = "fixture-bundle-AbCdEf1234"
_TIME_KEYS = {
    "startedAt",
    "stoppedAt",
    "maxDurationExpiresAt",
    "createdAt",
    "updatedAt",
    "analysisTimestamp",
    "versionCreatedAt",
}


def _fixture(name: str) -> dict[str, Any]:
    return json.loads((_RAW / name).read_text(encoding="utf-8"))


def _as_response(doc: Any) -> Any:
    """The fixture as botocore parses a response: timestamps as tz-aware datetimes."""
    if isinstance(doc, dict):
        return {
            k: datetime.fromisoformat(v)
            if k in _TIME_KEYS and isinstance(v, str)
            else _as_response(v)
            for k, v in doc.items()
        }
    if isinstance(doc, list):
        return [_as_response(v) for v in doc]
    return doc


@pytest.fixture
def clients(monkeypatch: pytest.MonkeyPatch) -> AwsClients:
    for key, value in {
        "AWS_ACCESS_KEY_ID": "testing",
        "AWS_SECRET_ACCESS_KEY": "testing",
        "AWS_DEFAULT_REGION": "us-west-2",
    }.items():
        monkeypatch.setenv(key, value)
    monkeypatch.delenv("AWS_PROFILE", raising=False)
    monkeypatch.delenv("CALIBRATED_AGENT_EVALUATION_AGENTCORE_MAX_BILLED_CALLS", raising=False)
    return audit_clients(region="us-west-2")


def _stub_test(stub: Stubber, doc: dict[str, Any]) -> None:
    stub.add_response("get_ab_test", _as_response(doc), {"abTestId": doc["abTestId"]})


def _stub_version(stub: Stubber, name: str) -> dict[str, Any]:
    doc = _fixture(name)
    stub.add_response(
        "get_configuration_bundle_version",
        _as_response(doc),
        {"bundleId": _BUNDLE_ID, "versionId": doc["versionId"]},
    )
    return doc


def _read(clients: AwsClients, test: dict[str, Any], versions: list[str]) -> Any:
    data = Stubber(clients.client("bedrock-agentcore"))
    control = Stubber(clients.client("bedrock-agentcore-control"))
    _stub_test(data, test)
    for name in versions:
        _stub_version(control, name)
    with data, control:
        record = read_ab_test(clients, test["abTestId"])
    data.assert_no_pending_responses()
    control.assert_no_pending_responses()
    return record


def _without_generated(record: Any) -> dict[str, Any]:
    doc = record.model_dump(mode="json")
    for key in ("record_id", "read_at", "provenance"):
        doc.pop(key)
    return doc


def test_an_ab_test_is_recorded_as_returned(clients: AwsClients) -> None:
    record = _read(
        clients,
        _fixture("abtest_stopped.json"),
        ["bundle_version_control.json", "bundle_version_treatment.json"],
    )

    expected = json.loads((_GOLDEN / "abtest_verdict.json").read_text(encoding="utf-8"))
    assert _without_generated(record) == expected
    assert record.claim_kind == "vendor-claim"
    assert record.read_at


def test_an_ab_test_without_results_records_no_claims(clients: AwsClients) -> None:
    record = _read(
        clients,
        _fixture("abtest_not_started.json"),
        ["bundle_version_control.json", "bundle_version_treatment.json"],
    )

    assert record.evaluator_claims == ()
    assert record.analysis_timestamp is None
    assert record.error_details == ("the online evaluation configuration is not enabled",)
    assert record.provisional is False


@pytest.mark.parametrize(
    "fixture", ["abtest_variant_both.json", "abtest_variant_neither.json"], ids=["both", "neither"]
)
def test_a_variant_naming_both_or_neither_configuration_is_refused(
    clients: AwsClients, fixture: str
) -> None:
    test = _fixture(fixture)
    data = Stubber(clients.client("bedrock-agentcore"))
    # No bundle-version response is queued, so reading one would fail the test.
    control = Stubber(clients.client("bedrock-agentcore-control"))
    _stub_test(data, test)

    with data, control, pytest.raises(AgentCoreError) as refused:
        read_ab_test(clients, test["abTestId"])

    message = str(refused.value)
    assert "GetABTest" in message
    assert test["abTestId"] in message
    assert "'T1'" in message


def test_identical_bundle_components_mark_an_a_a_test(clients: AwsClients) -> None:
    aa = _read(
        clients,
        _fixture("abtest_aa.json"),
        ["bundle_version_control.json", "bundle_version_aa.json"],
    )
    assert aa.identical_arms is True

    differing = _read(
        clients,
        _fixture("abtest_stopped.json"),
        ["bundle_version_control.json", "bundle_version_treatment.json"],
    )
    assert differing.identical_arms is False
    control, treatment = (v.bundle for v in differing.variants)
    assert control is not None
    assert treatment is not None
    changed_components = [
        a.component_id
        for a, b in zip(control.components, treatment.components, strict=True)
        if a.sha256 != b.sha256
    ]
    assert changed_components == [control.components[0].component_id]
    changed_keys = [
        a.key
        for a, b in zip(control.components[0].keys, treatment.components[0].keys, strict=True)
        if a.sha256 != b.sha256
    ]
    assert changed_keys == ["sql_generator_system_prompt"]

    control_stub = Stubber(clients.client("bedrock-agentcore-control"))
    _stub_version(control_stub, "bundle_version_string_config.json")
    with control_stub:
        version = read_bundle_version(
            clients,
            f"arn:aws:bedrock-agentcore:us-west-2:111122223333:configuration-bundle/{_BUNDLE_ID}",
            _fixture("bundle_version_string_config.json")["versionId"],
        )
    string_component = next(c for c in version.components if not c.keys)
    assert string_component.sha256
    assert string_component.length > 0


@pytest.mark.parametrize(
    ("status", "provisional"),
    [("RUNNING", True), ("PAUSED", True), ("STOPPED", False), ("NOT_STARTED", False)],
)
def test_running_and_paused_tests_are_provisional(
    clients: AwsClients, status: str, provisional: bool
) -> None:
    test = copy.deepcopy(_fixture("abtest_stopped.json"))
    test["executionStatus"] = status

    record = _read(clients, test, ["bundle_version_control.json", "bundle_version_treatment.json"])

    assert record.provisional is provisional
    assert record.execution_status == status


def test_a_target_variant_records_the_target_as_provenance(clients: AwsClients) -> None:
    record = _read(clients, _fixture("abtest_target.json"), ["bundle_version_control.json"])

    by_name = {v.name: v for v in record.variants}
    assert by_name["T1"].bundle is None
    assert by_name["T1"].target_name == "fixture-target-v2"
    assert by_name["C"].bundle is not None
    assert record.identical_arms is False


def test_a_response_missing_a_member_the_read_indexes_is_a_typed_error(
    clients: AwsClients, monkeypatch: pytest.MonkeyPatch
) -> None:
    # botocore validates requests, not responses, so a partial live response reaches the mapping.
    doc = _as_response(_fixture("abtest_stopped.json"))
    del doc["variants"][0]["variantConfiguration"]["configurationBundle"]["bundleArn"]
    monkeypatch.setattr(clients.client("bedrock-agentcore"), "get_ab_test", lambda **_: doc)

    with pytest.raises(AgentCoreError) as refused:
        read_ab_test(clients, doc["abTestId"])

    assert "GetABTest" in str(refused.value)
    assert "'bundleArn'" in str(refused.value)
    assert isinstance(refused.value.__cause__, KeyError)


@pytest.mark.parametrize(
    "code", ["ResourceNotFoundException", "AccessDeniedException", "ThrottlingException"]
)
def test_a_failed_read_is_a_typed_error_and_writes_nothing(clients: AwsClients, code: str) -> None:
    data = Stubber(clients.client("bedrock-agentcore"))
    data.add_client_error("get_ab_test", service_error_code=code, http_status_code=400)

    with data, pytest.raises(AgentCoreError) as refused:
        read_ab_test(clients, "fixture-abtest-missing")

    assert "GetABTest" in str(refused.value)
    assert "fixture-abtest-missing" in str(refused.value)
    assert code in str(refused.value)
    assert refused.value.__cause__ is not None


def test_a_connection_failure_is_a_typed_error(
    clients: AwsClients, monkeypatch: pytest.MonkeyPatch
) -> None:
    def fail(**_kwargs: Any) -> Any:
        raise EndpointConnectionError(endpoint_url="https://bedrock-agentcore.example")

    monkeypatch.setattr(clients.client("bedrock-agentcore"), "get_ab_test", fail)

    with pytest.raises(AgentCoreError) as refused:
        read_ab_test(clients, "fixture-abtest-stopped")

    assert "GetABTest" in str(refused.value)
    assert isinstance(refused.value.__cause__, EndpointConnectionError)


def test_no_prompt_text_reaches_a_stored_record(clients: AwsClients, tmp_path: Path) -> None:
    record = _read(
        clients,
        _fixture("abtest_stopped.json"),
        ["bundle_version_control.json", "bundle_version_treatment.json"],
    )

    dump_abtest_claims([record], tmp_path)

    written = [p.read_text(encoding="utf-8") for p in tmp_path.iterdir() if p.is_file()]
    assert written
    assert all(_CANARY not in text for text in written)


# --- the read-only guard -------------------------------------------------------------------


class _Sent(Exception):
    """Raised at ``before-send``: the call passed every guard and would have gone out."""


def _minimal(shape: Any) -> Any:
    """The smallest value the model's validation accepts for ``shape``."""
    kind = shape.type_name
    if kind == "structure":
        if getattr(shape, "is_tagged_union", False):
            name = next(iter(shape.members))
            return {name: _minimal(shape.members[name])}
        return {name: _minimal(shape.members[name]) for name in shape.required_members}
    if kind == "list":
        size = int(shape.metadata.get("min", 0))
        return [_minimal(shape.member) for _ in range(size)]
    if kind == "map":
        return {}
    if kind == "string":
        return "x" * max(1, int(shape.metadata.get("min", 1)))
    if kind in ("integer", "long"):
        return max(1, int(shape.metadata.get("min", 1)))
    if kind in ("float", "double"):
        return 1.0
    if kind == "boolean":
        return True
    if kind == "timestamp":
        return datetime.fromisoformat("2026-01-01T00:00:00+00:00")
    if kind == "blob":
        return b"x"
    return {}


def _send_or_refuse(clients: AwsClients, service: str, operation: str) -> list[str]:
    """Call ``operation`` with minimal valid input; returns the operations that reached the wire."""
    client = clients.client(service)
    sent: list[str] = []

    def record(**kwargs: Any) -> None:
        sent.append(operation)
        raise _Sent(operation)

    client.meta.events.register("before-send", record)
    model = client.meta.service_model.operation_model(operation)
    params = _minimal(model.input_shape) if model.input_shape is not None else {}
    method = next(m for m, op in client.meta.method_to_api_mapping.items() if op == operation)
    with contextlib.suppress(_Sent):
        getattr(client, method)(**params)
    return sent


@pytest.mark.parametrize(
    ("service", "operation"),
    [
        ("bedrock-agentcore", "StartRecommendation"),
        ("bedrock-agentcore", "CreateABTest"),
        ("bedrock-agentcore", "UpdateABTest"),
        ("bedrock-agentcore", "DeleteABTest"),
        ("bedrock-agentcore", "ListABTests"),
        ("bedrock-agentcore-control", "UpdateConfigurationBundle"),
        ("bedrock-agentcore-control", "ListConfigurationBundleVersions"),
        ("logs", "DescribeLogGroups"),
        ("xray", "GetTraceSummaries"),
    ],
)
def test_audit_clients_refuse_every_call_outside_the_four_reads_before_sending_it(
    clients: AwsClients, service: str, operation: str
) -> None:
    with pytest.raises(ReadOnlyViolation) as refused:
        _send_or_refuse(clients, service, operation)

    assert operation in str(refused.value)


@pytest.mark.parametrize(
    ("service", "operation"),
    sorted(AUDIT_READ_OPERATIONS | INGEST_READ_OPERATIONS),
)
def test_the_audit_reads_and_f029s_trace_reads_pass_the_guard(
    clients: AwsClients, service: str, operation: str
) -> None:
    assert _send_or_refuse(clients, service, operation) == [operation]


def _omni_round(stub: Stubber, rows: list[dict[str, Any]]) -> None:
    stub.add_response("start_telemetry_query_session", {"sessionId": "sess"})
    stub.add_response("start_telemetry_query", {"queryId": "q", "sessionId": "sess"})
    stub.add_response(
        "get_telemetry_query_results",
        {
            "status": "Complete",
            "rows": rows,
            "statistics": {"partialResults": {"partialResultsDetected": False}},
        },
    )
    stub.add_response("stop_telemetry_query_session", {})


def test_audit_clients_let_the_omni_trace_reader_finish_and_stop_its_session(
    clients: AwsClients,
) -> None:
    from calibrated_agent_evaluation.agentcore.readers import OmniReader
    from calibrated_agent_evaluation.agentcore.sources import AgentCoreSourceConfig, TimeWindow

    config = AgentCoreSourceConfig(
        runtime="fixture-runtime",
        window=TimeWindow(start="2026-10-28T19:00:00+00:00", end="2026-10-28T20:00:00+00:00"),
        omni=True,
    )
    stub = Stubber(clients.client("cloudwatchomni"))
    span = {"traceId": "t-1", "spanId": "sp-1", "session_id": "s-1", "@message": json.dumps({})}
    _omni_round(stub, [span])
    _omni_round(stub, [])

    with stub:
        traces = list(OmniReader(clients, config).read_traces())

    assert [t.trace_id for t in traces] == ["t-1"]
    # Both session stops were consumed: a guard without the stop would leave a response queued.
    stub.assert_no_pending_responses()


# --- recommendations ------------------------------------------------------------------------


def _recommendation(name: str, **over: Any) -> Any:
    from _rec_fixture import read_fixture

    return read_fixture(name, **over)


def test_a_recommendation_is_recorded_with_its_input_and_result_versions() -> None:
    record = _recommendation("rec_system_prompt.json")
    golden = json.loads((_GOLDEN / "recommendation.json").read_text(encoding="utf-8"))

    assert _without_generated(record) == golden
    assert [t.tool_name for t in (*record.result_texts, *record.explanations)] == [None, None]
    assert len(record.traces_inline_trace_ids) == 10

    tools = _recommendation("rec_tool_bundle.json")
    assert [(p.tool_name, p.json_path) for p in tools.input_json_paths] == [
        ("lookup_order", "$.configuration.tool_description"),
        ("list_refunds", "$.configuration.refund_tool_description"),
    ]
    by_tool = {t.tool_name: t for t in tools.result_texts}
    explained = {t.tool_name: t for t in tools.explanations}
    assert set(by_tool) == set(explained) == {"lookup_order", "list_refunds"}
    raw = json.loads(
        (_RAW.parent / "recommendations" / "rec_tool_bundle.json").read_text(encoding="utf-8")
    )
    for entry in raw["recommendationResult"]["toolDescriptionRecommendationResult"]["tools"]:
        text = entry["recommendedToolDescription"]
        assert by_tool[entry["toolName"]].length == len(text)
        assert explained[entry["toolName"]].length == len(entry["explanation"])
    assert tools.result_bundle is not None


def test_inline_and_failed_recommendations_are_recorded_without_text(tmp_path: Path) -> None:
    from calibrated_agent_evaluation.agentcore.audit_types import dump_recommendation_claims

    inline = _recommendation("rec_inline_text.json")
    assert inline.input_bundle is None
    [digest] = inline.input_inline
    assert digest.tool_name is None
    tools = _recommendation("rec_tool_inline.json")
    assert [t.tool_name for t in tools.input_inline] == ["lookup_order", "list_refunds"]
    failed = _recommendation("rec_failed.json")
    assert (failed.error_code, failed.status) == ("INSUFFICIENT_TRACES", "FAILED")
    assert failed.error_message
    assert failed.result_bundle is None
    assert failed.result_texts == failed.explanations == ()
    bare = _recommendation("rec_failed_no_result.json")
    assert (bare.status, bare.error_code) == ("FAILED", None)

    dump_recommendation_claims([inline, tools, failed, bare], tmp_path)
    written = [p.read_text(encoding="utf-8") for p in tmp_path.iterdir() if p.is_file()]
    assert written
    assert all(_CANARY not in text for text in written)


@pytest.mark.parametrize(
    ("name", "timeout", "prefixes"),
    [
        ("online_eval_timeout5.json", 5, ()),
        ("online_eval_no_session.json", None, ()),
        ("online_eval_prefix.json", 5, ("/aws/bedrock-agentcore/runtimes/fixture-",)),
    ],
)
def test_an_online_evaluation_config_is_read_for_its_window(
    clients: AwsClients, name: str, timeout: int | None, prefixes: tuple[str, ...]
) -> None:
    from _rec_fixture import online_config
    from calibrated_agent_evaluation.agentcore.optimization import read_online_eval_config

    doc = online_config(name)
    stub = Stubber(clients.client("bedrock-agentcore-control"))
    stub.add_response(
        "get_online_evaluation_config",
        _as_response(doc),
        {"onlineEvaluationConfigId": doc["onlineEvaluationConfigId"]},
    )
    with stub:
        config = read_online_eval_config(clients, doc["onlineEvaluationConfigArn"])

    assert config.session_timeout_minutes == timeout
    assert config.log_group_name_prefixes == prefixes
    assert config.service_name == "fixture-runtime"
    assert config.updated_at == doc["updatedAt"]


def test_a_live_recommendations_capture_reads_as_an_inline_window_recommendation() -> None:
    record = _recommendation("live/recommendation_live.json")

    assert record.input_bundle is None
    assert record.result_bundle is None
    assert record.traces_source == "cloudwatchLogs"
    assert record.traces_service_name == "fixture-runtime.DEFAULT"
    assert len(record.traces_log_group_arns) == 2
    assert record.evaluator_arn == "arn:aws:bedrock-agentcore:::evaluator/Builtin.Helpfulness"
