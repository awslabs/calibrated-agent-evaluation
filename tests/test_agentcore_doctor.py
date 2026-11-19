# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""`calibrated-agent-evaluation doctor --agentcore` checks report pass, fail or unknown from real
AWS response shapes."""

from __future__ import annotations

from contextlib import ExitStack
from datetime import UTC, datetime
from typing import Any

import pytest

pytest.importorskip("boto3")

import boto3
from botocore.exceptions import EndpointConnectionError, ReadTimeoutError, UnknownEndpointError
from botocore.stub import Stubber

from calibrated_agent_evaluation.agentcore import AwsClients
from calibrated_agent_evaluation.agentcore.doctor import (
    DEFAULT_CHECKS,
    CheckResult,
    ContentExtractionCheck,
    DoctorTarget,
    IndexingCheck,
    OmniDomainCheck,
    RegionCheck,
    RuntimeLogGroupCheck,
    SpanLogGroupCheck,
    read_only_guard,
    run_checks,
)
from calibrated_agent_evaluation.schema.errors import CalibratedAgentEvaluationDataError

_REGION = "us-west-2"
_RUNTIME_ID = "nl_sql_agent-AbCdEf1234"
_NOW = datetime(2026, 10, 20, tzinfo=UTC)


def _runtime_summary(name: str = "nl_sql_agent", runtime_id: str = _RUNTIME_ID) -> dict[str, Any]:
    return {
        "agentRuntimeArn": f"arn:aws:bedrock-agentcore:{_REGION}:000000000000:runtime/{runtime_id}",
        "agentRuntimeId": runtime_id,
        "agentRuntimeVersion": "24",
        "agentRuntimeName": name,
        "description": "fixture runtime",
        "lastUpdatedAt": _NOW,
        "status": "READY",
    }


def _runtime_detail(env: dict[str, str] | None) -> dict[str, Any]:
    detail: dict[str, Any] = {
        "agentRuntimeArn": _runtime_summary()["agentRuntimeArn"],
        "agentRuntimeName": "nl_sql_agent",
        "agentRuntimeId": _RUNTIME_ID,
        "agentRuntimeVersion": "24",
        "createdAt": _NOW,
        "lastUpdatedAt": _NOW,
        "roleArn": "arn:aws:iam::000000000000:role/runtime",
        "networkConfiguration": {"networkMode": "PUBLIC"},
        "status": "READY",
        "lifecycleConfiguration": {"idleRuntimeSessionTimeout": 900, "maxLifetime": 28800},
    }
    if env is not None:
        detail["environmentVariables"] = env
    return detail


def _domain(name: str = "nl-sql-agent", status: str = "ACTIVE") -> dict[str, Any]:
    return {
        "domainId": "dom-1234567890",
        "domainArn": f"arn:aws:cloudwatch:{_REGION}:000000000000:domain/dom-1234567890",
        "name": name,
        "region": _REGION,
        "createdAt": _NOW,
        "updatedAt": _NOW,
        "status": status,
    }


def _indexing(percent: float) -> dict[str, Any]:
    return {
        "IndexingRules": [
            {"Name": "Default", "Rule": {"Probabilistic": {"DesiredSamplingPercentage": percent}}}
        ]
    }


class Account:
    """Stubbed AWS clients for one region, guarded exactly as the doctor command guards them."""

    def __init__(self, raises: dict[str, Exception] | None = None) -> None:
        session = boto3.Session(
            region_name=_REGION, aws_access_key_id="testing", aws_secret_access_key="testing"
        )
        failures = raises or {}

        def network(**kwargs: Any) -> None:
            # Stands in for the transport failing on these operations; runs before the stub.
            failure = failures.get(kwargs["model"].name)
            if failure is not None:
                raise failure

        self.clients = AwsClients(session, handlers=[read_only_guard(DEFAULT_CHECKS), network])
        self.stubs: dict[str, Stubber] = {}
        self.respond(
            "sts",
            "get_caller_identity",
            {
                "UserId": "AROA:live",
                "Account": "000000000000",
                "Arn": "arn:aws:sts::000000000000:assumed-role/calibrated-agent-evaluation-live/s",
            },
        )

    def stub(self, service: str) -> Stubber:
        if service not in self.stubs:
            self.stubs[service] = Stubber(self.clients.client(service))
        return self.stubs[service]

    def respond(self, service: str, method: str, response: dict[str, Any]) -> Account:
        self.stub(service).add_response(method, response)
        return self

    def fail(self, service: str, method: str, code: str) -> Account:
        self.stub(service).add_client_error(method, service_error_code=code, http_status_code=400)
        return self

    def healthy(
        self, *, runtime_pages: int = 1, domain_pages: int = 1, span_pages: int = 1
    ) -> Account:
        self.respond("bedrock-agentcore-control", "list_evaluators", {"evaluators": []})
        self.respond(
            "xray",
            "get_trace_segment_destination",
            {"Destination": "CloudWatchLogs", "Status": "ACTIVE"},
        )
        self.respond("xray", "get_indexing_rules", _indexing(100.0))
        self._paged(
            "logs",
            "describe_log_groups",
            "logGroups",
            {"logGroupName": "aws/spans"},
            {"logGroupName": "aws/spans-archive"},
            span_pages,
            "nextToken",
        )
        self._runtime_listing(runtime_pages)
        self.respond(
            "logs",
            "describe_log_groups",
            {
                "logGroups": [
                    {"logGroupName": f"/aws/bedrock-agentcore/runtimes/{_RUNTIME_ID}-DEFAULT"}
                ]
            },
        )
        self._paged(
            "cloudwatchomni",
            "list_domains",
            "items",
            _domain(),
            _domain("other"),
            domain_pages,
            "nextToken",
        )
        self._runtime_listing(runtime_pages)
        self.respond(
            "bedrock-agentcore-control",
            "get_agent_runtime",
            _runtime_detail({"AWS_GENAI_CONTENT_EXTRACTION_OPT_OUT": "true"}),
        )
        return self

    def _runtime_listing(self, pages: int) -> None:
        self._paged(
            "bedrock-agentcore-control",
            "list_agent_runtimes",
            "agentRuntimes",
            _runtime_summary(),
            _runtime_summary("other", "other-0000000001"),
            pages,
            "nextToken",
        )

    def _paged(
        self, service: str, method: str, key: str, target: Any, filler: Any, pages: int, token: str
    ) -> None:
        for page in range(1, pages):
            self.respond(service, method, {key: [filler], token: f"page-{page + 1}"})
        self.respond(service, method, {key: [target]})

    def run(self, target: DoctorTarget, checks: tuple[Any, ...] = DEFAULT_CHECKS) -> Any:
        with ExitStack() as stack:
            for stub in self.stubs.values():
                stack.enter_context(stub)
            return run_checks(checks, self.clients, target)


def _target(**kw: Any) -> DoctorTarget:
    base: dict[str, Any] = {
        "region": _REGION,
        "runtime_name": "nl_sql_agent",
        "omni_domain": "nl-sql-agent",
    }
    return DoctorTarget(**{**base, **kw})


def _by_name(report: Any) -> dict[str, CheckResult]:
    return {r.name: r for r in report.results}


def test_a_healthy_account_passes_all_seven_checks_with_what_was_observed() -> None:
    report = Account().healthy().run(_target())

    results = _by_name(report)
    assert report.ok
    assert list(results) == [
        "region",
        "transaction_search",
        "indexing",
        "span_log_group",
        "runtime_log_group",
        "omni_domain",
        "content_extraction",
    ]
    assert results["transaction_search"].observed == "Destination=CloudWatchLogs Status=ACTIVE"
    assert "100.0%" in results["indexing"].observed
    assert _RUNTIME_ID in results["runtime_log_group"].observed
    assert results["content_extraction"].observed == "AWS_GENAI_CONTENT_EXTRACTION_OPT_OUT=true"
    assert report.identity.endswith("assumed-role/calibrated-agent-evaluation-live/s")


def test_targets_on_the_second_page_of_a_listing_are_found() -> None:
    report = Account().healthy(runtime_pages=2, domain_pages=2, span_pages=2).run(_target())

    assert report.ok, [r for r in report.results if r.status != "pass"]


def test_a_denied_call_is_unknown_naming_the_action_and_the_rest_still_run() -> None:
    account = Account()
    account.respond("bedrock-agentcore-control", "list_evaluators", {"evaluators": []})
    account.respond(
        "xray",
        "get_trace_segment_destination",
        {"Destination": "CloudWatchLogs", "Status": "ACTIVE"},
    )
    account.fail("xray", "get_indexing_rules", "AccessDeniedException")

    report = account.run(_target(), checks=DEFAULT_CHECKS[:3])

    results = _by_name(report)
    assert results["indexing"].status == "unknown"
    assert results["indexing"].next_step == "grant xray:GetIndexingRules"
    assert results["region"].status == "pass"
    assert results["transaction_search"].status == "pass"


def test_network_and_throttling_failures_are_unknown_and_the_other_checks_still_pass() -> None:
    account = Account(
        raises={
            "GetIndexingRules": ReadTimeoutError(
                endpoint_url=f"https://xray.{_REGION}.amazonaws.com"
            ),
            "DescribeLogGroups": EndpointConnectionError(
                endpoint_url=f"https://logs.{_REGION}.amazonaws.com"
            ),
        }
    ).healthy()
    account.stubs["cloudwatchomni"] = Stubber(account.clients.client("cloudwatchomni"))
    account.fail("cloudwatchomni", "list_domains", "ThrottlingException")

    results = _by_name(account.run(_target()))

    for name, observed in (
        ("indexing", "timeout"),
        ("span_log_group", "could not connect"),
        ("runtime_log_group", "could not connect"),
        ("omni_domain", "throttled"),
    ):
        assert results[name].status == "unknown", results[name]
        assert observed in results[name].observed.lower()
        assert results[name].next_step
    for name in ("region", "transaction_search", "content_extraction"):
        assert results[name].status == "pass", results[name]


class _FakeEndpointClient:
    def __init__(self, error: Exception | None) -> None:
        self.error = error
        self.meta = type(
            "Meta",
            (),
            {"endpoint_url": f"https://bedrock-agentcore-control.{_REGION}.amazonaws.com"},
        )

    def list_evaluators(self, **_: Any) -> dict[str, Any]:
        if self.error is not None:
            raise self.error
        return {"evaluators": []}


class _FakeClients:
    def __init__(self, error: Exception | None) -> None:
        self._client = _FakeEndpointClient(error)

    def client(self, _service: str) -> _FakeEndpointClient:
        return self._client


def _region(error: Exception | None, resolves: dict[str, bool] | None = None) -> CheckResult:
    table = resolves or {}

    def resolve(host: str) -> bool:
        return table.get(host, True)

    check = RegionCheck(resolves=resolve)
    return check.run(_FakeClients(error), _target())  # type: ignore[arg-type]


_AGENTCORE_HOST = f"bedrock-agentcore-control.{_REGION}.amazonaws.com"
_STS_HOST = f"sts.{_REGION}.amazonaws.com"


def test_region_classification_separates_a_missing_endpoint_from_a_dead_network() -> None:
    from botocore.exceptions import ClientError, ConnectTimeoutError

    denied = ClientError({"Error": {"Code": "AccessDeniedException"}}, "ListEvaluators")
    throttled = ClientError({"Error": {"Code": "ThrottlingException"}}, "ListEvaluators")
    dns = EndpointConnectionError(endpoint_url=f"https://{_AGENTCORE_HOST}")

    assert _region(None).status == "pass"
    assert _region(denied).status == "pass"
    assert _region(throttled).status == "unknown"
    assert _region(UnknownEndpointError(service_name="x", region_name=_REGION)).status == "fail"
    assert _region(dns, {_AGENTCORE_HOST: False, _STS_HOST: True}).status == "fail"
    assert _region(dns, {_AGENTCORE_HOST: False, _STS_HOST: False}).status == "unknown"
    assert _region(ConnectTimeoutError(endpoint_url="https://x")).status == "unknown"


def test_a_dns_lookup_error_during_classification_is_unknown() -> None:
    dns = EndpointConnectionError(endpoint_url="https://raise")

    def resolver_down(_host: str) -> bool:
        raise OSError("resolver down")

    check = RegionCheck(resolves=resolver_down)
    result = check.run(_FakeClients(dns), _target())  # type: ignore[arg-type]

    assert result.status == "unknown"
    assert result.next_step == "retry; network"


def _indexing_result(response: dict[str, Any]) -> CheckResult:
    account = Account().respond("xray", "get_indexing_rules", response)
    (result,) = account.run(_target(), checks=(IndexingCheck(),)).results
    return result


def test_one_percent_indexing_fails_with_the_command_that_raises_it() -> None:
    result = _indexing_result(_indexing(1.0))

    assert result.status == "fail"
    assert "1.0%" in result.observed
    assert result.next_step is not None
    assert "update-indexing-rule --name Default" in result.next_step
    assert 'DesiredSamplingPercentage":100' in result.next_step


def test_full_indexing_passes_and_a_missing_default_rule_fails() -> None:
    assert _indexing_result(_indexing(100.0)).status == "pass"
    other = {
        "IndexingRules": [
            {"Name": "Custom", "Rule": {"Probabilistic": {"DesiredSamplingPercentage": 100.0}}}
        ]
    }
    assert _indexing_result(other).status == "fail"


def _content_result(
    env: dict[str, str] | None, runtime_name: str | None = "nl_sql_agent"
) -> CheckResult:
    account = Account()
    if runtime_name is not None:
        account.respond(
            "bedrock-agentcore-control",
            "list_agent_runtimes",
            {"agentRuntimes": [_runtime_summary()]},
        )
        account.respond("bedrock-agentcore-control", "get_agent_runtime", _runtime_detail(env))
    (result,) = account.run(
        _target(runtime_name=runtime_name), checks=(ContentExtractionCheck(),)
    ).results
    return result


def test_content_extraction_passes_only_when_the_runtime_keeps_content_on_spans() -> None:
    var = "AWS_GENAI_CONTENT_EXTRACTION_OPT_OUT"

    assert _content_result({var: "true"}).status == "pass"
    off = _content_result({var: "false"})
    assert off.status == "fail"
    assert off.next_step is not None
    assert f"{var}=true" in off.next_step
    assert _content_result({}).status == "unknown"
    assert _content_result(None).status == "unknown"


def test_content_extraction_without_a_runtime_is_unknown_and_names_the_variable() -> None:
    result = _content_result(None, runtime_name=None)

    assert result.status == "unknown"
    assert result.next_step is not None
    assert "AWS_GENAI_CONTENT_EXTRACTION_OPT_OUT=true" in result.next_step
    assert "--runtime" in result.next_step


def test_an_unknown_runtime_name_fails_the_runtime_checks() -> None:
    account = Account()
    account.respond(
        "bedrock-agentcore-control",
        "list_agent_runtimes",
        {"agentRuntimes": [_runtime_summary("other")]},
    )

    (result,) = account.run(_target(), checks=(RuntimeLogGroupCheck(),)).results

    assert result.status == "fail"
    assert "nl_sql_agent" in result.observed


def test_a_missing_span_log_group_fails_with_a_next_step() -> None:
    account = Account().respond(
        "logs", "describe_log_groups", {"logGroups": [{"logGroupName": "aws/spans-old"}]}
    )

    (result,) = account.run(_target(), checks=(SpanLogGroupCheck(),)).results

    assert result.status == "fail"
    assert result.next_step


def test_an_inactive_or_missing_omni_domain_fails() -> None:
    def omni(items: list[dict[str, Any]]) -> CheckResult:
        account = Account().respond("cloudwatchomni", "list_domains", {"items": items})
        (result,) = account.run(_target(), checks=(OmniDomainCheck(),)).results
        return result

    assert omni([_domain()]).status == "pass"
    assert omni([_domain("other")]).status == "fail"


def test_a_non_pass_result_without_a_next_step_is_rejected() -> None:
    with pytest.raises(CalibratedAgentEvaluationDataError, match="no next step"):
        CheckResult(name="indexing", status="fail", observed="1%")
    with pytest.raises(CalibratedAgentEvaluationDataError):
        CheckResult(name="region", status="unknown", observed="?", next_step="")


def _cli(account: Account, monkeypatch: pytest.MonkeyPatch, capsys: Any, *extra: str) -> Any:
    from calibrated_agent_evaluation.cli import main

    monkeypatch.setattr(AwsClients, "from_profile", classmethod(lambda *_a, **_k: account.clients))
    with ExitStack() as stack:
        for stub in account.stubs.values():
            stack.enter_context(stub)
        argv = [
            "doctor",
            "--agentcore",
            "--runtime",
            "nl_sql_agent",
            "--omni-domain",
            "nl-sql-agent",
        ]
        code = main([*argv, *extra])
    return code, capsys.readouterr()


def test_the_cli_exits_zero_when_every_check_passes(
    monkeypatch: pytest.MonkeyPatch, capsys: Any
) -> None:
    code, out = _cli(Account().healthy(), monkeypatch, capsys)

    assert code == 0
    assert "7/7 checks passed" in out.out
    assert "[pass] indexing" in out.out


def _one_fail_one_unknown() -> Account:
    account = Account()
    account.respond("bedrock-agentcore-control", "list_evaluators", {"evaluators": []})
    account.respond(
        "xray",
        "get_trace_segment_destination",
        {"Destination": "CloudWatchLogs", "Status": "ACTIVE"},
    )
    account.respond("xray", "get_indexing_rules", _indexing(1.0))
    account.fail("logs", "describe_log_groups", "AccessDeniedException")
    account.respond(
        "bedrock-agentcore-control", "list_agent_runtimes", {"agentRuntimes": [_runtime_summary()]}
    )
    account.respond(
        "logs",
        "describe_log_groups",
        {"logGroups": [{"logGroupName": f"/aws/bedrock-agentcore/runtimes/{_RUNTIME_ID}-DEFAULT"}]},
    )
    account.respond("cloudwatchomni", "list_domains", {"items": [_domain()]})
    account.respond(
        "bedrock-agentcore-control", "list_agent_runtimes", {"agentRuntimes": [_runtime_summary()]}
    )
    account.respond(
        "bedrock-agentcore-control",
        "get_agent_runtime",
        _runtime_detail({"AWS_GENAI_CONTENT_EXTRACTION_OPT_OUT": "true"}),
    )
    return account


def test_a_failing_or_unknown_check_exits_one_and_each_carries_its_next_step(
    monkeypatch: pytest.MonkeyPatch, capsys: Any
) -> None:
    monkeypatch.delenv(
        "CALIBRATED_AGENT_EVALUATION_EXPLAIN"
    )  # the suite turns --explain on; this reads the default
    code, out = _cli(_one_fail_one_unknown(), monkeypatch, capsys)

    assert code == 1
    lines = out.err.splitlines()
    indexing = next(line for line in lines if line.startswith("[fail] indexing"))
    span = next(line for line in lines if line.startswith("[unknown] span_log_group"))
    assert "update-indexing-rule" in indexing
    assert span.endswith("grant logs:DescribeLogGroups")
    assert all(": " in line for line in lines if line.startswith(("[fail]", "[unknown]")))
    assert "1.0%" not in out.err


def test_explain_adds_what_each_check_observed(
    monkeypatch: pytest.MonkeyPatch, capsys: Any
) -> None:
    _code, out = _cli(_one_fail_one_unknown(), monkeypatch, capsys, "--explain")

    assert "indexing: Default rule indexes 1.0% of spans" in out.err
    assert "transaction_search: Destination=CloudWatchLogs Status=ACTIVE" in out.err
