# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Read-only readiness checks for an AgentCore account: `doctor --agentcore`.

Each check reports pass, fail or unknown with a next step. Unknown means the check couldn't decide
(denied, throttled, unreachable); it is never reported as pass. Every client the checks use carries
a :class:`~calibrated_agent_evaluation.agentcore.guards.ReadOnlyGuard` built from the operations the
checks declare, so a check can't change the account even by mistake.
"""

from __future__ import annotations

import socket
from collections.abc import Callable, Iterator, Sequence
from typing import Any, ClassVar, Literal, Protocol
from urllib.parse import urlparse

from botocore.exceptions import (
    BotoCoreError,
    ClientError,
    ConnectTimeoutError,
    EndpointConnectionError,
    EndpointResolutionError,
    ReadTimeoutError,
    UnknownEndpointError,
)
from pydantic import ConfigDict, Field, ValidationInfo, field_validator

from calibrated_agent_evaluation._model import CalibratedAgentEvaluationModel, replace
from calibrated_agent_evaluation.agentcore.errors import AgentCoreError
from calibrated_agent_evaluation.agentcore.guards import Operation, ReadOnlyGuard
from calibrated_agent_evaluation.agentcore.session import AwsClients

__all__ = [
    "DEFAULT_CHECKS",
    "CheckResult",
    "DoctorReport",
    "DoctorTarget",
    "allowed_operations",
    "run_checks",
]

CONTENT_EXTRACTION_VAR = "AWS_GENAI_CONTENT_EXTRACTION_OPT_OUT"
SPAN_LOG_GROUP = "aws/spans"

CheckStatus = Literal["pass", "fail", "unknown"]

# The IAM namespace each client's actions live in, for "grant <action>" next steps.
_IAM_PREFIX = {
    "bedrock-agentcore-control": "bedrock-agentcore",
    "bedrock-agentcore": "bedrock-agentcore",
    "cloudwatchomni": "cloudwatch",
    "logs": "logs",
    "xray": "xray",
    "sts": "sts",
}
_DENIED = frozenset(
    {
        "AccessDenied",
        "AccessDeniedException",
        "UnauthorizedOperation",
        "UnrecognizedClientException",
    }
)
_THROTTLED = frozenset(
    {"Throttling", "ThrottlingException", "TooManyRequestsException", "RequestLimitExceeded"}
)


class _DoctorModel(CalibratedAgentEvaluationModel):
    model_config = ConfigDict(frozen=True, extra="forbid", defer_build=True)
    invalid_error: ClassVar[type[Exception]] = AgentCoreError


class CheckResult(_DoctorModel):
    """One check's outcome. ``next_step`` is required unless the check passed."""

    name: str
    status: CheckStatus
    observed: str
    next_step: str | None = Field(default=None, validate_default=True)

    @field_validator("next_step")
    @classmethod
    def _non_pass_says_what_to_do(cls, value: str | None, info: ValidationInfo) -> str | None:
        status = info.data.get("status")
        if status not in (None, "pass") and not value:
            raise ValueError(f"check {info.data.get('name')!r} is {status} but gives no next step")
        return value


class DoctorTarget(_DoctorModel):
    """The account slice to check, from the CLI flags."""

    region: str
    profile: str | None = None
    runtime_name: str | None = None
    omni_domain: str | None = None


class DoctorReport(_DoctorModel):
    """Every check's result for one target."""

    target: DoctorTarget
    identity: str
    results: tuple[CheckResult, ...]

    @property
    def ok(self) -> bool:
        """True only when every check passed."""
        return all(r.status == "pass" for r in self.results)


class Check(Protocol):
    """A read-only readiness check."""

    name: ClassVar[str]
    operations: ClassVar[frozenset[Operation]]

    def run(self, clients: AwsClients, target: DoctorTarget) -> CheckResult:
        """Inspect the account and report."""
        ...


def _pages(client: Any, method: str, key: str, **params: Any) -> Iterator[dict[str, Any]]:
    """Every item under ``key`` across all pages; never just the first page."""
    for page in client.get_paginator(method).paginate(**params):
        yield from page.get(key, [])


def _find_runtime(clients: AwsClients, name: str) -> dict[str, Any] | None:
    control = clients.client("bedrock-agentcore-control")
    for runtime in _pages(control, "list_agent_runtimes", "agentRuntimes"):
        if runtime.get("agentRuntimeName") == name:
            return runtime
    return None


def _no_runtime(check: str, name: str | None, region: str) -> CheckResult:
    if name is None:
        return CheckResult(
            name=check,
            status="unknown",
            observed="no runtime named",
            next_step="pass --runtime NAME to check a deployed AgentCore Runtime",
        )
    return CheckResult(
        name=check,
        status="fail",
        observed=f"no runtime named {name!r} in {region}",
        next_step="check the runtime name and --region",
    )


class RegionCheck:
    """AgentCore has an endpoint in the region."""

    name: ClassVar[str] = "region"
    operations: ClassVar[frozenset[Operation]] = frozenset(
        {("bedrock-agentcore-control", "ListEvaluators")}
    )

    def __init__(self, resolves: Callable[[str], bool] | None = None) -> None:
        self._resolves = resolves or _resolves

    def run(self, clients: AwsClients, target: DoctorTarget) -> CheckResult:
        """Probe ``ListEvaluators``; an answer of any kind, even a denial, means it exists."""
        control = clients.client("bedrock-agentcore-control")
        try:
            control.list_evaluators(maxResults=1)  # pyright: ignore[reportAttributeAccessIssue]
        except ClientError as exc:
            code = exc.response.get("Error", {}).get("Code", "")
            if code in _DENIED:
                return self._available(target, f"endpoint answered ({code})")
            return CheckResult(
                name=self.name,
                status="unknown",
                observed=f"the endpoint returned {code or 'an error'}",
                next_step="retry; the service may be throttling",
            )
        except (UnknownEndpointError, EndpointResolutionError):
            return self._unavailable(target)
        except EndpointConnectionError:
            return self._classify_connection_failure(control.meta.endpoint_url, target)
        except (ConnectTimeoutError, ReadTimeoutError):
            return self._network_unknown("the endpoint timed out")
        return self._available(target, "endpoint answered")

    def _classify_connection_failure(self, endpoint_url: str, target: DoctorTarget) -> CheckResult:
        # An unsupported region and a dead network both surface as a connection error. STS exists in
        # every region, so if its host resolves and AgentCore's doesn't, AgentCore is absent.
        host = urlparse(endpoint_url).hostname or ""
        try:
            agentcore_resolves = self._resolves(host)
            sts_resolves = self._resolves(f"sts.{target.region}.amazonaws.com")
        except OSError:
            return self._network_unknown("DNS lookup failed")
        if sts_resolves and not agentcore_resolves:
            return self._unavailable(target)
        return self._network_unknown(f"could not reach {host}")

    def _available(self, target: DoctorTarget, observed: str) -> CheckResult:
        return CheckResult(name=self.name, status="pass", observed=f"{target.region}: {observed}")

    def _unavailable(self, target: DoctorTarget) -> CheckResult:
        return CheckResult(
            name=self.name,
            status="fail",
            observed=f"AgentCore has no endpoint in {target.region}",
            next_step="pick a region where AgentCore is offered and pass it with --region",
        )

    def _network_unknown(self, observed: str) -> CheckResult:
        return CheckResult(
            name=self.name, status="unknown", observed=observed, next_step="retry; network"
        )


def _resolves(host: str) -> bool:
    try:
        socket.getaddrinfo(host, 443)
    except socket.gaierror:
        return False
    return True


class TransactionSearchCheck:
    """X-Ray sends spans to CloudWatch Logs, which Transaction Search needs."""

    name: ClassVar[str] = "transaction_search"
    operations: ClassVar[frozenset[Operation]] = frozenset({("xray", "GetTraceSegmentDestination")})

    def run(self, clients: AwsClients, target: DoctorTarget) -> CheckResult:
        """Read the segment destination and its status."""
        out = clients.client("xray").get_trace_segment_destination()  # pyright: ignore[reportAttributeAccessIssue]
        destination, status = out.get("Destination"), out.get("Status")
        observed = f"Destination={destination} Status={status}"
        if destination == "CloudWatchLogs" and status == "ACTIVE":
            return CheckResult(name=self.name, status="pass", observed=observed)
        if destination == "CloudWatchLogs":
            return CheckResult(
                name=self.name,
                status="fail",
                observed=observed,
                next_step="Transaction Search is still turning on; check again in about 10 minutes",
            )
        return CheckResult(
            name=self.name,
            status="fail",
            observed=observed,
            next_step=(
                "turn on Transaction Search: "
                f"aws xray update-trace-segment-destination --destination CloudWatchLogs "
                f"--region {target.region}"
            ),
        )


class IndexingCheck:
    """The default X-Ray indexing rule indexes every span."""

    name: ClassVar[str] = "indexing"
    operations: ClassVar[frozenset[Operation]] = frozenset({("xray", "GetIndexingRules")})

    def run(self, clients: AwsClients, target: DoctorTarget) -> CheckResult:
        """Find the Default rule across all pages and read its sampling percentage."""
        xray = clients.client("xray")
        token: str | None = None
        while True:
            page = xray.get_indexing_rules(**({"NextToken": token} if token else {}))  # pyright: ignore[reportAttributeAccessIssue]
            for rule in page.get("IndexingRules", []):
                if rule.get("Name") == "Default":
                    return self._judge(rule, target)
            token = page.get("NextToken")
            if not token:
                break
        return CheckResult(
            name=self.name,
            status="fail",
            observed="no Default indexing rule",
            next_step="turn on Transaction Search, which creates the Default rule",
        )

    def _judge(self, rule: dict[str, Any], target: DoctorTarget) -> CheckResult:
        percent = rule.get("Rule", {}).get("Probabilistic", {}).get("DesiredSamplingPercentage")
        observed = f"Default rule indexes {percent}% of spans"
        if percent == 100:
            return CheckResult(name=self.name, status="pass", observed=observed)
        return CheckResult(
            name=self.name,
            status="fail",
            observed=observed,
            next_step=(
                "trace ingest through X-Ray sees only indexed spans; raise it with "
                "aws xray update-indexing-rule --name Default --rule "
                '\'{"Probabilistic":{"DesiredSamplingPercentage":100}}\' '
                f"--region {target.region}"
            ),
        )


class SpanLogGroupCheck:
    """The shared span log group exists."""

    name: ClassVar[str] = "span_log_group"
    operations: ClassVar[frozenset[Operation]] = frozenset({("logs", "DescribeLogGroups")})

    def run(self, clients: AwsClients, target: DoctorTarget) -> CheckResult:
        """Look for ``aws/spans`` by exact name across all pages."""
        groups = _pages(
            clients.client("logs"),
            "describe_log_groups",
            "logGroups",
            logGroupNamePrefix=SPAN_LOG_GROUP,
        )
        if any(g.get("logGroupName") == SPAN_LOG_GROUP for g in groups):
            return CheckResult(name=self.name, status="pass", observed=f"{SPAN_LOG_GROUP} exists")
        return CheckResult(
            name=self.name,
            status="fail",
            observed=f"no {SPAN_LOG_GROUP} log group in {target.region}",
            next_step="turn on Transaction Search; it creates the span log group",
        )


class RuntimeLogGroupCheck:
    """The named runtime writes to its own log group."""

    name: ClassVar[str] = "runtime_log_group"
    operations: ClassVar[frozenset[Operation]] = frozenset(
        {("bedrock-agentcore-control", "ListAgentRuntimes"), ("logs", "DescribeLogGroups")}
    )

    def run(self, clients: AwsClients, target: DoctorTarget) -> CheckResult:
        """Resolve the runtime id by name, then look for its log groups."""
        runtime = _find_runtime(clients, target.runtime_name) if target.runtime_name else None
        if runtime is None:
            return _no_runtime(self.name, target.runtime_name, target.region)
        # agentRuntimeId is required in the ListAgentRuntimes response shape.
        prefix = f"/aws/bedrock-agentcore/runtimes/{runtime['agentRuntimeId']}"
        groups = [
            g.get("logGroupName", "")
            for g in _pages(
                clients.client("logs"),
                "describe_log_groups",
                "logGroups",
                logGroupNamePrefix=prefix,
            )
        ]
        if groups:
            return CheckResult(name=self.name, status="pass", observed=", ".join(sorted(groups)))
        return CheckResult(
            name=self.name,
            status="fail",
            observed=f"no log group under {prefix}",
            next_step="invoke the runtime once; AgentCore creates its log groups on first use",
        )


class OmniDomainCheck:
    """A CloudWatch Omni domain is active."""

    name: ClassVar[str] = "omni_domain"
    operations: ClassVar[frozenset[Operation]] = frozenset({("cloudwatchomni", "ListDomains")})

    def run(self, clients: AwsClients, target: DoctorTarget) -> CheckResult:
        """Find the named domain (or any domain) across all pages."""
        domains = list(_pages(clients.client("cloudwatchomni"), "list_domains", "items"))
        wanted = [d for d in domains if target.omni_domain in (None, d.get("name"))]
        active = [d for d in wanted if d.get("status") == "ACTIVE"]
        if active:
            names = ", ".join(sorted(str(d.get("name")) for d in active))
            return CheckResult(name=self.name, status="pass", observed=f"active: {names}")
        if wanted:
            return CheckResult(
                name=self.name,
                status="fail",
                observed=f"{target.omni_domain or 'a domain'} exists but is not active",
                next_step="wait for the Omni domain to finish creating, then check again",
            )
        missing = (
            f"no Omni domain named {target.omni_domain!r}"
            if target.omni_domain
            else ("no Omni domain")
        )
        return CheckResult(
            name=self.name,
            status="fail",
            observed=f"{missing} in {target.region}",
            next_step="create an Omni domain in the CloudWatch console, or check --omni-domain",
        )


class ContentExtractionCheck:
    """The runtime keeps prompts and responses on its spans."""

    name: ClassVar[str] = "content_extraction"
    operations: ClassVar[frozenset[Operation]] = frozenset(
        {
            ("bedrock-agentcore-control", "ListAgentRuntimes"),
            ("bedrock-agentcore-control", "GetAgentRuntime"),
        }
    )

    def run(self, clients: AwsClients, target: DoctorTarget) -> CheckResult:
        """Read the runtime's environment for the content-extraction opt-out."""
        runtime = _find_runtime(clients, target.runtime_name) if target.runtime_name else None
        if runtime is None:
            result = _no_runtime(self.name, target.runtime_name, target.region)
            if result.status == "unknown":
                return replace(
                    result,
                    next_step=(
                        f"pass --runtime NAME, or make sure {CONTENT_EXTRACTION_VAR}=true is "
                        "set where the agent runs"
                    ),
                )
            return result
        runtime_id = runtime["agentRuntimeId"]
        detail = clients.client("bedrock-agentcore-control").get_agent_runtime(  # pyright: ignore[reportAttributeAccessIssue]
            agentRuntimeId=runtime_id
        )
        env: dict[str, str] = detail.get("environmentVariables") or {}
        fix = f"set {CONTENT_EXTRACTION_VAR}=true in the runtime's environment and redeploy"
        if CONTENT_EXTRACTION_VAR not in env:
            return CheckResult(
                name=self.name,
                status="unknown",
                observed=f"{CONTENT_EXTRACTION_VAR} is not set on the runtime",
                next_step=f"{fix} (unless the agent's image sets it)",
            )
        value = env[CONTENT_EXTRACTION_VAR]
        observed = f"{CONTENT_EXTRACTION_VAR}={value}"
        # The name reads inverted: "true" opts out of moving content into log events.
        if value.strip().lower() == "true":
            return CheckResult(name=self.name, status="pass", observed=observed)
        return CheckResult(name=self.name, status="fail", observed=observed, next_step=fix)


DEFAULT_CHECKS: tuple[Check, ...] = (
    RegionCheck(),
    TransactionSearchCheck(),
    IndexingCheck(),
    SpanLogGroupCheck(),
    RuntimeLogGroupCheck(),
    OmniDomainCheck(),
    ContentExtractionCheck(),
)


def allowed_operations(checks: Sequence[Check]) -> frozenset[Operation]:
    """Every operation ``checks`` declare, plus the identity call that runs first."""
    declared = {op for check in checks for op in check.operations}
    return frozenset({*declared, ("sts", "GetCallerIdentity")})


def read_only_guard(checks: Sequence[Check]) -> ReadOnlyGuard:
    """The guard every doctor client carries."""
    return ReadOnlyGuard(allowed_operations(checks))


def run_checks(checks: Sequence[Check], clients: AwsClients, target: DoctorTarget) -> DoctorReport:
    """Confirm the identity, then run every check, isolating each one's AWS failures."""
    identity = clients.caller_identity()
    results = tuple(_run_one(check, clients, target) for check in checks)
    return DoctorReport(target=target, identity=identity, results=results)


def _run_one(check: Check, clients: AwsClients, target: DoctorTarget) -> CheckResult:
    # Only AWS errors are turned into "unknown"; a guard refusal (an AgentCoreError) is a bug in
    # the check and propagates.
    try:
        return check.run(clients, target)
    except ClientError as exc:
        return _client_error_result(check, exc)
    except (EndpointConnectionError, ConnectTimeoutError, ReadTimeoutError) as exc:
        return CheckResult(
            name=check.name, status="unknown", observed=str(exc), next_step="retry; network"
        )
    except BotoCoreError as exc:
        return CheckResult(
            name=check.name,
            status="unknown",
            observed=f"{type(exc).__name__}: {exc}",
            next_step="retry; if it persists, run with --explain and report it",
        )


def _client_error_result(check: Check, exc: ClientError) -> CheckResult:
    name = check.name
    code = exc.response.get("Error", {}).get("Code", "")
    operation = exc.operation_name
    # ClientError carries no service name; the check's own declared operations say which it was.
    service = next((svc for svc, op in check.operations if op == operation), "aws")
    action = f"{_IAM_PREFIX.get(service, service)}:{operation}"
    if code in _DENIED:
        return CheckResult(
            name=name, status="unknown", observed=f"{action} denied", next_step=f"grant {action}"
        )
    if code in _THROTTLED:
        return CheckResult(
            name=name,
            status="unknown",
            observed=f"{action} throttled",
            next_step="retry; network or throttling",
        )
    return CheckResult(
        name=name,
        status="unknown",
        observed=f"{action} failed: {code or 'unknown error'}",
        next_step="retry; if it persists, run with --explain and report it",
    )
