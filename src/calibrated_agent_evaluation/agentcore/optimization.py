# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""The A/B audit's AWS reads: an A/B test and its bundle versions, mapped to vendor-free records.

The vendor seam of the audit. Each read is one call through :class:`AwsClients`, and nothing from
botocore leaves this module. Bundle components and texts are digested here and never returned.
Every audit client carries a :class:`ReadOnlyGuard` over :data:`AUDIT_READ_OPERATIONS` and the
trace reads the AgentCore ingest's readers make, so a write, a list or any other call is refused
before it is sent.
"""

from __future__ import annotations

import functools
import os
import uuid
from collections.abc import Callable, Mapping
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any, ParamSpec, TypeVar

from calibrated_agent_evaluation.agentcore.audit_types import (
    AbTestClaimRecord,
    BundleVersionRef,
    ComponentDigest,
    EvaluatorClaim,
    JsonPathRef,
    KeyDigest,
    RecommendationClaimRecord,
    TextDigest,
    VariantRef,
    VariantStats,
    canonical_json,
    sha256_text,
)
from calibrated_agent_evaluation.agentcore.errors import AgentCoreError
from calibrated_agent_evaluation.agentcore.guards import BilledCallBudget, Operation, ReadOnlyGuard

if TYPE_CHECKING:
    from calibrated_agent_evaluation.agentcore.replay import OnlineEvalConfig
    from calibrated_agent_evaluation.agentcore.session import AwsClients

__all__ = [
    "AUDIT_READ_OPERATIONS",
    "INGEST_READ_OPERATIONS",
    "audit_clients",
    "read_ab_test",
    "read_bundle_version",
    "read_online_eval_config",
    "read_recommendation",
]

_DATA = "bedrock-agentcore"
_CONTROL = "bedrock-agentcore-control"

AUDIT_READ_OPERATIONS: frozenset[Operation] = frozenset(
    {
        (_DATA, "GetABTest"),
        (_DATA, "GetRecommendation"),
        (_CONTROL, "GetConfigurationBundleVersion"),
        (_CONTROL, "GetOnlineEvaluationConfig"),
    }
)

# The trace reads the AgentCore ingest's readers make on the paths the audit uses. The ingest
# exports no such list and builds no guard of its own, so the audit holds it. The Omni session
# stop is easy to miss: the ingest calls it in a ``finally`` that swallows errors, so a list
# without it fails no read and leaves the session open.
INGEST_READ_OPERATIONS: frozenset[Operation] = frozenset(
    {
        ("logs", "StartQuery"),
        ("logs", "GetQueryResults"),
        ("logs", "FilterLogEvents"),
        ("cloudwatchomni", "StartTelemetryQuerySession"),
        ("cloudwatchomni", "StartTelemetryQuery"),
        ("cloudwatchomni", "GetTelemetryQueryResults"),
        ("cloudwatchomni", "StopTelemetryQuerySession"),
    }
)

_PROVISIONAL = frozenset({"RUNNING", "PAUSED"})


def audit_clients(region: str | None = None, profile: str | None = None) -> AwsClients:
    """AWS clients that can make only the audit's reads and the AgentCore ingest's trace reads.

    The billed-call budget comes from ``CALIBRATED_AGENT_EVALUATION_AGENTCORE_MAX_BILLED_CALLS``
    when it is set.

    Raises:
        AgentCoreError: The profile does not exist or cannot be opened, or the budget variable
            is not a non-negative whole number.
    """
    # Through the package's lazy attribute, which turns a missing AWS SDK into
    # AgentCoreExtraMissing instead of a bare ModuleNotFoundError.
    from calibrated_agent_evaluation import agentcore

    guard = ReadOnlyGuard(AUDIT_READ_OPERATIONS | INGEST_READ_OPERATIONS)
    budget = BilledCallBudget.from_env(os.environ)
    return agentcore.AwsClients.from_profile(
        region=region, profile=profile, handlers=[guard, budget]
    )


def _iso(value: object) -> str | None:
    if value is None:
        return None
    if isinstance(value, datetime):
        # botocore parses a response time into the machine's local zone; a stored record keeps
        # UTC, so the same response reads the same on every machine.
        stamped = value if value.utcoffset() is not None else value.replace(tzinfo=UTC)
        return stamped.astimezone(UTC).isoformat()
    return str(value)


_P = ParamSpec("_P")
_R = TypeVar("_R")


def _shaped(operation: str) -> Callable[[Callable[_P, _R]], Callable[_P, _R]]:
    """Refuse, as a typed error, a response that lacks a member the read indexes.

    botocore validates requests against the service model, not responses, so a partial response
    would otherwise surface as a bare ``KeyError`` from the mapping below the call.
    """

    def wrap(read: Callable[_P, _R]) -> Callable[_P, _R]:
        @functools.wraps(read)
        def guarded(*args: _P.args, **kwargs: _P.kwargs) -> _R:
            try:
                return read(*args, **kwargs)
            except KeyError as exc:
                raise AgentCoreError(
                    f"{operation} returned a response without the member {exc.args[0]!r} that "
                    "the audit reads, so nothing was recorded; the service's response shape may "
                    "have changed, so check the installed botocore model against the live response"
                ) from exc

        return guarded

    return wrap


def _call(operation: str, resource: str, call: Any, **params: Any) -> dict[str, Any]:
    from botocore.exceptions import BotoCoreError, ClientError

    try:
        response = call(**params)
    except ClientError as exc:
        code = exc.response.get("Error", {}).get("Code") or type(exc).__name__
        raise AgentCoreError(f"{operation} failed for {resource!r}: {code}") from exc
    except BotoCoreError as exc:
        raise AgentCoreError(f"{operation} failed for {resource!r}: {type(exc).__name__}") from exc
    return dict(response)


def _component_digest(component_id: str, configuration: object) -> ComponentDigest:
    text = canonical_json(configuration)
    keys: tuple[KeyDigest, ...] = ()
    if isinstance(configuration, Mapping):
        keys = tuple(
            KeyDigest(key=str(key), sha256=sha256_text(value_text), length=len(value_text))
            for key, value_text in sorted(
                (k, v if isinstance(v, str) else canonical_json(v))
                for k, v in configuration.items()
            )
        )
    return ComponentDigest(
        component_id=component_id, sha256=sha256_text(text), length=len(text), keys=keys
    )


@_shaped("GetConfigurationBundleVersion")
def read_bundle_version(clients: AwsClients, bundle_arn: str, version_id: str) -> BundleVersionRef:
    """Read one configuration bundle version, keeping its components as digests only.

    Args:
        clients: The audit's clients.
        bundle_arn: The bundle's ARN; its last path segment is the bundle id.
        version_id: The version to read.

    Returns:
        The version's lineage and a digest of each component, of each top-level key of a
        component whose configuration is a JSON object, and of the whole components map.

    Raises:
        AgentCoreError: The call failed, naming ``GetConfigurationBundleVersion`` and the version.
    """
    bundle_id = bundle_arn.rsplit("/", 1)[-1]
    response = _call(
        "GetConfigurationBundleVersion",
        f"{bundle_id}/{version_id}",
        clients.client(_CONTROL).get_configuration_bundle_version,
        bundleId=bundle_id,
        versionId=version_id,
    )
    components: dict[str, Any] = dict(response.get("components") or {})
    lineage: dict[str, Any] = dict(response.get("lineageMetadata") or {})
    created_by = lineage.get("createdBy") or {}
    return BundleVersionRef(
        bundle_arn=str(response.get("bundleArn") or bundle_arn),
        version_id=str(response.get("versionId") or version_id),
        version_created_at=_iso(response.get("versionCreatedAt")) or "",
        parent_version_ids=tuple(str(v) for v in lineage.get("parentVersionIds") or ()),
        branch_name=lineage.get("branchName"),
        created_by=created_by.get("name") if isinstance(created_by, Mapping) else None,
        components=tuple(
            _component_digest(cid, (components[cid] or {}).get("configuration"))
            for cid in sorted(components)
        ),
        components_sha256=sha256_text(canonical_json(components)),
    )


def _stats(doc: Mapping[str, Any]) -> VariantStats:
    interval = doc.get("confidenceInterval") or {}
    return VariantStats(
        variant_name=str(doc.get("variantName")),
        sample_size=int(doc.get("sampleSize", 0)),
        mean=float(doc.get("mean", 0.0)),
        absolute_change=doc.get("absoluteChange"),
        percent_change=doc.get("percentChange"),
        p_value=doc.get("pValue"),
        ci_lower=interval.get("lower"),
        ci_upper=interval.get("upper"),
        is_significant=doc.get("isSignificant"),
    )


def _evaluation_config_arns(config: Mapping[str, Any]) -> tuple[str, ...]:
    shared = config.get("onlineEvaluationConfigArn")
    if shared:
        return (str(shared),)
    return tuple(
        str(v["onlineEvaluationConfigArn"])
        for v in config.get("perVariantOnlineEvaluationConfig") or ()
    )


@_shaped("GetABTest")
def read_ab_test(
    clients: AwsClients, ab_test_id: str, *, now: datetime | None = None
) -> AbTestClaimRecord:
    """Read an A/B test and each arm's bundle version as a vendor-claim record.

    Every variant must name exactly one of a bundle version and a gateway target; the model
    allows both or neither, so such a response is refused before any bundle version is read. A
    target arm is kept by name and reads no bundle.

    Args:
        clients: The audit's clients.
        ab_test_id: The A/B test to read.
        now: The read time, for tests; the current UTC time otherwise.

    Returns:
        The claim record: the test's state, both arms, AgentCore's per-evaluator figures as
        returned, whether the two bundle arms are identical, and whether the test is still running.

    Raises:
        AgentCoreError: A call failed, naming the operation and the id, or a variant names both
            or neither configuration.
    """
    from calibrated_agent_evaluation._provenance import capture_provenance

    response = _call(
        "GetABTest", ab_test_id, clients.client(_DATA).get_ab_test, abTestId=ab_test_id
    )
    raw_variants = list(response.get("variants") or [])
    for variant in raw_variants:
        config = variant.get("variantConfiguration") or {}
        if ("configurationBundle" in config) == ("target" in config):
            names = (
                "both a bundle version and a target"
                if config
                else "neither a bundle version nor a target"
            )
            raise AgentCoreError(
                f"GetABTest for {ab_test_id!r}: variant {variant.get('name')!r} names {names}; "
                "an arm must name exactly one, so the test is refused"
            )
    variants: list[VariantRef] = []
    for variant in raw_variants:
        config = variant["variantConfiguration"]
        bundle_ref = config.get("configurationBundle")
        target = config.get("target")
        variants.append(
            VariantRef(
                name=variant.get("name"),
                weight=int(variant.get("weight", 0)),
                bundle=(
                    read_bundle_version(
                        clients, str(bundle_ref["bundleArn"]), str(bundle_ref["bundleVersion"])
                    )
                    if bundle_ref
                    else None
                ),
                target_name=str(target["name"]) if target else None,
            )
        )
    results = response.get("results") or {}
    claims = tuple(
        EvaluatorClaim(
            evaluator_arn=str(metric.get("evaluatorArn")),
            control=_stats(metric.get("controlStats") or {}),
            variants=tuple(_stats(v) for v in metric.get("variantResults") or ()),
        )
        for metric in results.get("evaluatorMetrics") or ()
    )
    bundles = [v.bundle for v in variants if v.bundle is not None]
    identical = len(bundles) == 2 and bundles[0].components_sha256 == bundles[1].components_sha256
    execution_status = str(response.get("executionStatus"))
    read_at = (now or datetime.now(tz=UTC)).astimezone(UTC).isoformat()
    return AbTestClaimRecord(
        record_id=f"abtest-{uuid.uuid4().hex[:16]}",
        ab_test_arn=str(response.get("abTestArn")),
        ab_test_id=str(response.get("abTestId") or ab_test_id),
        name=str(response.get("name")),
        status=str(response.get("status")),
        execution_status=execution_status,
        gateway_arn=str(response.get("gatewayArn")),
        variants=tuple(sorted(variants, key=lambda v: v.name)),  # type: ignore[arg-type]
        evaluation_config_arns=_evaluation_config_arns(response.get("evaluationConfig") or {}),
        current_run_id=response.get("currentRunId"),
        error_details=tuple(str(d) for d in response.get("errorDetails") or ()),
        started_at=_iso(response.get("startedAt")),
        stopped_at=_iso(response.get("stoppedAt")),
        max_duration_expires_at=_iso(response.get("maxDurationExpiresAt")),
        analysis_timestamp=_iso(results.get("analysisTimestamp")),
        updated_at=_iso(response.get("updatedAt")) or "",
        evaluator_claims=claims,
        identical_arms=identical,
        provisional=execution_status in _PROVISIONAL,
        read_at=read_at,
        provenance=capture_provenance(now=now),
    )


def _text(tool_name: str | None, text: object) -> TextDigest:
    value = text if isinstance(text, str) else ""
    return TextDigest(tool_name=tool_name, sha256=sha256_text(value), length=len(value))


def _traces_fields(traces: Mapping[str, Any]) -> dict[str, Any]:
    """The record fields for a recommendation's agent-traces source, whichever kind it is."""
    fields: dict[str, Any] = {
        "traces_window_start": None,
        "traces_window_end": None,
        "traces_service_name": None,
        "traces_log_group_arns": (),
        "traces_batch_evaluation_arn": None,
        "traces_online_evaluation_config_arn": None,
        "traces_inline_trace_ids": (),
    }
    if "sessionSpans" in traces:
        ids = {
            str(span.get("traceId"))
            for span in traces["sessionSpans"] or ()
            if isinstance(span, Mapping) and span.get("traceId")
        }
        return {
            **fields,
            "traces_source": "sessionSpans",
            "traces_inline_trace_ids": tuple(sorted(ids)),
        }
    if "cloudwatchLogs" in traces:
        logs = traces["cloudwatchLogs"]
        names = list(logs.get("serviceNames") or ())
        return {
            **fields,
            "traces_source": "cloudwatchLogs",
            "traces_window_start": _iso(logs.get("startTime")),
            "traces_window_end": _iso(logs.get("endTime")),
            "traces_service_name": str(names[0]) if names else None,
            "traces_log_group_arns": tuple(str(a) for a in logs.get("logGroupArns") or ()),
        }
    if "batchEvaluation" in traces:
        arn = traces["batchEvaluation"].get("batchEvaluationArn")
        return {**fields, "traces_source": "batchEvaluation", "traces_batch_evaluation_arn": arn}
    online = traces.get("onlineEvaluation") or {}
    return {
        **fields,
        "traces_source": "onlineEvaluation",
        "traces_window_start": _iso(online.get("startTime")),
        "traces_window_end": _iso(online.get("endTime")),
        "traces_online_evaluation_config_arn": online.get("onlineEvaluationConfigArn"),
    }


@_shaped("GetRecommendation")
def read_recommendation(
    clients: AwsClients, recommendation_id: str, *, now: datetime | None = None
) -> RecommendationClaimRecord:
    """Read a recommendation, its input and result bundle versions, as a vendor-claim record.

    The result's bundle, texts, explanations and error are read whenever the response carries a
    result, whatever its status: the model documents a result only on ``COMPLETED`` and its error
    code as a failed recommendation's. Texts are kept as digests and lengths only.

    Raises:
        AgentCoreError: A call failed, naming the operation and the id.
    """
    from calibrated_agent_evaluation._provenance import capture_provenance

    response = _call(
        "GetRecommendation",
        recommendation_id,
        clients.client(_DATA).get_recommendation,
        recommendationId=recommendation_id,
    )
    rtype = str(response.get("type"))
    tool = rtype == "TOOL_DESCRIPTION_RECOMMENDATION"
    config_all = response.get("recommendationConfig") or {}
    config = (
        config_all.get(
            "toolDescriptionRecommendationConfig" if tool else "systemPromptRecommendationConfig"
        )
        or {}
    )
    source = config.get("toolDescription" if tool else "systemPrompt") or {}
    input_bundle = None
    paths: tuple[JsonPathRef, ...] = ()
    inline: tuple[TextDigest, ...] = ()
    if "configurationBundle" in source:
        ref = source["configurationBundle"]
        input_bundle = read_bundle_version(clients, str(ref["bundleArn"]), str(ref["versionId"]))
        if tool:
            paths = tuple(
                JsonPathRef(
                    tool_name=str(e["toolName"]), json_path=str(e["toolDescriptionJsonPath"])
                )
                for e in ref.get("tools") or ()
            )
        else:
            paths = (JsonPathRef(tool_name=None, json_path=str(ref["systemPromptJsonPath"])),)
    elif tool:
        inline = tuple(
            _text(str(e["toolName"]), (e.get("toolDescription") or {}).get("text"))
            for e in (source.get("toolDescriptionText") or {}).get("tools") or ()
        )
    else:
        inline = (_text(None, source.get("text")),)
    evaluators = (config.get("evaluationConfig") or {}).get("evaluators") or ()
    result_all = response.get("recommendationResult") or {}
    result = (
        result_all.get(
            "toolDescriptionRecommendationResult" if tool else "systemPromptRecommendationResult"
        )
        or {}
    )
    result_ref = result.get("configurationBundle")
    result_bundle = (
        read_bundle_version(clients, str(result_ref["bundleArn"]), str(result_ref["versionId"]))
        if result_ref
        else None
    )
    if tool:
        tools = list(result.get("tools") or ())
        texts = tuple(
            _text(str(e["toolName"]), e["recommendedToolDescription"])
            for e in tools
            if "recommendedToolDescription" in e
        )
        explanations = tuple(
            _text(str(e["toolName"]), e["explanation"]) for e in tools if "explanation" in e
        )
    else:
        texts = (
            (_text(None, result["recommendedSystemPrompt"]),)
            if "recommendedSystemPrompt" in result
            else ()
        )
        explanations = (_text(None, result["explanation"]),) if "explanation" in result else ()
    read_at = (now or datetime.now(tz=UTC)).astimezone(UTC).isoformat()
    return RecommendationClaimRecord(
        record_id=f"recommendation-{uuid.uuid4().hex[:16]}",
        recommendation_arn=str(response.get("recommendationArn")),
        recommendation_id=str(response.get("recommendationId") or recommendation_id),
        name=str(response.get("name")),
        type=rtype,  # type: ignore[arg-type]
        status=str(response.get("status")),
        created_at=_iso(response.get("createdAt")) or "",
        updated_at=_iso(response.get("updatedAt")) or "",
        input_bundle=input_bundle,
        input_json_paths=paths,
        input_inline=inline,
        **_traces_fields(config.get("agentTraces") or {}),
        evaluator_arn=(evaluators[0].get("evaluatorArn") if evaluators else None),
        result_bundle=result_bundle,
        result_texts=texts,
        explanations=explanations,
        error_code=result.get("errorCode"),
        error_message=result.get("errorMessage"),
        read_at=read_at,
        provenance=capture_provenance(now=now),
    )


@_shaped("GetOnlineEvaluationConfig")
def read_online_eval_config(clients: AwsClients, config_arn: str) -> OnlineEvalConfig:
    """Read the parts of an online evaluation configuration a development window depends on.

    Raises:
        AgentCoreError: The call failed, naming ``GetOnlineEvaluationConfig`` and the id.
    """
    from calibrated_agent_evaluation.agentcore.replay import OnlineEvalConfig

    config_id = config_arn.rsplit("/", 1)[-1]
    response = _call(
        "GetOnlineEvaluationConfig",
        config_id,
        clients.client(_CONTROL).get_online_evaluation_config,
        onlineEvaluationConfigId=config_id,
    )
    logs = (response.get("dataSourceConfig") or {}).get("cloudWatchLogs") or {}
    session = (response.get("rule") or {}).get("sessionConfig") or {}
    names = list(logs.get("serviceNames") or ())
    timeout = session.get("sessionTimeoutMinutes")
    return OnlineEvalConfig(
        config_arn=str(response.get("onlineEvaluationConfigArn") or config_arn),
        service_name=str(names[0]) if names else "",
        log_group_names=tuple(str(g) for g in logs.get("logGroupNames") or ()),
        log_group_name_prefixes=tuple(str(g) for g in logs.get("logGroupNamePrefixes") or ()),
        session_timeout_minutes=int(timeout) if timeout is not None else None,
        updated_at=_iso(response.get("updatedAt")) or "",
    )
