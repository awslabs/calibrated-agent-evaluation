# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""The A/B audit's records: what AgentCore claimed about a test, and which arm each trace ran in.

Vendor-free: plain pydantic models and their stores, so they import with no AWS SDK. AgentCore's
figures are kept as a vendor claim, copied as returned and never computed on; prompt and bundle
texts are kept only as digests and lengths, never as text.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Sequence
from datetime import datetime
from pathlib import Path
from typing import ClassVar, Literal, Self

from pydantic import ConfigDict, Field, field_validator, model_validator

from calibrated_agent_evaluation._model import CalibratedAgentEvaluationModel
from calibrated_agent_evaluation._provenance import ProvenanceStamp
from calibrated_agent_evaluation._recordstore import RecordStore
from calibrated_agent_evaluation.agentcore.errors import AgentCoreError

__all__ = [
    "AUDIT_SCHEMA_VERSION",
    "AbTestClaimRecord",
    "ArmAssignment",
    "ArmStatus",
    "AuditRecord",
    "BundleVersionRef",
    "ComponentDigest",
    "EvaluatorClaim",
    "JsonPathRef",
    "KeyDigest",
    "RecommendationClaimRecord",
    "RunProvenance",
    "TextDigest",
    "VariantRef",
    "VariantStats",
    "canonical_json",
    "dump_abtest_claims",
    "dump_arm_assignments",
    "dump_audit_records",
    "dump_recommendation_claims",
    "dump_run_provenance",
    "load_abtest_claims",
    "load_arm_assignments",
    "load_audit_records",
    "load_recommendation_claims",
    "load_run_provenance",
    "sha256_text",
]

AUDIT_SCHEMA_VERSION = "1"

_FROZEN = ConfigDict(frozen=True, extra="forbid", defer_build=True)


def canonical_json(value: object) -> str:
    """The one JSON form every audit digest is taken over, as the dataset codec writes it."""
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def sha256_text(text: str) -> str:
    """The lowercase hex SHA-256 of ``text``'s UTF-8 bytes."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _tz_aware(value: str | None) -> str | None:
    if value is None:
        return None
    try:
        parsed = datetime.fromisoformat(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{value!r} is not ISO-8601") from exc
    if parsed.utcoffset() is None:
        raise ValueError(f"{value!r} has no timezone")
    return value


class _AuditModel(CalibratedAgentEvaluationModel):
    model_config = _FROZEN
    invalid_error: ClassVar[type[Exception]] = AgentCoreError


class TextDigest(_AuditModel):
    """One text, kept as its digest and length: the text itself is never stored."""

    tool_name: str | None
    sha256: str
    length: int = Field(ge=0)


class KeyDigest(_AuditModel):
    """One top-level key of a component's configuration object, digested."""

    key: str
    sha256: str
    length: int = Field(ge=0)


class ComponentDigest(_AuditModel):
    """One bundle component's configuration, digested, with one digest per top-level key."""

    component_id: str
    sha256: str
    length: int = Field(ge=0)
    keys: tuple[KeyDigest, ...] = ()

    @model_validator(mode="after")
    def _unique_keys(self) -> Self:
        names = [k.key for k in self.keys]
        if len(names) != len(set(names)):
            raise AgentCoreError(f"component {self.component_id!r} repeats a key digest")
        return self


class BundleVersionRef(_AuditModel):
    """One configuration bundle version, as read, with its components digested."""

    bundle_arn: str
    version_id: str
    version_created_at: str
    parent_version_ids: tuple[str, ...]
    branch_name: str | None
    created_by: str | None
    components: tuple[ComponentDigest, ...]
    components_sha256: str

    @field_validator("version_created_at")
    @classmethod
    def _time(cls, value: str) -> str:
        _tz_aware(value)
        return value

    @model_validator(mode="after")
    def _unique_components(self) -> Self:
        ids = [c.component_id for c in self.components]
        if len(ids) != len(set(ids)):
            raise AgentCoreError(f"bundle version {self.version_id!r} repeats a component id")
        return self


class VariantRef(_AuditModel):
    """One arm of an A/B test: a bundle version, or a gateway target, never both."""

    name: Literal["C", "T1"]
    weight: int
    bundle: BundleVersionRef | None
    target_name: str | None

    @model_validator(mode="after")
    def _exactly_one(self) -> Self:
        if (self.bundle is None) == (self.target_name is None):
            raise AgentCoreError(
                f"variant {self.name!r} must name exactly one of a bundle version and a gateway "
                "target"
            )
        return self


class VariantStats(_AuditModel):
    """One arm's figures for one evaluator, copied as AgentCore returned them."""

    variant_name: str
    sample_size: int
    mean: float
    absolute_change: float | None = None
    percent_change: float | None = None
    p_value: float | None = None
    ci_lower: float | None = None
    ci_upper: float | None = None
    is_significant: bool | None = None


class EvaluatorClaim(_AuditModel):
    """AgentCore's result for one evaluator: the control's figures and each variant's."""

    evaluator_arn: str
    control: VariantStats
    variants: tuple[VariantStats, ...]


class AbTestClaimRecord(_AuditModel):
    """What AgentCore said about an A/B test when it was read: a vendor claim, not evidence."""

    record_id: str
    claim_kind: Literal["vendor-claim"] = "vendor-claim"
    ab_test_arn: str
    ab_test_id: str
    name: str
    status: str
    execution_status: str
    gateway_arn: str
    variants: tuple[VariantRef, VariantRef]
    evaluation_config_arns: tuple[str, ...]
    current_run_id: str | None
    error_details: tuple[str, ...]
    started_at: str | None
    stopped_at: str | None
    max_duration_expires_at: str | None
    analysis_timestamp: str | None
    updated_at: str
    evaluator_claims: tuple[EvaluatorClaim, ...]
    identical_arms: bool
    provisional: bool
    read_at: str
    schema_version: str = AUDIT_SCHEMA_VERSION
    provenance: ProvenanceStamp

    @field_validator(
        "started_at",
        "stopped_at",
        "max_duration_expires_at",
        "analysis_timestamp",
        "updated_at",
        "read_at",
    )
    @classmethod
    def _times(cls, value: str | None) -> str | None:
        return _tz_aware(value)

    @model_validator(mode="after")
    def _one_of_each_arm(self) -> Self:
        if {v.name for v in self.variants} != {"C", "T1"}:
            raise AgentCoreError(f"A/B test {self.ab_test_id!r} must carry one C and one T1 arm")
        return self


ArmStatus = Literal["assigned", "unassigned", "refused", "other-experiment", "unread"]


class ArmAssignment(_AuditModel):
    """Which arm of the audited A/B test one delivered trace ran in, or why it has none."""

    trace_id: str
    otlp_trace_ids: tuple[str, ...]
    read_id: str
    runtime_session_id: str | None
    span_session_ids: tuple[str, ...]
    ab_test_arn: str
    runtime: str
    window_start: str
    window_end: str
    omni: bool
    rekey_sessions: bool
    shared_id: Literal["repeated-source-id", "rekey-collision"] | None = None
    status: ArmStatus
    experiment_arn: str | None
    variant: Literal["C", "T1"] | None
    spans_total: int = Field(ge=0)
    spans_stamped: int = Field(ge=0)
    spans_half_stamped: int = Field(default=0, ge=0)
    applied_bundle_attribute: str | None = None
    applied_bundle_require: dict[str, str] = Field(default_factory=dict)
    applied_bundle_values: dict[str, str] = Field(default_factory=dict)
    reason: str | None = None
    content_missing: bool = False
    schema_version: str = AUDIT_SCHEMA_VERSION

    @field_validator("window_start", "window_end")
    @classmethod
    def _times(cls, value: str) -> str:
        _tz_aware(value)
        return value

    @model_validator(mode="after")
    def _consistent(self) -> Self:
        where = f"trace {self.trace_id!r}"
        if self.spans_stamped + self.spans_half_stamped > self.spans_total:
            raise AgentCoreError(f"{where} counts more stamped spans than it has spans")
        if self.status == "assigned":
            if self.variant is None or self.experiment_arn != self.ab_test_arn:
                raise AgentCoreError(f"{where} is assigned without its variant or the test's ARN")
            if self.reason is not None:
                raise AgentCoreError(f"{where} is assigned but carries a reason")
        elif not self.reason:
            raise AgentCoreError(f"{where} has status {self.status!r} and no reason")
        if len(self.otlp_trace_ids) > 1 and self.shared_id is None:
            raise AgentCoreError(f"{where} covers several traces without saying why")
        if self.shared_id is not None and (
            self.status != "refused"
            or self.experiment_arn is not None
            or self.variant is not None
            or self.runtime_session_id is not None
        ):
            raise AgentCoreError(f"{where} shares its id and so must be refused with no arm")
        return self


class JsonPathRef(_AuditModel):
    """Where in a bundle a recommendation read its text: ``None`` for the system prompt's path."""

    tool_name: str | None
    json_path: str


def _unique_tools(items: tuple[TextDigest, ...] | tuple[JsonPathRef, ...], what: str) -> None:
    names = [i.tool_name for i in items]
    if len(names) != len(set(names)):
        raise AgentCoreError(f"{what} names one tool twice")


class RecommendationClaimRecord(_AuditModel):
    """What AgentCore said about a recommendation when it was read: a vendor claim, not evidence."""

    record_id: str
    claim_kind: Literal["vendor-claim"] = "vendor-claim"
    recommendation_arn: str
    recommendation_id: str
    name: str
    type: Literal["SYSTEM_PROMPT_RECOMMENDATION", "TOOL_DESCRIPTION_RECOMMENDATION"]
    status: str
    created_at: str
    updated_at: str
    input_bundle: BundleVersionRef | None
    input_json_paths: tuple[JsonPathRef, ...]
    input_inline: tuple[TextDigest, ...]
    traces_source: Literal["sessionSpans", "cloudwatchLogs", "batchEvaluation", "onlineEvaluation"]
    traces_window_start: str | None
    traces_window_end: str | None
    traces_service_name: str | None
    traces_log_group_arns: tuple[str, ...]
    traces_batch_evaluation_arn: str | None
    traces_online_evaluation_config_arn: str | None
    traces_inline_trace_ids: tuple[str, ...]
    evaluator_arn: str | None
    result_bundle: BundleVersionRef | None
    result_texts: tuple[TextDigest, ...]
    explanations: tuple[TextDigest, ...]
    error_code: str | None
    error_message: str | None
    read_at: str
    schema_version: str = AUDIT_SCHEMA_VERSION
    provenance: ProvenanceStamp

    @field_validator(
        "created_at", "updated_at", "traces_window_start", "traces_window_end", "read_at"
    )
    @classmethod
    def _times(cls, value: str | None) -> str | None:
        return _tz_aware(value)

    @model_validator(mode="after")
    def _consistent(self) -> Self:
        where = f"recommendation {self.recommendation_id!r}"
        if (self.input_bundle is None) == (not self.input_inline):
            raise AgentCoreError(
                f"{where} must name exactly one of an input bundle and inline text"
            )
        tool = self.type == "TOOL_DESCRIPTION_RECOMMENDATION"
        for group, what in (
            (self.input_json_paths, "the input JSON paths"),
            (self.input_inline, "the inline input"),
            (self.result_texts, "the recommended texts"),
            (self.explanations, "the explanations"),
        ):
            _unique_tools(group, f"{where}: {what}")
            if any((item.tool_name is None) == tool for item in group):
                raise AgentCoreError(
                    f"{where}: {what} must carry a tool name on a tool-description "
                    "recommendation and none on a system-prompt one"
                )
        return self


class RunProvenance(_AuditModel):
    """Which bundle version one replayed trace ran under, as its executor recorded it."""

    trace_id: str
    bundle_arn: str
    bundle_version: str
    runtime_arn: str | None = None
    qualifier: str | None = None
    schema_version: str = AUDIT_SCHEMA_VERSION


class AuditRecord(_AuditModel):
    """One re-test of an A/B test's or a recommendation's change, and everything it rested on."""

    audit_id: str
    subject: Literal["ab-test", "recommendation"]
    subject_arn: str
    claim_record_id: str
    improve_id: str
    target_metric: str
    before_arm: BundleVersionRef
    after_arm: BundleVersionRef
    before_root: str
    after_root: str
    before_provenance_sha256: str
    after_provenance_sha256: str
    confirmation_trace_ids: tuple[str, ...]
    confirmation_prompt_digests: tuple[str, ...]
    confirmation_slice_id: str
    prompt_key: Literal["single-string-input/1"] = "single-string-input/1"
    development_source: str
    development_window_start: str | None
    development_window_end: str | None
    development_margin_minutes: int | None
    development_source_updated_at: str | None = None
    development_digest_count: int
    development_log_groups: tuple[str, ...] = ()
    development_groups_read: tuple[str, ...] = ()
    arms_read_id: str | None
    prereg_frozen_at: str
    prereg_order: dict[str, str]
    arm_counts: dict[str, int]
    applied_bundle_checked: Literal["none", "version-named", "version-and-value"]
    target_success_threshold: float
    verdict_source: Literal["judge", "scorer", "human"]
    target_rater: str
    guard_modes: dict[str, str]
    unguarded_modes: tuple[str, ...]
    inputs: dict[str, str]
    schema_version: str = AUDIT_SCHEMA_VERSION
    provenance: ProvenanceStamp

    @model_validator(mode="after")
    def _consistent(self) -> Self:
        where = f"audit record {self.audit_id!r}"
        if not self.inputs:
            raise AgentCoreError(f"{where} needs non-empty inputs")
        guarded = {self.target_metric, *self.guard_modes}
        if guarded & set(self.unguarded_modes):
            raise AgentCoreError(f"{where} lists a guarded mode as unguarded")
        raters = [self.target_rater, *self.guard_modes.values()]
        if self.verdict_source == "judge":
            for rater in raters:
                judge, _, fingerprint = rater.rpartition("@")
                if not judge or not fingerprint:
                    raise AgentCoreError(f"{where}: judge rater {rater!r} has no judge@fingerprint")
        elif any(r != self.verdict_source for r in raters):
            raise AgentCoreError(f"{where}: every rater must be {self.verdict_source!r}")
        return self


def _store(model: type[_AuditModel], stem: str, noun: str) -> RecordStore[_AuditModel]:
    return RecordStore(
        model,
        stem=stem,
        schema_version=AUDIT_SCHEMA_VERSION,
        error=AgentCoreError,
        noun=noun,
        rerun="calibrated-agent-evaluation audit",
    )


_ABTEST_STORE = _store(AbTestClaimRecord, "agentcore_abtest_record", "A/B test claim record")
_ARMS_STORE = _store(ArmAssignment, "agentcore_arms_record", "arm assignment")
_RECOMMENDATION_STORE = _store(
    RecommendationClaimRecord, "agentcore_recommendation_record", "recommendation claim record"
)
_AUDIT_STORE = _store(AuditRecord, "agentcore_audit_record", "audit record")
_PROVENANCE_STORE = _store(RunProvenance, "agentcore_run_provenance", "run provenance record")


def dump_abtest_claims(
    records: Sequence[AbTestClaimRecord], path: str | Path, fmt: str = "csv"
) -> None:
    """Write A/B test claim records under ``path`` (both CSV and JSON are written).

    Raises:
        AgentCoreError: A bad ``fmt`` or any filesystem failure.
    """
    _ABTEST_STORE.dump(records, path, fmt)


def load_abtest_claims(path: str | Path, fmt: str = "csv") -> list[AbTestClaimRecord]:
    """Read A/B test claim records back from ``path``; a missing file is an empty store.

    Raises:
        AgentCoreError: Another schema version, or any read, parse or validation failure.
    """
    return [r for r in _ABTEST_STORE.load(path, fmt) if isinstance(r, AbTestClaimRecord)]


def dump_arm_assignments(
    records: Sequence[ArmAssignment], path: str | Path, fmt: str = "csv"
) -> None:
    """Write arm assignments under ``path`` (both CSV and JSON are written).

    Raises:
        AgentCoreError: A bad ``fmt`` or any filesystem failure.
    """
    _ARMS_STORE.dump(records, path, fmt)


def load_arm_assignments(path: str | Path, fmt: str = "csv") -> list[ArmAssignment]:
    """Read arm assignments back from ``path``; a missing file is an empty store.

    Raises:
        AgentCoreError: Another schema version, or any read, parse or validation failure.
    """
    return [r for r in _ARMS_STORE.load(path, fmt) if isinstance(r, ArmAssignment)]


def dump_recommendation_claims(
    records: Sequence[RecommendationClaimRecord], path: str | Path, fmt: str = "csv"
) -> None:
    """Write recommendation claim records under ``path``."""
    _RECOMMENDATION_STORE.dump(records, path, fmt)


def load_recommendation_claims(
    path: str | Path, fmt: str = "csv"
) -> list[RecommendationClaimRecord]:
    """Read recommendation claim records back from ``path``; a missing file is an empty store."""
    return [
        r for r in _RECOMMENDATION_STORE.load(path, fmt) if isinstance(r, RecommendationClaimRecord)
    ]


def dump_audit_records(records: Sequence[AuditRecord], path: str | Path, fmt: str = "csv") -> None:
    """Write audit records under ``path``."""
    _AUDIT_STORE.dump(records, path, fmt)


def load_audit_records(path: str | Path, fmt: str = "csv") -> list[AuditRecord]:
    """Read audit records back from ``path``; a missing file is an empty store."""
    return [r for r in _AUDIT_STORE.load(path, fmt) if isinstance(r, AuditRecord)]


def dump_run_provenance(
    records: Sequence[RunProvenance], path: str | Path, fmt: str = "csv"
) -> None:
    """Write run provenance records under ``path``."""
    _PROVENANCE_STORE.dump(records, path, fmt)


def load_run_provenance(path: str | Path, fmt: str = "csv") -> list[RunProvenance]:
    """Read run provenance records back from ``path``; a missing file is an empty store."""
    return [r for r in _PROVENANCE_STORE.load(path, fmt) if isinstance(r, RunProvenance)]
