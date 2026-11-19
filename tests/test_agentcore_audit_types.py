# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""The A/B audit's records: they round-trip, refuse another schema, and refuse impossible shapes."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from _arms_fixture import MAIN, VERSIONS, Window, deliver
from calibrated_agent_evaluation._provenance import capture_provenance
from calibrated_agent_evaluation.agentcore.arms import read_arms
from calibrated_agent_evaluation.agentcore.audit_types import (
    AbTestClaimRecord,
    ArmAssignment,
    AuditRecord,
    BundleVersionRef,
    ComponentDigest,
    KeyDigest,
    RecommendationClaimRecord,
    RunProvenance,
    VariantRef,
    dump_abtest_claims,
    dump_arm_assignments,
    dump_audit_records,
    dump_recommendation_claims,
    dump_run_provenance,
    load_abtest_claims,
    load_arm_assignments,
    load_audit_records,
    load_recommendation_claims,
    load_run_provenance,
)
from calibrated_agent_evaluation.agentcore.errors import AgentCoreError

_GOLDEN = Path(__file__).resolve().parents[1] / "fixtures" / "expected" / "audit"
_ARN = "arn:aws:bedrock-agentcore:us-west-2:111122223333:ab-test/fixture-abtest-stopped"


def _claim() -> AbTestClaimRecord:
    doc = json.loads((_GOLDEN / "abtest_verdict.json").read_text(encoding="utf-8"))
    return AbTestClaimRecord.model_validate(
        {
            **doc,
            "record_id": "abtest-0001",
            "read_at": "2026-11-02T10:00:00+00:00",
            "provenance": capture_provenance().model_dump(mode="json"),
        }
    )


def _assignments() -> tuple[ArmAssignment, ...]:
    from _arms_fixture import Double

    return read_arms(
        Double(deliver(MAIN)),
        ab_test_arn=_ARN,
        runtime="fixture-runtime",
        window=Window(),
        bundle_versions=VERSIONS,
    ).assignments


@pytest.mark.parametrize("fmt", ["csv", "json"])
def test_audit_records_round_trip_through_csv_and_json(tmp_path: Path, fmt: str) -> None:
    claim = _claim()
    dump_abtest_claims([claim], tmp_path / "claims", fmt=fmt)
    assert load_abtest_claims(tmp_path / "claims", fmt=fmt) == [claim]

    arms = list(_assignments())
    dump_arm_assignments(arms, tmp_path / "arms", fmt=fmt)
    assert load_arm_assignments(tmp_path / "arms", fmt=fmt) == arms

    recs = [recommendation_claim_record()]
    dump_recommendation_claims(recs, tmp_path / "recs", fmt=fmt)
    assert load_recommendation_claims(tmp_path / "recs", fmt=fmt) == recs
    audits = [audit_record()]
    dump_audit_records(audits, tmp_path / "audits", fmt=fmt)
    assert load_audit_records(tmp_path / "audits", fmt=fmt) == audits
    runs = [run_provenance()]
    dump_run_provenance(runs, tmp_path / "runs", fmt=fmt)
    assert load_run_provenance(tmp_path / "runs", fmt=fmt) == runs


def test_a_stored_audit_record_of_another_schema_version_is_refused(tmp_path: Path) -> None:
    dump_abtest_claims([_claim()], tmp_path)
    stored = tmp_path / "agentcore_abtest_record.json"
    docs = json.loads(stored.read_text(encoding="utf-8"))
    docs[0]["schema_version"] = "0"
    stored.write_text(json.dumps(docs), encoding="utf-8")

    with pytest.raises(AgentCoreError) as refused:
        load_abtest_claims(tmp_path, fmt="json")
    assert "calibrated-agent-evaluation audit" in str(refused.value)

    stored.write_text("[{", encoding="utf-8")
    with pytest.raises(AgentCoreError):
        load_abtest_claims(tmp_path, fmt="json")


def _bundle(**over: Any) -> dict[str, Any]:
    base: dict[str, Any] = {
        "bundle_arn": "arn:b",
        "version_id": "v",
        "version_created_at": "2026-10-28T18:00:00+00:00",
        "parent_version_ids": (),
        "branch_name": None,
        "created_by": None,
        "components": (),
        "components_sha256": "x",
    }
    return {**base, **over}


def _assignment(**over: Any) -> dict[str, Any]:
    base: dict[str, Any] = {
        "trace_id": "t",
        "otlp_trace_ids": ("t",),
        "read_id": "arms-1",
        "runtime_session_id": None,
        "span_session_ids": (),
        "ab_test_arn": _ARN,
        "runtime": "r",
        "window_start": "2026-10-28T19:00:00+00:00",
        "window_end": "2026-10-28T20:00:00+00:00",
        "omni": False,
        "rekey_sessions": False,
        "status": "assigned",
        "experiment_arn": _ARN,
        "variant": "C",
        "spans_total": 2,
        "spans_stamped": 2,
    }
    return {**base, **over}


def claim_record() -> AbTestClaimRecord:
    """One fixed claim record, equal on every call (for the record-kind round-trip test)."""
    from datetime import UTC, datetime

    doc = json.loads((_GOLDEN / "abtest_verdict.json").read_text(encoding="utf-8"))
    when = datetime(2026, 11, 2, 10, 0, tzinfo=UTC)
    return AbTestClaimRecord.model_validate(
        {
            **doc,
            "record_id": "abtest-fixed",
            "read_at": when.isoformat(),
            "provenance": capture_provenance(now=when).model_dump(mode="json"),
        }
    )


def arm_assignment() -> ArmAssignment:
    """One fixed arm assignment, equal on every call."""
    return ArmAssignment.model_validate(_assignment())


_FIXED = "2026-11-02T10:00:00+00:00"


def _text(tool: str | None) -> dict[str, Any]:
    return {"tool_name": tool, "sha256": "a", "length": 1}


def _recommendation(**over: Any) -> dict[str, Any]:
    from datetime import UTC, datetime

    doc = json.loads((_GOLDEN / "recommendation.json").read_text(encoding="utf-8"))
    when = datetime(2026, 11, 2, 10, 0, tzinfo=UTC)
    base = {
        **doc,
        "record_id": "recommendation-fixed",
        "read_at": _FIXED,
        "provenance": capture_provenance(now=when).model_dump(mode="json"),
    }
    return {**base, **over}


def _audit(**over: Any) -> dict[str, Any]:
    from datetime import UTC, datetime

    claim = claim_record()
    before, after = (v.bundle for v in claim.variants)
    base: dict[str, Any] = {
        "audit_id": "audit-fixed",
        "subject": "ab-test",
        "subject_arn": claim.ab_test_arn,
        "claim_record_id": claim.record_id,
        "improve_id": "improve-fixed",
        "target_metric": "missing_date_range",
        "before_arm": before,
        "after_arm": after,
        "before_root": "/replay/control",
        "after_root": "/replay/treatment",
        "before_provenance_sha256": "b",
        "after_provenance_sha256": "a",
        "confirmation_trace_ids": ("h1-c00",),
        "confirmation_prompt_digests": ("d",),
        "confirmation_slice_id": "confirm-0123456789ab",
        "development_source": "root:/replay/development",
        "development_window_start": None,
        "development_window_end": None,
        "development_margin_minutes": None,
        "development_digest_count": 10,
        "arms_read_id": "arms-1",
        "prereg_frozen_at": "2026-10-01T09:00:00+00:00",
        "prereg_order": {"ab_test_started_at": "before", "analysis_timestamp": "before"},
        "arm_counts": {"C.traces": 5, "T1.traces": 4},
        "applied_bundle_checked": "none",
        "target_success_threshold": 0.10,
        "verdict_source": "judge",
        "target_rater": "target-judge@3f1c",
        "guard_modes": {"wrong_join": "guard-judge@8e2d"},
        "unguarded_modes": ("unbounded_scan",),
        "inputs": {"improve_id": "improve-fixed"},
        "provenance": capture_provenance(now=datetime(2026, 11, 2, 10, 0, tzinfo=UTC)),
    }
    return {**base, **over}


def recommendation_claim_record() -> RecommendationClaimRecord:
    """One fixed recommendation claim record, equal on every call."""
    return RecommendationClaimRecord.model_validate(_recommendation())


def audit_record() -> AuditRecord:
    """One fixed audit record, equal on every call."""
    return AuditRecord.model_validate(_audit())


def run_provenance() -> RunProvenance:
    """One fixed run provenance record."""
    return RunProvenance(
        trace_id="h1-c00",
        bundle_arn="arn:aws:bedrock-agentcore:us-west-2:111122223333:configuration-bundle/b",
        bundle_version="0a1b2c3d-0000-4000-8000-000000000002",
    )


@pytest.mark.parametrize(
    ("model", "doc"),
    [
        (VariantRef, {"name": "C", "weight": 50, "bundle": _bundle(), "target_name": "t"}),
        (VariantRef, {"name": "C", "weight": 50, "bundle": None, "target_name": None}),
        (
            BundleVersionRef,
            _bundle(
                components=(
                    {"component_id": "a", "sha256": "1", "length": 1},
                    {"component_id": "a", "sha256": "2", "length": 1},
                )
            ),
        ),
        (
            ComponentDigest,
            {
                "component_id": "a",
                "sha256": "1",
                "length": 1,
                "keys": [KeyDigest(key="k", sha256="1", length=1)] * 2,
            },
        ),
        (ArmAssignment, _assignment(variant=None)),
        (ArmAssignment, _assignment(experiment_arn="arn:other")),
        (ArmAssignment, _assignment(status="unassigned", variant=None, experiment_arn=None)),
        (ArmAssignment, _assignment(otlp_trace_ids=("a", "b"))),
        (
            ArmAssignment,
            _assignment(
                status="refused",
                otlp_trace_ids=("a", "b"),
                shared_id="repeated-source-id",
                reason="two traces",
            ),
        ),
        (ArmAssignment, _assignment(spans_stamped=2, spans_half_stamped=1)),
        (
            RecommendationClaimRecord,
            _recommendation(result_texts=[_text(None), _text(None)]),
        ),
        (RecommendationClaimRecord, _recommendation(explanations=[_text("lookup_order")])),
        (
            RecommendationClaimRecord,
            _recommendation(
                type="TOOL_DESCRIPTION_RECOMMENDATION",
                input_json_paths=[],
                result_texts=[],
                explanations=[_text(None)],
            ),
        ),
        (RecommendationClaimRecord, _recommendation(input_inline=[_text(None)])),
        (RecommendationClaimRecord, _recommendation(input_bundle=None, input_json_paths=[])),
        (AuditRecord, _audit(inputs={})),
        (AuditRecord, _audit(unguarded_modes=("wrong_join",))),
        (AuditRecord, _audit(unguarded_modes=("missing_date_range",))),
        (AuditRecord, _audit(target_rater="target-judge")),
        (AuditRecord, _audit(guard_modes={"wrong_join": "guard-judge"})),
    ],
    ids=[
        "variant-both",
        "variant-neither",
        "repeated-component",
        "repeated-key",
        "assigned-without-variant",
        "assigned-other-arn",
        "status-without-reason",
        "two-traces-without-shared-id",
        "shared-id-with-variant",
        "more-stamped-than-spans",
        "repeated-tool-name",
        "tool-name-on-a-system-prompt",
        "no-tool-name-on-a-tool-description",
        "bundle-and-inline-input",
        "neither-bundle-nor-inline-input",
        "audit-without-inputs",
        "unguarded-guard-mode",
        "unguarded-target",
        "judge-target-rater-without-fingerprint",
        "judge-guard-rater-without-fingerprint",
    ],
)
def test_audit_record_shapes_that_cannot_be_true_are_refused(
    model: Any, doc: dict[str, Any]
) -> None:
    with pytest.raises(AgentCoreError):
        model.model_validate(doc)
