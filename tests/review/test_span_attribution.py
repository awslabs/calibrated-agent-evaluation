# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Span-level crew attribution — must reproduce the gold-label shapes."""

from __future__ import annotations

from calibrated_agent_evaluation._model import replace
from calibrated_agent_evaluation.review import build_label
from calibrated_agent_evaluation.schema import Label


def _gold(labels: list[Label], trace_id: str, span_id: str | None) -> Label:
    return next(lab for lab in labels if lab.trace_id == trace_id and lab.span_id == span_id)


def test_trace_level_attribution_reproduces_gold_row(gold_labels: list[Label]) -> None:
    gold = _gold(gold_labels, "CLM-4471", None)
    label = build_label(
        trace_id="CLM-4471",
        outcome="FAIL",
        critique="counterfactual walk-back lands on intake",
        failure_mode=gold.failure_mode,
        culprit_span=gold.culprit_span,
        manifestation_span=gold.manifestation_span,
        handoff_implicated=gold.handoff_implicated,
        expected=gold.expected,
        codebook_version=gold.codebook_version,
        rater_id=gold.rater_id,
    )
    assert label.span_id is None
    assert label.culprit_span == "CLM-4471/intake"
    assert label.manifestation_span == "CLM-4471/payout"
    assert label.handoff_implicated == "intake->policy-checker"
    assert label.span_defect is False
    # The UI can reconstruct the gold row exactly (modulo the human's free-text critique).
    assert replace(gold, critique=label.critique) == label


def test_span_level_defect_reproduces_gold_row(gold_labels: list[Label]) -> None:
    gold = _gold(gold_labels, "CLM-4471", "CLM-4471/intake")
    label = build_label(
        trace_id="CLM-4471",
        span_id="CLM-4471/intake",
        outcome="FAIL",
        critique="intake returned occupancy_status=null",
        failure_mode=gold.failure_mode,
        culprit_span=gold.culprit_span,
        span_defect=True,
        codebook_version=gold.codebook_version,
        rater_id=gold.rater_id,
    )
    assert label.span_id == "CLM-4471/intake"
    assert label.span_defect is True
    assert label.manifestation_span is None  # the span-level defect row has no manifestation
    assert replace(gold, critique=label.critique) == label


def test_label_is_identified_only_by_trace_and_span_ids() -> None:
    label = build_label(
        trace_id="CLM-4471", span_id="CLM-4471/intake", outcome="FAIL", critique="x"
    )
    field_names = set(type(label).model_fields)
    # Identity is trace-central: trace_id + optional span_id — no backend/vendor id field.
    assert "trace_id" in field_names
    assert "span_id" in field_names
    assert not (field_names & {"id", "pk", "_id", "rowid", "langfuse_id", "observation_id"})
    assert label.trace_id == "CLM-4471"
    assert label.span_id == "CLM-4471/intake"
