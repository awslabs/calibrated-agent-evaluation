# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""The lines ``audit`` prints: an A/B test's vendor claim, and an arms read's traffic.

Vendor-free; reads records only. AgentCore's figures are printed under a vendor-claim heading as
returned: they feed no Calibrated Agent Evaluation verdict.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Literal

from calibrated_agent_evaluation.agentcore.audit_types import (
    AbTestClaimRecord,
    ArmAssignment,
    AuditRecord,
    BundleVersionRef,
    RecommendationClaimRecord,
)

if TYPE_CHECKING:
    from calibrated_agent_evaluation.improve.report import ImproveRecord

__all__ = [
    "VENDOR_CLAIM_HEADING",
    "ReadSummary",
    "arms_lines",
    "bundle_differences",
    "claim_lines",
    "read_summary",
    "recommendation_lines",
    "render_report",
]

VENDOR_CLAIM_HEADING = "AgentCore reported (vendor claim, not evidence)"


def bundle_differences(control: BundleVersionRef, treatment: BundleVersionRef) -> list[str]:
    """Each component, and each top-level key of a component, whose digest differs between two."""
    before = {c.component_id: c for c in control.components}
    after = {c.component_id: c for c in treatment.components}
    out: list[str] = []
    for cid in sorted(set(before) | set(after)):
        a, b = before.get(cid), after.get(cid)
        if a is None or b is None:
            out.append(f"component {cid} is only in {'T1' if a is None else 'C'}")
            continue
        if a.sha256 == b.sha256:
            continue
        keys_a = {k.key: k.sha256 for k in a.keys}
        keys_b = {k.key: k.sha256 for k in b.keys}
        changed = sorted(k for k in set(keys_a) | set(keys_b) if keys_a.get(k) != keys_b.get(k))
        out.append(f"component {cid}" + (f": key {', '.join(changed)}" if changed else ""))
    return out


def _fmt(value: float | None) -> str:
    return "n/a" if value is None else f"{value:g}"


def claim_lines(record: AbTestClaimRecord) -> list[str]:
    """The vendor-claim block for one A/B test read."""
    lines = [
        VENDOR_CLAIM_HEADING,
        f"  A/B test {record.ab_test_id} ({record.name}): status {record.status}, execution "
        f"{record.execution_status}",
        f"  read at {record.read_at}; claim record {record.record_id}",
    ]
    if record.provisional:
        lines.append(
            f"  provisional: the test is {record.execution_status}; analysis time "
            f"{record.analysis_timestamp or 'none yet'}"
        )
    for variant in record.variants:
        if variant.bundle is not None:
            what = f"bundle {variant.bundle.bundle_arn} version {variant.bundle.version_id}"
        else:
            what = f"gateway target {variant.target_name}"
        lines.append(f"  arm {variant.name} (weight {variant.weight}): {what}")
    bundles = [v.bundle for v in record.variants if v.bundle is not None]
    if record.identical_arms:
        lines.append("  A/A comparison: both arms' bundle versions have identical components")
    elif len(bundles) == 2:
        for diff in bundle_differences(bundles[0], bundles[1]):
            lines.append(f"  differs between the arms: {diff}")
    if not record.evaluator_claims:
        lines.append(
            f"  AgentCore reported no results (status {record.status}, execution "
            f"{record.execution_status})"
        )
    else:
        lines.append(
            f"  {len(record.evaluator_claims)} evaluator comparison(s) by an undocumented test, "
            "with no documented correction across them"
        )
    for claim in record.evaluator_claims:
        lines.append(f"  evaluator {claim.evaluator_arn}")
        c = claim.control
        lines.append(
            f"    {c.variant_name}: mean {_fmt(c.mean)}, sample size {c.sample_size} "
            "(the number of sessions evaluated)"
        )
        for v in claim.variants:
            lines.append(
                f"    {v.variant_name}: mean {_fmt(v.mean)}, sample size {v.sample_size}; change "
                f"{_fmt(v.absolute_change)} ({_fmt(v.percent_change)}%), interval "
                f"[{_fmt(v.ci_lower)}, {_fmt(v.ci_upper)}] (level not stated by AgentCore), "
                f"p {_fmt(v.p_value)}, significant {v.is_significant}"
            )
    if record.error_details:
        lines.append(f"  error details: {'; '.join(record.error_details)}")
    return lines


def arms_lines(
    *,
    read_id: str,
    counts: Mapping[str, int],
    stored: int,
    omni: bool,
    rekey_sessions: bool,
    missing_from_store: Sequence[str] | None = None,
    arns_seen: Iterable[str] = (),
) -> list[str]:
    """The summary of one arms read."""
    path = "Omni" if omni else "Logs Insights"
    lines = [
        f"arms read {read_id}: {stored} assignment(s) stored"
        if stored
        else f"arms read {read_id}: unchanged, nothing stored",
    ]
    for arm in ("C", "T1"):
        lines.append(
            f"  {arm}: {counts.get(f'{arm}.traces', 0)} trace(s), "
            f"{counts.get(f'{arm}.runtime_sessions', 0)} runtime session(s), "
            f"{counts.get(f'{arm}.without_session', 0)} without a session"
        )
    lines.append(
        f"  refused {counts.get('refused', 0)} (shared source id "
        f"{counts.get('refused.shared_id', 0)}, half-stamped "
        f"{counts.get('refused.half_stamped', 0)}), unassigned {counts.get('unassigned', 0)}, "
        f"other experiment {counts.get('other-experiment', 0)}, unread {counts.get('unread', 0)}"
    )
    others = list(arns_seen)
    if counts.get("other-experiment", 0) and others:
        lines.append(f"  experiments seen: {', '.join(others)}")
    if counts.get("content_missing", 0):
        lines.append(
            f"  {counts['content_missing']} trace(s) came back without their message content, "
            "which the ingest refuses; their arms were read from their spans' stamps, and they "
            "cannot be replayed or re-tested from this window"
        )
    if missing_from_store is not None:
        lines.append(
            f"  {len(missing_from_store)} assigned trace id(s) not in this root's trace store "
            f"(read path {path}, re-key sessions {'on' if rekey_sessions else 'off'})"
            + (f": {', '.join(missing_from_store)}" if missing_from_store else "")
        )
    return lines


AppliedBundle = Literal["none", "version-named", "version-and-value"]


@dataclass(frozen=True)
class ReadSummary:
    """The traffic counts of one stored arms read, and how far it checked the applied bundle."""

    read_id: str | None
    counts: dict[str, int] = field(default_factory=dict)
    applied_bundle_checked: AppliedBundle = "none"


def read_summary(assignments: Sequence[ArmAssignment]) -> ReadSummary:
    """The one place a stored read's assignments become the counts a report prints."""
    from calibrated_agent_evaluation.agentcore.arms import arm_counts

    if not assignments:
        return ReadSummary(read_id=None)
    first = assignments[0]
    # A required value without the version attribute doesn't show which version the agent ran,
    # so only the version attribute moves the setting off "none".
    applied: AppliedBundle = (
        "none"
        if first.applied_bundle_attribute is None
        else "version-and-value"
        if first.applied_bundle_require
        else "version-named"
    )
    return ReadSummary(
        read_id=first.read_id, counts=arm_counts(assignments), applied_bundle_checked=applied
    )


def _traffic_lines(claim: AbTestClaimRecord, summary: ReadSummary) -> list[str]:
    if summary.read_id is None:
        return [
            "  traffic: no arms read is stored (`calibrated-agent-evaluation audit arms` records "
            "one)"
        ]
    counts = summary.counts
    lines = [
        f"  traffic in arms read {summary.read_id} (counted by Calibrated Agent Evaluation, not "
        "tested):"
    ]
    for arm in ("C", "T1"):
        samples = [
            f"{s.sample_size}"
            for c in claim.evaluator_claims
            for s in (c.control, *c.variants)
            if s.variant_name == arm
        ]
        beside = (
            f"; AgentCore sampleSize {', '.join(samples)} (the number of sessions evaluated)"
            if samples
            else ""
        )
        lines.append(
            f"    {arm}: {counts.get(f'{arm}.traces', 0)} trace(s) in "
            f"{counts.get(f'{arm}.runtime_sessions', 0)} runtime session(s){beside}"
        )
    lines.append(
        f"    traces without a runtime session: "
        f"{sum(counts.get(f'{arm}.without_session', 0) for arm in ('C', 'T1'))}"
    )
    lines.append(
        f"    refused {counts.get('refused', 0)} (shared source id "
        f"{counts.get('refused.shared_id', 0)}, half-stamped "
        f"{counts.get('refused.half_stamped', 0)}), unassigned {counts.get('unassigned', 0)}, "
        f"other experiment {counts.get('other-experiment', 0)}, unread {counts.get('unread', 0)}"
    )
    if summary.applied_bundle_checked == "none":
        lines.append("    the agent's use of each arm's bundle was not checked")
    elif summary.applied_bundle_checked == "version-named":
        lines.append(
            "    a matching bundle version shows only that the agent named it; whether the bundle "
            "was applied was not checked"
        )
    return lines


def recommendation_lines(record: RecommendationClaimRecord) -> list[str]:
    """The vendor-claim block for one recommendation read."""
    lines = [
        VENDOR_CLAIM_HEADING,
        f"  recommendation {record.recommendation_id} ({record.name}): {record.type}, status "
        f"{record.status}",
        f"  read at {record.read_at}; claim record {record.record_id}",
    ]
    if record.input_bundle is not None:
        bundle = record.input_bundle
        lines.append(f"  input: bundle {bundle.bundle_arn} version {bundle.version_id}")
    else:
        lines.append(f"  input: {len(record.input_inline)} inline text(s), no bundle version")
    if record.result_bundle is not None:
        bundle = record.result_bundle
        parents = ", ".join(bundle.parent_version_ids) or "none"
        lines.append(
            f"  result: bundle version {bundle.version_id} (parents {parents}, created by "
            f"{bundle.created_by or 'unknown'})"
        )
    else:
        lines.append("  result: no bundle version")
    for text in record.explanations:
        who = f"tool {text.tool_name}" if text.tool_name else "the system prompt"
        lines.append(f"  explanation for {who}: {text.length} characters")
    lines.append(f"  agent traces: {record.traces_source}")
    if record.error_code:
        lines.append(f"  error: {record.error_code}")
    return lines


_UNITS = (
    "Units: AgentCore measures a per-session mean evaluator score on live traffic; Calibrated "
    "Agent Evaluation "
    "measures a paired per-input failure verdict on a held-out replay"
)


def _p(value: float | None) -> str:
    return "undefined" if value is None else f"{value:.4g}"


def _point(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.3f}"


def _calibrated_agent_evaluation_lines(audit: AuditRecord, improve: ImproveRecord) -> list[str]:
    effect = improve.paired.effect
    level = f"{effect.level:.0%} {effect.sided}"
    lines = [
        "Calibrated Agent Evaluation re-test (paired McNemar on one held-out replay, Holm guard "
        "panel)",
        f"  verdict: {improve.verdict} (improve record {improve.improve_id}, audit "
        f"{audit.audit_id})",
        f"  target {audit.target_metric!r}, rater {audit.target_rater}; registered minimum "
        f"reduction {audit.target_success_threshold:.2f}",
        f"  McNemar {improve.paired.method}, {improve.paired.n_discordant} discordant pair(s), "
        f"p {_p(improve.paired.p_value)}",
        f"  effect on the fail rate (after - before) {_point(effect.point)} "
        f"[{effect.lo:.3f}, {effect.hi:.3f}] ({level}, {effect.method})",
        f"  guard panel: {improve.panel.method}, alpha 0.05, m={improve.panel.m}",
    ]
    for mode, rater in sorted(audit.guard_modes.items()):
        lines.append(f"    guard {mode!r}, rater {rater}")
    lines.append(
        "  not guarded: "
        + (
            ", ".join(audit.unguarded_modes)
            if audit.unguarded_modes
            else "no labelled mode (a mode no label names is not listed)"
        )
    )
    order = ", ".join(f"{k} {v}" for k, v in sorted(audit.prereg_order.items())) or "not recorded"
    lines.append(f"  pre-registration frozen at {audit.prereg_frozen_at}: {order}")
    for reason in improve.reasons:
        lines.append(f"  reason: {reason}")
    if audit.development_log_groups:
        lines.append(
            f"  development source log groups: {', '.join(audit.development_log_groups)}; "
            f"read: {', '.join(audit.development_groups_read)}"
        )
    return lines


def render_report(
    claim: AbTestClaimRecord | RecommendationClaimRecord,
    summary: ReadSummary,
    *,
    audit: AuditRecord | None = None,
    improve: ImproveRecord | None = None,
    retest_hint: str = "calibrated-agent-evaluation audit retest",
) -> list[str]:
    """The side-by-side report: AgentCore's claim, and the library's re-test when one is stored."""
    if isinstance(claim, AbTestClaimRecord):
        lines = [*claim_lines(claim), *_traffic_lines(claim, summary)]
    else:
        lines = recommendation_lines(claim)
    lines.append(_UNITS)
    if audit is not None and improve is not None:
        lines += _calibrated_agent_evaluation_lines(audit, improve)
    else:
        lines.append(f"No re-test is stored for this subject; `{retest_hint}` makes one.")
    return lines
