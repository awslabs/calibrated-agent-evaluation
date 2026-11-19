# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Critique-first label construction for the review surface.

Turns a labeler's form submission into a validated, human-provenance ``Label``. This is the
enforcement point for the binary/critique-first discipline, the tiered critique
policy, and the human-gold boundary:

  * every label is ``kind="human"`` with a non-empty ``critique``;
  * a FAIL with no critique is refused; the "why" is the asset;
  * a PASS with no critique gets a canonical default so throughput is not taxed;
  * a label derived from an LLM ``Suggestion`` is refused unless a human ratified it.

Ratification is recorded via the enforced ``kind="human"`` provenance, not a label field; the
``Label`` has no metadata field by design (``kind`` is the boundary mechanism; the durable
``ratified_by`` audit record belongs to the gold-set assembly). This
module is pure (no I/O): the app calls it, then submits the returned ``Label`` through the backend.
"""

from __future__ import annotations

from collections.abc import Collection
from typing import Any, get_args

from calibrated_agent_evaluation.review.errors import RatificationError, ReviewError
from calibrated_agent_evaluation.schema import Label, Outcome

# The canonical PASS critique, applied when a labeler passes a trace without typing one.
# A fixed sentinel (not human prose) so calibration can distinguish a defaulted PASS from a
# written critique if it ever needs to.
PASS_DEFAULT_CRITIQUE = "Reviewed: outcome PASS, no defect observed."  # noqa: S105  # nosec B105 - a critique string, not a secret

_VALID_OUTCOMES: tuple[Outcome, ...] = get_args(Outcome)


def build_label(
    *,
    trace_id: str,
    outcome: str,
    critique: str,
    span_id: str | None = None,
    failure_mode: str | None = None,
    culprit_span: str | None = None,
    manifestation_span: str | None = None,
    handoff_implicated: str | None = None,
    span_defect: bool = False,
    expected: dict[str, Any] | None = None,
    codebook_version: str | None = None,
    from_suggestion: bool = False,
    ratified: bool = False,
    rater_id: str | None = None,
    span_ids: Collection[str] | None = None,
    failure_modes: Collection[str] | None = None,
) -> Label:
    """Build a validated ``kind="human"`` ``Label`` from a review submission.

    The arguments mirror the labeling form. ``from_suggestion``/``ratified`` carry the LLM-assist
    provenance; ``rater_id`` names the human in session (the review app always passes one).
    ``span_ids``, when given, is the trace's span ids; every span the label names must be one.
    ``failure_modes``, when given and non-empty, is the codebook's failure-mode list: a FAIL must
    name one of them and a PASS names none. The returned ``Label`` still passes through the
    backend's ``validate`` on write.

    Raises:
        ReviewError: ``outcome`` is not a valid binary verdict, a FAIL has an empty
            ``critique``, a named span is not in ``span_ids``, or the failure mode breaks the
            ``failure_modes`` rule.
        RatificationError: The values came from a ``Suggestion`` the human did not ratify.
    """
    if not trace_id:
        raise ReviewError("A label requires a non-empty trace_id (trace-central).")

    if outcome not in _VALID_OUTCOMES:
        raise ReviewError(f"outcome must be a binary verdict {_VALID_OUTCOMES}; got {outcome!r}.")

    # Refuse an LLM-suggested label the human never explicitly accepted/overrode,
    # before any critique defaulting; no LLM-origin label reaches the gold path unratified.
    if from_suggestion and not ratified:
        raise RatificationError(
            "This label was pre-filled by an LLM suggestion and has not been ratified by a "
            "human. Accept or override the suggestion before submitting."
        )

    if span_ids is not None:
        # A span id that is not in the trace is refused, never stored:
        # `calibrated-agent-evaluation run --ci` derives the gated agent from `culprit_span`, so a
        # typo minted a failure mode for an agent that does not exist.
        for field_name, value in (
            ("span_id", span_id),
            ("culprit_span", culprit_span),
            ("manifestation_span", manifestation_span),
        ):
            if value is not None and value not in span_ids:
                raise ReviewError(
                    f"{field_name} {value!r} is not a span of trace {trace_id!r}; "
                    f"pick one of: {', '.join(sorted(span_ids)) or '(no spans)'}."
                )

    critique = (critique or "").strip()
    if outcome == "FAIL" and not critique:
        # A hard refusal; the FAIL "why / first failure" is the irreducible asset;
        # a critique-less FAIL is the exact anti-pattern the surface exists to kill.
        raise ReviewError(
            "A FAIL requires a non-empty critique; record the failure reason before submitting "
            "(critique-first)."
        )
    if outcome == "PASS" and not critique:
        critique = PASS_DEFAULT_CRITIQUE  # tiered: a default, not a hard block, on PASS.

    if failure_modes:
        _check_failure_mode(outcome, failure_mode, failure_modes)

    return Label(
        trace_id=trace_id,
        span_id=span_id,
        kind="human",  # the review surface only ever writes human-provenance labels.
        outcome=outcome,  # type: ignore[arg-type]  # validated against _VALID_OUTCOMES above
        critique=critique,
        failure_mode=failure_mode,
        culprit_span=culprit_span,
        manifestation_span=manifestation_span,
        handoff_implicated=handoff_implicated,
        span_defect=span_defect,
        codebook_version=codebook_version,
        expected=expected,
        rater_id=rater_id,
    )


def _check_failure_mode(outcome: str, failure_mode: str | None, modes: Collection[str]) -> None:
    # A closed list, like the span pickers: a mistyped mode would count as a new mode in the
    # saturation check and split one failure across two gold sets.
    listed = ", ".join(sorted(modes))
    if outcome == "PASS":
        if failure_mode:
            raise ReviewError(f"A PASS carries no failure mode; got {failure_mode!r}.")
        return
    if not failure_mode:
        raise ReviewError(f"A FAIL needs a failure mode from the codebook; pick one of: {listed}.")
    if failure_mode not in modes:
        raise ReviewError(
            f"failure mode {failure_mode!r} is not in the codebook; pick one of: {listed}."
        )
