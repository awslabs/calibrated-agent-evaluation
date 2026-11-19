# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Map AgentCore evaluator results to advisory Calibrated Agent Evaluation labels.

AgentCore batch, online and console evaluations produce result records (``evaluatorArn``,
``value``, ``label``, ``explanation``, a ``spanContext`` address). This module maps each to an
**advisory** ``Label`` (``kind="llm-judge"``): it is never gold and never a gate — certifying a
judge is the calibration engine's job. The raw ``value`` and ``label`` are preserved verbatim in
the critique, so the real verdict survives for later calibration and the mapped binary is only a
provisional, inspectable convenience.

Vendor-free by contract: the AWS read of the result records (the batch result log stream, named
after the ``batchEvaluationId``, and the online/console result log groups) lives in
:mod:`calibrated_agent_evaluation.agentcore.readers`; this module maps already-decoded result dicts,
so re-reading the same results yields identical labels (stable identity) and nothing here touches
boto3. The reader carries a batch record's job id on the dict under ``jobId``;
:func:`result_to_label` stamps it into the provenance (an online/console result has none).
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator
from typing import Any

from calibrated_agent_evaluation.agentcore.errors import AgentCoreScoreIncomplete
from calibrated_agent_evaluation.agentcore.judge import (
    ScaleValue,
    outcome_for_value,
    scores_inverted,
)
from calibrated_agent_evaluation.schema.types import Label, Outcome

# Result labels that read as a pass across the correctness-family built-ins. An evaluator that
# scores failures high (FailureAnalysis, Toxicity, Bias) is flipped by the direction lookup in
# ``agentcore.judge`` rather than read against these labels. The binary is advisory only; the raw
# value/label in the critique is what a later calibration reads.
_PASS_LABELS = frozenset({"pass", "correct", "perfectly correct", "yes", "true", "compliant"})


def _evaluator_names(result: dict[str, Any]) -> set[str]:
    """The names a result's evaluator goes by: its id, its name, and the tail of its ARN.

    An ``Evaluate`` result carries the id and ARN; a result-log record carries only the name and
    ARN, so all three are read for the direction lookup to work on both paths.
    """
    names = {str(result[key]) for key in ("evaluatorId", "evaluatorName") if result.get(key)}
    arn = result.get("evaluatorArn")
    if isinstance(arn, str) and "evaluator/" in arn:
        names.add(arn.rsplit("evaluator/", 1)[1])
    return names


def _advisory_outcome(value: float | None, label: str | None, *, inverted: bool) -> Outcome:
    # A categorical evaluator can report a non-numeric score; only a real number is comparable, so
    # guard the comparison rather than let a str/float TypeError escape the typed-error boundary.
    number = value if isinstance(value, (int, float)) and not isinstance(value, bool) else None
    if inverted:
        # Mirror the default's "only a perfect score passes": for an evaluator that scores failures
        # high, only a perfectly clean 0.0 passes. Its labels name the bad outcome, so they are not
        # read against the pass labels.
        return "PASS" if number is not None and number <= 0.0 else "FAIL"
    if label is not None and label.strip().lower() in _PASS_LABELS:
        return "PASS"
    if number is not None and number >= 1.0:
        return "PASS"
    return "FAIL"


def result_to_label(
    result: dict[str, Any],
    *,
    job_id: str | None = None,
    rating_scale: tuple[ScaleValue, ...] = (),
    inverted: bool | None = None,
) -> Label:
    """Map one AgentCore evaluation result to an advisory ``Label``.

    When ``rating_scale`` is given and the result carries a numeric value, the binary outcome is
    read from that value against the evaluator's published scale; otherwise the polarity default
    reads it from the label text. A value off the published scale is refused.

    ``inverted`` says the evaluator scores failures high, so the outcome is flipped. The judge path
    passes its judge's direction (a derived evaluator's results name only itself, not its base);
    left ``None``, the direction is looked up from the evaluator the result names.

    Raises :class:`AgentCoreScoreIncomplete` when the record lacks a trace/session id or has no
    value or label, rather than storing a label with an empty or inferred field, and
    :class:`AgentCoreNoPassDirection` when the evaluator's score has no pass direction.
    """
    # An explicit ``job_id`` wins; otherwise fall back to the one the reader carried on the record
    # (a batch result's log stream name). An online/console result carries none, so it stays None.
    if job_id is None:
        carried = result.get("jobId")
        job_id = carried if isinstance(carried, str) and carried else None
    span_context = (result.get("context") or {}).get("spanContext") or {}
    trace_id = span_context.get("traceId") or span_context.get("sessionId")
    if not trace_id:
        raise AgentCoreScoreIncomplete(
            f"evaluation result from {result.get('evaluatorArn')!r} has no traceId or sessionId; "
            "it cannot be addressed to a trace, so it is refused rather than stored unaddressed"
        )
    value = result.get("value")
    label_str = result.get("label")
    if value is None and not label_str:
        raise AgentCoreScoreIncomplete(
            f"evaluation result for trace {trace_id!r} has neither a value nor a label; there is "
            "no verdict to record, so it is refused rather than stored empty"
        )
    # A tool-level result addresses a span; a trace/session-level result does not.
    span_id = span_context.get("spanId") or None
    arn = result.get("evaluatorArn") or result.get("evaluatorId") or "unknown-evaluator"
    explanation = (result.get("explanation") or "").strip()
    name = result.get("evaluatorName") or arn
    body = explanation or f"AgentCore evaluator {name} produced no explanation"
    # The raw verdict rides in the critique verbatim: nothing the evaluator reported is lost, the
    # advisory binary stays auditable, and the identity is a pure function of the result (so a
    # re-read dedups). A job id, when present, distinguishes a batch result from an online one.
    provenance = f"[agentcore evaluator={arn} value={value} label={label_str!r}"
    provenance += f" job={job_id}]" if job_id else "]"
    if inverted is None:
        inverted = scores_inverted(_evaluator_names(result))
    if rating_scale and isinstance(value, (int, float)) and not isinstance(value, bool):
        outcome = outcome_for_value(float(value), rating_scale, inverted=inverted)
    else:
        outcome = _advisory_outcome(value, label_str, inverted=inverted)
    return Label(
        trace_id=str(trace_id),
        span_id=str(span_id) if span_id else None,
        kind="llm-judge",
        outcome=outcome,
        critique=f"{body} {provenance}",
    )


def read_scores(results: Iterable[dict[str, Any]], *, job_id: str | None = None) -> Iterator[Label]:
    """Map a stream of AgentCore evaluation results to advisory labels.

    Raises, at the record that cannot be mapped, the error :func:`result_to_label` raises:
    :class:`AgentCoreScoreIncomplete` for a record with no trace or session id or no verdict, and
    :class:`AgentCoreNoPassDirection` for an evaluator with no pass direction. Nothing is skipped
    silently; a caller that wants to skip such records maps each one with :func:`result_to_label`
    and counts the ones it skips, as
    :func:`calibrated_agent_evaluation.agentcore.ingest.ingest_agentcore` does.
    """
    for result in results:
        yield result_to_label(result, job_id=job_id)
