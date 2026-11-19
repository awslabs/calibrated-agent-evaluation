# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Publish an online judge's calibrated pass rate, and its interval, as CloudWatch metrics.

AgentCore already publishes the raw scores of its online evaluations. What it cannot publish is
the rate corrected for the judge's measured errors: :func:`publish_pass_rate` takes the online
verdicts, corrects their failure rate with the judge's calibration record (Rogan-Gladen, through
:func:`calibrated_agent_evaluation.calibrate.engine.corrected_rate`), and publishes one minus it,
with the interval's bounds swapped, so the operator can alarm when the lower bound falls. Nothing is
published for a judge that is no longer the one certified, or when the correction is refused; the
raw pass rate is never published.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import TYPE_CHECKING

from calibrated_agent_evaluation.agentcore.errors import AgentCoreError, AgentCoreMetricNotPublished
from calibrated_agent_evaluation.agentcore.judge import AgentCoreJudge
from calibrated_agent_evaluation.agentcore.online import (
    AttachedStatus,
    aws_error_code,
    call_aws,
    check_attached,
)
from calibrated_agent_evaluation.agentcore.online_record import (
    AttachedEvaluator,
    OnlineEvaluationRecord,
)
from calibrated_agent_evaluation.calibrate.record import CalibrationRecord
from calibrated_agent_evaluation.schema.types import Label

if TYPE_CHECKING:
    from calibrated_agent_evaluation.agentcore.session import AwsClients

__all__ = ["METRIC_NAMES", "NAMESPACE", "PublishedMetrics", "publish_pass_rate"]

NAMESPACE = "CalibratedAgentEvaluation/Judges"
METRIC_NAMES = ("CorrectedPassRate", "CorrectedPassRateLower95", "CorrectedPassRateUpper95")


@dataclass(frozen=True)
class PublishedMetrics:
    """What one publish sent, or why it sent nothing.

    Attributes:
        judge_id: The judge the metrics are for.
        fingerprint: The evaluator fingerprint they were published under.
        gold_source: Where the calibration gold came from, published as a dimension.
        status: Whether the attached evaluator is still the one certified.
        published: Each metric name and the value CloudWatch accepted; empty when nothing was.
        refusal: Why nothing was published, in plain words, or ``None`` when the three values
            were.
        detail: The statistics behind a refused correction, for ``--explain``; ``None`` otherwise.
    """

    judge_id: str
    fingerprint: str
    gold_source: str
    status: AttachedStatus
    published: tuple[tuple[str, float], ...]
    refusal: str | None
    detail: str | None = None


def publish_pass_rate(
    record: OnlineEvaluationRecord,
    judge: AgentCoreJudge,
    calib_record: CalibrationRecord,
    *,
    clients: AwsClients,
    rate_labels: Sequence[Label],
    gold_source: str,
    rate_cluster_ids: Sequence[str] | None = None,
    clock: Callable[[], datetime] = lambda: datetime.now(UTC),
) -> PublishedMetrics:
    """Publish ``judge``'s corrected pass rate over ``rate_labels`` and its 95% interval.

    ``rate_labels`` must all be this judge's own verdicts as it is now (their ``rater_id`` is
    ``<judge_id>@<fingerprint>``), one per trace. The evaluator is checked first
    (:func:`~calibrated_agent_evaluation.agentcore.online.check_attached`); one that changed or is
    gone publishes nothing, and so does a ``calib_record`` or ``judge`` that is not the one the
    config attached. ``rate_cluster_ids`` (each verdict's session, when sessions hold several
    traces) widens the interval for verdicts that are not independent. The failure rate of
    ``rate_labels`` (the online verdicts the read-back stored, one per scored trace) is corrected
    with ``calib_record``; when the correction is refused, too small a sample or a rate the judge's
    errors do not transfer to, nothing is published and the refusal is returned. Otherwise
    ``CorrectedPassRate``, ``CorrectedPassRateLower95`` and ``CorrectedPassRateUpper95`` go to
    ``CalibratedAgentEvaluation/Judges`` in one synchronous ``PutMetricData``, with ``JudgeId``,
    ``EvaluatorFingerprint`` and ``GoldSource`` dimensions.

    Raises:
        AgentCoreMetricNotPublished: CloudWatch rejected the call; none of the three values was
            published.
        AgentCoreError: ``judge`` is not attached to ``record``; ``rate_labels`` holds a verdict
            by another rater or another version of the judge, or two for one trace; reading the
            evaluator failed; or the publish call failed in a way that leaves unknown whether
            CloudWatch recorded it.
    """
    from calibrated_agent_evaluation.calibrate.engine import corrected_rate, pin_mismatch
    from calibrated_agent_evaluation.calibrate.errors import (
        CalibrationError,
        RateSampleTooSmallError,
    )
    from calibrated_agent_evaluation.judges.verdict import verdict_rater_id

    _check_labels(rate_labels, verdict_rater_id(judge))
    attached = _attached(record, judge)
    (status,) = check_attached(
        record.model_copy(update={"evaluators": (attached,)}), clients=clients
    )

    def refused(reason: str, detail: str | None = None) -> PublishedMetrics:
        return PublishedMetrics(
            judge.judge_id, judge.fingerprint, gold_source, status, (), reason, detail
        )

    if not status.certified:
        return refused(status.reason)
    mismatch = _record_mismatch(attached, judge, calib_record, pin_mismatch(calib_record, judge))
    if mismatch is not None:
        return refused(mismatch)
    verdicts = [1 if label.outcome == "FAIL" else 0 for label in rate_labels]
    try:
        scored = corrected_rate(
            calib_record,
            verdicts,
            f"online-{record.config_id}",
            rate_trace_ids=[label.trace_id for label in rate_labels],
            rate_cluster_ids=rate_cluster_ids,
        ).corrected_rate
    except RateSampleTooSmallError as exc:
        return refused(f"there are too few online verdicts to publish a pass rate yet: {exc}")
    except CalibrationError as exc:
        return refused(_why_not_corrected(exc), str(exc))
    if scored is None or scored.point is None:
        return refused(_NOT_CORRECTED, scored.refusal_reason if scored is not None else None)
    values = (1.0 - scored.point, 1.0 - scored.hi, 1.0 - scored.lo)
    dimensions = [
        {"Name": "JudgeId", "Value": judge.judge_id},
        {"Name": "EvaluatorFingerprint", "Value": judge.fingerprint},
        {"Name": "GoldSource", "Value": gold_source},
    ]
    when = clock()
    request = {
        "Namespace": NAMESPACE,
        "MetricData": [
            {
                "MetricName": name,
                "Dimensions": dimensions,
                "Value": value,
                "Unit": "None",
                "Timestamp": when,
            }
            for name, value in zip(METRIC_NAMES, values, strict=True)
        ],
    }
    try:
        call_aws(clients, "cloudwatch", "PutMetricData", request)
    except AgentCoreError as exc:
        code = aws_error_code(exc)
        if code is None:
            # Not an answer from CloudWatch: the request may still have been recorded.
            raise AgentCoreError(
                f"CloudWatch did not confirm the metrics {', '.join(METRIC_NAMES)}, and they may "
                f"or may not have been recorded, so they are not reported as published: {exc}"
            ) from exc
        raise AgentCoreMetricNotPublished(", ".join(METRIC_NAMES), code) from exc
    published = tuple(zip(METRIC_NAMES, values, strict=True))
    return PublishedMetrics(judge.judge_id, judge.fingerprint, gold_source, status, published, None)


_NOT_CORRECTED = (
    "the judge's measured mistakes do not fit these online verdicts, so no corrected pass rate "
    "can be given and nothing was published; check that the agent's traffic is like the traces "
    "the judge was calibrated on (run with --explain for the numbers)"
)


def _why_not_corrected(exc: Exception) -> str:
    """A plain sentence for a refused correction; the statistics stay in the detail."""
    from calibrated_agent_evaluation.calibrate.errors import (
        RateUnsafeError,
        SampleMisuseError,
        UncalibratedJudgeError,
    )

    if isinstance(exc, SampleMisuseError):
        return (
            "the online verdicts cannot be used to measure the pass rate (they include traces the "
            "judge was calibrated on, or none at all), so nothing was published"
        )
    if isinstance(exc, RateUnsafeError):
        return (
            "the judge's calibration had too few gold labels to correct a rate safely, so nothing "
            "was published; calibrate it on more gold first"
        )
    if isinstance(exc, UncalibratedJudgeError):
        return "the judge is not certified, so nothing was published"
    return _NOT_CORRECTED


def _check_labels(rate_labels: Sequence[Label], rater_id: str) -> None:
    """Refuse verdicts that are not ``rater_id``'s, or two verdicts for one trace."""
    seen: set[str] = set()
    for label in rate_labels:
        if label.rater_id != rater_id:
            raise AgentCoreError(
                f"verdict on trace {label.trace_id} is by {label.rater_id}, not {rater_id}; only "
                "the attached judge's own verdicts measure its pass rate"
            )
        if label.trace_id in seen:
            raise AgentCoreError(
                f"trace {label.trace_id} has more than one verdict; each trace counts once"
            )
        seen.add(label.trace_id)


def _attached(record: OnlineEvaluationRecord, judge: AgentCoreJudge) -> AttachedEvaluator:
    """The evaluator ``record`` attached for ``judge``."""
    for attached in record.evaluators:
        if attached.evaluator_id == judge.evaluator_id:
            return attached
    raise AgentCoreError(
        f"evaluator {judge.evaluator_id} is not attached to online config {record.config_id}, "
        "so there is nothing of it to publish"
    )


def _record_mismatch(
    attached: AttachedEvaluator,
    judge: AgentCoreJudge,
    calib_record: CalibrationRecord,
    pin: str | None,
) -> str | None:
    """Why ``judge`` and ``calib_record`` are not the ones the config attached, if they are not."""
    if calib_record.calib_id != attached.calib_id:
        return (
            f"calibration {calib_record.calib_id} is not the one evaluator {attached.evaluator_id} "
            f"was attached with ({attached.calib_id}), so its error rates would correct the wrong "
            "judge; nothing was published"
        )
    if judge.fingerprint != attached.fingerprint:
        return (
            f"the judge given is not the evaluator {attached.evaluator_id} was attached as "
            "(its fingerprint differs); nothing was published"
        )
    if pin is not None:
        return f"calibration {calib_record.calib_id} does not certify this judge: {pin}"
    return None
