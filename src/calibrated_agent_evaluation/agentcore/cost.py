# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""What an online evaluation run cost, counted from what was stored and AgentCore wrote.

The calls a judge's calibration and bias runs made are the entries of the session logs those
runs stored (a run refused part-way stores none, so its calls cannot be counted); the online
config's own work is the results in its result group. Each is a custom evaluation, billed at the
Price List's per-evaluation rate. The evaluator's own model usage (or Lambda) is billed on top by
Bedrock, and neither record holds it, so the report is a floor and says so. Whether AWS bills an
evaluation it could not finish is not documented, so error results are counted and left unpriced.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from decimal import Decimal

from calibrated_agent_evaluation.agentcore.online_record import (
    AgentCoreSessionLog,
    OnlineEvaluationRecord,
)
from calibrated_agent_evaluation.agentcore.pricing import EvaluationRates

__all__ = ["CostReport", "JudgeCost", "online_run_cost"]


@dataclass(frozen=True)
class JudgeCost:
    """One attached judge's calibration and bias calls.

    Attributes:
        evaluator_id: The attached evaluator.
        calibration_calls: Entries of the session log of the calibration it was attached with.
        bias_calls: Entries of the session log of its bias battery.
        missing_logs: The runs whose session log is not stored (``calibration``, ``bias``).
    """

    evaluator_id: str
    calibration_calls: int
    bias_calls: int
    missing_logs: tuple[str, ...]


@dataclass(frozen=True)
class CostReport:
    """The evaluations an online run made that the library can count, and their price.

    Attributes:
        config_id: The online config.
        judges: Each attached judge's calibration and bias calls.
        results: The scored results in the config's result group.
        error_results: The results AgentCore could not score.
        per_evaluation: The Price List's per-evaluation rate for custom evaluators, or ``None``
            when the lookup failed.
        unpriced_reason: Why there is no price, when there is none.
    """

    config_id: str
    judges: tuple[JudgeCost, ...]
    results: int
    error_results: int
    per_evaluation: Decimal | None
    unpriced_reason: str | None

    @property
    def priced_evaluations(self) -> int:
        """The calibration, bias and scored online evaluations together."""
        runs = sum(j.calibration_calls + j.bias_calls for j in self.judges)
        return runs + self.results

    @property
    def cost(self) -> Decimal | None:
        """What those evaluations cost at the per-evaluation rate, or ``None`` with no rate."""
        if self.per_evaluation is None:
            return None
        return self.per_evaluation * self.priced_evaluations

    def describe(self) -> list[str]:
        """The report in plain words."""
        lines = [f"Online config {self.config_id}:"]
        for judge in self.judges:
            lines.append(
                f"  {judge.evaluator_id}: {judge.calibration_calls} calibration and "
                f"{judge.bias_calls} bias evaluation(s)"
            )
            for run in judge.missing_logs:
                lines.append(
                    f"    no session log is stored for its {run} run, so its calls are not counted"
                )
        lines.append(f"  {self.results} scored online result(s)")
        if self.error_results:
            lines.append(
                f"  {self.error_results} error result(s), not priced: whether AWS bills an "
                "evaluation it could not finish is not documented"
            )
        total = self.priced_evaluations
        if self.cost is None:
            lines.append(
                f"  {total} evaluation(s) in all; no price is shown because the Price List "
                f"lookup failed ({self.unpriced_reason or 'no rate'})"
            )
        else:
            lines.append(
                f"  {total} evaluation(s) at ${self.per_evaluation} each: ${self.cost:.4f}"
            )
        lines.append(
            "  The evaluator's own model usage (or Lambda), billed on top, is not included."
        )
        return lines


def online_run_cost(
    record: OnlineEvaluationRecord,
    session_logs: Sequence[AgentCoreSessionLog],
    *,
    results: int,
    error_results: int,
    rates: EvaluationRates | None,
    unpriced_reason: str | None = None,
) -> CostReport:
    """Count and price the evaluations ``record``'s judges and config made.

    Only the session logs of the runs each judge was attached with are counted (its record's
    ``calib_id`` and ``battery_id``), one evaluation per entry; an earlier run of the same judge
    is not part of this config's cost.

    Args:
        record: The online config.
        session_logs: The stored session logs.
        results: The scored results in the config's result group.
        error_results: The results AgentCore could not score.
        rates: The Price List rates, or ``None`` when the lookup failed.
        unpriced_reason: Why the lookup failed, when it did.
    """
    by_run = {(log.run_kind, log.run_id): log for log in session_logs}
    judges = []
    for attached in record.evaluators:
        calibration = by_run.get(("calibration", attached.calib_id))
        bias = by_run.get(("bias", attached.battery_id))
        judges.append(
            JudgeCost(
                evaluator_id=attached.evaluator_id,
                calibration_calls=len(calibration.session_ids) if calibration else 0,
                bias_calls=len(bias.session_ids) if bias else 0,
                missing_logs=tuple(
                    run
                    for run, log in (("calibration", calibration), ("bias", bias))
                    if log is None
                ),
            )
        )
    per_evaluation = rates.custom_per_evaluation if rates is not None else None
    return CostReport(
        config_id=record.config_id,
        judges=tuple(judges),
        results=results,
        error_results=error_results,
        per_evaluation=per_evaluation,
        unpriced_reason=None if per_evaluation is not None else unpriced_reason,
    )
