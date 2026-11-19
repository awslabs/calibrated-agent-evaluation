# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Three ways to read the same rate sample, and how many gate decisions change between them.

For each failure mode, over one judge's calibration counts and one rate sample:

1. ``naive_point``: the raw FAIL fraction, compared to the ceiling as a bare number.
2. ``naive_wilson``: the raw FAIL fraction's two-sided 95% Wilson upper edge against the ceiling.
3. ``calibrated``: certify the judge on its calibration counts, then the Rogan-Gladen corrected
   rate's interval upper edge against the ceiling. An uncertified judge or a refused correction
   is a refusal, never a pass.

A mode passes an arm when its value is at or under ``max_fail_rate``. Everything is computed by
the shipped estimators (``certify``, ``corrected_rate``, ``raw_rate_interval``).
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict

from calibrated_agent_evaluation.calibrate import InMemoryLockedTestLedger
from calibrated_agent_evaluation.calibrate.engine import certify, corrected_rate
from calibrated_agent_evaluation.calibrate.record import CalibrationSet, LabelPair
from calibrated_agent_evaluation.run.trend import raw_rate_interval
from calibrated_agent_evaluation.schema.types import Label

__all__ = ["ARMS", "evaluate_mode", "flips"]

ARMS = ("naive_point", "naive_wilson", "calibrated")


class _Judge(BaseModel):
    model_config = ConfigDict(frozen=True)

    judge_id: str
    model: str
    prompt_version: str


def _pairs(n_pos: int, tp: int, n_neg: int, tn: int) -> tuple[LabelPair, ...]:
    """Locked-test pairs realising exactly the given confusion counts (FAIL is positive)."""
    out = []
    for i in range(n_pos):
        out.append(
            LabelPair(
                gold=Label(
                    trace_id=f"p{i}", span_id=None, kind="human", outcome="FAIL", critique="gold"
                ),
                verdict=Label(
                    trace_id=f"p{i}",
                    span_id=None,
                    kind="llm-judge",
                    outcome="FAIL" if i < tp else "PASS",
                    critique="v",
                ),
            )
        )
    for i in range(n_neg):
        out.append(
            LabelPair(
                gold=Label(
                    trace_id=f"n{i}", span_id=None, kind="human", outcome="PASS", critique="gold"
                ),
                verdict=Label(
                    trace_id=f"n{i}",
                    span_id=None,
                    kind="llm-judge",
                    outcome="PASS" if i < tn else "FAIL",
                    critique="v",
                ),
            )
        )
    return tuple(out)


def evaluate_mode(
    mode: dict[str, Any], *, max_fail_rate: float, seed: int, n_resamples: int
) -> dict[str, Any]:
    """One failure mode's row: the raw rate, each arm's decision, and the corrected interval.

    Raises:
        ValueError: The rate sample is empty, or the corrected rate came back without an interval.
    """
    name = mode["name"]
    cal = mode["calibration"]
    sample = mode["rate_sample"]
    verdicts = [1] * int(sample["fail"]) + [0] * (int(sample["n"]) - int(sample["fail"]))
    raw = raw_rate_interval(verdicts)
    if raw.point is None:
        raise ValueError(f"mode {name!r}: the rate sample is empty")

    judge = _Judge(judge_id=name, model=mode.get("model", "unspecified"), prompt_version="v")
    locked = CalibrationSet(
        pairs=_pairs(cal["n_pos"], cal["tp"], cal["n_neg"], cal["tn"]),
        split="locked_test",
        set_id=f"locked-{name}",
    )
    record = certify(judge, locked, ledger=InMemoryLockedTestLedger())
    row: dict[str, Any] = {
        "mode": name,
        "raw_point": raw.point,
        "raw_lo": raw.lo,
        "raw_hi": raw.hi,
        "naive_point": "pass" if raw.point <= max_fail_rate else "fail",
        "naive_wilson": "pass" if raw.hi <= max_fail_rate else "fail",
    }
    if record.result != "CERTIFIED":
        row.update(corrected_point=None, corrected_lo=None, corrected_hi=None)
        row["calibrated"] = "refused: judge not certified"
        return row
    scored = corrected_rate(
        record,
        verdicts,
        rate_set_id=f"rate-{name}",
        rate_trace_ids=[f"rate-{name}-{i}" for i in range(len(verdicts))],
        seed=seed,
        n_resamples=n_resamples,
    )
    iv = scored.corrected_rate
    if iv is None:
        raise ValueError(f"mode {name!r}: corrected_rate attached no interval")
    row.update(corrected_point=iv.point, corrected_lo=iv.lo, corrected_hi=iv.hi)
    if iv.point is None:
        row["calibrated"] = f"refused: {iv.refusal_reason}"
    else:
        row["calibrated"] = "pass" if iv.hi <= max_fail_rate else "fail"
    return row


def _passed(decision: str) -> bool:
    return decision == "pass"


def flips(rows: list[dict[str, Any]]) -> dict[str, int]:
    """Modes whose pass/not-pass decision differs between each pair of arms."""
    out = {}
    for i, a in enumerate(ARMS):
        for b in ARMS[i + 1 :]:
            out[f"{a}_vs_{b}"] = sum(_passed(r[a]) != _passed(r[b]) for r in rows)
    return out
