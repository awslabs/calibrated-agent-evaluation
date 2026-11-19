# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Monte Carlo coverage study for ``rg_interval``, the corrected-rate interval.

The grid and the acceptance criterion were fixed before the study ran, so none of the parameters
below are chosen here:

* ``(p_hat, n_class)`` is a paired axis: ``n_class = plan_calibration_size(p_hat, 0.80)`` for
  ``p_hat`` in 0.85, 0.88, 0.90, 0.92, 0.95, 0.98. The judge is simulated at ``TPR = TNR = p_hat``.
  The pairs are read from the shipped planner, so a later planner change moves the grid with it.
* prevalence in 0.02, 0.05, 0.06, 0.10, 0.20, 0.35, 0.50; eval size in 100, 500, 1000, 5000.
* 2000 replicates per cell, ``n_resamples=9999`` (the shipped default).

Per replicate: ``cal_pos ~ Bern(TPR)`` and ``cal_neg ~ Bern(1 - TNR)``, each of size ``n_class``;
``eval_preds ~ Bern(pi * TPR + (1 - pi) * (1 - TNR))`` of size ``n_eval``. Each arm's interval is
asked whether it contains ``pi``; a miss is recorded by direction (interval above or below).

The built-in arms are ``rg_interval`` (the shipped interval, the one the acceptance criterion
judges) and ``naive_wilson`` (a two-sided Wilson interval on the raw judge rate, no correction).
A caller can add arms of its own, such as a comparison package, through ``extra_arms``.

Acceptance: a cell passes when the Wilson 95% interval on its coverage count contains 0.95. At 2000
replicates that is roughly 0.940 to 0.959, a tolerance set by the replicate count rather than
picked (:func:`acceptance_band`).

Every replicate's data and bootstrap seed derive from the base seed and the cell's own parameters
through ``numpy.random.SeedSequence``, so a cell's numbers do not depend on worker count,
scheduling, or which other cells are in the grid.
"""

from __future__ import annotations

import math
import time
from collections.abc import Callable, Mapping, Sequence
from typing import Any

import numpy as np
from pydantic import BaseModel, ConfigDict

from calibrated_agent_evaluation.calibrate.errors import (
    NonIdentifiableRateError,
    StatisticsInputError,
)
from calibrated_agent_evaluation.calibrate.plan import plan_calibration_size
from calibrated_agent_evaluation.stats.proportions import wilson_lower, wilson_upper
from calibrated_agent_evaluation.stats.rogan_gladen import rg_interval

__all__ = [
    "ACCEPTANCE_ARM",
    "ARMS",
    "BASE_SEED",
    "EVAL_SIZES",
    "NOMINAL",
    "N_RESAMPLES",
    "PREVALENCES",
    "P_HATS",
    "REPLICATES",
    "TAU",
    "Cell",
    "IntervalArm",
    "acceptance_band",
    "asymmetric_grid",
    "grid",
    "simulate_cell",
    "summarize_arm",
]

TAU = 0.80
"""The gate threshold the planner sizes calibration sets for."""
P_HATS: tuple[float, ...] = (0.85, 0.88, 0.90, 0.92, 0.95, 0.98)
PREVALENCES: tuple[float, ...] = (0.02, 0.05, 0.06, 0.10, 0.20, 0.35, 0.50)
EVAL_SIZES: tuple[int, ...] = (100, 500, 1000, 5000)
REPLICATES = 2000
N_RESAMPLES = 9999
NOMINAL = 0.95
BASE_SEED = 20260926

ARMS: tuple[str, ...] = ("rg_interval", "naive_wilson")
"""The built-in arms; :func:`simulate_cell` takes more through ``extra_arms``."""
ACCEPTANCE_ARM = "rg_interval"

IntervalArm = Callable[[np.ndarray, np.ndarray, np.ndarray, int, int], "tuple[float, float] | None"]
"""An arm: ``(cal_pos, cal_neg, eval_preds, n_resamples, seed)`` to an interval, or ``None`` for a
refusal (counted apart from misses)."""


def _rate(x: float) -> str:
    """A rate for a cell key: two decimals, or three where two would merge it with a neighbor."""
    two = f"{x:.2f}"
    return two if abs(float(two) - x) < 1e-9 else f"{x:.3f}"


class Cell(BaseModel):
    """One grid cell: the judge, its calibration size, the true prevalence, and the eval size.

    The judge has ``TPR = p_hat`` and ``TNR = tnr``; ``tnr`` None is the symmetric judge,
    ``TNR = p_hat``, and such a cell keeps the key, seeds and JSON it had before ``tnr`` existed.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    p_hat: float
    n_class: int
    prevalence: float
    n_eval: int
    tnr: float | None = None

    @property
    def judge_tnr(self) -> float:
        """The simulated judge's TNR."""
        return self.p_hat if self.tnr is None else self.tnr

    @property
    def key(self) -> str:
        """The cell's stable identifier in checkpoints and maps, e.g. ``p0.90-n35-pi0.06-e100``."""
        judge = f"p{_rate(self.p_hat)}" + ("" if self.tnr is None else f"-tnr{_rate(self.tnr)}")
        return f"{judge}-n{self.n_class}-pi{self.prevalence:.2f}-e{self.n_eval}"

    def seed_key(self) -> tuple[int, ...]:
        """The ``SeedSequence`` spawn key, from the cell's own parameters so it survives edits."""
        base = (
            round(self.p_hat * 1000),
            self.n_class,
            round(self.prevalence * 1000),
            self.n_eval,
        )
        return base if self.tnr is None else (*base, round(self.tnr * 1000))

    def as_dict(self) -> dict[str, Any]:
        """The cell as JSON-ready fields; a symmetric cell carries no ``tnr``."""
        return self.model_dump(exclude={"tnr"} if self.tnr is None else set())


class _ArmTally(BaseModel):
    """Running counts for one arm in one cell."""

    covered: int = 0
    miss_above: int = 0
    miss_below: int = 0
    refused: int = 0
    width_sum: float = 0.0

    def record(self, interval: tuple[float, float] | None, truth: float) -> None:
        if interval is None:
            self.refused += 1
            return
        lo, hi = interval
        self.width_sum += hi - lo
        if lo > truth:
            self.miss_above += 1
        elif hi < truth:
            self.miss_below += 1
        else:
            self.covered += 1


def grid(n_class_override: Sequence[int] | None = None) -> list[Cell]:
    """Return the 168 decided cells, with ``n_class`` read from the shipped planner.

    ``n_class_override`` replaces the planner pairing with a free ``p_hat x n_class`` cross. It is
    not the decided study; it exists for deriving the planner's coverage floor, which needs
    calibration sizes the planner does not return.
    """
    if n_class_override:
        return [
            Cell(p_hat=p, n_class=n, prevalence=pi, n_eval=n_eval)
            for p in P_HATS
            for n in n_class_override
            for pi in PREVALENCES
            for n_eval in EVAL_SIZES
        ]
    return [
        Cell(p_hat=p, n_class=plan_calibration_size(p, TAU), prevalence=pi, n_eval=n_eval)
        for p in P_HATS
        for pi in PREVALENCES
        for n_eval in EVAL_SIZES
    ]


def asymmetric_grid() -> list[Cell]:
    """Return the asymmetric judges' cells (840 of them).

    Every ordered ``(TPR, TNR)`` pair of distinct values from the decided ``p_hat`` axis, at the
    planner's size for that pair, crossed with the decided prevalences and eval sizes.

    Both orders are needed: swapping TPR and TNR is the same as reading prevalence ``pi`` as
    ``1 - pi``, and the decided prevalences stop at 0.50.
    """
    return [
        Cell(
            p_hat=tpr,
            n_class=plan_calibration_size(tpr, TAU, tnr_hat=tnr),
            prevalence=pi,
            n_eval=n_eval,
            tnr=tnr,
        )
        for tpr in P_HATS
        for tnr in P_HATS
        if tpr != tnr
        for pi in PREVALENCES
        for n_eval in EVAL_SIZES
    ]


def _rg(
    cal_pos: np.ndarray, cal_neg: np.ndarray, eval_preds: np.ndarray, n_resamples: int, seed: int
) -> tuple[float, float] | None:
    try:
        got = rg_interval(
            cal_pos.tolist(),
            cal_neg.tolist(),
            eval_preds.tolist(),
            conf=NOMINAL,
            n_resamples=n_resamples,
            seed=seed,
        )
    except NonIdentifiableRateError:
        # rg_interval refuses when too many draws have no corrected rate: a refusal, not a miss.
        return None
    return got.lo, got.hi


def _wilson(k: int, n: int) -> tuple[float, float]:
    """Two-sided Wilson interval at the nominal level (``alpha`` is the two-sided level)."""
    alpha = 1 - NOMINAL
    return wilson_lower(k, n, alpha=alpha), wilson_upper(k, n, alpha=alpha)


def simulate_cell(
    cell: Cell,
    *,
    replicates: int = REPLICATES,
    n_resamples: int = N_RESAMPLES,
    base_seed: int = BASE_SEED,
    arms: Sequence[str] = ARMS,
    extra_arms: Mapping[str, IntervalArm] | None = None,
) -> dict[str, Any]:
    """Run one cell and return its tallies as a JSON-ready dict.

    Replicate ``r`` of a cell always sees the same data and bootstrap seeds for a given
    ``base_seed``, so running the first ``k`` replicates reproduces the first ``k`` of a full run.
    ``rg_interval`` draws its bootstrap from the replicate's first seed and every extra arm from
    its second, so adding an arm never changes another arm's numbers.

    Raises:
        StatisticsInputError: ``arms`` names an arm that is neither built in nor in ``extra_arms``.
    """
    extra = dict(extra_arms or {})
    unknown = set(arms) - set(ARMS) - set(extra)
    if unknown:
        raise StatisticsInputError(f"unknown arms: {sorted(unknown)}")
    tpr = cell.p_hat
    fpr = 1.0 - cell.judge_tnr
    measured_p = cell.prevalence * tpr + (1.0 - cell.prevalence) * fpr
    tallies = {arm: _ArmTally() for arm in arms}
    started = time.perf_counter()
    for rep in range(replicates):
        ss = np.random.SeedSequence(base_seed, spawn_key=(*cell.seed_key(), rep))
        data_rng = np.random.default_rng(ss)
        rg_seed, extra_seed = (int(x) for x in ss.generate_state(2))
        cal_pos = data_rng.binomial(1, tpr, cell.n_class)
        cal_neg = data_rng.binomial(1, fpr, cell.n_class)
        eval_preds = data_rng.binomial(1, measured_p, cell.n_eval)
        for arm, tally in tallies.items():
            if arm == "rg_interval":
                interval = _rg(cal_pos, cal_neg, eval_preds, n_resamples, rg_seed)
            elif arm == "naive_wilson":
                interval = _wilson(int(eval_preds.sum()), len(eval_preds))
            else:
                interval = extra[arm](cal_pos, cal_neg, eval_preds, n_resamples, extra_seed)
            tally.record(interval, cell.prevalence)
    return {
        "cell": cell.as_dict(),
        "key": cell.key,
        "replicates": replicates,
        "n_resamples": n_resamples,
        "base_seed": base_seed,
        "seconds": round(time.perf_counter() - started, 2),
        "arms": {arm: t.model_dump() for arm, t in tallies.items()},
    }


def summarize_arm(tally: Mapping[str, float], replicates: int) -> dict[str, Any]:
    """Coverage, its Monte Carlo error and Wilson CI, the verdict, miss split, and mean width."""
    answered = replicates - int(tally["refused"])
    covered = int(tally["covered"])
    out: dict[str, Any] = {
        "answered": answered,
        "refused": int(tally["refused"]),
        "covered": covered,
        "miss_above": int(tally["miss_above"]),
        "miss_below": int(tally["miss_below"]),
    }
    if answered == 0:
        out.update(coverage=None, mc_se=None, coverage_ci=None, verdict="no-intervals")
        out["mean_width"] = None
        return out
    cov = covered / answered
    lo, hi = _wilson(covered, answered)
    if lo > NOMINAL:
        verdict = "over"
    elif hi < NOMINAL:
        verdict = "under"
    else:
        verdict = "nominal"
    out.update(
        coverage=round(cov, 5),
        mc_se=round(math.sqrt(cov * (1 - cov) / answered), 5),
        coverage_ci=[round(float(lo), 5), round(float(hi), 5)],
        verdict=verdict,
        mean_width=round(float(tally["width_sum"]) / answered, 5),
    )
    return out


def acceptance_band(replicates: int) -> tuple[float, float]:
    """The coverage range a cell can observe and still have its Wilson CI contain the nominal.

    Derived from the replicate count, so the tolerance is not a picked constant.
    """
    lo_k = next(k for k in range(replicates + 1) if _wilson(k, replicates)[1] >= NOMINAL)
    hi_k = max(k for k in range(replicates + 1) if _wilson(k, replicates)[0] <= NOMINAL)
    return lo_k / replicates, hi_k / replicates
