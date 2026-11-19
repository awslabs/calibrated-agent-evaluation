# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""The pass@k estimator: the probability that at least one of k sampled solutions passes.

This is the HumanEval metric (Chen et al. 2021), ``pass@k = 1 - C(n-c, k) / C(n, k)`` for a task
with ``n`` samples of which ``c`` passed, averaged over tasks. It is a DIFFERENT quantity from the
tau-bench ``pass^k`` in :mod:`calibrated_agent_evaluation.probe.robustness` (``C(c, k) / C(n, k)``,
"all k hold"); the two share a combinatorial shape and a symbol collision but measure opposite
things, so they are kept as separate, cross-referenced functions.

The caller supplies the per-sample PASS/FAIL booleans it already has (from scoring k samples);
pass@k does not run anything itself.

The point estimate is paired with a two-sided 95% **bootstrap percentile** confidence interval
(the project requires every headline rate to carry its confidence treatment). The interval is a
nested resample -- tasks with replacement, then each task's samples with replacement -- so it
captures both across-task and within-task (finite-sample) variability and is non-degenerate even
for a single task. It is deterministic given ``seed`` so tests and docs are reproducible, and uses
only the standard library (no new dependency, and nothing from the statsmodels-backed
``stats.proportions``).
"""

from __future__ import annotations

import math
import random
from collections.abc import Sequence
from dataclasses import dataclass

from calibrated_agent_evaluation.calibrate.record import Interval
from calibrated_agent_evaluation.scorers.errors import ScorerError

_LEVEL = 0.95
_DEFAULT_RESAMPLES = 2000
METHOD_BOOTSTRAP = "bootstrap-percentile"


@dataclass(frozen=True)
class PassAtK:
    """A pass@k point estimate with its bootstrap interval.

    ``interval`` carries the two-sided 95% bootstrap percentile bound (an :class:`Interval`, so it
    reads the same as every other headline rate). ``n_tasks`` is the number of tasks averaged and
    ``k`` the draw size.
    """

    point: float
    interval: Interval
    k: int
    n_tasks: int


def _pass_at_k_one(c: int, n: int, k: int) -> float:
    """``pass@k`` for a single task: ``1 - C(n-c, k) / C(n, k)``.

    When fewer than ``k`` samples failed (``n - c < k``) every size-``k`` draw must contain a pass,
    so ``C(n-c, k) = 0`` and the value is 1. ``math.comb`` returns 0 for that case, so the formula
    holds without a special branch.
    """
    return 1.0 - math.comb(n - c, k) / math.comb(n, k)


def pass_at_k(
    samples: Sequence[Sequence[bool]],
    k: int,
    *,
    resamples: int = _DEFAULT_RESAMPLES,
    seed: int = 0,
) -> PassAtK:
    """The unbiased pass@k over ``samples`` (one inner sequence per task) with a bootstrap interval.

    Each inner sequence is a task's per-sample PASS (``True``) / FAIL (``False``) results. The point
    is the mean over tasks of ``1 - C(n-c, k) / C(n, k)``; the interval is the two-sided 95%
    bootstrap percentile bound over a nested resample (tasks, then samples within task).

    Raises:
        ScorerError: there are no tasks, a task is empty, or ``k`` is outside ``1..n`` for any task
            (``k`` samples cannot be drawn from fewer than ``k``).
    """
    if not samples:
        raise ScorerError("pass_at_k: need at least one task")
    tasks: list[list[bool]] = []
    for i, task in enumerate(samples):
        n = len(task)
        if not 1 <= k <= n:
            raise ScorerError(
                f"pass_at_k: k={k} must lie in [1, n] for task {i}, which has n={n} samples"
            )
        tasks.append(list(task))

    point = sum(_pass_at_k_one(sum(t), len(t), k) for t in tasks) / len(tasks)
    interval = _bootstrap_interval(tasks, k, resamples=resamples, seed=seed)
    return PassAtK(point=point, interval=interval, k=k, n_tasks=len(tasks))


def _bootstrap_interval(tasks: list[list[bool]], k: int, *, resamples: int, seed: int) -> Interval:
    """Two-sided 95% bootstrap percentile interval for the mean pass@k.

    Nested resample: draw ``n_tasks`` tasks with replacement, and within each draw its ``n`` samples
    with replacement, recompute the mean pass@k, and take the 2.5/97.5 percentiles over the
    ``resamples`` replicates. Deterministic given ``seed``.
    """
    # Non-crypto PRNG is correct here: this is statistical resampling that must be reproducible.
    rng = random.Random(seed)  # noqa: S311 - statistical bootstrap, not security
    n_tasks = len(tasks)
    total_samples = sum(len(t) for t in tasks)
    means: list[float] = []
    for _ in range(resamples):
        acc = 0.0
        for _ in range(n_tasks):
            task = tasks[rng.randrange(n_tasks)]
            n = len(task)
            c = sum(1 for _ in range(n) if task[rng.randrange(n)])
            acc += _pass_at_k_one(c, n, k)
        means.append(acc / n_tasks)
    means.sort()
    tail = (1.0 - _LEVEL) / 2.0
    lo = means[int(tail * (resamples - 1))]
    hi = means[int((1.0 - tail) * (resamples - 1))]
    return Interval(
        point=sum(_pass_at_k_one(sum(t), len(t), k) for t in tasks) / n_tasks,
        lo=lo,
        hi=hi,
        method=METHOD_BOOTSTRAP,
        level=_LEVEL,
        sided="two-sided",
        n=total_samples,
    )
