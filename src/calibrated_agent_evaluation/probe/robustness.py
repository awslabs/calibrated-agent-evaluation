# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Repeated-trial robustness aggregation.

Two directions, deliberately opposite:
  * **Capability** (:func:`capability_robustness`) -- the tau-bench ``pass^k`` estimator (Yao et al.
    2024, arXiv:2406.12045): the probability that ``k`` trials drawn without replacement from the
    ``n`` run all held, estimated without bias by ``C(c, k) / C(n, k)`` with ``c`` the trials that
    held; :func:`pass_hat_k` averages it over tasks. The strict "every one of the ``n`` trials
    held" flag is :func:`all_trials_held`, which is ``pass^n == 1``.
  * **Security** (:func:`security_result`) -- vulnerable-if-any: the target is flagged vulnerable
    if ANY of the ``k`` trials elicited the unsafe behavior (worst-case; averaging would hide a
    real exposure).

Both carry the per-trial rate (held rate for capability, break rate for security) with its exact
two-sided 95% Clopper-Pearson interval, so "0 of 3 broke" reads as a break rate bounded by 0.71,
not as a clean bill. The interval is computed with the stdlib so probing stays dependency-free: the
binomial tails are summed in log space (``math.lgamma``) and inverted by bisection, which stays
finite at any ``n`` (``math.comb(n, i) * p**i`` overflowed a float from about ``n = 1030``).
"""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Literal

from calibrated_agent_evaluation.calibrate.record import Interval
from calibrated_agent_evaluation.probe.errors import ProbeError

METHOD_CLOPPER_PEARSON = "clopper_pearson"
_LEVEL = 0.95


def _binom_sf(x: int, n: int, p: float) -> float:
    """``P(X >= x)`` for ``X ~ Binomial(n, p)``, for ``0 < p < 1``, summed in log space."""
    if x <= 0:
        return 1.0
    log_p, log_q = math.log(p), math.log1p(-p)
    head = math.lgamma(n + 1)
    terms = [
        head - math.lgamma(i + 1) - math.lgamma(n - i + 1) + i * log_p + (n - i) * log_q
        for i in range(x, n + 1)
    ]
    top = max(terms)
    return min(1.0, math.exp(top) * math.fsum(math.exp(t - top) for t in terms))


def _bisect(f: Callable[[float], float], target: float, *, increasing: bool) -> float:
    """Solve ``f(p) = target`` on [0, 1] for a monotone ``f`` by bisection (80 halvings)."""
    lo, hi = 0.0, 1.0
    for _ in range(80):
        mid = (lo + hi) / 2.0
        value = f(mid)
        if (value < target) == increasing:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2.0


def clopper_pearson(x: int, n: int, *, level: float = _LEVEL) -> Interval:
    """The exact two-sided Clopper-Pearson interval for ``x`` successes in ``n`` trials.

    The lower edge solves ``P(X >= x | p) = (1 - level)/2`` and the upper edge
    ``P(X <= x | p) = (1 - level)/2`` (0 and 1 at ``x = 0`` and ``x = n``). Raises
    :class:`ProbeError` on ``n < 1``, ``x`` outside ``[0, n]`` or ``level`` outside ``(0, 1)``.
    """
    if n < 1 or not 0 <= x <= n:
        raise ProbeError(f"clopper_pearson: need 0 <= x <= n and n >= 1, got x={x}, n={n}")
    if not 0.0 < level < 1.0:
        raise ProbeError(f"clopper_pearson: level must lie in (0, 1), got {level}")
    tail = (1.0 - level) / 2.0
    lo = 0.0 if x == 0 else _bisect(lambda p: _binom_sf(x, n, p), tail, increasing=True)
    hi = 1.0 if x == n else _bisect(lambda p: 1.0 - _binom_sf(x + 1, n, p), tail, increasing=False)
    return Interval(
        point=x / n,
        lo=max(0.0, lo),
        hi=min(1.0, hi),
        method=METHOD_CLOPPER_PEARSON,
        level=level,
        sided="two-sided",
        n=n,
    )


@dataclass(frozen=True)
class RobustnessResult:
    """The aggregated result over ``n`` repeated trials.

    ``rate`` is the per-trial rate with its Clopper-Pearson interval: the held rate for
    ``kind="capability"``, the break rate for ``kind="security"``. For capability, ``k`` is the
    ``pass^k`` order, ``pass_hat_k`` the tau-bench estimate, and ``passed`` whether all ``n``
    trials held. For security, ``k == n`` and ``passed`` is NOT vulnerable.
    """

    kind: Literal["capability", "security"]
    k: int
    passed: bool
    vulnerable_if_any: bool  # True when the security aggregation was used
    n: int
    rate: Interval
    pass_hat_k: float | None = None


def all_trials_held(trial_held: Sequence[bool]) -> bool:
    """Whether every trial held -- the strict flag, equal to ``pass^n == 1``."""
    if len(trial_held) == 0:
        raise ProbeError("all_trials_held: need at least one trial")
    return all(trial_held)


def _pass_hat_k_one(c: int, n: int, k: int) -> float:
    return math.comb(c, k) / math.comb(n, k)


def pass_hat_k(tasks: Sequence[Sequence[bool]], k: int) -> float:
    """The tau-bench ``pass^k`` over tasks: ``mean_i C(c_i, k) / C(n_i, k)``.

    Each task contributes its own trial count ``n_i``; ``k`` must satisfy ``1 <= k <= n_i`` for
    every task. Raises :class:`ProbeError` on no tasks, an empty task, or an out-of-range ``k``.
    """
    if not tasks:
        raise ProbeError("pass_hat_k: need at least one task")
    total = 0.0
    for i, trials in enumerate(tasks):
        n = len(trials)
        if not 1 <= k <= n:
            raise ProbeError(f"pass_hat_k: k={k} must lie in [1, n] for task {i}, which has n={n}")
        total += _pass_hat_k_one(sum(1 for t in trials if t), n, k)
    return total / len(tasks)


def capability_robustness(trial_held: Sequence[bool], *, k: int | None = None) -> RobustnessResult:
    """``pass^k`` for one capability over its ``n`` trials (tau-bench, arXiv:2406.12045).

    ``k`` defaults to ``n``, where ``pass^n`` is 1 exactly when every trial held. A smaller ``k``
    asks how likely ``k`` fresh trials are to all hold. Raises :class:`ProbeError` on no trials or
    ``k`` outside ``[1, n]``.
    """
    n = len(trial_held)
    if n == 0:
        raise ProbeError("capability_robustness: need at least one trial")
    order = n if k is None else k
    if not 1 <= order <= n:
        raise ProbeError(f"capability_robustness: k={order} must lie in [1, n={n}]")
    held = sum(1 for t in trial_held if t)
    return RobustnessResult(
        kind="capability",
        k=order,
        passed=held == n,
        vulnerable_if_any=False,
        n=n,
        rate=clopper_pearson(held, n),
        pass_hat_k=_pass_hat_k_one(held, n, order),
    )


def security_result(trial_broke: Sequence[bool]) -> RobustnessResult:
    """Vulnerable-if-any -- flag the target vulnerable if ANY of the ``k`` trials broke."""
    k = len(trial_broke)
    if k == 0:
        raise ProbeError("security_result: need at least one trial")
    broke = sum(1 for t in trial_broke if t)
    return RobustnessResult(
        kind="security",
        k=k,
        passed=broke == 0,
        vulnerable_if_any=True,
        n=k,
        rate=clopper_pearson(broke, k),
    )
