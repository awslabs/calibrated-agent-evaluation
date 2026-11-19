# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Bradley-Terry maximum-likelihood ratings for subjective dimensions, with Wald intervals.

The estimator of record for pairwise preferences. Sequential Elo (:func:`compute_ratings`) depends
on the order the pairs arrive in and carries no interval; the Bradley-Terry MLE (Bradley and Terry
1952) depends only on the win counts, so any ordering of the same pairs gives the same ratings.

  * Fit: Hunter's (2004) MM iteration ``pi_i <- W_i / sum_j n_ij / (pi_i + pi_j)`` on strengths
    ``pi_i = exp(theta_i)``, normalized so the log-strengths sum to zero.
  * Interval: Wald, ``theta_i +- z * SE_i``, with the covariance the Moore-Penrose inverse of the
    Fisher information ``L = sum_{pairs} n_ij p_ij (1 - p_ij) (e_i - e_j)(e_i - e_j)^T``. ``L``
    is a weighted graph Laplacian, so its pseudo-inverse is ``(L + J/k)^-1 - J/k`` (``J`` all-ones),
    which is the covariance under the sum-to-zero constraint.
  * Scale: reported on the Elo scale, ``1500 + 400 * theta / ln 10``, a linear transform of the
    log-strength (a 400-point gap is 10:1 odds), so the numbers read like Elo without being it.

The MLE exists only when the directed win graph is strongly connected (Ford 1957): a trace that
never lost (or never won) to the rest has an infinite rating. That case is refused rather than
reported. Ties are not modelled (a :class:`PreferencePair` has no tie outcome); a tie-aware
extension (Rao and Kupper 1967; Davidson 1970) is out of scope.

Stdlib only.
"""

from __future__ import annotations

import math
from collections import defaultdict
from collections.abc import Sequence

from calibrated_agent_evaluation.calibrate.record import Interval
from calibrated_agent_evaluation.subjective.errors import SubjectiveError
from calibrated_agent_evaluation.subjective.types import (
    BradleyTerryRating,
    BradleyTerryRatings,
    PreferencePair,
)

METHOD_BRADLEY_TERRY = "bradley_terry_mle_wald"
ELO_ANCHOR = 1500.0
ELO_PER_NAT = 400.0 / math.log(10.0)
_Z_95 = 1.959963984540054
_MAX_ITER = 100_000
_TOL = 1e-12


def _strongly_connected(nodes: list[str], beats: dict[str, set[str]]) -> bool:
    """Whether every node reaches every other along winner -> loser edges (two DFS passes)."""

    def reach(start: str, edges: dict[str, set[str]]) -> set[str]:
        seen = {start}
        stack = [start]
        while stack:
            for nxt in edges.get(stack.pop(), ()):
                if nxt not in seen:
                    seen.add(nxt)
                    stack.append(nxt)
        return seen

    reverse: dict[str, set[str]] = defaultdict(set)
    for w, losers in beats.items():
        for loser in losers:
            reverse[loser].add(w)
    everyone = set(nodes)
    return reach(nodes[0], beats) == everyone and reach(nodes[0], reverse) == everyone


def _invert(matrix: list[list[float]]) -> list[list[float]]:
    """Gauss-Jordan inverse with partial pivoting; raises on a singular matrix."""
    k = len(matrix)
    aug = [row[:] + [1.0 if i == j else 0.0 for j in range(k)] for i, row in enumerate(matrix)]
    for col in range(k):
        pivot = max(range(col, k), key=lambda r: abs(aug[r][col]))
        if abs(aug[pivot][col]) < 1e-14:
            raise SubjectiveError("bradley_terry: the information matrix is singular")
        aug[col], aug[pivot] = aug[pivot], aug[col]
        scale = aug[col][col]
        aug[col] = [v / scale for v in aug[col]]
        for r in range(k):
            if r != col and aug[r][col] != 0.0:
                factor = aug[r][col]
                aug[r] = [a - factor * b for a, b in zip(aug[r], aug[col], strict=True)]
    return [row[k:] for row in aug]


def bradley_terry(dimension: str, pairs: Sequence[PreferencePair]) -> BradleyTerryRatings:
    """Fit Bradley-Terry ratings with 95% Wald intervals for one subjective dimension.

    Returns ratings sorted best first, each an :class:`Interval` on the Elo scale with
    ``method="bradley_terry_mle_wald"`` and ``n`` = the comparisons that trace took part in.
    Order-invariant: any permutation of ``pairs`` gives the same result.

    Raises :class:`SubjectiveError` on no pairs, a pair for another dimension, a self-comparison,
    fewer than two traces, or a win graph that is not strongly connected (some trace never lost,
    or never won, against the rest, so its MLE is infinite).
    """
    if not pairs:
        raise SubjectiveError(f"bradley_terry({dimension!r}): no preference pairs to fit")
    wins: dict[str, float] = defaultdict(float)
    games: dict[tuple[str, str], int] = defaultdict(int)
    beats: dict[str, set[str]] = defaultdict(set)
    n_comparisons: dict[str, int] = defaultdict(int)
    for pair in pairs:
        if pair.dimension != dimension:
            raise SubjectiveError(
                f"bradley_terry({dimension!r}) received a pair for dimension {pair.dimension!r}"
            )
        a, b = pair.trace_id_a, pair.trace_id_b
        if a == b:
            raise SubjectiveError(f"bradley_terry: self-comparison in pair (trace_id={a!r})")
        if pair.winner not in ("a", "b"):
            raise SubjectiveError(f"winner must be 'a' or 'b'; got {pair.winner!r}")
        winner, loser = (a, b) if pair.winner == "a" else (b, a)
        wins[winner] += 1.0
        wins.setdefault(loser, 0.0)
        games[(min(a, b), max(a, b))] += 1
        beats[winner].add(loser)
        n_comparisons[a] += 1
        n_comparisons[b] += 1

    nodes = sorted(n_comparisons)
    if len(nodes) < 2:
        raise SubjectiveError("bradley_terry: need at least two traces")
    if not _strongly_connected(nodes, beats):
        raise SubjectiveError(
            f"bradley_terry({dimension!r}): the win graph is not strongly connected (some trace "
            "never lost, or never won, against the rest), so its maximum-likelihood rating is "
            "infinite; collect more comparisons rather than report an unbounded number"
        )

    index = {t: i for i, t in enumerate(nodes)}
    k = len(nodes)
    neighbours: dict[int, list[tuple[int, int]]] = defaultdict(list)
    for (a, b), n_ab in games.items():
        neighbours[index[a]].append((index[b], n_ab))
        neighbours[index[b]].append((index[a], n_ab))
    w = [wins[t] for t in nodes]

    strength = [1.0] * k
    for _ in range(_MAX_ITER):
        updated = [
            w[i] / sum(n_ij / (strength[i] + strength[j]) for j, n_ij in neighbours[i])
            for i in range(k)
        ]
        geo = math.exp(sum(math.log(s) for s in updated) / k)
        updated = [s / geo for s in updated]
        delta = max(abs(math.log(u) - math.log(s)) for u, s in zip(updated, strength, strict=True))
        strength = updated
        if delta < _TOL:
            break
    else:
        raise SubjectiveError("bradley_terry: the MM iteration did not converge")
    theta = [math.log(s) for s in strength]

    info = [[0.0] * k for _ in range(k)]
    for (a, b), n_ab in games.items():
        i, j = index[a], index[b]
        p = 1.0 / (1.0 + math.exp(theta[j] - theta[i]))
        weight = n_ab * p * (1.0 - p)
        info[i][i] += weight
        info[j][j] += weight
        info[i][j] -= weight
        info[j][i] -= weight
    shifted = [[info[i][j] + 1.0 / k for j in range(k)] for i in range(k)]
    inverse = _invert(shifted)
    variance = [max(inverse[i][i] - 1.0 / k, 0.0) for i in range(k)]

    ratings = []
    for t in nodes:
        i = index[t]
        half = _Z_95 * math.sqrt(variance[i])
        ratings.append(
            BradleyTerryRating(
                trace_id=t,
                rating=Interval(
                    point=ELO_ANCHOR + ELO_PER_NAT * theta[i],
                    lo=ELO_ANCHOR + ELO_PER_NAT * (theta[i] - half),
                    hi=ELO_ANCHOR + ELO_PER_NAT * (theta[i] + half),
                    method=METHOD_BRADLEY_TERRY,
                    level=0.95,
                    sided="two-sided",
                    n=n_comparisons[t],
                ),
                n_comparisons=n_comparisons[t],
            )
        )
    ratings.sort(key=lambda r: (-(r.rating.point or 0.0), r.trace_id))
    return BradleyTerryRatings(
        dimension=dimension, ratings=tuple(ratings), n_total_comparisons=len(pairs)
    )
