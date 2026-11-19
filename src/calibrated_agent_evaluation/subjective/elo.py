# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Sequential Elo ratings for subjective dimensions, kept as a display transform.

Order-dependent: the same pairs in a different order give different ratings, and there is no
interval. Use :func:`calibrated_agent_evaluation.subjective.bradley_terry` for the ratings of
record.
"""

from __future__ import annotations

from calibrated_agent_evaluation.subjective.errors import SubjectiveError
from calibrated_agent_evaluation.subjective.types import EloRating, EloRatings, PreferencePair


def compute_ratings(
    dimension: str,
    pairs: list[PreferencePair],
    *,
    k_factor: float = 32.0,
    initial_rating: float = 1500.0,
) -> EloRatings:
    """Compute Elo ratings from preference pairs.

    Returns EloRatings with treatment="tracked" (never "gated"), carrying ``k_factor`` and
    ``initial_rating``. Returns empty ratings if pairs is empty. The result depends on the order
    of ``pairs``; :func:`calibrated_agent_evaluation.subjective.bradley_terry` does not.
    """
    if k_factor <= 0:
        raise SubjectiveError(f"k_factor must be positive; got {k_factor}")

    if not pairs:
        return EloRatings(
            dimension=dimension,
            ratings=(),
            n_total_comparisons=0,
            treatment="tracked",
            k_factor=k_factor,
            initial_rating=initial_rating,
        )

    ratings: dict[str, float] = {}
    comparisons: dict[str, int] = {}

    for pair in pairs:
        if pair.dimension != dimension:
            raise SubjectiveError(
                f"compute_ratings({dimension!r}) received a pair for dimension "
                f"{pair.dimension!r}; all pairs must match the requested dimension"
            )

        if pair.winner not in ("a", "b"):
            raise SubjectiveError(f"winner must be 'a' or 'b'; got {pair.winner!r}")

        if pair.trace_id_a == pair.trace_id_b:
            raise SubjectiveError(
                f"Self-comparison in pair (trace_id={pair.trace_id_a!r}); "
                f"cannot compute Elo for a trace against itself"
            )

        for tid in (pair.trace_id_a, pair.trace_id_b):
            if tid not in ratings:
                ratings[tid] = initial_rating
                comparisons[tid] = 0

        ra = ratings[pair.trace_id_a]
        rb = ratings[pair.trace_id_b]

        ea = 1.0 / (1.0 + 10.0 ** ((rb - ra) / 400.0))
        eb = 1.0 - ea

        if pair.winner == "a":
            sa, sb = 1.0, 0.0
        else:
            sa, sb = 0.0, 1.0

        ratings[pair.trace_id_a] = ra + k_factor * (sa - ea)
        ratings[pair.trace_id_b] = rb + k_factor * (sb - eb)
        comparisons[pair.trace_id_a] += 1
        comparisons[pair.trace_id_b] += 1

    sorted_ratings = sorted(ratings.items(), key=lambda x: x[1], reverse=True)
    elo_ratings = tuple(
        EloRating(trace_id=tid, rating=score, n_comparisons=comparisons[tid])
        for tid, score in sorted_ratings
    )

    return EloRatings(
        dimension=dimension,
        ratings=elo_ratings,
        n_total_comparisons=len(pairs),
        treatment="tracked",
        k_factor=k_factor,
        initial_rating=initial_rating,
    )
