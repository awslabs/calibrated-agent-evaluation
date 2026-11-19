# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Data types for the subjective-outcome module."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from calibrated_agent_evaluation.calibrate.record import Interval

# Who judged a preference pair; a policy engine's labels never judge one, so its kind is absent.
PairJudgeKind = Literal["human", "llm-judge", "code"]


@dataclass(frozen=True)
class Atom:
    """One objective binary dimension (gatable via the calibration gate)."""

    name: str
    description: str = ""


@dataclass(frozen=True)
class SubjectiveDimension:
    """One subjective dimension (tracked via preference/Elo only)."""

    name: str
    description: str = ""


@dataclass(frozen=True)
class AtomSet:
    """Declares the gate/track split for one evaluation criterion."""

    criterion: str
    objective_atoms: tuple[Atom, ...]
    subjective_dimensions: tuple[SubjectiveDimension, ...]


@dataclass(frozen=True)
class PreferencePair:
    """One pairwise preference judgment."""

    trace_id_a: str
    trace_id_b: str
    dimension: str
    winner: Literal["a", "b"]
    judge_kind: str


@dataclass(frozen=True)
class EloRating:
    """One trace's sequential-Elo display score for one dimension.

    Display only: sequential Elo depends on the order the pairs were processed in and has no
    interval. :func:`calibrated_agent_evaluation.subjective.bradley_terry` is the estimator of
    record.
    """

    trace_id: str
    rating: float
    n_comparisons: int


@dataclass(frozen=True)
class EloRatings:
    """All sequential-Elo display ratings for one subjective dimension.

    ``k_factor`` and ``initial_rating`` are the update hyperparameters, recorded so the numbers
    can be reproduced given the same pairs in the same order.
    """

    dimension: str
    ratings: tuple[EloRating, ...]
    n_total_comparisons: int
    treatment: str = "tracked"
    k_factor: float = 32.0
    initial_rating: float = 1500.0


@dataclass(frozen=True)
class BradleyTerryRating:
    """One trace's Bradley-Terry rating (Elo scale) with its 95% Wald interval."""

    trace_id: str
    rating: Interval
    n_comparisons: int


@dataclass(frozen=True)
class BradleyTerryRatings:
    """All Bradley-Terry ratings for one subjective dimension, best first. Tracked, never gated."""

    dimension: str
    ratings: tuple[BradleyTerryRating, ...]
    n_total_comparisons: int
    treatment: str = "tracked"
