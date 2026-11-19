# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Judge-biased upper-bound rate.

A black-box target has no calibration set, so its true failure rate is not identifiable -- only a
judge-biased UPPER BOUND is honest. :func:`upper_bound_rate` produces one from per-trial
outcomes: the exact one-sided Clopper-Pearson upper bound on the rate the judge (or benchmark
scorer) reported, with no correction for that judge's own errors. :class:`UpperBoundRate` carries
the bound but refuses a prevalence-corrected point via :meth:`require_point` (mirrors the
calibration ``Interval``'s ``require_point``: a caller wanting the scalar must go through the
accessor that raises the typed refusal, so a "corrected rate" can never silently leak out of a
probe).
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from calibrated_agent_evaluation.probe.errors import ProbeError, ProbeUncorrectableError
from calibrated_agent_evaluation.probe.robustness import clopper_pearson

METHOD_CLOPPER_PEARSON_UPPER = "clopper_pearson_one_sided_upper"


@dataclass(frozen=True)
class UpperBoundRate:
    """A judge-biased upper bound on a failure rate over ``k`` trials.

    ``upper_bound`` is the honest signal; there is no corrected point estimate for a black-box
    target. :meth:`require_point` raises rather than fabricating one. ``failures`` is the number
    of the ``k`` trials the judge flagged (``None`` when the bound came from elsewhere), and
    ``level`` the one-sided confidence of the bound.
    """

    upper_bound: float
    k: int
    failures: int | None = None
    level: float = 0.95
    method: str = METHOD_CLOPPER_PEARSON_UPPER

    def require_point(self) -> float:
        """Refuse a corrected point rate -- none exists for a black-box target."""
        raise ProbeUncorrectableError(
            f"no prevalence-corrected point rate exists for a black-box target (k={self.k}); "
            f"the judge-biased upper bound {self.upper_bound:.4f} is the honest signal "
            ". Calibrate against human gold to correct a rate."
        )


def upper_bound_rate(trial_failed: Sequence[bool], *, level: float = 0.95) -> UpperBoundRate:
    """The judge-biased one-sided upper bound on a black-box target's failure rate.

    ``trial_failed`` holds one judge (or benchmark scorer) outcome per trial, ``True`` for a
    failure. The bound is the exact one-sided Clopper-Pearson upper limit at ``level`` on the rate
    of flagged trials. It inherits the judge's errors, which cannot be corrected without a
    calibration set, and it never yields a point: 0 failures in 5 trials bounds the rate at 0.451
    at 95%, not at 0.

    Raises:
        ProbeError: No trials, or ``level`` outside ``(0, 1)``.
    """
    n = len(trial_failed)
    if n == 0:
        raise ProbeError("upper_bound_rate: need at least one trial to bound a rate")
    if not 0.0 < level < 1.0:
        raise ProbeError(f"upper_bound_rate: level must lie in (0, 1), got {level}")
    failures = sum(1 for t in trial_failed if t)
    # The two-sided interval at 2 * level - 1 has the one-sided ``level`` upper limit as its hi.
    upper = 1.0 if failures == n else clopper_pearson(failures, n, level=2.0 * level - 1.0).hi
    return UpperBoundRate(upper_bound=upper, k=n, failures=failures, level=level)
