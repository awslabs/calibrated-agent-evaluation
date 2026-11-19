# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""The agreement gate's thresholds and vocabulary, importable without the statistics engine.

The stored records (:mod:`calibrated_agent_evaluation.review.record`) need these names, and reading
a record must not pull in numpy; :mod:`calibrated_agent_evaluation.review.agreement`, which computes
the statistics, re-exports them.
"""

from __future__ import annotations

from typing import Literal

__all__ = [
    "ALPHA_LOWER_BOUND_GATE",
    "ALPHA_REVISE_FLOOR",
    "MINIMUM_OVERLAP",
    "AgreementAxis",
    "GateVerdict",
]

AgreementAxis = Literal["outcome", "failure_mode"]
GateVerdict = Literal["usable", "top_up_overlap", "revise_codebook", "codebook_broken"]

#: The gold set is usable when the bootstrap lower bound of alpha reaches this.
ALPHA_LOWER_BOUND_GATE = 0.80
#: A point estimate from here up to the gate means the codebook is ambiguous: revise it and
#: re-label the discordant strata. Below it the codebook is broken, not the raters.
ALPHA_REVISE_FLOOR = 0.667
#: Below this many units coded by two or more raters the verdict is never ``usable``, and is
#: ``top_up_overlap`` unless alpha's upper bound is under a revision band's bar:
#: ``calibrated_agent_evaluation.review.overlap.plan_overlap()`` at its defaults (raters expected at
# alpha : 0.90 on balanced PASS/FAIL gold), recorded here so reading a record needs no numpy; a test
#: recomputes it.
MINIMUM_OVERLAP = 130
