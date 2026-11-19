# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Statistics used by the calibration gate and the later stages.

Wilson lower bounds and Holm/BY multiplicity come from statsmodels. The Rogan-Gladen interval
(Jeffreys posteriors), Youden's J and Cohen's kappa are implemented here, kappa to avoid a large
scikit-learn dependency. Saturation (Good-Turing, Kish), McNemar and drift live in their own
modules in this package. Pure functions, no vendor trace type, no I/O.
"""

from __future__ import annotations

from calibrated_agent_evaluation.stats.agreement import (
    cohen_kappa,
    minority_fraction,
    should_hide_raw_agreement,
)
from calibrated_agent_evaluation.stats.drift import (
    jensen_shannon,
    permutation_p,
    psi,
    psi_permutation_p,
    wasserstein,
    wasserstein_normalized,
)
from calibrated_agent_evaluation.stats.mcnemar import McNemarStat, mcnemar_paired
from calibrated_agent_evaluation.stats.multiplicity import multiplicity
from calibrated_agent_evaluation.stats.proportions import (
    newcombe_paired_diff,
    wilson_lower,
    wilson_upper,
)
from calibrated_agent_evaluation.stats.rogan_gladen import rg_interval, rogan_gladen, youden_j

__all__ = [
    "McNemarStat",
    "cohen_kappa",
    "jensen_shannon",
    "mcnemar_paired",
    "minority_fraction",
    "multiplicity",
    "newcombe_paired_diff",
    "permutation_p",
    "psi",
    "psi_permutation_p",
    "rg_interval",
    "rogan_gladen",
    "should_hide_raw_agreement",
    "wasserstein",
    "wasserstein_normalized",
    "wilson_lower",
    "wilson_upper",
    "youden_j",
]
