# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""The tiered enforcement / lint-policy layer -- "enforcement, not enablement".

The methodology-plane surface that makes the right habit the default and the wrong habit the hard
path, tiering each intervention to the strength of the evidence: **hard refusals** only
where a pattern is provably invalid, **warnings + nudges** where it is merely
directional, and **display defaults** that shape output. It reads --
never recomputes -- the calibration verdict to gate scoring, and reuses the shared
``cohen_kappa`` for the imbalance display.

Pure methodology plane: no vendor SDK type appears here, and no new runtime dependency is
introduced. Every decision is captured as an auditable
:class:`~calibrated_agent_evaluation.enforce.decision.EnforcementDecision`. It does NOT run judges,
parse CLI flags, or compute the calibration statistics.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from calibrated_agent_evaluation.enforce.checklist import ChecklistResult, collapse_checklist
from calibrated_agent_evaluation.enforce.context import EnforcementContext
from calibrated_agent_evaluation.enforce.decision import EnforcementDecision, Tier
from calibrated_agent_evaluation.enforce.errors import (
    CritiqueOrderError,
    EnforcementError,
    EnforcementStoreError,
    GenericMetricInCIError,
    LikertNotAllowedError,
    MultiAxisLikertError,
    UncalibratedScoreError,
    WeightedCollapseError,
)
from calibrated_agent_evaluation.enforce.gate import enforce_score_gate, warn_error_analysis_first
from calibrated_agent_evaluation.enforce.judgedef import (
    Axis,
    ChecklistItem,
    JudgeDefinition,
    Scale,
    critique_before_verdict,
    likert_axes,
)
from calibrated_agent_evaluation.enforce.lint import (
    GENERIC_GENERATION_METRICS,
    lint_judge_definition,
    lint_scorer_lanes,
)
from calibrated_agent_evaluation.enforce.store import (
    dump_decisions,
    dump_definitions,
    load_decisions,
    load_definitions,
)
from calibrated_agent_evaluation.enforce.version import ENFORCE_SCHEMA_VERSION

if TYPE_CHECKING:
    from calibrated_agent_evaluation.enforce.display import AgreementDisplay, agreement_display

# The agreement display needs the [stats] extra (numpy, scipy, statsmodels); loading it on first
# use keeps the rest of this layer, such as the error-analysis-first warning a stdlib-only scorer
# applies, importable on a bare install.
_LAZY_DISPLAY = frozenset({"AgreementDisplay", "agreement_display"})


def __getattr__(name: str) -> Any:
    if name in _LAZY_DISPLAY:
        from calibrated_agent_evaluation.enforce import display

        return getattr(display, name)
    raise AttributeError(f"module 'calibrated_agent_evaluation.enforce' has no attribute {name!r}")


__all__ = [
    "ENFORCE_SCHEMA_VERSION",
    "GENERIC_GENERATION_METRICS",
    "AgreementDisplay",
    "Axis",
    "ChecklistItem",
    "ChecklistResult",
    "CritiqueOrderError",
    "EnforcementContext",
    "EnforcementDecision",
    "EnforcementError",
    "EnforcementStoreError",
    "GenericMetricInCIError",
    "JudgeDefinition",
    "LikertNotAllowedError",
    "MultiAxisLikertError",
    "Scale",
    "Tier",
    "UncalibratedScoreError",
    "WeightedCollapseError",
    "agreement_display",
    "collapse_checklist",
    "critique_before_verdict",
    "dump_decisions",
    "dump_definitions",
    "enforce_score_gate",
    "likert_axes",
    "lint_judge_definition",
    "lint_scorer_lanes",
    "load_decisions",
    "load_definitions",
    "warn_error_analysis_first",
]
