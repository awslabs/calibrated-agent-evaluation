# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Check whether a proposed fix really worked.

The last stage of the loop. Four guards combine into one :class:`ImproveVerdict`:

- a pre-registration ledger, which freezes the metric and threshold before the fix;
- a held-out confirmation slice, fresh and paired, that the fix was not tuned on;
- a paired McNemar test on before and after verdicts (with a mid-p variant, and an explicit
  ``undefined`` result when no verdict changed);
- a Holm-corrected panel of guard slices that catches a fix that only moves failures elsewhere,
  which for a crew means gating on the workflow outcome rate.

No trace-store or observability SDK and no backend writes; the only external library is
``statsmodels``, used in :mod:`calibrated_agent_evaluation.stats.mcnemar`.
"""

from __future__ import annotations

from calibrated_agent_evaluation.improve.confirm import assert_held_out, pair_observations, slice_id
from calibrated_agent_evaluation.improve.crew import (
    CrewComparison,
    CrewFile,
    CrewSide,
    compare_crew_sides,
    count_crew_side,
    crew_definition_from_file,
    crew_outcomes,
    not_measured_rate,
)
from calibrated_agent_evaluation.improve.errors import (
    AdjudicationRequiredError,
    ConfirmationContaminationError,
    ImproveError,
    ImproveStoreError,
    LedgerIntegrityError,
)
from calibrated_agent_evaluation.improve.guard import evaluate_improvement, record_of
from calibrated_agent_evaluation.improve.paired import (
    adjudicate_pairs,
    adjudication_problem,
    discordant_pairs,
    discovery_enqueue,
    paired_mcnemar,
    require_adjudication,
)
from calibrated_agent_evaluation.improve.panel import build_crew_slices, crew_panel, guard_panel
from calibrated_agent_evaluation.improve.prereg import (
    EMPTY_LEDGER_HEAD,
    LEDGER_HEAD_FIELD,
    freeze,
    ledger_head,
    register_prereg,
    require_frozen_before,
    require_prereg,
    verify_anchors,
    verify_heads,
    verify_ledger,
)
from calibrated_agent_evaluation.improve.report import (
    IMPROVE_SCHEMA_VERSION,
    GuardPanelReport,
    GuardSlice,
    ImproveRecord,
    ImproveRequest,
    ImproveVerdict,
    McNemarResult,
    PairedObs,
    PreRegEntry,
    PreRegLedger,
)
from calibrated_agent_evaluation.improve.store import (
    dump_improve_records,
    dump_prereg_ledger,
    load_improve_records,
    load_prereg_ledger,
)

__all__ = [
    "EMPTY_LEDGER_HEAD",
    "IMPROVE_SCHEMA_VERSION",
    "LEDGER_HEAD_FIELD",
    "AdjudicationRequiredError",
    "ConfirmationContaminationError",
    "CrewComparison",
    "CrewFile",
    "CrewSide",
    "GuardPanelReport",
    "GuardSlice",
    "ImproveError",
    "ImproveRecord",
    "ImproveRequest",
    "ImproveStoreError",
    "ImproveVerdict",
    "LedgerIntegrityError",
    "McNemarResult",
    "PairedObs",
    "PreRegEntry",
    "PreRegLedger",
    "adjudicate_pairs",
    "adjudication_problem",
    "assert_held_out",
    "build_crew_slices",
    "compare_crew_sides",
    "count_crew_side",
    "crew_definition_from_file",
    "crew_outcomes",
    "crew_panel",
    "discordant_pairs",
    "discovery_enqueue",
    "dump_improve_records",
    "dump_prereg_ledger",
    "evaluate_improvement",
    "freeze",
    "guard_panel",
    "ledger_head",
    "load_improve_records",
    "load_prereg_ledger",
    "not_measured_rate",
    "pair_observations",
    "paired_mcnemar",
    "record_of",
    "register_prereg",
    "require_adjudication",
    "require_frozen_before",
    "require_prereg",
    "slice_id",
    "verify_anchors",
    "verify_heads",
    "verify_ledger",
]
