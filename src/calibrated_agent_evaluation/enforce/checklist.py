# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Checklist collapse -- any critical "no" is overall FAIL.

The pre-registered collapse rule: each item is tagged critical/non-critical at
authoring time; a single ``critical`` item verdict of FAIL collapses the whole judge to FAIL
regardless of the rest, while non-critical items are tracked as non-gating quality sub-metrics.
Weighted-threshold collapse is refused at lint -- it is never offered here.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

from calibrated_agent_evaluation.enforce.judgedef import JudgeDefinition
from calibrated_agent_evaluation.schema.types import Outcome


@dataclass(frozen=True)
class ChecklistResult:
    """The collapsed verdict plus the non-gating sub-metrics.

    ``overall`` is FAIL iff at least one ``critical`` item is FAIL; ``non_gating`` records every
    non-critical item's verdict (tracked, never gating).
    """

    overall: Outcome
    non_gating: dict[str, Outcome]


def collapse_checklist(
    defn: JudgeDefinition, item_verdicts: Mapping[str, Outcome]
) -> ChecklistResult:
    """Collapse per-item verdicts to one overall outcome.

    ``item_verdicts`` maps each checklist item name to its ``PASS``/``FAIL`` verdict. Any critical
    item at ``FAIL`` -> overall ``FAIL``; otherwise overall ``PASS``. Non-critical items are
    returned as sub-metrics and never affect ``overall``. An item named in the definition but
    absent from ``item_verdicts`` is treated as unresolved and, if critical, fails closed (a
    missing critical verdict cannot certify a PASS).
    """
    overall: Outcome = "PASS"
    non_gating: dict[str, Outcome] = {}
    for item in defn.checklist:
        verdict = item_verdicts.get(item.name)
        if item.critical:
            # Fail closed: a critical item that is FAIL or unresolved collapses to FAIL.
            if verdict != "PASS":
                overall = "FAIL"
        elif verdict is not None:
            non_gating[item.name] = verdict
    return ChecklistResult(overall=overall, non_gating=non_gating)
