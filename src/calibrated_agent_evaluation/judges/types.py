# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Data model for the judge-level layer.

Pure, frozen data holders. The three judge *levels* and the deterministic process judge's verdict;
the operations are free functions in ``levels``/``process``. No vendor type appears here.
``JudgeDefinition`` is the minimal level-carrying identity the level guards read; the judge-runtime
layer owns the full executable judge object and its calibration wiring.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

JudgeLevel = Literal["workflow", "span", "process"]


@dataclass(frozen=True)
class JudgeDefinition:
    """The minimal level-carrying judge identity the level guards read.

    ``level`` is what a judge attributes at; a ``"span"`` judge is calibrated over one agent's
    input+output, a ``"workflow"`` judge over whole traces; a workflow judge may not compute a
    per-agent rate. ``model``/``prompt_version`` mirror the calibration ``Judge`` protocol so a
    definition can be handed straight to the calibration gate.
    """

    judge_id: str
    level: JudgeLevel
    model: str
    prompt_version: str


@dataclass(frozen=True)
class ProcessVerdict:
    """The deterministic process-judge output.

    Computed by expected-vs-actual agent set-difference (no LLM call). ``rule_violations`` lists the
    rule ids whose deterministic clause failed; ``undecided_rules`` lists the non-deterministic
    rules, which nothing here can decide; reported as undecided, never a silent deterministic pass.
    ``passed`` is True only when nothing was skipped/unexpected and no rule was violated.

    ``ordering_undetermined`` is True because a supervisor-fanout crew's pipeline order is not in a
    normalized trace (siblings under the supervisor; ``seq`` is emission order, not the logical
    route), so the deterministic judge does not assert an ordering violation; it records that order
    could not be verified. A mis-route is still caught by the set differences + the rule clauses.
    """

    skipped_agents: tuple[str, ...]
    unexpected_agents: tuple[str, ...]
    ordering_undetermined: bool
    rule_violations: tuple[str, ...]
    undecided_rules: tuple[str, ...]
    passed: bool
