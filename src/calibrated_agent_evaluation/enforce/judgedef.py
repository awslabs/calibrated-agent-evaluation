# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""The lint target: a typed, frozen ``JudgeDefinition``.

This module owns the *authored* judge shape it lints; the judge-execution runtime later
constructs and runs these. Pure, frozen data holders in the "typed
functional core" house style: validation and the tiered refusals live in
:mod:`calibrated_agent_evaluation.enforce.lint`, never in ``__init__`` -- so a
deliberately-malformed definition can be constructed in a test and handed to the linter (the linter
is the enforcement point).

The type carries only what the lint rules read: per-axis scale, the output-field
order, the checklist items with critical tags, the scorer id classified for
the CI ban, and the weighted-collapse flag that is refused. No vendor type appears
here.
"""

from __future__ import annotations

from typing import ClassVar, Literal

from pydantic import ConfigDict

from calibrated_agent_evaluation._model import CalibratedAgentEvaluationModel
from calibrated_agent_evaluation.enforce.errors import EnforcementError
from calibrated_agent_evaluation.enforce.version import ENFORCE_SCHEMA_VERSION

Scale = Literal["binary", "likert"]


class Axis(CalibratedAgentEvaluationModel):
    """One scoring axis of a judge. ``scale`` defaults to ``binary``."""

    model_config = ConfigDict(frozen=True, extra="forbid")
    invalid_error: ClassVar[type[Exception]] = EnforcementError

    name: str
    scale: Scale = "binary"


class ChecklistItem(CalibratedAgentEvaluationModel):
    """One checklist item, tagged critical or non-critical at authoring time.

    A ``critical`` item's "no" collapses the whole judge to FAIL; a non-critical item's "no" is
    tracked as a non-gating quality sub-metric.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")
    invalid_error: ClassVar[type[Exception]] = EnforcementError

    name: str
    critical: bool


class JudgeDefinition(CalibratedAgentEvaluationModel):
    """An authored judge definition -- the lint target.

    ``output_fields`` is the ordered field list of the judge's forced-structured output; the
    critique-first discipline requires a non-empty ``"critique"`` ordered before ``"verdict"``
    (checked by the linter). ``scorer`` names the scoring mechanism (e.g. ``"rouge"``,
    ``"exact_match"``); :func:`calibrated_agent_evaluation.enforce.lint` classifies it as
    generic-generation vs structured-target for the CI ban. ``checklist`` items carry critical tags;
    ``weighted_collapse`` is refused at lint.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")
    invalid_error: ClassVar[type[Exception]] = EnforcementError

    judge_id: str
    axes: tuple[Axis, ...]
    output_fields: tuple[str, ...]
    scorer: str | None = None
    checklist: tuple[ChecklistItem, ...] = ()
    weighted_collapse: bool = False
    schema_version: str = ENFORCE_SCHEMA_VERSION


def critique_before_verdict(defn: JudgeDefinition) -> bool:
    """Return whether ``output_fields`` contains ``critique`` strictly before ``verdict``.

    The pure predicate behind the critique-first refusal: a definition missing ``critique``, missing
    ``verdict``, or ordering ``verdict`` first fails it. The linter turns a ``False`` here into a
    typed :class:`~calibrated_agent_evaluation.enforce.errors.CritiqueOrderError`.
    """
    fields = defn.output_fields
    if "critique" not in fields or "verdict" not in fields:
        return False
    return fields.index("critique") < fields.index("verdict")


def likert_axes(defn: JudgeDefinition) -> tuple[Axis, ...]:
    """Return the non-binary (Likert) axes of ``defn``."""
    return tuple(ax for ax in defn.axes if ax.scale == "likert")
