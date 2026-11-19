# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""The three judge levels + the calibrate-at-the-level-you-attribute-at guards.

A crew is judged at three levels -- **workflow** (whole trace), **span** (one agent's input+output),
**process** (the span tree) -- and each must be calibrated at the level it attributes at. Two
misuses are provably invalid and refused here:

  * :func:`assert_span_gold_human` -- span-level calibration gold must be **per-span human gold**
    (a ``Label`` with a ``span_id`` and ``kind="human"``), hunted per-span. Gold derived from the
    trace outcome (no ``span_id``) cannot certify a span judge -- you cannot read span correctness
    off the outcome.
  * :func:`assert_not_workflow_for_agent_rate` -- a workflow judge scores the whole trace, so it
    cannot compute a per-agent rate; using one for per-agent attribution is refused.

These guards hold the calibration discipline the judge-execution runtime relies on.
"""

from __future__ import annotations

from collections.abc import Sequence

from calibrated_agent_evaluation.judges.errors import JudgeLevelError, SpanGoldError
from calibrated_agent_evaluation.judges.types import JudgeDefinition
from calibrated_agent_evaluation.schema.types import Label


def assert_span_gold_human(gold: Sequence[Label]) -> None:
    """Refuse span-judge calibration gold that is not per-span human gold.

    Raises :class:`SpanGoldError` if any label lacks a ``span_id`` (it is trace-outcome gold, not a
    per-span judgment) or is not ``kind="human"`` (an LLM/code verdict cannot be gold). The
    fail hunted for a span judge is the *span* fail, so its gold is keyed on a span, not on outcome.
    """
    for label in gold:
        if label.span_id is None:
            raise SpanGoldError(
                f"span-level calibration gold for trace {label.trace_id!r} has span_id=None; span "
                f"gold must be per-span human gold, not derived from the trace outcome. "
                f"Label the span itself."
            )
        if label.kind != "human":
            raise SpanGoldError(
                f"span-level gold for {label.span_id!r} is kind={label.kind!r}; calibration gold "
                f"must be human."
            )


def assert_not_workflow_for_agent_rate(judge: JudgeDefinition) -> None:
    """Refuse a workflow-level judge used to compute a per-agent rate.

    Raises :class:`JudgeLevelError` if ``judge.level == "workflow"``. A workflow judge scores the
    whole trace and cannot attribute per agent; per-agent rates require a span-level judge.
    """
    if judge.level == "workflow":
        raise JudgeLevelError(
            f"judge {judge.judge_id!r} is level='workflow' and cannot compute a per-agent rate; "
            f"per-agent attribution requires a span-level judge."
        )
