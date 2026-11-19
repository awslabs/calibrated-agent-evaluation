# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Build the forced-structured-output schema for a judge.

The schema the judge model is forced to fill: a non-empty ``critique`` string ordered strictly
before a binary ``verdict``. The critique-before-verdict rule is not
re-implemented here; it is read from the :func:`critique_before_verdict` predicate (the single
source of truth for the ordering rule), so a mis-ordered judge is rejected at build time with the
same rule the linter enforces at authoring time.
"""

from __future__ import annotations

from typing import Any

from calibrated_agent_evaluation.enforce import JudgeDefinition as LintJudgeDefinition
from calibrated_agent_evaluation.enforce import critique_before_verdict
from calibrated_agent_evaluation.judges.errors import JudgeRuntimeError
from calibrated_agent_evaluation.judges.executable import ExecutableJudge

_VERDICT_VALUES = ["pass", "fail"]


def build_verdict_schema(judge: ExecutableJudge) -> dict[str, Any]:
    """Return the JSON schema for ``judge``'s forced-structured verdict.

    Guarantees the critique-before-verdict ordering by reusing ``critique_before_verdict``: a judge
    whose ``output_fields`` place ``verdict`` before ``critique`` (or omit either) is rejected with
    a :class:`JudgeRuntimeError` rather than emitting a schema; a mis-ordered judge must never
    reach the model. The emitted schema requires a non-empty ``critique`` (``minLength: 1``) and a
    binary ``verdict`` enum, so a blank critique or non-binary verdict fails schema-side too.
    """
    # Reuse the shared predicate as the single source of the ordering rule. axes=(); this
    # shape carries only the output_fields the predicate reads; the axis/lint checks live upstream.
    lint_shape = LintJudgeDefinition(
        judge_id=judge.judge_id, axes=(), output_fields=judge.output_fields
    )
    if not critique_before_verdict(lint_shape):
        raise JudgeRuntimeError(
            f"judge {judge.judge_id!r} output_fields {judge.output_fields!r} must place a "
            "non-empty 'critique' strictly before 'verdict'; refusing to build "
            "a verdict schema"
        )
    properties: dict[str, Any] = {
        "critique": {
            "type": "string",
            "minLength": 1,
            "description": (
                "The judge's reasoning, written before deciding the verdict "
                "(P-003)."  # allow-id: model-facing text that recorded cassettes key on
            ),
        },
        "verdict": {
            "type": "string",
            "enum": _VERDICT_VALUES,
            "description": "The binary verdict: 'pass' or 'fail'.",
        },
    }
    return {
        "type": "object",
        "properties": properties,
        "required": ["critique", "verdict"],
        "additionalProperties": False,
    }
