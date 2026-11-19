# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Judge-level layer.

The three judge *levels* a crew is judged at -- **workflow** (whole trace), **span** (one agent's
input+output), **process** (the span tree) -- with the "calibrate at the level you attribute at"
discipline enforced, plus the **deterministic process judge** (expected-vs-actual agent
set-difference + ordering, no LLM call).

Pure methodology plane: reads normalized ``Trace``/``Span`` and the crew Define artifact; no vendor
type appears here. It does NOT execute LLM workflow/span judges -- that batch fan-out
runtime is the judge-execution layer; this package owns the level model + the deterministic
process judge only.

Public API re-exports from submodules.
"""

from __future__ import annotations

from calibrated_agent_evaluation.judges.cache import VerdictCache
from calibrated_agent_evaluation.judges.errors import (
    JudgeLevelError,
    JudgeRuntimeError,
    SpanGoldError,
)
from calibrated_agent_evaluation.judges.executable import ExecutableJudge
from calibrated_agent_evaluation.judges.levels import (
    assert_not_workflow_for_agent_rate,
    assert_span_gold_human,
)
from calibrated_agent_evaluation.judges.probe_runner import JudgeProbeRunner, render_probe_input
from calibrated_agent_evaluation.judges.process import process_judge
from calibrated_agent_evaluation.judges.run_record import (
    JudgeRunRecord,
    dump_run_records,
    load_run_records,
)
from calibrated_agent_evaluation.judges.runtime import (
    JudgeRunResult,
    JudgeTarget,
    locked_test_set,
    run_judge,
)
from calibrated_agent_evaluation.judges.targets import (
    ancestor_path,
    render_span,
    render_trace,
    span_target,
    workflow_target,
)
from calibrated_agent_evaluation.judges.types import JudgeDefinition, JudgeLevel, ProcessVerdict
from calibrated_agent_evaluation.judges.verdict import reply_to_label, verdict_rater_id
from calibrated_agent_evaluation.judges.verdict_schema import build_verdict_schema
from calibrated_agent_evaluation.judges.version import JUDGES_SCHEMA_VERSION

__all__ = [
    "JUDGES_SCHEMA_VERSION",
    "ExecutableJudge",
    "JudgeDefinition",
    "JudgeLevel",
    "JudgeLevelError",
    "JudgeProbeRunner",
    "JudgeRunRecord",
    "JudgeRunResult",
    "JudgeRuntimeError",
    "JudgeTarget",
    "ProcessVerdict",
    "SpanGoldError",
    "VerdictCache",
    "ancestor_path",
    "assert_not_workflow_for_agent_rate",
    "assert_span_gold_human",
    "build_verdict_schema",
    "dump_run_records",
    "load_run_records",
    "locked_test_set",
    "process_judge",
    "render_probe_input",
    "render_span",
    "render_trace",
    "reply_to_label",
    "run_judge",
    "span_target",
    "verdict_rater_id",
    "workflow_target",
]
