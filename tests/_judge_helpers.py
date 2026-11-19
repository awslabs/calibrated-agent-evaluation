# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Test doubles for judge-runtime tests — a spyable fake JudgeModelBackend.

Mirrors the real ``JudgeModelBackend`` surface (``supports_response_schema`` +
``generate_structured``) so unit tests exercise the runtime's real gate/cache/retry/write-back paths
without a network call (rubric-blessed stub: it mirrors the verified seam contract; the live Bedrock
path is an opt-in integration test).
"""

from __future__ import annotations

from typing import Any

from _helpers import TEST_STAMP
from calibrated_agent_evaluation.calibrate.record import Bound, CalibrationRecord
from calibrated_agent_evaluation.execute import StructuredReply
from calibrated_agent_evaluation.execute.errors import ExecutionError
from calibrated_agent_evaluation.judges.executable import ExecutableJudge


class FakeJudgeModel:
    """A JudgeModelBackend that returns a scripted reply and counts calls (the spy).

    ``reply`` is returned from every ``generate_structured``; ``call_count`` records how many times
    the model was invoked (so a test can assert a cache hit did NOT call it). ``fail_times`` makes
    the first N calls raise ``ExecutionError`` (transient), then succeed — for the retry test.
    """

    def __init__(
        self,
        reply: StructuredReply | None = None,
        *,
        supports_schema: bool = True,
        fail_times: int = 0,
    ) -> None:
        self._reply = reply or StructuredReply(
            fields={"critique": "out-of-window refund wrongly issued", "verdict": "fail"},
            via="response_schema" if supports_schema else "tool_call",
        )
        self._supports_schema = supports_schema
        self._fail_times = fail_times
        self.call_count = 0

    def supports_response_schema(self) -> bool:
        return self._supports_schema

    def generate_structured(
        self, *, system: str, user: str, schema: dict[str, Any], sample: int = 0
    ) -> StructuredReply:
        self.call_count += 1
        if self.call_count <= self._fail_times:
            raise ExecutionError(f"transient model failure #{self.call_count}")
        return self._reply


def judge(**over: Any) -> ExecutableJudge:
    """A default certified-shaped OrderBot judge; override any field via kwargs."""
    base: dict[str, Any] = {
        "judge_id": "orderbot-refund-resolved",
        "model": "bedrock/global.anthropic.claude-sonnet-4-6",
        "prompt_version": "v1",
        "level": "workflow",
        "prompt_template": "Was the refund resolved correctly?",
    }
    base.update(over)
    return ExecutableJudge(**base)


def certified_record(j: ExecutableJudge, **over: Any) -> CalibrationRecord:
    """A CERTIFIED CalibrationRecord pinned to ``j``'s version; override to force drift/refusal."""
    base: dict[str, Any] = {
        "calib_id": "calib-1",
        "judge_id": j.judge_id,
        "model": j.model,
        "prompt_version": j.prompt_version,
        "judge_fingerprint": j.fingerprint,
        "hunted_set_id": "hunted-1",
        "n_pos": 80,
        "n_neg": 80,
        "tpr": Bound(point=0.95, lb95=0.90, n=80),
        "tnr": Bound(point=0.95, lb95=0.90, n=80),
        "youden_j": 0.90,
        "result": "CERTIFIED",
    }
    base.update(over)
    return CalibrationRecord(provenance=TEST_STAMP, **base)
