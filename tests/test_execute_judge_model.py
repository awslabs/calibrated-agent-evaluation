# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the judge-model seam in execute/.

Exercises the real Inspect extraction paths (_reply_from_tool_call / _reply_from_response_schema)
against real inspect_ai.model.ModelOutput objects — no network. The live Bedrock parity path is the
opt-in integration test (test_judges_live.py).
"""

from __future__ import annotations

import pytest

from calibrated_agent_evaluation.execute.errors import ExecutionError
from calibrated_agent_evaluation.execute.judge_model import (
    _VERDICT_TOOL_NAME,
    _reply_from_response_schema,
    _reply_from_tool_call,
)


def test_tool_call_reply_extracted_from_arguments() -> None:
    """A forced-tool-call ModelOutput yields a StructuredReply built from the tool-call
    ARGUMENTS (via='tool_call'), not from any completion text."""
    from inspect_ai.model import ModelOutput

    output = ModelOutput.for_tool_call(
        model="bedrock/amazon.titan-text",
        tool_name=_VERDICT_TOOL_NAME,
        tool_arguments={"critique": "out-of-window refund", "verdict": "fail"},
    )
    reply = _reply_from_tool_call(output, "bedrock/amazon.titan-text")
    assert reply.via == "tool_call"
    assert reply.fields == {"critique": "out-of-window refund", "verdict": "fail"}


def test_no_tool_call_raises_execution_error() -> None:
    """A reply carrying no verdict tool call raises ExecutionError — the runtime
    refuses to parse a verdict from free text."""
    from inspect_ai.model import ModelOutput

    output = ModelOutput.from_content(
        model="bedrock/amazon.titan-text", content="Sure, the refund looks fine to me!"
    )
    with pytest.raises(ExecutionError, match="no record_verdict tool call"):
        _reply_from_tool_call(output, "bedrock/amazon.titan-text")


def test_response_schema_reply_parsed_from_json_completion() -> None:
    """A native response_schema completion (JSON text) yields via='response_schema'."""
    from inspect_ai.model import ModelOutput

    output = ModelOutput.from_content(
        model="bedrock/global.anthropic.claude-sonnet-4-6",
        content='{"critique": "wrong", "verdict": "fail"}',
    )
    reply = _reply_from_response_schema(output, "bedrock/…claude")
    assert reply.via == "response_schema"
    assert reply.fields["verdict"] == "fail"


def test_non_json_response_schema_completion_raises() -> None:
    """A non-JSON response_schema completion raises — no free-text guess after."""
    from inspect_ai.model import ModelOutput

    output = ModelOutput.from_content(
        model="bedrock/…claude", content="I think it is fine, no JSON here"
    )
    with pytest.raises(ExecutionError, match="non-JSON"):
        _reply_from_response_schema(output, "bedrock/…claude")


def test_supports_response_schema_gating() -> None:
    """Only Bedrock Claude reports native response_schema support; others tool-force."""
    from calibrated_agent_evaluation.execute import InspectJudgeModel

    claude = InspectJudgeModel("bedrock/global.anthropic.claude-sonnet-4-6")
    assert claude.supports_response_schema()
    assert not InspectJudgeModel("bedrock/amazon.titan-text").supports_response_schema()
    assert not InspectJudgeModel("sagemaker/my-vllm-endpoint").supports_response_schema()
