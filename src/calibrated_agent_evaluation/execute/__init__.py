# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Execution plane: run agents programmatically and get normalized Traces.

The public surface: ``ExecutorBackend`` (the protocol), ``InspectExecutor`` (Inspect-backed) and
``StrandsExecutor``/``StrandsTask`` (Strands-backed), ``ExecutionError`` (the typed error
family), and the ``Cassette`` record/replay layer. The vendor SDKs are imported lazily so
``import calibrated_agent_evaluation.execute`` succeeds without the ``[inspect]``/``[strands]``
extras installed.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from calibrated_agent_evaluation.execute.base import ExecutorBackend
from calibrated_agent_evaluation.execute.cassette import (
    Cassette,
    CassetteError,
    CassetteMiss,
    CassetteModelMismatch,
    Mode,
    model_stamp,
)
from calibrated_agent_evaluation.execute.errors import (
    ExecutionError,
    JudgeConfigurationError,
    SandboxError,
)
from calibrated_agent_evaluation.execute.judge_model import (
    CassetteJudgeModel,
    InspectJudgeModel,
    JudgeModelBackend,
    StructuredReply,
)
from calibrated_agent_evaluation.execute.sandbox import (
    LocalSubprocessSandbox,
    SandboxBackend,
    SandboxResult,
)

if TYPE_CHECKING:
    # Type-checkers resolve these statically; at runtime they are served lazily by __getattr__
    # below so `import calibrated_agent_evaluation.execute` stays cheap and does not require the
    # vendor extras.
    from calibrated_agent_evaluation.execute.bedrock_embedder import BedrockEmbedder
    from calibrated_agent_evaluation.execute.bedrock_renderer import BedrockRenderer
    from calibrated_agent_evaluation.execute.inspect_executor import InspectExecutor
    from calibrated_agent_evaluation.execute.strands_capture import (
        CaptureProviderError,
        SpanCapture,
        capture_spans,
        spans_to_otlp_json,
        write_otlp_json,
    )
    from calibrated_agent_evaluation.execute.strands_executor import StrandsExecutor, StrandsTask
    from calibrated_agent_evaluation.execute.task_builders import (
        InspectTaskBuilder,
        StrandsTaskBuilder,
    )

__all__ = [
    "BedrockEmbedder",
    "BedrockRenderer",
    "CaptureProviderError",
    "Cassette",
    "CassetteError",
    "CassetteJudgeModel",
    "CassetteMiss",
    "CassetteModelMismatch",
    "ExecutionError",
    "ExecutorBackend",
    "InspectExecutor",
    "InspectJudgeModel",
    "InspectTaskBuilder",
    "JudgeConfigurationError",
    "JudgeModelBackend",
    "LocalSubprocessSandbox",
    "Mode",
    "SandboxBackend",
    "SandboxError",
    "SandboxResult",
    "SpanCapture",
    "StrandsExecutor",
    "StrandsTask",
    "StrandsTaskBuilder",
    "StructuredReply",
    "capture_spans",
    "model_stamp",
    "spans_to_otlp_json",
    "write_otlp_json",
]


_CAPTURE_NAMES = frozenset(
    {
        "CaptureProviderError",
        "SpanCapture",
        "capture_spans",
        "spans_to_otlp_json",
        "write_otlp_json",
    }
)


def __getattr__(name: str) -> object:
    if name == "InspectExecutor":
        from calibrated_agent_evaluation.execute.inspect_executor import InspectExecutor

        return InspectExecutor
    if name in ("StrandsExecutor", "StrandsTask"):
        # StrandsTask is stdlib-only (safe to import), but strands_executor.py imports nothing at
        # module top that needs the [strands] extra; the SDK is loaded inside run().
        from calibrated_agent_evaluation.execute import strands_executor

        return getattr(strands_executor, name)
    if name in ("InspectTaskBuilder", "StrandsTaskBuilder"):
        # Concrete TaskBuilders. Vendor SDK is imported inside build(), so importing this
        # module needs no extra.
        from calibrated_agent_evaluation.execute import task_builders

        return getattr(task_builders, name)
    if name == "BedrockRenderer":
        from calibrated_agent_evaluation.execute.bedrock_renderer import BedrockRenderer

        return BedrockRenderer
    if name in _CAPTURE_NAMES:
        # strands_capture imports the OpenTelemetry SDK (the [strands] extra).
        from calibrated_agent_evaluation.execute import strands_capture

        return getattr(strands_capture, name)
    if name == "BedrockEmbedder":
        from calibrated_agent_evaluation.execute.bedrock_embedder import BedrockEmbedder

        return BedrockEmbedder
    raise AttributeError(f"module 'calibrated_agent_evaluation.execute' has no attribute {name!r}")
