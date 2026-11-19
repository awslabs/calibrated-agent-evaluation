# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""The typed error the AgentCore Runtime executor raises. Vendor-free.

It is also an execution-plane error, so it lives apart from
:mod:`calibrated_agent_evaluation.agentcore.errors`: importing
:mod:`calibrated_agent_evaluation.execute` there would make a bare
``import calibrated_agent_evaluation.agentcore`` load the execution plane. The package's lazy names
load it only when first asked for.
"""

from __future__ import annotations

from calibrated_agent_evaluation.agentcore.errors import AgentCoreError
from calibrated_agent_evaluation.execute.errors import ExecutionError

__all__ = ["AgentReportedError", "RuntimeInvocationError"]


class RuntimeInvocationError(ExecutionError, AgentCoreError):
    """A run against an AgentCore Runtime agent was refused or failed.

    An ``ExecutionError`` and an ``AgentCoreError``, so a caller of any executor and a caller of
    the AgentCore plug-in both catch it. The underlying error is chained as ``__cause__``. The
    message names identifiers and metadata only (error codes, operations, session and trace IDs,
    paths), never a payload, a response body or the agent's error text, with one exception: when
    a span reader fails with an ``AgentCoreError`` other than a billed-call or read-only guard's
    refusal, the message quotes that error's text after ``The reader's message:``. The span reader
    contract keeps a reader's typed refusals to IDs, log groups, counts and times. An
    ``AgentCoreError`` from a caller's own guard on the reader's clients looks the same, so its
    text is quoted too, whatever it holds.
    """


class AgentReportedError(RuntimeInvocationError):
    """The agent's reply reported an error, so the run has no answer.

    A caller that treats such a run as a result rather than a failure (a policy battery, where a
    denied tool call can come back as an agent error) tells it apart from every other refusal by
    type. ``error_code`` is the report's ``error_type`` when that is an identifier of at most 100
    characters, else ``"agent_error"``; the error text is never kept. ``trace_id`` is the run's
    trace ID when the raiser knows it, else ``None``.
    """

    def __init__(self, message: str, *, error_code: str, trace_id: str | None = None) -> None:
        super().__init__(message)
        self.error_code = error_code
        self.trace_id = trace_id
