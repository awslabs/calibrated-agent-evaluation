# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""The span reader contract: how the AgentCore Runtime executor gets the span tree of a run.

The agent's span tree is written by the runtime's telemetry, so a :class:`RunSpanReader` fetches
it. After an invocation succeeds, and before the run's session is stopped, the executor calls
``read_spans`` once with a :class:`RunSpanQuery`. The query names the run by the IDs the executor
generated for it, so the spans are found by ID rather than by time window.

A reader promises four things.

- **Normalized spans keyed to the run.** Every
  :class:`~calibrated_agent_evaluation.schema.types.Span` it returns has ``query.trace_id`` as its
  ``trace_id``.
- **A complete tree or a refusal, never a partial list.** The reader owns a bounded wait for the
  telemetry to settle. When the wait runs out, results are partial, span content is missing, or
  the spans carry more than one session ID, it raises a typed subclass of
  :class:`~calibrated_agent_evaluation.agentcore.errors.AgentCoreError`. The executor repeats that
  error's message, so the message names IDs, log groups, counts and times, never span content or
  credentials.
- **Billed calls go through the clients it is given.** Every AWS call goes through the
  :class:`~calibrated_agent_evaluation.agentcore.session.AwsClients` the reader is built from. Given
  the executor's clients, its billed calls (the operations
  ``calibrated_agent_evaluation.agentcore.guards.BILLED_OPERATIONS`` lists) count against the same
  billed-call budget as the invocations and their retries. A reader that polls documents an upper
  bound on the billed calls one read makes, such as its most polls times the billed calls per poll,
  so a budget sized with it covers invocations, retries and reads. CloudWatch Logs Insights queries
  (``logs:StartQuery``) are not in that list today, so they spend none of the budget, although Logs
  Insights bills each query by the data it scans.
- **Guard refusals stay visible.** A refusal from a guard on those clients, such as
  :class:`~calibrated_agent_evaluation.agentcore.errors.BilledCallBudgetExceeded`, propagates as it
  is, or chained as the cause of the reader's own error, so the executor can name the guard and the
  operation.

The executor checks that some spans came back and that each carries the run's trace ID; trace
validation then checks the tree's shape. A run that fails these checks, or whose reader raises,
returns no trace. Its invocation was already sent, and nothing reads a finished run's spans again
yet, so running the task again is a new billed invocation.

The AgentCore per-run span reader that implements this contract is
``calibrated_agent_evaluation.agentcore.AgentCoreSpanReader``, built as
``AgentCoreSpanReader(clients)``; the window ingest
(``calibrated-agent-evaluation ingest --source agentcore``) is a separate reader for a time window.
Offline runs replay a cassette and need no reader.
"""

from __future__ import annotations

from typing import ClassVar, Protocol, runtime_checkable

from pydantic import ConfigDict

from calibrated_agent_evaluation._model import CalibratedAgentEvaluationModel
from calibrated_agent_evaluation.agentcore.runtime_errors import RuntimeInvocationError
from calibrated_agent_evaluation.schema.types import Span

__all__ = ["RunSpanQuery", "RunSpanReader"]


class RunSpanQuery(CalibratedAgentEvaluationModel):
    """The run whose spans to read, named by the IDs the executor generated for it.

    Attributes:
        trace_id: The Calibrated Agent Evaluation trace ID of the run's ``Trace``; every span
            returned must carry it.
        w3c_trace_id: The W3C trace ID every invocation of the run carried in ``traceparent``,
            which the executor always marks sampled (flag ``01``): the runtime records no spans for
            an unsampled trace, so such a run could never be read.
        session_id: The runtime session ID every invocation of the run was sent with.
        runtime_arn: The runtime invoked.
        qualifier: The endpoint invoked.
        started_at: When the invocation started: ISO-8601 in UTC, with its offset.
        ended_at: When the invocation ended, in the same form.
    """

    model_config = ConfigDict(frozen=True, extra="forbid", defer_build=True)
    invalid_error: ClassVar[type[Exception]] = RuntimeInvocationError

    trace_id: str
    w3c_trace_id: str
    session_id: str
    runtime_arn: str
    qualifier: str
    started_at: str
    ended_at: str


@runtime_checkable
class RunSpanReader(Protocol):
    """Reads the span tree of one AgentCore Runtime run, under the contract in this module."""

    def read_spans(self, query: RunSpanQuery) -> list[Span]:
        """The run's whole span tree, every span keyed to ``query.trace_id``.

        Args:
            query: The run to read.

        Returns:
            The run's spans: never empty, never partial.

        Raises:
            AgentCoreError: A typed subclass when the tree can't be read whole, or a guard's
                refusal, as it is or chained.
        """
        ...
