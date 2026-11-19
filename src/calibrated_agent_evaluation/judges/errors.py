# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Typed error hierarchy for the judge-level layer.

Every judge-level guard is a typed, explained refusal (conventions: "no silent failure").
All errors subclass
:class:`~calibrated_agent_evaluation.schema.errors.CalibratedAgentEvaluationDataError` so callers
can catch the whole data-plane family at the boundary.
"""

from __future__ import annotations

from calibrated_agent_evaluation.schema.errors import CalibratedAgentEvaluationDataError


class SpanGoldError(CalibratedAgentEvaluationDataError):
    """A span-level judge was calibrated on gold derived from the trace outcome, not the span."""


class JudgeLevelError(CalibratedAgentEvaluationDataError):
    """A workflow-level judge was used for a per-agent rate, which it can't attribute."""


class JudgeRuntimeError(CalibratedAgentEvaluationDataError):
    """A failure in the judge-execution runtime.

    Covers a model call that failed after exhausting retries, a reply that carried neither a
    schema-valid structured output nor a valid forced-tool-call payload, and a structured payload
    that violated the verdict schema (empty ``critique`` / non-binary ``verdict``). Wraps the
    underlying vendor/``execute`` exception via chaining (``raise JudgeRuntimeError(...) from exc``)
    so no raw ``inspect_ai``/boto3/stdlib exception escapes the ``judges/`` boundary
    (conventions: "wrap external errors at the boundary"). The runtime never fabricates a verdict:
    on any of these it raises rather than persisting a ``Label``.
    """
