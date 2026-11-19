# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Call guards for AWS clients: a read-only allowlist and an optional billed-call budget.

Both are botocore ``before-call`` handlers for
:class:`~calibrated_agent_evaluation.agentcore.session.AwsClients`. Raising from one aborts the call
before any request is sent.
"""

from __future__ import annotations

import threading
from collections.abc import Iterable, Mapping
from typing import Any

from calibrated_agent_evaluation.agentcore.errors import (
    AgentCoreError,
    BilledCallBudgetExceeded,
    ReadOnlyViolation,
)

__all__ = [
    "BILLED_OPERATIONS",
    "BUDGET_ENV_VAR",
    "BilledCallBudget",
    "Operation",
    "ReadOnlyGuard",
]

BUDGET_ENV_VAR = "CALIBRATED_AGENT_EVALUATION_AGENTCORE_MAX_BILLED_CALLS"

Operation = tuple[str, str]
"""``(client service name, operation name)``, e.g. ``("xray", "GetIndexingRules")``."""

# Each call to these is charged. New billed calls an AgentCore feature makes are added here.
BILLED_OPERATIONS: frozenset[Operation] = frozenset(
    {
        ("bedrock-agentcore", "Evaluate"),
        ("bedrock-agentcore", "StartBatchEvaluation"),
        ("bedrock-agentcore", "InvokeAgentRuntime"),
        ("bedrock-agentcore", "InvokeCodeInterpreter"),
        ("cloudwatchomni", "StartTelemetryQuery"),
    }
)


def _operation(kwargs: dict[str, Any]) -> Operation:
    model = kwargs["model"]
    return (model.service_model.service_name, model.name)


class ReadOnlyGuard:
    """Refuses every operation not in ``allowed``."""

    def __init__(self, allowed: Iterable[Operation]) -> None:
        self.allowed = frozenset(allowed)

    def __call__(self, **kwargs: Any) -> None:
        """Raise :class:`ReadOnlyViolation` unless the call is on the allowlist."""
        operation = _operation(kwargs)
        if operation not in self.allowed:
            service, name = operation
            raise ReadOnlyViolation(
                f"refused {service}:{name}: this command only makes the calls it lists",
                operation=operation,
            )


class BilledCallBudget:
    """Counts billed operations and refuses the first one past ``limit``; no limit when ``None``."""

    def __init__(self, limit: int | None = None) -> None:
        if limit is not None and limit < 0:
            raise ValueError(f"a billed-call budget can't be negative (got {limit})")
        self.limit = limit
        self.used = 0
        # Charged from every thread a run sends calls on, so the check and the count are one step.
        self._lock = threading.Lock()

    @classmethod
    def from_env(cls, environ: Mapping[str, str]) -> BilledCallBudget:
        """The budget the environment sets; no limit when unset.

        The variable is ``CALIBRATED_AGENT_EVALUATION_AGENTCORE_MAX_BILLED_CALLS``.
        """
        raw = environ.get(BUDGET_ENV_VAR, "").strip()
        if not raw:
            return cls()
        try:
            return cls(int(raw))
        except ValueError as exc:
            raise AgentCoreError(
                f"{BUDGET_ENV_VAR}={raw!r} is not a non-negative whole number"
            ) from exc

    def __call__(self, **kwargs: Any) -> None:
        """Count a billed call, or raise :class:`BilledCallBudgetExceeded` past the limit.

        The botocore ``before-call`` entry point; delegates to :meth:`charge`.
        """
        self.charge(_operation(kwargs))

    @property
    def remaining(self) -> int | None:
        """Billed calls left before the limit, or ``None`` with no limit; charges nothing."""
        return None if self.limit is None else max(self.limit - self.used, 0)

    def charge(self, operation: Operation) -> None:
        """Count one ``operation`` against the budget, or refuse it past the limit.

        The explicit entry point for callers that do not go through a botocore ``before-call``
        handler (an adapter that builds its own client, e.g. the Code Interpreter sandbox). A
        non-billed operation is ignored; a billed one past ``limit`` raises before it is sent.
        """
        if operation not in BILLED_OPERATIONS:
            return
        with self._lock:
            if self.limit is not None and self.used >= self.limit:
                service, name = operation
                raise BilledCallBudgetExceeded(
                    f"refused {service}:{name}: the billed-call budget of {self.limit} is used up "
                    "(raise CALIBRATED_AGENT_EVALUATION_AGENTCORE_MAX_BILLED_CALLS or unset it to "
                    "remove the limit)",
                    operation=operation,
                )
            self.used += 1
