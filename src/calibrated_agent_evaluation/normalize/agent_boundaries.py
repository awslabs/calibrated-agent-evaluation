# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Which spans are agent steps, beyond the standard ``invoke_agent`` boundary, per vendor family.

The GenAI convention marks an agent step with ``gen_ai.operation.name = invoke_agent``, and the
generic reader builds one node per such span. Not every instrumentor puts the boundary there.
Traceloop's LangChain instrumentor names a whole compiled LangGraph ``invoke_agent LangGraph`` and
emits every graph *node* (the crew's actual roles: a planner, a writer, a reviewer) as an
``execute_task`` span. Read by the standard rule alone, gpt-researcher's seven-role LangGraph crew
normalized to three agents all named ``LangGraph``, none carrying output, and was refused. The
roles and their content were on the wire the whole time.

So a vendor family registers a rule naming the extra spans that are agent steps in its dialect, the
same way it registers content placements
(:mod:`~calibrated_agent_evaluation.normalize.content_placements`), and the convention-neutral
reader asks the registry without naming any vendor. Rules only *add* agent boundaries: a span the
standard rule already treats as an agent stays one, and no rule can remove it. A rule returns the
agent name for a span it claims, or ``None``.

Shared-mutable-state contract mirrors the placement registry: built-ins are registered once at
import, re-registering a family is refused, and the vendor modules are imported on first use so a
reader that never imported the convention package still sees their rules.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from calibrated_agent_evaluation.normalize.errors import NormalizationError

#: Given one OTLP-JSON span, the agent name if this family treats it as an agent step, else None.
BoundaryRule = Callable[[dict[str, Any]], str | None]

_RULES: dict[str, BoundaryRule] = {}


def register_agent_boundary(family: str, rule: BoundaryRule) -> None:
    """Register ``family``'s agent-boundary rule. Refuses a second registration under one name."""
    if family in _RULES:
        raise NormalizationError(
            f"an agent-boundary rule is already registered for {family!r}; choose a distinct name"
        )
    _RULES[family] = rule


def _ensure_vendor_families() -> None:
    """Import the shipped convention modules so their rules are registered.

    See the placements registry's twin for why the registry, not its callers, owns this.
    """
    from calibrated_agent_evaluation.normalize.conventions import (  # noqa: F401 - imported for the side effect
        openllmetry_adapter,
    )


def boundary_name(span: dict[str, Any]) -> str | None:
    """The agent name the first registered rule claiming ``span`` gives it, else ``None``."""
    _ensure_vendor_families()
    for rule in _RULES.values():
        name = rule(span)
        if name:
            return name
    return None
