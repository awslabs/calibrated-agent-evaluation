# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Capability-discovery output types.

The capability map is deliberately a *lower bound*: it lists only tools the agent was observed
USING (self-report is unreliable), carrying :class:`LimitsFraming` so no reader mistakes it for a
complete inventory.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from calibrated_agent_evaluation.probe.framing import LimitsFraming


@dataclass(frozen=True)
class FailureHypothesis:
    """A candidate failure the probe surfaced, with the trace(s) that evidence it."""

    description: str
    evidence_trace_ids: tuple[str, ...] = ()


@dataclass(frozen=True)
class CapabilityMap:
    """What probing observed about an unowned agent -- a lower bound, not an inventory.

    ``observed_tools`` are tools seen USED in a probe response; ``refusal_boundaries`` are prompts
    the agent declined. ``framing`` is mandatory so the lower-bound nature is structural.
    """

    observed_tools: tuple[str, ...]
    refusal_boundaries: tuple[str, ...]
    framing: LimitsFraming = field(default_factory=LimitsFraming)
