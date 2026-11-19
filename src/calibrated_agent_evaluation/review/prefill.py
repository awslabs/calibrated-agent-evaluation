# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Optional LLM pre-fill seam for the review surface.

A ``Prefill`` proposes a verdict + critique for a trace; the human MUST explicitly accept or
override it before a label is written (an LLM may draft, a human ratifies). The
default is a no-op, so the surface runs with zero LLM dependency (offline) and the
assist is strictly opt-in.

No ``Prefill`` implementation ships. A library caller may pass one to ``build_app(prefill=...)``;
``calibrated-agent-evaluation label`` has none (amended 2026-10-13). The first implementation is
planned as a library-only pre-fill from ``strands_evals`` detector output.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable

from calibrated_agent_evaluation.schema import Outcome, Trace


@dataclass(frozen=True)
class Suggestion:
    """An LLM proposal for a trace. NEVER a label until a human ratifies it."""

    outcome: Outcome
    critique: str


@runtime_checkable
class Prefill(Protocol):
    """Proposes a ``Suggestion`` for a trace, or ``None`` when it has no proposal."""

    def suggest(self, trace: Trace) -> Suggestion | None:
        """Return a pre-fill suggestion for ``trace``, or ``None`` to leave the form blank."""
        ...


class NoPrefill:
    """The default: no LLM assist. Every trace is labeled from scratch by the human."""

    def suggest(self, trace: Trace) -> Suggestion | None:  # noqa: ARG002 - protocol shape
        """Return ``None``: no suggestion (the offline, zero-dependency default)."""
        return None
