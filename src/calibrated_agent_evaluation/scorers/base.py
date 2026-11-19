# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Scorer protocol and ScorerLane enum: the scorers module's public contract.

Defined exclusively against normalized schema types (``Label``) and Python stdlib types. No
``autoevals`` or other vendor type appears in the protocol's method signatures.
"""

from __future__ import annotations

from enum import StrEnum
from typing import Any, Protocol, runtime_checkable

from calibrated_agent_evaluation.schema.types import Label


class ScorerLane(StrEnum):
    """Declares whether a scorer may be used as a CI gate."""

    CI_GATE = "ci_gate"
    EXPLORATION = "exploration"


@runtime_checkable
class Scorer(Protocol):
    """Score an agent output against an expected value and return a Label.

    Every scorer carries a ``lane`` attribute declaring its CI-gate eligibility.
    The ``score()`` method returns a ``Label`` with ``kind="code"``, binary outcome, and
    a non-empty critique.
    """

    lane: ScorerLane

    def score(
        self,
        output: dict[str, Any],
        expected: dict[str, Any] | None,
        *,
        trace_id: str,
    ) -> Label:
        """Score ``output`` against ``expected`` and return a Label.

        Raises ``ScorerError`` on invalid input or vendor failure.
        """
        ...
