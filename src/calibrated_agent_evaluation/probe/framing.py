# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Limits-of-inference framing.

Attached as STRUCTURED fields to every probing output (never prose alone) so a reader -- or a
downstream consumer -- cannot skip the caveats: the tool inventory is a lower bound, a refusal is
not an inability, and every verdict is a judge output bounded by the judge-reliability ceiling.
"""

from __future__ import annotations

from typing import ClassVar

from pydantic import ConfigDict

from calibrated_agent_evaluation._model import CalibratedAgentEvaluationModel
from calibrated_agent_evaluation.probe.errors import ProbeError


class LimitsFraming(CalibratedAgentEvaluationModel):
    """The three mandatory limits-of-inference flags on every probing output."""

    model_config = ConfigDict(frozen=True, extra="forbid")
    invalid_error: ClassVar[type[Exception]] = ProbeError

    tool_inventory_is_lower_bound: bool = True
    refusal_is_not_inability: bool = True
    verdict_bounded_by_judge_ceiling: bool = True
