# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Typed errors for test scenarios, dataset slices and answer keys."""

from __future__ import annotations

from calibrated_agent_evaluation._errors import CalibratedAgentEvaluationDataError

__all__ = ["ScenarioError"]


class ScenarioError(CalibratedAgentEvaluationDataError):
    """A scenario, slice file or answer key is invalid, unreadable, or can't be built as asked."""
