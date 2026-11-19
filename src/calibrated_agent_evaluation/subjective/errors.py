# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Typed error hierarchy for the subjective-outcome module."""

from __future__ import annotations

from calibrated_agent_evaluation.schema.errors import CalibratedAgentEvaluationDataError


class SubjectiveError(CalibratedAgentEvaluationDataError):
    """Invalid input or state in the subjective-outcome module."""


class SubjectiveGateError(SubjectiveError):
    """Attempt to gate a CI decision on a subjective dimension."""


class SubjectiveStoreError(SubjectiveError):
    """A preference record file could not be read or written, or holds a malformed record."""
