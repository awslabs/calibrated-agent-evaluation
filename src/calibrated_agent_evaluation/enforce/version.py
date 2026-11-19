# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Schema version for enforcement artifacts (leaf module -- no intra-package deps).

Stamped on every :class:`~calibrated_agent_evaluation.enforce.judgedef.JudgeDefinition` and
:class:`~calibrated_agent_evaluation.enforce.decision.EnforcementDecision`; the store refuses to
load an artifact whose ``schema_version`` differs from the running value (no coercion).
"""

from __future__ import annotations

ENFORCE_SCHEMA_VERSION = "1"
