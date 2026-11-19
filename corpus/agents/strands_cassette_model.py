# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Strands cassette Model — re-exports from
calibrated_agent_evaluation.execute.strands_cassette_model.

``CassetteModel`` was promoted to
``src/calibrated_agent_evaluation/execute/strands_cassette_model.py`` so a pip-installed user can
run ``StrandsExecutor`` with offline replay. This module re-exports it so existing corpus reference
agents continue to work unchanged.
"""

from calibrated_agent_evaluation.execute.strands_cassette_model import CassetteModel

__all__ = ["CassetteModel"]
