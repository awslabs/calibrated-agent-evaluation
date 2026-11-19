# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Failure detection and draft naming over the ``strands-agents-evals`` failure detector.

The one sub-package of the AgentCore plug-in that imports the ``strands_evals`` and ``strands``
SDKs. Its names are reached through :mod:`calibrated_agent_evaluation.agentcore`'s lazy loader,
which raises :class:`~calibrated_agent_evaluation.agentcore.AgentCoreExtraMissing` when the
``[agentcore]`` extra is absent. This file imports nothing, so importing the vendor-free ``types``
module loads no vendor code.
"""
