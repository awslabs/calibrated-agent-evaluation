# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Embedding and naming protocols for the clustering pipeline.

The caller injects both. ``Embedder`` has a concrete implementation in the execute plane
(``calibrated_agent_evaluation.execute.bedrock_embedder.BedrockEmbedder``, the ``[bedrock]`` extra).
``DraftNamer`` has one in the AgentCore plug-in:
``calibrated_agent_evaluation.agentcore.DetectorDraftNamer`` (the ``[agentcore]`` extra) names
clusters from the ``strands_evals`` failure detector's category vocabulary. This module does not
import it, so clustering stays vendor-free. Without a namer, ``build_draft`` names clusters
``mode_0``, ``mode_1``, and so on.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable


@runtime_checkable
class Embedder(Protocol):
    """Injectable embedding interface."""

    def embed(self, texts: list[str]) -> list[list[float]]:
        """Convert texts to embedding vectors."""
        ...


@runtime_checkable
class DraftNamer(Protocol):
    """Injectable cluster-naming interface.

    Optional; if not provided, clusters get auto-generated names. The shipped implementation is
    ``calibrated_agent_evaluation.agentcore.DetectorDraftNamer`` (the ``[agentcore]`` extra), which
    satisfies this protocol structurally; any object with a matching ``name_clusters`` works.
    """

    def name_clusters(self, clusters: list[list[str]]) -> list[str]:
        """Propose a name for each cluster given its member critiques."""
        ...
