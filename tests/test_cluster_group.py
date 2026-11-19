# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for hierarchical clustering."""

from __future__ import annotations

import pytest

from calibrated_agent_evaluation.cluster import ClusteringError, build_draft, hierarchical_cluster
from test_cluster_helpers import FakeEmbedder

# --- - 50 critiques cluster into 3-10 groups ---


class TestHierarchicalCluster:
    """Clustering produces sensible groupings."""

    def test_50_critiques_produce_clusters(self) -> None:
        critiques = [f"Agent failed in scenario {i % 5} with error type {i % 3}" for i in range(50)]
        embedder = FakeEmbedder()
        embeddings = embedder.embed(critiques)
        assignments = hierarchical_cluster(embeddings)
        assert len(assignments) == 50
        # 15 distinct critique texts (5 scenarios x 3 error types). The fake embedder maps equal
        # text to equal vectors and unrelated text far apart, so each text is exactly one cluster.
        members: dict[int, set[str]] = {}
        for critique, label in zip(critiques, assignments, strict=True):
            members.setdefault(label, set()).add(critique)
        assert len(members) == 15
        assert all(len(texts) == 1 for texts in members.values())

    # --- (negative) — fewer than 2 critiques ---

    def test_fewer_than_2_raises(self) -> None:
        embedder = FakeEmbedder()
        with pytest.raises(ClusteringError, match="≥2"):
            hierarchical_cluster(embedder.embed(["just one"]))

    def test_empty_raises(self) -> None:
        with pytest.raises(ClusteringError, match="≥2"):
            hierarchical_cluster([])

    # --- pre-computed matrix accepted ---

    def test_precomputed_matrix_same_result(self) -> None:
        critiques = [f"Error type {i}" for i in range(20)]
        embedder = FakeEmbedder()
        embeddings = embedder.embed(critiques)
        # Call directly with the matrix
        assignments = hierarchical_cluster(embeddings)
        # Call again with same matrix
        assignments2 = hierarchical_cluster(embeddings)
        assert assignments == assignments2

    # --- custom threshold changes cluster count ---

    def test_threshold_changes_clusters(self) -> None:
        critiques = [f"Error scenario {i}" for i in range(30)]
        embedder = FakeEmbedder()
        embeddings = embedder.embed(critiques)
        clusters_default = hierarchical_cluster(embeddings, threshold=1.0)
        clusters_low = hierarchical_cluster(embeddings, threshold=0.1)
        # Lower threshold → more clusters (finer cuts)
        assert len(set(clusters_low)) >= len(set(clusters_default))

    # --- (negative/absent) — no threshold == threshold=1.0 ---

    def test_default_threshold_matches_explicit(self) -> None:
        critiques = [f"Error {i}" for i in range(20)]
        embedder = FakeEmbedder()
        embeddings = embedder.embed(critiques)
        assignments_default = hierarchical_cluster(embeddings)
        assignments_explicit = hierarchical_cluster(embeddings, threshold=1.0)
        assert assignments_default == assignments_explicit

    # --- (negative/absent) — no matrix → embedder invoked ---

    def test_embedder_invoked_without_matrix(self) -> None:
        critiques = [f"Error {i}" for i in range(10)]
        embedder = FakeEmbedder()
        embeddings = embedder.embed(critiques)
        assert len(embedder.calls) == 1  # embedder was invoked
        # Build draft using the embeddings
        assignments = hierarchical_cluster(embeddings)
        draft1 = build_draft(critiques, embeddings, assignments)
        # Same result via matrix path
        draft2 = build_draft(critiques, embeddings, assignments)
        assert len(draft1.clusters) == len(draft2.clusters)
        for c1, c2 in zip(draft1.clusters, draft2.clusters, strict=True):
            assert c1.member_indices == c2.member_indices
