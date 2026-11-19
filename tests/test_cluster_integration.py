# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""End-to-end integration test: embed → cluster → ratify → saturation → coverage."""

from __future__ import annotations

from calibrated_agent_evaluation.cluster import (
    build_draft,
    coverage_report,
    hierarchical_cluster,
    ratify,
    saturation_check,
    tag_label,
)
from calibrated_agent_evaluation.sampling.types import TaxonomyGrid
from calibrated_agent_evaluation.schema.types import Label
from test_cluster_helpers import FakeEmbedder


class TestEndToEnd:
    """Full pipeline composition proof."""

    def test_full_pipeline(self) -> None:
        # 1. Generate critiques across different failure patterns
        critiques = (
            [f"Agent ignored the policy tool output in case {i}" for i in range(15)]
            + [f"Agent calculated wrong amount: {10 + i}" for i in range(10)]
            + [f"Agent hallucinated entity ID-{i}" for i in range(8)]
            + [f"Agent timed out waiting for response {i}" for i in range(5)]
            + [f"Unique failure pattern {i}" for i in range(2)]
        )
        assert len(critiques) == 40

        # 2. Embed
        embedder = FakeEmbedder()
        embeddings = embedder.embed(critiques)
        assert len(embeddings) == 40

        # 3. Cluster
        assignments = hierarchical_cluster(embeddings, threshold=0.5)
        n_clusters = len(set(assignments))
        assert n_clusters >= 2

        # 4. Build draft
        draft = build_draft(critiques, embeddings, assignments)
        assert len(draft.clusters) >= 2
        total_members = sum(len(c.member_indices) for c in draft.clusters)
        assert total_members == 40

        # 5. Ratify
        codebook = ratify(draft, ratified_by="integration-test", version="1")
        assert codebook.ratified_by == "integration-test"

        # 6. Saturation check
        from collections import Counter

        mode_counts = list(Counter(assignments).values())
        result = saturation_check(mode_counts)
        assert result.total_failures == 40
        assert result.total_modes == n_clusters
        # With 40 observations, we meet the floor

        # 7. Coverage report
        grid = TaxonomyGrid(
            dimensions={
                "feature": ["refund", "cancel"],
                "scenario": ["happy", "edge"],
            }
        )
        observations = [
            {"feature": "refund", "scenario": "happy"},
            {"feature": "refund", "scenario": "edge"},
            {"feature": "cancel", "scenario": "happy"},
            # cancel/edge is a blind spot
        ]
        cov = coverage_report(grid, observations)
        assert cov.total_cells == 4
        assert cov.covered_cells == 3
        assert len(cov.blind_spots) == 1

        # 8. Tag a label
        label = Label(
            trace_id="test-1",
            span_id=None,
            kind="human",
            outcome="FAIL",
            critique="test critique",
        )
        mode_name = codebook.modes[0].name
        tagged = tag_label(label, mode_name, codebook)
        assert tagged.failure_mode == mode_name
        assert tagged.codebook_version == "1"
