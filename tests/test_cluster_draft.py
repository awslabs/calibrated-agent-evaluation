# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for build_draft."""

from __future__ import annotations

import pytest

from calibrated_agent_evaluation.cluster import build_draft
from calibrated_agent_evaluation.cluster.errors import ClusteringError

# --- build_draft produces valid DraftCodebook ---


class TestBuildDraft:
    """build_draft produces complete, non-overlapping clusters."""

    def test_all_indices_accounted_for(self) -> None:
        critiques = [f"critique {i}" for i in range(20)]
        embeddings = [[float(i)] * 4 for i in range(20)]
        assignments = [i % 4 for i in range(20)]  # 4 clusters

        draft = build_draft(critiques, embeddings, assignments)

        assert len(draft.clusters) == 4
        all_indices: set[int] = set()
        for cluster in draft.clusters:
            assert len(cluster.member_indices) >= 1
            assert len(cluster.member_critiques) >= 1
            all_indices.update(cluster.member_indices)

        assert all_indices == set(range(20))

    def test_auto_generated_names(self) -> None:
        critiques = ["a", "b", "c", "d"]
        embeddings = [[1.0]] * 4
        assignments = [0, 0, 1, 1]

        draft = build_draft(critiques, embeddings, assignments)

        names = [c.name for c in draft.clusters]
        assert names == ["mode_0", "mode_1"]

    def test_custom_namer(self) -> None:
        class TestNamer:
            def name_clusters(self, clusters: list[list[str]]) -> list[str]:
                return [f"custom_{i}" for i in range(len(clusters))]

        critiques = ["a", "b", "c"]
        embeddings = [[1.0]] * 3
        assignments = [0, 1, 1]

        draft = build_draft(critiques, embeddings, assignments, namer=TestNamer())

        names = [c.name for c in draft.clusters]
        assert "custom_0" in names
        assert "custom_1" in names

    def test_namer_input_aligns_with_members(self) -> None:
        """The name a namer proposes for a cluster must land on THAT cluster's members.

        scipy's ``fcluster`` labels are not in first-appearance order, so build_draft must feed
        the namer clusters in the same order it assembles DraftClusters. This namer derives each
        name from the cluster's own member text, so any misalignment is directly observable.
        """

        class ContentNamer:
            def name_clusters(self, clusters: list[list[str]]) -> list[str]:
                return [f"named_{members[0]}" for members in clusters]

        # Labels deliberately out of sorted order (2, 0, 1) — mimics real scipy output.
        critiques = ["belongs_to_2", "belongs_to_0", "belongs_to_1"]
        embeddings = [[1.0]] * 3
        assignments = [2, 0, 1]

        draft = build_draft(critiques, embeddings, assignments, namer=ContentNamer())

        for cluster in draft.clusters:
            # e.g. name "named_belongs_to_0" must own the member critique "belongs_to_0".
            assert cluster.name == f"named_{cluster.member_critiques[0]}"

    def test_namer_returning_too_few_names_is_refused(self) -> None:
        """A namer that under-returns must NOT silently drop the unnamed clusters (and
        every critique in them) — the whole point of is to not lose failure modes."""

        class UnderReturningNamer:
            def name_clusters(self, clusters: list[list[str]]) -> list[str]:
                return ["only_one"]  # 1 name for however many clusters there are

        critiques = ["a", "b", "c", "d"]
        embeddings = [[1.0]] * 4
        assignments = [0, 1, 2, 3]  # 4 clusters, 1 name → would drop 3 clusters under lenient zip

        with pytest.raises(ClusteringError, match="every cluster must be named"):
            build_draft(critiques, embeddings, assignments, namer=UnderReturningNamer())

    def test_a_namer_that_repeats_a_name_is_refused(self) -> None:
        """Two clusters given one name would collapse into one mode at ratification, so the
        draft is refused, naming the repeat; distinct names still build it."""

        class ScriptedNamer:
            def __init__(self, names: list[str]) -> None:
                self.names = names

            def name_clusters(self, clusters: list[list[str]]) -> list[str]:
                return self.names

        critiques = ["a", "b", "c"]
        embeddings = [[1.0]] * 3
        assignments = [0, 1, 1]

        with pytest.raises(ClusteringError, match="'tool-misuse'"):
            build_draft(
                critiques,
                embeddings,
                assignments,
                namer=ScriptedNamer(["tool-misuse", "tool-misuse"]),
            )

        draft = build_draft(
            critiques,
            embeddings,
            assignments,
            namer=ScriptedNamer(["tool-misuse", "hallucination"]),
        )
        assert [c.name for c in draft.clusters] == ["tool-misuse", "hallucination"]
        assert [c.member_critiques for c in draft.clusters] == [("a",), ("b", "c")]
