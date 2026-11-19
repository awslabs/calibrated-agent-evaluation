# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for codebook ratification and label tagging."""

from __future__ import annotations

import pytest

from calibrated_agent_evaluation.cluster import (
    ClusteringError,
    Codebook,
    DraftCodebook,
    UnratifiedCodebookError,
    ratify,
    tag_label,
)
from calibrated_agent_evaluation.cluster.types import DraftCluster
from calibrated_agent_evaluation.schema.types import Label


def _make_draft(n_clusters: int = 5) -> DraftCodebook:
    clusters = tuple(
        DraftCluster(
            name=f"mode_{i}",
            member_indices=(i,),
            member_critiques=(f"critique {i}",),
        )
        for i in range(n_clusters)
    )
    return DraftCodebook(clusters=clusters)


def _draft_named(*names: str) -> DraftCodebook:
    return DraftCodebook(
        clusters=tuple(
            DraftCluster(
                name=name,
                member_indices=(i,),
                member_critiques=(f"critique of {name}",),
            )
            for i, name in enumerate(names)
        )
    )


def _make_label() -> Label:
    return Label(
        trace_id="test-trace",
        span_id=None,
        kind="human",
        outcome="FAIL",
        critique="Agent did something wrong",
    )


# --- ratify with merge/rename ---


class TestRatify:
    """Ratify transforms a DraftCodebook into a Codebook."""

    def test_merge_and_rename(self) -> None:
        draft = _make_draft(5)
        codebook = ratify(
            draft,
            ratified_by="alice",
            merges={"merged_mode": ["mode_0", "mode_1"]},
            renames={"mode_2": "policy-ignored"},
            version="1",
        )
        assert isinstance(codebook, Codebook)
        assert codebook.ratified_by == "alice"
        assert codebook.version == "1"
        mode_names = [m.name for m in codebook.modes]
        assert "merged_mode" in mode_names
        assert "policy-ignored" in mode_names
        assert "mode_0" not in mode_names
        assert "mode_1" not in mode_names
        assert "mode_2" not in mode_names
        # 5 original - 2 merged into 1 - 1 renamed = 4 modes
        assert len(codebook.modes) == 4

    # --- version propagates ---

    def test_version_set(self) -> None:
        draft = _make_draft(3)
        codebook = ratify(draft, ratified_by="bob", version="42")
        assert codebook.version == "42"

    # --- (negative) — DraftCodebook has no ratified_by ---

    def test_draft_has_no_ratified_by(self) -> None:
        draft = _make_draft(3)
        with pytest.raises(AttributeError):
            _ = draft.ratified_by  # type: ignore[attr-defined]

    # --- (negative) — version <= parent_version ---

    def test_version_monotonicity_enforced(self) -> None:
        draft = _make_draft(3)
        with pytest.raises(ClusteringError, match="monotonicity"):
            ratify(draft, ratified_by="x", version="1", parent_version="2")

    def test_version_equal_to_parent_rejected(self) -> None:
        draft = _make_draft(3)
        with pytest.raises(ClusteringError, match="monotonicity"):
            ratify(draft, ratified_by="x", version="1", parent_version="1")

    def test_splits_are_refused_naming_the_way_to_split(self) -> None:
        """Ratify never splits; the refusal names re-clustering instead."""
        draft = _make_draft(3)
        with pytest.raises(ClusteringError, match="lower threshold"):
            ratify(draft, ratified_by="x", splits={"mode_0": [[0], [1]]}, version="1")

    def test_version_after_parent_accepted(self) -> None:
        draft = _make_draft(3)
        codebook = ratify(draft, ratified_by="x", version="2", parent_version="1")
        assert codebook.version == "2"
        assert codebook.parent_version == "1"

    def test_numeric_version_bump_across_digit_boundary_accepted(self) -> None:
        # a normal sequential bump 9 -> 10 must be ACCEPTED. A pure lexicographic string
        # compare gets this backwards ("10" < "9"), spuriously refusing a legitimate bump.
        draft = _make_draft(3)
        codebook = ratify(draft, ratified_by="x", version="10", parent_version="9")
        assert codebook.version == "10"

    def test_numeric_version_regression_across_digit_boundary_refused(self) -> None:
        # (negative): 10 -> 9 is a regression and must be REFUSED — a lexicographic compare
        # would silently admit it ("9" > "10"), letting a codebook regress.
        draft = _make_draft(3)
        with pytest.raises(ClusteringError, match="monotonicity"):
            ratify(draft, ratified_by="x", version="9", parent_version="10")

    def test_ratify_attaches_descriptions_by_final_mode_name(self) -> None:
        """Definitions land on the mode that ends up with the vocabulary name, after merges and
        renames; every other mode keeps its description, and no descriptions changes nothing."""
        draft = _make_draft(5)
        merges = {"tool-misuse": ["mode_0", "mode_1"]}
        renames = {"mode_2": "hallucination", "mode_3": "policy-ignored"}
        # The whole vocabulary is passed, as a caller would; "other" names no mode here.
        descriptions = {
            "tool-misuse": "Calls a tool the task does not need.",
            "hallucination": "States a fact no source supports.",
            "other": "Fits no category.",
        }

        described = ratify(
            draft,
            ratified_by="alice",
            merges=merges,
            renames=renames,
            descriptions=descriptions,
        )
        plain = ratify(draft, ratified_by="alice", merges=merges, renames=renames)

        assert {m.name: m.description for m in described.modes} == {
            "mode_4": "",
            "tool-misuse": "Calls a tool the task does not need.",
            "hallucination": "States a fact no source supports.",
            "policy-ignored": "",
        }
        assert {m.name: m.description for m in plain.modes} == {
            "mode_4": "",
            "tool-misuse": "",
            "hallucination": "",
            "policy-ignored": "",
        }
        assert [(m.name, m.example_critiques) for m in plain.modes] == [
            (m.name, m.example_critiques) for m in described.modes
        ]

    def test_ratify_refuses_a_draft_with_repeated_cluster_names(self) -> None:
        """Two draft clusters with one name are refused, not collapsed into the last of them."""
        draft = _draft_named("x", "y", "x")

        with pytest.raises(ClusteringError, match="'x'") as raised:
            ratify(draft, ratified_by="alice")

        assert "merges=" in str(raised.value)
        assert "'y'" not in str(raised.value)

    def test_a_merge_or_rename_onto_another_modes_name_is_refused(self) -> None:
        """A merge or rename onto a name a surviving mode holds would silently replace that mode;
        it is refused, and listing both modes in the merge combines them."""
        draft = _draft_named("x", "x #2")

        with pytest.raises(ClusteringError, match="'x'") as renamed:
            ratify(draft, ratified_by="alice", renames={"x #2": "x"})
        assert "merges=" in str(renamed.value)

        with pytest.raises(ClusteringError, match="'x'") as merged:
            ratify(draft, ratified_by="alice", merges={"x": ["x #2"]})
        assert "merges=" in str(merged.value)

        codebook = ratify(draft, ratified_by="alice", merges={"x": ["x", "x #2"]})
        assert [(m.name, m.example_critiques) for m in codebook.modes] == [
            ("x", ("critique of x", "critique of x #2"))
        ]


# --- tag_label with valid mode ---


class TestTagLabel:
    """tag_label sets failure_mode and codebook_version."""

    def test_valid_tag(self) -> None:
        draft = _make_draft(3)
        codebook = ratify(draft, ratified_by="alice", version="1")
        label = _make_label()
        tagged = tag_label(label, "mode_0", codebook)
        assert tagged.failure_mode == "mode_0"
        assert tagged.codebook_version == "1"
        # Original unchanged (frozen)
        assert label.failure_mode is None

    # --- (negative) — unratified draft → error ---

    def test_unratified_draft_raises(self) -> None:
        draft = _make_draft(3)
        label = _make_label()
        with pytest.raises(UnratifiedCodebookError, match="ratif"):
            tag_label(label, "mode_0", draft)

    # --- (negative) — mode not in codebook ---

    def test_mode_not_found_raises(self) -> None:
        draft = _make_draft(3)
        codebook = ratify(draft, ratified_by="alice", version="1")
        label = _make_label()
        with pytest.raises(ClusteringError, match="not found"):
            tag_label(label, "nonexistent_mode", codebook)

    def test_version_propagates_to_label(self) -> None:
        """Complementary: codebook_version on label matches codebook.version."""
        draft = _make_draft(3)
        codebook = ratify(draft, ratified_by="bob", version="7")
        label = _make_label()
        tagged = tag_label(label, "mode_0", codebook)
        assert tagged.codebook_version == "7"
