# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""`calibrated-agent-evaluation cluster` reports saturation and coverage over the human FAIL
labels, and drafts failure modes when given an embedder.

The embedder is the one injection the tests replace (Bedrock is a paid call); saturation,
coverage, clustering and the draft run on the shipped engines.
"""

from __future__ import annotations

import hashlib
import json

from calibrated_agent_evaluation.cli import main
from calibrated_agent_evaluation.cli.config import Degradation
from calibrated_agent_evaluation.cli.dispatch import EXIT_DOMAIN_ERROR, EXIT_OK


class _HashEmbedder:
    """Deterministic, offline: similar strings do not embed close, but the draft is reproducible."""

    def embed(self, texts: list[str]) -> list[list[float]]:
        return [[b / 255.0 for b in hashlib.sha256(t.encode()).digest()[:8]] for t in texts]


def test_cluster_reports_saturation_with_its_bounds(seeded_local_root, capsys):
    assert main(["cluster"]) == EXIT_OK
    out = capsys.readouterr().out
    assert "human FAIL label(s) on 3 stored trace(s)" in out
    assert "ratified failure modes (failing traces per mode): " in out
    assert "NOT saturated (insufficient_observations" in out
    assert "one-sided 95% lower bound" in out
    assert "Chao1 richness estimate" in out
    assert "95% CI [" in out


def test_cluster_reports_coverage_blind_spots(seeded_local_root, capsys):
    assert main(["cluster"]) == EXIT_OK
    out = capsys.readouterr().out
    assert "taxonomy cell(s);" in out
    assert "blind spot" in out


def test_cluster_with_a_taxonomy_file_covers_that_grid(seeded_local_root, tmp_path, capsys):
    grid = tmp_path / "taxonomy.json"
    grid.write_text(json.dumps({"feature": ["refund_handling", "claim_adjudication"]}))
    assert main(["cluster", "--taxonomy", str(grid)]) == EXIT_OK
    assert "of 2 taxonomy cell(s)" in capsys.readouterr().out


def test_without_an_embedder_the_draft_is_skipped_and_said_so(seeded_local_root, capsys):
    assert main(["cluster"]) == EXIT_OK
    assert "draft modes: skipped (needs an embedder" in capsys.readouterr().out


def test_with_an_embedder_the_draft_is_written_unratified(
    seeded_local_root, tmp_path, monkeypatch, capsys
):
    import calibrated_agent_evaluation.cli.stages.cluster as stage

    monkeypatch.setattr(stage, "_embedder", lambda _args: _HashEmbedder())
    out = tmp_path / "draft.json"
    assert main(["cluster", "--embedder", "bedrock", "--out", str(out)]) == EXIT_OK
    summary = capsys.readouterr().out
    assert "draft modes (unratified):" in summary
    doc = json.loads(out.read_text())
    assert doc["ratified"] is False
    members = [m for c in doc["clusters"] for m in c["members"]]
    # Every FAIL critique lands in exactly one draft cluster: none dropped, none duplicated.
    n_fail = int(summary.split("cluster over ")[1].split(" ")[0])
    assert len(members) == n_fail
    assert all(m["critique"].strip() for m in members)


def test_the_bedrock_embedder_without_its_extra_is_a_typed_hint(
    seeded_local_root, monkeypatch, capsys
):
    from calibrated_agent_evaluation.cli.config import probe_extra as real_probe

    def _probe(module: str) -> Degradation:
        if module == "boto3":
            return Degradation(module, False, "pip install calibrated-agent-evaluation[bedrock]")
        return real_probe(module)

    monkeypatch.setattr("calibrated_agent_evaluation.cli.config.probe_extra", _probe)
    assert main(["cluster", "--embedder", "bedrock"]) == EXIT_DOMAIN_ERROR
    assert "--embedder bedrock needs the bedrock extra" in capsys.readouterr().err


def test_trace_narrows_the_labels_clustered(seeded_local_root, capsys):
    assert main(["cluster", "--trace", "ORD-1001"]) == EXIT_OK
    out = capsys.readouterr().out
    assert "on trace 'ORD-1001'" in out


def test_a_trace_with_no_fail_label_has_nothing_to_cluster(seeded_local_root, capsys):
    """ORD-1002 is the corpus's passing control."""
    assert main(["cluster", "--trace", "ORD-1002"]) == EXIT_DOMAIN_ERROR
    assert "no human FAIL labels" in capsys.readouterr().err


def test_saturation_counts_each_failing_trace_once_per_mode(tmp_path, monkeypatch, capsys):
    """A span defect repeating its trace's mode, or a second rater, is not a second sighting."""
    from _helpers import load_corpus_labels, load_corpus_trace
    from calibrated_agent_evaluation.backends.local import LocalBackend
    from calibrated_agent_evaluation.schema.types import Label

    monkeypatch.delenv("CALIBRATED_AGENT_EVALUATION_LOCAL_ROOT", raising=False)
    root = tmp_path / "store"
    with LocalBackend(root=root) as backend:
        for tid in ("ORD-1001", "ORD-1002", "CLM-4471", "CLM-4473", "CLM-4475", "CLM-4477"):
            backend.put_trace(load_corpus_trace(tid))
        for label in load_corpus_labels():
            backend.put_score(label)
        backend.put_score(
            Label(
                trace_id="CLM-4477",
                span_id=None,
                kind="human",
                outcome="FAIL",
                critique="second rater, same mode",
                failure_mode="vacancy_threshold_margin_mishandled",
                rater_id="second-rater",
            )
        )
    assert main(["--root", str(root), "--explain", "cluster"]) == EXIT_OK
    out = capsys.readouterr().out
    assert "trusted_broken_policy_tool (2)" in out
    assert "dropped_material_fact (1)" in out
    assert "vacancy_threshold_margin_mishandled (1)" in out
    assert "misread_amount (1)" in out
    assert "(n=5 observations, one per failing trace and mode)" in out
    assert "3 seen once" in out
