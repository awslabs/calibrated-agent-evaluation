# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""The committed detector and namer cassettes replay offline to their committed oracles.

The cassettes under ``fixtures/raw/cassettes/`` hold one billed recording against Bedrock: the
detector on ORD-1001, ORD-1002 and CLM-4471, and the namer on the clusters ``build_draft`` forms
from the seven gold FAIL critiques. The oracles under ``fixtures/expected/detectors/`` hold the
oracle view of each replayed result: every field except the stamp's run-environment values. A
replay needs no AWS configuration and opens no connection, and neither detecting nor naming nor
ratifying writes a label.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest

pytest.importorskip("strands_evals")

from _detector_fakes import ScriptedModel, spy_connections, unset_aws_env
from _helpers import FIXTURES, load_corpus_labels, load_corpus_trace
from calibrated_agent_evaluation._provenance import STATS_DISTRIBUTIONS
from calibrated_agent_evaluation.agentcore.detectors import models
from calibrated_agent_evaluation.agentcore.detectors.detect import FailureDetector
from calibrated_agent_evaluation.agentcore.detectors.models import detector_bedrock_model
from calibrated_agent_evaluation.agentcore.detectors.namer import DetectorDraftNamer
from calibrated_agent_evaluation.agentcore.detectors.vocabulary import failure_vocabulary
from calibrated_agent_evaluation.backends.local import LocalBackend
from calibrated_agent_evaluation.cluster import build_draft, hierarchical_cluster, ratify
from calibrated_agent_evaluation.execute import Mode
from calibrated_agent_evaluation.schema.types import Label
from test_cluster_helpers import FakeEmbedder

NOW = datetime(2026, 10, 27, 18, 0, tzinfo=UTC)
RECORDED_MODEL_ID = "global.anthropic.claude-sonnet-4-6"
VERIFIED_DETECTOR_VERSION = "1.6.0"
VERIFIED_TEMPLATE_SHA256 = "30d6baa2559a514b87e9477f58afefb977371cc413f5c841eaa3ad9798c809a9"
DETECTED_TRACE_IDS = ("ORD-1001", "ORD-1002", "CLM-4471")
DETECTOR_CASSETTE = FIXTURES / "raw" / "cassettes" / "detector_fixture.cassette.json"
NAMER_CASSETTE = FIXTURES / "raw" / "cassettes" / "namer_gold_fails.cassette.json"
ORACLES = FIXTURES / "expected" / "detectors"


@pytest.fixture
def offline(monkeypatch: pytest.MonkeyPatch) -> list[Any]:
    """No AWS configuration, no Bedrock model built, and every connection refused and recorded."""
    unset_aws_env(monkeypatch)

    def never_built(**_kwargs: Any) -> None:
        raise AssertionError("a replay built a Bedrock model")

    monkeypatch.setattr(models, "BedrockModel", never_built)
    return spy_connections(monkeypatch)


def _oracle(name: str) -> dict[str, Any]:
    return json.loads((ORACLES / name).read_text(encoding="utf-8"))


def _replaying_detector() -> FailureDetector:
    return FailureDetector(
        detector_bedrock_model(RECORDED_MODEL_ID),
        cassette_path=DETECTOR_CASSETTE,
        cassette_mode=Mode.REPLAY,
        now=NOW,
    )


def _gold_fail_clusters() -> tuple[list[str], list[list[float]], list[int]]:
    """The gold FAIL critiques in file order, their test-embedder vectors and their clusters."""
    critiques = [label.critique for label in load_corpus_labels() if label.outcome == "FAIL"]
    embeddings = FakeEmbedder().embed(critiques)
    return critiques, embeddings, hierarchical_cluster(embeddings, threshold=0.5)


def _labels_by_trace(store: LocalBackend, trace_ids: list[str]) -> dict[str, list[Label]]:
    return {trace_id: store.get_scores(trace_id) for trace_id in trace_ids}


def test_the_oracle_view_drops_every_numeric_library_version_the_stamp_records(
    offline: list[Any],
) -> None:
    detection = _replaying_detector().detect(load_corpus_trace(DETECTED_TRACE_IDS[0]))

    recorded = detection.provenance.stamp.dep_versions
    kept = detection.oracle_view()["provenance"]["stamp"]["dep_versions"]
    assert offline == []
    assert set(STATS_DISTRIBUTIONS) <= set(recorded)
    assert not set(STATS_DISTRIBUTIONS) & set(kept)


def test_committed_detector_cassettes_replay_offline_to_their_oracles(offline: list[Any]) -> None:
    detector = _replaying_detector()

    detections = {
        trace_id: detector.detect(load_corpus_trace(trace_id)) for trace_id in DETECTED_TRACE_IDS
    }

    assert offline == []
    assert {trace_id: d.oracle_view() for trace_id, d in detections.items()} == {
        trace_id: _oracle(f"{trace_id}.detection.json") for trace_id in DETECTED_TRACE_IDS
    }
    assert [d.provenance.stamp.run_at for d in detections.values()] == [
        "2026-10-27T18:00:00+00:00"
    ] * len(DETECTED_TRACE_IDS)
    # The recording found failures: at least one oracle shows what a detected failure looks like.
    assert any(d.failures for d in detections.values())


def test_committed_results_carry_their_provenance(offline: list[Any]) -> None:
    critiques, embeddings, assignments = _gold_fail_clusters()
    namer = DetectorDraftNamer(
        detector_bedrock_model(RECORDED_MODEL_ID),
        failure_vocabulary(),
        cassette_path=NAMER_CASSETTE,
        cassette_mode=Mode.REPLAY,
        now=NOW,
    )

    draft = build_draft(critiques, embeddings, assignments, namer=namer)

    assert offline == []
    oracles = [_oracle(f"{trace_id}.detection.json") for trace_id in DETECTED_TRACE_IDS]
    oracles.append(_oracle("namer_gold_fails.json"))
    for oracle in oracles:
        provenance = oracle["provenance"]
        assert (
            provenance["detector_version"],
            provenance["prompt_template"],
            provenance["prompt_sha256"],
            provenance["model_id"],
        ) == (VERIFIED_DETECTOR_VERSION, "v0", VERIFIED_TEMPLATE_SHA256, RECORDED_MODEL_ID)
        assert provenance["stamp"]["dep_versions"] == {
            "strands-agents-evals": VERIFIED_DETECTOR_VERSION
        }
    (naming,) = namer.results
    assert naming.namer_prompt_version == "namer-v1"
    assert naming.oracle_view() == _oracle("namer_gold_fails.json")
    assert [cluster.name for cluster in draft.clusters] == list(naming.names)
    assert naming.shown == ((1, 1),) * len(draft.clusters)


def test_detection_and_naming_write_no_label(tmp_path: Path) -> None:
    trace_ids = sorted(path.stem for path in (FIXTURES / "normalized").glob("*.json"))
    vocabulary = failure_vocabulary()
    with LocalBackend(tmp_path / "store") as store:
        for trace_id in trace_ids:
            store.put_trace(load_corpus_trace(trace_id))
        for label in load_corpus_labels():
            store.put_score(label)
        seeded = _labels_by_trace(store, trace_ids)

        detector = _replaying_detector()
        detections = [detector.detect(store.get_trace(trace_id)) for trace_id in DETECTED_TRACE_IDS]
        critiques = [
            label.critique
            for trace_id in trace_ids
            for label in store.get_scores(trace_id)
            if label.outcome == "FAIL"
        ]
        embeddings = FakeEmbedder().embed(critiques)
        assignments = hierarchical_cluster(embeddings, threshold=0.5)
        names = vocabulary.names()[: len(set(assignments))]
        namer = DetectorDraftNamer(
            ScriptedModel([json.dumps(list(names))]), vocabulary, max_live_calls=1, now=NOW
        )
        draft = build_draft(critiques, embeddings, assignments, namer=namer)
        codebook = ratify(draft, ratified_by="replay-test", descriptions=vocabulary.descriptions())

        after = _labels_by_trace(store, trace_ids)

    assert any(detection.failures for detection in detections)
    assert [mode.name for mode in codebook.modes] == list(names)
    assert sum(len(labels) for labels in seeded.values()) == len(load_corpus_labels())
    assert after == seeded
