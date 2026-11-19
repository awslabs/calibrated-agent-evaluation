# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""BedrockEmbedder tests.

The embedder satisfies the ``Embedder`` protocol via boto3 Titan ``invoke_model``. Tests run
against a stub client that mirrors the real ``invoke_model(...)`` → ``{"body": <readable JSON>}``
surface and assert on the round-tripped vectors (1:1, in order) and the typed refusals (unreachable
/ count-mismatch → ClusteringError, never truncation). The live check is the opt-in integration
test.
"""

from __future__ import annotations

import json

import pytest

from calibrated_agent_evaluation.cluster.embed import Embedder
from calibrated_agent_evaluation.cluster.errors import ClusteringError
from calibrated_agent_evaluation.execute.bedrock_embedder import BedrockEmbedder


class _StubBody:
    def __init__(self, payload: dict) -> None:
        self._raw = json.dumps(payload).encode()

    def read(self) -> bytes:
        return self._raw


class _StubClient:
    """Mirrors bedrock-runtime.invoke_model: returns {"body": readable JSON} or raises."""

    def __init__(self, *, vector: list[float] | None = None, raises: Exception | None = None):
        self._vector = vector
        self._raises = raises
        self.calls = 0

    def invoke_model(self, *, modelId: str, contentType: str, accept: str, body: str):
        self.calls += 1
        if self._raises is not None:
            raise self._raises
        return {"body": _StubBody({"embedding": self._vector})}


def _embedder_with(client: _StubClient) -> BedrockEmbedder:
    e = BedrockEmbedder()
    e._client = client  # type: ignore[attr-defined]
    return e


def test_embed_returns_one_vector_per_input_in_order():
    """Embed(texts) returns exactly len(texts) vectors, in order."""
    embedder = _embedder_with(_StubClient(vector=[0.1, 0.2, 0.3]))

    vectors = embedder.embed(["a", "b", "c"])

    assert len(vectors) == 3
    assert all(v == [0.1, 0.2, 0.3] for v in vectors)


def test_satisfies_the_embedder_protocol():
    """BedrockEmbedder structurally satisfies the Embedder protocol (contract)."""
    embedder = _embedder_with(_StubClient(vector=[0.0]))
    assert isinstance(embedder, Embedder)


def test_unreachable_backend_raises_clustering_error():
    """Negative: an erroring client → ClusteringError (wrapped), never a partial list."""
    boom = ConnectionError("bedrock unreachable")
    embedder = _embedder_with(_StubClient(raises=boom))

    with pytest.raises(ClusteringError) as ei:
        embedder.embed(["a", "b"])
    assert ei.value.__cause__ is boom


def test_missing_embedding_key_is_refused():
    """Negative: a response with no 'embedding' list → ClusteringError, no fabricated
    vector."""
    embedder = _embedder_with(_StubClient(vector=None))  # payload {"embedding": None}

    with pytest.raises(ClusteringError, match="embedding"):
        embedder.embed(["a"])


def test_a_missing_aws_profile_is_a_clustering_error(monkeypatch: pytest.MonkeyPatch) -> None:
    """boto3 raised a raw ProfileNotFound from client construction, past the typed boundary."""
    pytest.importorskip("boto3")
    monkeypatch.setenv("AWS_PROFILE", "calibrated-agent-evaluation-no-such-profile")
    with pytest.raises(ClusteringError, match="AWS_PROFILE") as caught:
        BedrockEmbedder().embed(["a"])
    assert type(caught.value.__cause__).__name__ == "ProfileNotFound"
