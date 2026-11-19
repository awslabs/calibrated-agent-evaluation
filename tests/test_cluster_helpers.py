# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shared test helpers for the clustering module.

FakeEmbedder produces deterministic vectors from critique text.
"""

from __future__ import annotations

import hashlib


class FakeEmbedder:
    """Deterministic hash-based embedder for tests.

    Produces 16-dimensional vectors from SHA-256 of the input text.
    Records inputs for invocation evidence.
    """

    def __init__(self) -> None:
        self.calls: list[list[str]] = []

    def embed(self, texts: list[str]) -> list[list[float]]:
        self.calls.append(texts)
        return [self._hash_to_vector(t) for t in texts]

    def _hash_to_vector(self, text: str) -> list[float]:
        digest = hashlib.sha256(text.encode()).digest()
        return [b / 255.0 for b in digest[:16]]
