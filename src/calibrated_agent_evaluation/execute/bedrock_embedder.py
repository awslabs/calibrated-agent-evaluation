# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""``BedrockEmbedder``: the concrete Titan embedder deferred to the caller.

Satisfies the ``Embedder`` protocol (``embed(texts) -> list[list[float]]``), shared with the
dedup/diversity guards. It calls Bedrock Titan Text Embeddings v2
(``amazon.titan-embed-text-v2:0``) via boto3 ``invoke_model``; vendor-isolated in ``execute/`` (the
vendor-adapter exemption package), opt-in behind the ``[bedrock]`` extra. The default
clustering/synth path injects a caller-provided embedder or none; this is the concrete one for
production use.

Verified surface (introspected against the installed boto3): ``bedrock-runtime`` exposes
``invoke_model(modelId=, body=, contentType=, accept=)`` returning ``{"body", "contentType"}``; the
body JSON for Titan v2 carries the float list under ``"embedding"`` (AWS-documented; the live test
``tests/integration/test_bedrock_live.py`` is the honest end-to-end check of the body keys).

An unreachable backend or a returned vector count that does not match the input count raises
:class:`~calibrated_agent_evaluation.cluster.errors.ClusteringError`; the result is NEVER truncated
or misaligned (no silent truncation).
"""

from __future__ import annotations

import json
from typing import Any

from calibrated_agent_evaluation._install import install_hint
from calibrated_agent_evaluation.cluster.errors import ClusteringError

_DEFAULT_MODEL_ID = "amazon.titan-embed-text-v2:0"
_DEFAULT_REGION = "us-west-2"
_DEFAULT_TIMEOUT_SECONDS = 30.0


class BedrockEmbedder:
    """Concrete ``Embedder`` on Bedrock Titan v2 via boto3 ``invoke_model``.

    Construction is cheap (the boto3 client is built lazily on first :meth:`embed`, so importing
    this module needs no AWS session and importing ``calibrated_agent_evaluation.execute`` needs no
    boto3).
    """

    def __init__(
        self,
        *,
        model_id: str = _DEFAULT_MODEL_ID,
        region: str = _DEFAULT_REGION,
        timeout_seconds: float = _DEFAULT_TIMEOUT_SECONDS,
    ) -> None:
        self._model_id = model_id
        self._region = region
        self._timeout_seconds = timeout_seconds
        self._client: Any = None

    def embed(self, texts: list[str]) -> list[list[float]]:
        """Return one embedding vector per input text, in order.

        Raises :class:`ClusteringError` if the backend is unreachable, a call fails, or the number
        of returned vectors does not equal ``len(texts)``; never a truncated/misaligned list.
        """
        client = self._get_client()
        vectors: list[list[float]] = []
        for text in texts:
            vectors.append(self._embed_one(client, text))
        if len(vectors) != len(texts):
            # Defensive: the loop is 1:1 by construction, but assert the invariant explicitly so a
            # future batch-API change can never silently drop a vector.
            raise ClusteringError(
                f"embedder produced {len(vectors)} vectors for {len(texts)} inputs; "
                "refusing a misaligned result"
            )
        return vectors

    def _embed_one(self, client: Any, text: str) -> list[float]:
        """Embed a single text via Titan v2 ``invoke_model``; wrap any failure typed."""
        try:
            response = client.invoke_model(
                modelId=self._model_id,
                contentType="application/json",
                accept="application/json",
                body=json.dumps({"inputText": text}),
            )
            payload = json.loads(response["body"].read())
        except Exception as exc:
            raise ClusteringError(
                f"Titan embedding call failed ({type(exc).__name__}: {exc}); "
                "refusing to return a partial/misaligned embedding set"
            ) from exc
        embedding = payload.get("embedding")
        if not isinstance(embedding, list) or not embedding:
            raise ClusteringError(
                f"Titan response carried no 'embedding' list (got keys {list(payload)}); "
                "refusing to fabricate a vector"
            )
        return [float(x) for x in embedding]

    def _get_client(self) -> Any:
        """Build the boto3 bedrock-runtime client lazily, with a bounded timeout."""
        if self._client is not None:
            return self._client
        try:
            import boto3
            from botocore.config import Config as BotocoreConfig
        except ImportError as exc:
            raise ClusteringError(
                f"the Titan embedder needs the bedrock extra; {install_hint('bedrock')}"
            ) from exc
        # Bound the connect/read so an unreachable Bedrock peer raises within the window rather than
        # hanging on a shifting SDK default (every network call is bounded).
        config = BotocoreConfig(
            connect_timeout=self._timeout_seconds, read_timeout=self._timeout_seconds
        )
        try:
            self._client = boto3.client("bedrock-runtime", region_name=self._region, config=config)
        except Exception as exc:
            # boto3 resolves the AWS profile and region here, so a named profile that does not
            # exist (ProfileNotFound) or a missing region fails before any embedding call.
            raise ClusteringError(
                f"could not create the Bedrock client for the Titan embedder "
                f"({type(exc).__name__}: {exc}); check AWS_PROFILE and the region"
            ) from exc
        return self._client
