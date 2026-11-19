# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""A cassette-backed Bedrock Converse client with a defect-injection seam.

This is the one place the corpus talks to a live model. It wraps the real boto3 ``bedrock-runtime``
``converse`` API (the transport Strands' ``BedrockModel`` and a hand-coded loop both ride on) with:

  * the record/replay :class:`~corpus.generate.cassette.Cassette` (one live pass mints cassettes;
    CI replays offline and deterministically; the executor reuses this layer), and
  * a **defect-injection** interceptor (``corpus-companion.md`` §2): a documented seam that forces
    the canonical worked-example defects deterministically (ORD-1001's out-of-window over-refund;
    CLM-4471's intake-drops-vacancy) instead of hoping a sampled model reproduces them.

Credentials come from the environment / a named AWS profile (``AWS_PROFILE=<your-profile>``) — never
hardcoded, never written to a cassette (conventions: "secrets stay in the process"). Every network
call sets an explicit timeout (conventions: bounded I/O), so a dead peer can't hang the capture.
"""

from __future__ import annotations

import logging
import os
from collections.abc import Callable
from pathlib import Path
from typing import Any

from calibrated_agent_evaluation.flagship.runner import live_bedrock_model_id
from corpus.generate.cassette import Cassette, Mode

logger = logging.getLogger("calibrated_agent_evaluation.corpus.bedrock")

# Default model: Claude Sonnet 4.6 on Bedrock. A cross-region inference profile id; the one Bedrock
# model knob shared with the live tests and the flagship script overrides it.
DEFAULT_MODEL_ID = live_bedrock_model_id(os.environ)
DEFAULT_REGION = os.environ.get("AWS_REGION", "us-west-2")
# Explicit, self-set bound on every Converse call (never rely on a library default that can change).
_CONNECT_TIMEOUT_S = 10.0
_READ_TIMEOUT_S = 60.0


class BedrockConverseClient:
    """Cassette-backed wrapper over ``bedrock-runtime.converse``.

    In REPLAY mode the boto client is never created, so the corpus replays with no AWS credentials
    and no network — a fresh clone runs the agents offline. The live client is lazily built only
    when a real call is actually needed (RECORD, or an AUTO miss).
    """

    def __init__(
        self,
        cassette: Cassette,
        *,
        model_id: str = DEFAULT_MODEL_ID,
        region: str = DEFAULT_REGION,
    ) -> None:
        self._cassette = cassette
        self._model_id = model_id
        self._region = region
        self._client: Any | None = None  # lazily created boto3 client (only when live)

    @property
    def model_id(self) -> str:
        return self._model_id

    def _live_client(self) -> Any:
        if self._client is None:
            try:
                import boto3
                from botocore.config import Config
            except ImportError as exc:  # corpus group not installed
                raise RuntimeError(
                    "boto3 is required for a live Bedrock call; install the corpus group: "
                    "`uv sync --group corpus`"
                ) from exc
            # AWS_PROFILE / AWS_REGION are honored by Session(); creds stay in the process.
            session = boto3.Session(region_name=self._region)
            self._client = session.client(
                "bedrock-runtime",
                config=Config(
                    connect_timeout=_CONNECT_TIMEOUT_S,
                    read_timeout=_READ_TIMEOUT_S,
                    retries={"max_attempts": 2, "mode": "standard"},
                ),
            )
        return self._client

    def converse(self, request: dict[str, Any]) -> dict[str, Any]:
        """Run a Converse request through the cassette; returns the Converse response dict.

        ``request`` is the kwargs dict passed to ``bedrock-runtime.converse`` (``messages``,
        ``system``, ``toolConfig``, ...) **without** ``modelId`` (added here) — so the fingerprint
        is stable across a model-id override and keys only on the conversation content.
        """

        def live_call() -> dict[str, Any]:
            client = self._live_client()
            try:
                resp = client.converse(modelId=self._model_id, **request)
            except Exception as exc:  # boto/botocore error — surface with context, never silently
                raise RuntimeError(
                    f"live Bedrock converse failed (model={self._model_id}, "
                    f"region={self._region}): {type(exc).__name__}: {exc}"
                ) from exc
            # Drop the SDK metadata envelope (request ids, headers) — not content, varies per call.
            return {k: v for k, v in resp.items() if k != "ResponseMetadata"}

        return self._cassette.play(request, live_call)


def make_client(
    cassette_path: Path,
    *,
    mode: Mode = Mode.AUTO,
    model_id: str = DEFAULT_MODEL_ID,
    region: str = DEFAULT_REGION,
) -> BedrockConverseClient:
    """Build a cassette-backed Converse client for a capture run."""
    return BedrockConverseClient(
        Cassette(cassette_path, mode=mode), model_id=model_id, region=region
    )


# --- Defect injection ------------------------------------------------------------------------

# A defect is a deterministic transform applied to a tool result before the agent sees it (the seam
# that forces the canonical worked-example failures). Keyed by (trace_id, tool_name); identity for
# anything not registered. This is the documented interceptor `corpus-companion.md` §2 calls for.
DefectFn = Callable[[dict[str, Any]], dict[str, Any]]
_DEFECTS: dict[tuple[str, str], DefectFn] = {}


def register_defect(trace_id: str, tool_name: str, fn: DefectFn) -> None:
    """Register a deterministic defect transform for ``(trace_id, tool_name)``."""
    _DEFECTS[(trace_id, tool_name)] = fn


def apply_defect(trace_id: str, tool_name: str, result: dict[str, Any]) -> dict[str, Any]:
    """Apply a registered defect to a tool ``result`` (identity if none registered)."""
    fn = _DEFECTS.get((trace_id, tool_name))
    if fn is None:
        return result
    injected = fn(dict(result))
    logger.info("defect injected: trace=%s tool=%s", trace_id, tool_name)
    return injected
