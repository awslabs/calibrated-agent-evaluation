# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""A Strands ``Model`` provider that records/replays the wrapped model's stream via a cassette.

Promoted from ``corpus/agents/strands_cassette_model.py`` into the shipped package so an
installed user can run ``StrandsExecutor`` with deterministic offline replay. Writing a custom
``strands.models.model.Model`` is the framework's documented extension point: :class:`CassetteModel`
**wraps a real ``BedrockModel``** and, on each ``stream()``, records the genuine sequence of
``StreamEvent``s once and replays them deterministically thereafter. The wrapped model is unmodified
and Strands is unaware; no monkey-patching, just composition behind the public ``Model`` interface,
so the ``Agent``/``Graph`` machinery, the tracer, and the emitted span tree stay genuine (only the
model's token stream is captured; a VCR-style harness around the app).

Re-pointed at the promoted ``calibrated_agent_evaluation.execute.cassette`` (the framework-agnostic
record/replay store), consolidating on one cassette. ``strands`` is imported at module top, so this
module is lazy-imported by ``StrandsExecutor``; never eagerly from
``calibrated_agent_evaluation.execute.__init__`` (that would break the lazy-import guarantee).
"""

from __future__ import annotations

import json
from collections.abc import AsyncGenerator, AsyncIterable, Callable
from typing import Any

from strands.models.model import Model

from calibrated_agent_evaluation.execute.cassette import Cassette, model_stamp


class CassetteModel(Model):
    """Wrap a real Strands ``Model``, recording/replaying its ``stream`` output through a cassette.

    Delegates ``get_config`` / ``update_config`` / ``structured_output`` straight to the wrapped
    model; only ``stream`` is intercepted (that is the one live network call). The request
    fingerprint is built from the serialized ``messages`` + ``tool_specs`` + ``system_prompt`` the
    agent passes, so the same agent turn is a deterministic cassette hit.
    """

    def __init__(self, inner: Model, cassette: Cassette) -> None:
        self._inner = inner
        self._cassette = cassette

    def get_config(self) -> Any:
        """The wrapped model's configuration."""
        return self._inner.get_config()

    def update_config(self, **model_config: Any) -> None:
        """Update the wrapped model's configuration."""
        self._inner.update_config(**model_config)

    def structured_output(self, *args: Any, **kwargs: Any) -> AsyncGenerator[dict[str, Any], None]:
        """Delegate structured output to the wrapped model; it is not recorded."""
        # The crew does not use structured_output; delegate so the provider is still a faithful
        # Model. Returns the inner generator directly to preserve the exact base signature.
        return self._inner.structured_output(*args, **kwargs)

    async def stream(
        self,
        messages: Any,
        tool_specs: Any = None,
        system_prompt: str | None = None,
        **kwargs: Any,
    ) -> AsyncIterable[Any]:
        """Yield the wrapped model's ``StreamEvent``s, recording them once and replaying offline."""
        request = {
            "messages": _jsonable(messages),
            "tool_specs": _jsonable(tool_specs),
            "system_prompt": system_prompt,
        }

        async def live() -> dict[str, Any]:
            # Drain the real model's async stream into a recordable list of StreamEvent dicts.
            events: list[Any] = []
            async for event in self._inner.stream(messages, tool_specs, system_prompt, **kwargs):
                events.append(event)
            return {"events": events}

        recorded = await self._cassette.aplay(request, live, model=self._model_stamp())
        for event in recorded["events"]:
            yield event

    def _model_stamp(self) -> dict[str, Any] | None:
        """The wrapped model's id and decoding config, which each recorded entry carries."""
        config = dict(_jsonable(self._inner.get_config()) or {})
        model_id = config.pop("model_id", None)
        return model_stamp(None if model_id is None else str(model_id), config)


class DeferredModel(Model):
    """A Strands ``Model`` that builds the real model on its first live call.

    Building ``BedrockModel`` creates a boto3 session, which resolves the AWS profile and region. A
    cassette replay makes no live call, so it must not need credentials; deferring the build means
    a replay works with no AWS configuration at all, and a recording still builds the real model
    when the first cassette miss reaches it. ``config`` is held here so it is readable before then.
    """

    def __init__(self, build: Callable[[], Model], config: dict[str, Any]) -> None:
        self._build = build
        self._model: Model | None = None
        self.config = dict(config)

    def _real(self) -> Model:
        if self._model is None:
            self._model = self._build()
            self._model.update_config(**self.config)
        return self._model

    def get_config(self) -> Any:
        """The configuration the real model will be built with."""
        return self.config

    def update_config(self, **model_config: Any) -> None:
        """Update the configuration, and the real model's too if it is already built."""
        self.config.update(model_config)
        if self._model is not None:
            self._model.update_config(**model_config)

    def structured_output(self, *args: Any, **kwargs: Any) -> AsyncGenerator[dict[str, Any], None]:
        """Build the real model if needed and delegate structured output to it."""
        return self._real().structured_output(*args, **kwargs)

    def stream(self, *args: Any, **kwargs: Any) -> AsyncIterable[Any]:
        """Build the real model if needed and stream from it."""
        return self._real().stream(*args, **kwargs)


def _jsonable(value: Any) -> Any:
    """Best-effort canonicalization of a request part for fingerprinting (stable, content-only)."""
    return json.loads(json.dumps(value, default=str, sort_keys=True))
