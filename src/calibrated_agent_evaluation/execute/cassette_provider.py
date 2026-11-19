# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""The cassette-backed Inspect model provider: shipped so replay works for installed users.

Inspect's extension point for models is a ``@modelapi`` provider (a ``ModelAPI`` subclass). This is
exactly that: a provider registered as ``cassette/<inner-model>`` that **delegates to Inspect's real
Bedrock provider** and records its ``ModelOutput`` once, replaying it deterministically thereafter.

So the OrderBot/ClaimsCrew ``.eval`` logs are produced by a genuine Inspect ``eval()`` run; real
``Task``/solver/tool machinery, real event transcript (the ``ModelEvent``/``ToolEvent`` tree the
``inspect_adapter`` reads via ``event_tree``); with only the model's output captured for
offline/deterministic replay (a VCR-style harness around the eval, not a change to Inspect).

Promoted from ``corpus/agents/inspect_cassette_provider.py`` into the shipped package: the
``InspectExecutor`` registers this provider at runtime, so a pip-installed user
(who has no ``examples/`` tree) can still run ``model="cassette/..."`` replays in CI. Inspect is
imported at module top level, so this module is imported lazily by the executor; never eagerly
from ``calibrated_agent_evaluation.execute.__init__`` (that would break the lazy-import guarantee).
"""

from __future__ import annotations

from typing import Any

from inspect_ai.model import (
    ChatMessage,
    GenerateConfig,
    ModelAPI,
    ModelOutput,
    get_model,
    modelapi,
)
from inspect_ai.tool import ToolChoice, ToolInfo

from calibrated_agent_evaluation.execute.cassette import (
    Cassette,
    CassetteMiss,
    Mode,
    fingerprint,
    model_stamp,
)

# The default inner model recorded against when RECORD mode falls through to a live call. Callers
# override it per run via ``InspectExecutor(cassette_inner_model=...)`` -> ``configure()``.
DEFAULT_INNER_MODEL = "bedrock/global.anthropic.claude-sonnet-4-6"

# Module-level cassette handle: the eval() runner sets this before launching the task, because
# Inspect constructs the provider by name and we cannot thread the cassette through that boundary.
_ACTIVE_CASSETTE: Cassette | None = None
_INNER_MODEL: str = DEFAULT_INNER_MODEL


def configure(cassette: Cassette, inner_model: str) -> None:
    """Set the cassette + inner model the provider should use for the next ``eval()`` run."""
    global _ACTIVE_CASSETTE, _INNER_MODEL
    _ACTIVE_CASSETTE = cassette
    _INNER_MODEL = inner_model


@modelapi(name="cassette")
def cassette_provider() -> type[ModelAPI]:
    """Register the ``cassette/<...>`` Inspect provider (called by Inspect's model registry)."""
    return CassetteModelAPI


class CassetteModelAPI(ModelAPI):
    """Wrap Inspect's real Bedrock provider, recording/replaying ``generate`` via the cassette."""

    def __init__(self, model_name: str, *args: Any, **kwargs: Any) -> None:
        # model_name arrives as the part after "cassette/"; we ignore it and use the configured
        # inner model so the eval is launched simply as eval(model="cassette/orderbot").
        super().__init__(model_name, *args, **kwargs)

    @property
    def _cassette(self) -> Cassette:
        # Read the module-global lazily, NOT at construction: Inspect caches a model instance by
        # name ("cassette/..."), so one instance is reused across multiple eval() calls. Binding the
        # cassette in __init__ would route every later eval into the first run's cassette. Reading
        # the live global on each call lets configure() retarget per eval.
        if _ACTIVE_CASSETTE is None:
            raise RuntimeError(
                "CassetteModelAPI used before configure(); call "
                "cassette_provider.configure(cassette, inner_model) before eval()"
            )
        return _ACTIVE_CASSETTE

    async def generate(
        self,
        input: list[ChatMessage],
        tools: list[ToolInfo],
        tool_choice: ToolChoice,
        config: GenerateConfig,
    ) -> ModelOutput:
        """Return the inner provider's ``ModelOutput``, recorded once and replayed offline."""
        cassette = self._cassette
        request = {
            "messages": [_stable_message(m) for m in input],
            "tools": [t.model_dump(mode="json") for t in tools],
            "tool_choice": str(tool_choice),
        }
        fp = fingerprint(request)
        model = model_stamp(_INNER_MODEL, _decoding(config))
        hit = cassette.lookup(fp, model=model)
        if hit is not None:
            return ModelOutput.model_validate(hit)
        # Miss: honor the cassette mode; REPLAY must not fall through to a live call (offline CI).
        if cassette.mode is Mode.REPLAY:
            raise CassetteMiss(
                f"cassette {cassette.path.name}: no recorded ModelOutput for fingerprint {fp}"
            )
        output = await get_model(_INNER_MODEL).generate(input, tools, tool_choice, config)
        cassette.record(fp, request, output.model_dump(mode="json"), model=model)
        return output


# The GenerateConfig fields that change what the model returns. Retries, timeouts and connection
# limits are left out, so changing them does not invalidate a recording.
_DECODING_FIELDS = (
    "temperature",
    "top_p",
    "top_k",
    "max_tokens",
    "seed",
    "stop_seqs",
    "frequency_penalty",
    "presence_penalty",
    "reasoning_effort",
    "reasoning_tokens",
)


def _decoding(config: GenerateConfig) -> dict[str, Any]:
    """The decoding settings of ``config`` that are set, for the cassette's model stamp."""
    dumped = config.model_dump(mode="json", exclude_none=True)
    return {name: dumped[name] for name in _DECODING_FIELDS if name in dumped}


def _stable_message(message: ChatMessage) -> dict[str, Any]:
    """Serialize a chat message for fingerprinting, dropping per-process-volatile fields.

    Inspect assigns every ``ChatMessage`` a random ``id`` (and an ``internal`` field) that varies
    across processes but is irrelevant to the model request. Including it would make the fingerprint
    non-deterministic across runs, breaking offline replay. We drop those and keep only the
    semantically-meaningful content (role, content, tool calls).
    """
    data = message.model_dump(mode="json")
    for volatile in ("id", "internal"):
        data.pop(volatile, None)
    # inspect-ai 0.3.282 added an unset-by-default cache_breakpoint to content blocks; dropping it
    # while None keeps cassettes recorded under earlier releases replayable. A set value is kept.
    content = data.get("content")
    if isinstance(content, list):
        for block in content:
            if isinstance(block, dict) and block.get("cache_breakpoint", ...) is None:
                del block["cache_breakpoint"]
    return data
