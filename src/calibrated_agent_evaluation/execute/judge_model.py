# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Judge-model seam: the vendor-isolation boundary for the forced-structured verdict call.

The judge runtime lives in the methodology plane (``judges/``), which must not import a vendor
SDK. The existing ``ExecutorBackend.run() -> list[Trace]`` runs a *whole agent eval* and is
the wrong shape for "call an LLM once and get one forced-structured verdict".
So this module adds a narrow ``JudgeModelBackend`` protocol here in ``execute/`` (the blessed
vendor-adapter dir), keeping ``inspect_ai`` entirely on this side of the boundary. ``judges/`` calls
:meth:`JudgeModelBackend.generate_structured` and never sees an Inspect type.

Two implementations:

* :class:`InspectJudgeModel`: real, on ``inspect_ai==0.3.242``. Uses native ``response_schema``
  where the model supports it (Bedrock **Claude** only, verified ``_providers/bedrock.py:642``),
  else a **forced single-tool call** whose arguments ARE the verdict schema. Never falls
  back to free-text JSON parsing. The model call is synchronous-raising (Inspect/boto3
  propagate inside ``generate``), so no reachability probe is needed on this path (unlike the
  Langfuse write path); ``Model.generate`` is async, so the seam drives the
  event loop and presents a synchronous surface to ``judges/``.
* :class:`CassetteJudgeModel`: offline test double backed by the shipped :class:`Cassette`. It
  mirrors the real surface for both paths so unit tests are deterministic and network-free
  (no network in unit tests); a live Bedrock parity test is opt-in.

``StructuredReply`` is a normalized dataclass (no vendor type), so what crosses back into
``judges/`` is plane-clean.
"""

from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal, Protocol, runtime_checkable

from calibrated_agent_evaluation.execute.cassette import (
    Cassette,
    CassetteError,
    CassetteMiss,
    Mode,
    fingerprint,
    model_stamp,
)
from calibrated_agent_evaluation.execute.errors import ExecutionError, JudgeConfigurationError

if TYPE_CHECKING:
    from inspect_ai.model import ModelOutput

# The forced tool the tool-forcing path uses to extract the verdict. The tool's arguments
# ARE the verdict schema, so a compliant model returns the structured fields as tool-call args.
_VERDICT_TOOL_NAME = "record_verdict"


@dataclass(frozen=True)
class StructuredReply:
    """A model's forced-structured verdict, normalized off any vendor type.

    ``fields`` is the structured payload (e.g. ``{"critique": ..., "verdict": "pass"|"fail"}``);
    ``via`` records which path produced it; ``"response_schema"`` (native structured output) or
    ``"tool_call"`` (the forced-tool fallback); for audit. ``judges/`` validates ``fields`` against
    the verdict schema; this seam does not judge correctness, only that a structured payload exists.
    """

    fields: dict[str, Any]
    via: Literal["response_schema", "tool_call"]


@runtime_checkable
class JudgeModelBackend(Protocol):
    """Call a judge model and return a forced-structured reply: the judge model seam.

    ``judges/`` programs against this; it never imports ``inspect_ai``. Implementations raise
    :class:`ExecutionError` on any model failure (the ``judges/`` boundary re-wraps it as
    ``JudgeRuntimeError``).
    """

    def supports_response_schema(self) -> bool:
        """Whether the selected model honors native ``response_schema``.

        Drives the path choice: ``True`` → native structured output; ``False`` → forced tool call.
        """
        ...

    def generate_structured(
        self,
        *,
        system: str,
        user: str,
        schema: dict[str, Any],
        sample: int = 0,
    ) -> StructuredReply:
        """Return a forced-structured reply for the ``(system, user)`` prompt against ``schema``.

        ``schema`` is a plain JSON-schema dict describing the verdict object (critique-before
        -verdict is guaranteed by the caller). ``sample`` numbers repeats of an identical request
        (0 is the first): a backend that answers by request content keeps repeats apart, so a
        repeat is a fresh draw and not the first reply played back. Raises
        :class:`ExecutionError` on model failure or when no structured payload can be extracted
        (enforced with the typed error at the ``judges/`` boundary).
        """
        ...


def _json_schema_from_dict(schema: dict[str, Any]) -> Any:
    """Build an Inspect ``JSONSchema`` from a plain schema dict (verified surface, 0.3.242).

    Recurses ``properties``/``items`` so a nested object/array schema is preserved, and carries the
    verdict schema's constraints to the model side: ``minLength`` (non-empty critique) and
    ``additionalProperties`` (no extra fields). All mapped keys are verified ``JSONSchema`` fields.
    """
    from inspect_ai.util import JSONSchema

    kwargs: dict[str, Any] = {}
    if "type" in schema:
        kwargs["type"] = schema["type"]
    if "description" in schema:
        kwargs["description"] = schema["description"]
    if "enum" in schema:
        kwargs["enum"] = schema["enum"]
    if "required" in schema:
        kwargs["required"] = schema["required"]
    if "minLength" in schema:
        kwargs["minLength"] = schema["minLength"]
    if "additionalProperties" in schema:
        kwargs["additionalProperties"] = schema["additionalProperties"]
    if "properties" in schema:
        kwargs["properties"] = {
            k: _json_schema_from_dict(v) for k, v in schema["properties"].items()
        }
    if "items" in schema:
        kwargs["items"] = _json_schema_from_dict(schema["items"])
    return JSONSchema(**kwargs)


class InspectJudgeModel:
    """Real judge-model backend on Inspect model-runners.

    Constructed with an Inspect model string (e.g. a ``"bedrock/…claude-sonnet-4-6"`` id).
    A model whose provider/name does not natively support ``response_schema`` is served through the
    forced-tool path; never free text.

    ``temperature`` is sent only when set. A set temperature the provider did not send (Inspect
    drops it for Claude 4.7 and later on Bedrock) is refused as a
    :class:`JudgeConfigurationError`, so a judge never runs at a temperature other than the one
    recorded for it.
    """

    def __init__(
        self,
        model: str,
        *,
        timeout_seconds: float = 60.0,
        temperature: float | None = None,
    ) -> None:
        self._model_str = model
        self._timeout = timeout_seconds
        self._temperature = temperature

    def supports_response_schema(self) -> bool:
        """True only for Bedrock Claude models (verified ``_providers/bedrock.py:642-657``).

        Bedrock honors ``response_schema`` for Claude via the Converse ``output_config.format``, and
        **warn-drops it for non-Claude Bedrock**; SageMaker has no support. Rather than trust a sent
        schema was honored, we gate on the known-supporting family and route everything else through
        the forced-tool path. Conservative by design: a model we can't prove supports it
        uses the tool path, which is always structured.
        """
        model = self._model_str.lower()
        provider, _, name = model.partition("/")
        return provider == "bedrock" and "claude" in name

    def generate_text(self, *, system: str, user: str) -> str:
        """Return the model's plain-text completion for a ``(system, user)`` prompt.

        The free-text sibling of :meth:`generate_structured`; used by the ``BedrockRenderer``
        to render a taxonomy cell into a natural-language input. Wraps the verified Inspect model
        seam (``get_model(model).generate(messages).completion``, the exact call this class already
        makes internally). Raises :class:`ExecutionError` on any model failure (wrapped from the
        vendor exception); never returns a fabricated or partial string.
        """
        try:
            return asyncio.run(self._agenerate_text(system=system, user=user))
        except (CassetteMiss, ExecutionError):
            raise
        except Exception as exc:
            raise ExecutionError(
                f"judge model {self._model_str!r} failed to generate text: {exc}"
            ) from exc

    async def _agenerate_text(self, *, system: str, user: str) -> str:
        from inspect_ai.model import ChatMessageSystem, ChatMessageUser, GenerateConfig, get_model

        messages = [ChatMessageSystem(content=system), ChatMessageUser(content=user)]
        model = get_model(self._model_str)
        timeout_s = max(1, round(self._timeout))
        output = await model.generate(messages, config=GenerateConfig(timeout=timeout_s))
        return output.completion

    def generate_structured(
        self,
        *,
        system: str,
        user: str,
        schema: dict[str, Any],
        sample: int = 0,  # noqa: ARG002  # part of the port; see the note below
    ) -> StructuredReply:
        """Drive the async Inspect model call under a sync boundary; extract a structured reply.

        ``sample`` needs nothing here: Inspect does not cache by default, so every call is a draw.

        Refuses up front when an event loop is already running. `asyncio.run` raises `RuntimeError`
        there, which the broad `except Exception` below wrapped as `ExecutionError` -- and
        `judges/runtime._judge_one` treats any `ExecutionError` as transient and retries it. So
        calling a judge from inside an event loop burned `max_retries` attempts and then failed with
        "model failed after 4 attempts", naming a permanent config error as flakiness.
        `JudgeConfigurationError` is outside the retry family, so it surfaces on the first attempt.
        """
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            pass  # no running loop: the sync boundary is safe to cross
        else:
            raise JudgeConfigurationError(
                f"judge model {self._model_str!r} was called from inside a running event loop; "
                "generate_structured() crosses a sync boundary with asyncio.run(), which cannot "
                "nest. Call it from a synchronous context, or await the async model directly"
            )
        try:
            return asyncio.run(self._agenerate(system=system, user=user, schema=schema))
        except (CassetteMiss, ExecutionError):
            raise
        except Exception as exc:
            raise ExecutionError(
                f"judge model {self._model_str!r} failed to generate a structured verdict: {exc}"
            ) from exc

    async def _agenerate(
        self,
        *,
        system: str,
        user: str,
        schema: dict[str, Any],
    ) -> StructuredReply:
        from inspect_ai.model import (
            ChatMessageSystem,
            ChatMessageUser,
            GenerateConfig,
            ResponseSchema,
            get_model,
        )

        messages = [ChatMessageSystem(content=system), ChatMessageUser(content=user)]
        model = get_model(self._model_str)
        # Inspect's timeout is int-seconds; floor at 1s so a sub-second timeout_seconds does not
        # truncate to 0 (which some providers read as "no timeout").
        timeout_s = max(1, round(self._timeout))

        if self.supports_response_schema():
            config = GenerateConfig(
                response_schema=ResponseSchema(
                    name="verdict", json_schema=_json_schema_from_dict(schema)
                ),
                timeout=timeout_s,
                temperature=self._temperature,
            )
            output = await model.generate(messages, config=config)
            self._confirm_temperature_sent()
            return _reply_from_response_schema(output, self._model_str)

        tool = _verdict_tool(schema)
        config = GenerateConfig(timeout=timeout_s, temperature=self._temperature)
        from inspect_ai.tool import ToolFunction

        output = await model.generate(
            messages, tools=[tool], tool_choice=ToolFunction(name=_VERDICT_TOOL_NAME), config=config
        )
        self._confirm_temperature_sent()
        return _reply_from_tool_call(output, self._model_str)

    def _confirm_temperature_sent(self) -> None:
        """Refuse when a set temperature is missing from the request the provider sent.

        Reads the request Inspect recorded for the call just made (the last model event in its
        transcript). With no temperature set there is nothing to confirm.
        """
        if self._temperature is None:
            return
        from inspect_ai.event import ModelEvent
        from inspect_ai.log import transcript

        event = next((e for e in reversed(transcript().events) if isinstance(e, ModelEvent)), None)
        request = event.call.request if event is not None and event.call is not None else None
        if request is None or not _has_key(request, "temperature"):
            raise JudgeConfigurationError(
                f"judge model {self._model_str!r} was asked for temperature {self._temperature} "
                "but the request sent to the provider carries none (the provider drops it for "
                "this model); remove temperature from the judge file"
            )


def _has_key(doc: object, key: str) -> bool:
    """Whether ``key`` names a non-null value anywhere in a nested request dict."""
    if isinstance(doc, dict):
        return any((k == key and v is not None) or _has_key(v, key) for k, v in doc.items())
    if isinstance(doc, list):
        return any(_has_key(v, key) for v in doc)
    return False


def _verdict_tool(schema: dict[str, Any]) -> Any:
    """Build the forced verdict tool whose parameters ARE the verdict schema."""
    from inspect_ai.tool import ToolInfo
    from inspect_ai.tool._tool_params import ToolParams

    props = schema.get("properties", {})
    params = ToolParams(
        properties={k: _json_schema_from_dict(v) for k, v in props.items()},
        required=list(schema.get("required", [])),
    )
    return ToolInfo(
        name=_VERDICT_TOOL_NAME,
        description="Record the binary judge verdict with its critique. Call this tool once.",
        parameters=params,
    )


def _reply_from_response_schema(output: ModelOutput, model_str: str) -> StructuredReply:
    """Parse a native structured-output completion into a StructuredReply.

    Inspect does not guarantee schema adherence (verified docstring), so a completion that is not
    valid JSON is an ``ExecutionError`` here; never a free-text parse guess handed downstream.
    """
    text = output.completion
    try:
        fields = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ExecutionError(
            f"judge model {model_str!r} returned a non-JSON response_schema completion "
            f"({text[:120]!r}...); no structured verdict to extract"
        ) from exc
    if not isinstance(fields, dict):
        raise ExecutionError(
            f"judge model {model_str!r} response_schema completion was not a JSON object"
        )
    return StructuredReply(fields=fields, via="response_schema")


def _reply_from_tool_call(output: ModelOutput, model_str: str) -> StructuredReply:
    """Extract the forced tool-call arguments into a StructuredReply.

    Reads the verdict from the tool-call arguments (the structured channel) not from any text the
    model also emitted. A reply carrying no matching tool call is an ``ExecutionError`` (then
    surfaced as the typed ``JudgeRuntimeError`` at the ``judges/`` boundary).
    """
    for choice in output.choices:
        for call in choice.message.tool_calls or []:
            if call.function == _VERDICT_TOOL_NAME:
                if call.parse_error:
                    raise ExecutionError(
                        f"judge model {model_str!r} emitted an unparseable {_VERDICT_TOOL_NAME} "
                        f"tool call: {call.parse_error}"
                    )
                return StructuredReply(fields=dict(call.arguments), via="tool_call")
    raise ExecutionError(
        f"judge model {model_str!r} returned no {_VERDICT_TOOL_NAME} tool call; "
        "refusing to parse a verdict from free text"
    )


class CassetteJudgeModel:
    """Offline judge-model backend backed by a :class:`Cassette` (test double + deterministic CI).

    Mirrors :class:`InspectJudgeModel`'s surface so ``judges/`` code is identical in both worlds.
    ``supports_response_schema`` is explicit (defaults to True) so a test can exercise either path;
    the recorded ``ModelOutput`` JSON is replayed by content fingerprint. A REPLAY miss raises
    :class:`CassetteMiss`.

    A repeat of an identical request (``sample`` above 0) and a set ``temperature`` are part of
    the fingerprint, so a repeat replays its own recorded draw. Neither is in the fingerprint at
    its default, so cassettes recorded before either existed still replay.

    ``model_id`` names the model this run judges with (it defaults to ``record_model``). Entries
    are recorded under it, and replaying an entry recorded with a different model or temperature
    raises :class:`~calibrated_agent_evaluation.execute.cassette.CassetteModelMismatch`.
    """

    def __init__(
        self,
        cassette: Cassette,
        *,
        supports_schema: bool = True,
        record_model: str | None = None,
        temperature: float | None = None,
        model_id: str | None = None,
    ) -> None:
        self._cassette = cassette
        self._supports_schema = supports_schema
        # Only needed in RECORD/AUTO mode to mint a real ModelOutput; unit tests use REPLAY.
        self._record_model = record_model
        self._temperature = temperature
        config = {} if temperature is None else {"temperature": temperature}
        self._model = model_stamp(model_id or record_model, config)

    def supports_response_schema(self) -> bool:
        """Whether this backend honors a response schema."""
        return self._supports_schema

    def generate_structured(
        self,
        *,
        system: str,
        user: str,
        schema: dict[str, Any],
        sample: int = 0,
    ) -> StructuredReply:
        """Return the recorded structured reply for this request."""
        request: dict[str, Any] = {
            "system": system,
            "user": user,
            "schema": schema,
            "via": "response_schema" if self._supports_schema else "tool_call",
        }
        if sample:
            request["sample"] = sample
        if self._temperature is not None:
            request["temperature"] = self._temperature
        fp = fingerprint(request)
        # RECORD means re-record: a stored entry is not consulted, or re-recording a cassette would
        # silently keep every old verdict. A model mismatch on lookup propagates as its own type.
        hit = (
            None
            if self._cassette.mode is Mode.RECORD
            else self._cassette.lookup(fp, model=self._model)
        )
        if hit is not None:
            return StructuredReply(fields=hit["fields"], via=hit["via"])
        if self._cassette.mode is Mode.REPLAY:
            raise CassetteMiss(
                f"judge cassette {self._cassette.path.name}: no recorded verdict for fingerprint "
                f"{fp} (REPLAY mode does not fall through to a live call)"
            )
        # RECORD/AUTO miss: delegate to a real Inspect call, capture the normalized reply.
        if self._record_model is None:
            raise ExecutionError(
                "CassetteJudgeModel miss in RECORD/AUTO mode but no record_model was set; "
                "pass record_model= to capture a live verdict"
            )
        reply = InspectJudgeModel(
            self._record_model, temperature=self._temperature
        ).generate_structured(system=system, user=user, schema=schema)
        try:
            self._cassette.record(
                fp, request, {"fields": reply.fields, "via": reply.via}, model=self._model
            )
        except CassetteError as exc:
            # Permanent (an unwritable or corrupt cassette), so not retried as a model failure.
            raise JudgeConfigurationError(f"the live verdict could not be recorded: {exc}") from exc
        return reply
