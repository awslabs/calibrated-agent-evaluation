# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""The model stack the failure detector and the draft namer call, and the guards they share.

The detector library calls the model it is given and sends any exception the call raises through
its own context check, which reads certain substrings of the message as "the input was too long"
and then retries the session in chunks. So the model the library sees is a stack of wrappers, each
a strands ``Model`` that hands every failure up as a
:class:`~calibrated_agent_evaluation.agentcore.DetectorError` with a fixed message the check cannot
misread:

- the capturing wrapper, which lets through only the one call a detection or a naming makes,
  records its text, and refuses any other call before it is sent;
- a ``CassetteModel``, when a cassette path is given, which records and replays the calls;
- the live-call cap, which refuses a live call past ``max_live_calls`` before it is sent and stamps
  the resolved model id on every cassette entry;
- the caller's model.

After the call, :func:`call_error` turns whatever the library raised into the error the caller
gets, with the original exception chained.

Each detector or namer builds its own stack with :func:`prepare_model` and makes one call through
it at a time, so the wrappers keep their counts without locks of their own.
"""

from __future__ import annotations

import asyncio
import contextvars
import json
import math
from collections.abc import (
    AsyncGenerator,
    AsyncIterable,
    AsyncIterator,
    Callable,
    Mapping,
    Sequence,
)
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Literal

from botocore.config import Config as BotocoreConfig
from strands.models.bedrock import BedrockModel
from strands.models.model import Model
from strands.types.content import Messages, SystemContentBlock
from strands.types.exceptions import ContextWindowOverflowException
from strands.types.streaming import StreamEvent
from strands.types.tools import ToolChoice, ToolSpec

from calibrated_agent_evaluation.agentcore.errors import (
    DetectorCallError,
    DetectorCallLimitExceeded,
    DetectorConfigError,
    DetectorError,
    DetectorSessionTooLarge,
)
from calibrated_agent_evaluation.execute.cassette import (
    Cassette,
    CassetteError,
    Mode,
    mode_from_env,
)
from calibrated_agent_evaluation.execute.strands_cassette_model import CassetteModel, DeferredModel

__all__ = [
    "CONTEXT_SUBSTRINGS",
    "PreparedModel",
    "call_error",
    "detector_bedrock_model",
    "prepare_model",
    "run_stream",
    "wrap_model",
]

#: The message substrings the detector library reads as a context overflow, in lower case. It
#: matches them against the lower-cased message of any exception a model call raises.
CONTEXT_SUBSTRINGS: tuple[str, ...] = (
    "context window",
    "context length",
    "context_length",
    "too long",
    "max_tokens",
    "input_limit",
    "input too large",
)

# A local copy: the vendor-free types module, which names the levels too, may not import this one.
_CONFIDENCE_LEVELS = ("low", "medium", "high")

_DEFAULT_TIMEOUT_SECONDS = 120.0


class _CallFailed(DetectorError):
    """Carries an exception raised beneath the detector library, chained, past its context check.

    ``origin`` is ``"model"`` when the caller's model raised it and ``"stack"`` when anything
    between the capturing wrapper and the model did (the cassette, for example). Only a model
    exception can mean the input was over the model's limit.
    """

    def __init__(self, origin: Literal["model", "stack"], *, cause: BaseException) -> None:
        self.origin: Literal["model", "stack"] = origin
        super().__init__("a model call failed; the original error is chained as the cause")
        self.__cause__ = cause


class _CallRefused(DetectorError):
    """A model call other than the one the capturing wrapper was armed for; never sent."""

    def __init__(self) -> None:
        super().__init__(
            "a model call was refused before it was sent: only the one expected call is allowed"
        )


class _LiveCallCap(Model):
    """Sits directly on the caller's model, so every call that reaches it is a live call.

    It refuses the call past ``max_live_calls`` before delegating it, counting over its whole
    lifetime, and carries any other exception the model raises as ``_CallFailed("model")``. Its
    configuration carries the resolved model id, which a cassette above it stamps on each entry.
    """

    def __init__(self, inner: Model, *, model_id: str, max_live_calls: int) -> None:
        self._inner = inner
        self._model_id = model_id
        self._max_live_calls = max_live_calls
        self._live_calls = 0

    def get_config(self) -> dict[str, Any]:
        """The caller's model configuration, with ``model_id`` set to the resolved id."""
        return {**dict(self._inner.get_config() or {}), "model_id": self._model_id}

    def update_config(self, **model_config: Any) -> None:
        """Update the caller's model configuration."""
        self._inner.update_config(**model_config)

    def structured_output(self, *args: Any, **kwargs: Any) -> AsyncGenerator[dict[str, Any], None]:
        """Delegate structured output to the caller's model."""
        return self._inner.structured_output(*args, **kwargs)

    async def stream(
        self,
        messages: Messages,
        tool_specs: list[ToolSpec] | None = None,
        system_prompt: str | None = None,
        *,
        tool_choice: ToolChoice | None = None,
        system_prompt_content: list[SystemContentBlock] | None = None,
        invocation_state: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> AsyncIterable[StreamEvent]:
        """Stream from the caller's model, unless the call would exceed ``max_live_calls``."""
        if self._live_calls >= self._max_live_calls:
            raise DetectorCallLimitExceeded(self._max_live_calls)
        self._live_calls += 1
        extra = _given(tool_choice, system_prompt_content, invocation_state)
        call = _Call(
            lambda: self._inner.stream(messages, tool_specs, system_prompt, **extra, **kwargs)
        )
        while (event := await call.next("model")) is not _END:
            yield event


class _CapturingModel(Model):
    """The model the detector library calls: one expected call per :meth:`arm`, captured.

    After ``arm(expected_prompt)``, the next ``stream`` call goes through only if it is the first
    call since then and its messages are exactly one user message whose one content block is the
    text ``expected_prompt``. Any other call raises ``_CallRefused`` before it reaches the model.
    ``text`` holds the call's joined text deltas and ``completed`` says whether its stream ended
    without an error. A :class:`~calibrated_agent_evaluation.agentcore.DetectorError` from below
    passes through; any other exception is carried as ``_CallFailed("stack")``.
    """

    def __init__(self, inner: Model) -> None:
        self._inner = inner
        self._expected: str | None = None
        self._calls_since_arm = 0
        self.text = ""
        self.completed = False

    def arm(self, expected_prompt: str) -> None:
        """Start one detection or naming: allow one call carrying ``expected_prompt``."""
        self._expected = expected_prompt
        self._calls_since_arm = 0
        self.text = ""
        self.completed = False

    def get_config(self) -> Any:
        """The configuration of the stack below, which carries the resolved model id."""
        return self._inner.get_config()

    def update_config(self, **model_config: Any) -> None:
        """Update the configuration of the stack below."""
        self._inner.update_config(**model_config)

    def structured_output(self, *args: Any, **kwargs: Any) -> AsyncGenerator[dict[str, Any], None]:
        """Delegate structured output to the stack below."""
        return self._inner.structured_output(*args, **kwargs)

    async def stream(
        self,
        messages: Messages,
        tool_specs: list[ToolSpec] | None = None,
        system_prompt: str | None = None,
        *,
        tool_choice: ToolChoice | None = None,
        system_prompt_content: list[SystemContentBlock] | None = None,
        invocation_state: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> AsyncIterable[StreamEvent]:
        """Stream the one expected call from the stack below, recording its text."""
        self._calls_since_arm += 1
        allowed = (
            self._expected is not None
            and self._calls_since_arm == 1
            and _is_one_user_prompt(messages, self._expected)
        )
        if not allowed:
            raise _CallRefused()
        extra = _given(tool_choice, system_prompt_content, invocation_state)
        call = _Call(
            lambda: self._inner.stream(messages, tool_specs, system_prompt, **extra, **kwargs)
        )
        chunks: list[str] = []
        try:
            while (event := await call.next("stack")) is not _END:
                try:
                    text = _text_delta(event)
                except TypeError as exc:
                    raise _CallFailed("stack", cause=exc) from exc
                if text is not None:
                    chunks.append(text)
                yield event
        finally:
            # Recorded whether or not the stream finished; ``completed`` tells the two apart.
            self.text = "".join(chunks)
        self.completed = True


_END: Any = object()


class _Call:
    """One ``stream`` call on the model below, started lazily and read one event at a time.

    Only what the model below raises, while the call is started or read, is carried as
    ``_CallFailed``; an exception thrown into the wrapper's own generator is left alone.
    """

    def __init__(self, start: Callable[[], AsyncIterable[Any]]) -> None:
        self._start = start
        self._events: AsyncIterator[Any] | None = None

    async def next(self, origin: Literal["model", "stack"]) -> Any:
        """The next event, or ``_END`` when the stream is over."""
        try:
            if self._events is None:
                self._events = aiter(self._start())
            return await anext(self._events)
        except StopAsyncIteration:
            return _END
        except DetectorError:
            raise
        except Exception as exc:
            raise _CallFailed(origin, cause=exc) from exc


def detector_bedrock_model(
    model_id: str,
    *,
    region: str | None = None,
    timeout_seconds: float = _DEFAULT_TIMEOUT_SECONDS,
    temperature: float | None = None,
) -> Model:
    """A Bedrock model for the detector or the namer, built only when a live call needs it.

    The ``BedrockModel`` and its client are built on the first live call, with connect and read
    timeouts of ``timeout_seconds`` each, so a cassette replay builds no client, resolves no AWS
    credentials and opens no connection. The client sends each request once, whatever retries the
    environment or the AWS profile configure, so one live call is one billed ``ConverseStream``
    request; a throttled or failed call raises instead of being retried.

    Args:
        model_id: The Bedrock model id, for example ``"global.anthropic.claude-sonnet-4-6"``.
        region: The AWS region of the Bedrock runtime; ``None`` uses the configured default.
        timeout_seconds: The connect and read timeout of the Bedrock client, in seconds.
        temperature: The sampling temperature; ``None`` leaves it out of the request.

    Returns:
        A strands ``Model`` whose configuration is ``{"model_id": model_id}``, plus
        ``temperature`` when one is given.

    Raises:
        DetectorConfigError: ``timeout_seconds`` is not a positive, finite number.
    """
    if (
        isinstance(timeout_seconds, bool)
        or not isinstance(timeout_seconds, int | float)
        or not math.isfinite(timeout_seconds)
        or timeout_seconds <= 0
    ):
        raise DetectorConfigError(
            f"timeout_seconds={timeout_seconds!r} is not a usable timeout: pass a positive number "
            f"of seconds (the default is {_DEFAULT_TIMEOUT_SECONDS:g})"
        )
    # A null temperature is left out rather than sent: the models that refuse an explicit
    # temperature refuse a null one too.
    decoding: dict[str, Any] = {} if temperature is None else {"temperature": temperature}

    def build() -> Model:
        # Bounded connect and read, so an unreachable Bedrock endpoint raises within the window
        # instead of waiting on an SDK default that can change between releases. One attempt,
        # so one live call is one billed request: each SDK retry would be another, uncounted by
        # max_live_calls. The mode is named: left out, it would come from the environment or
        # the AWS profile.
        client_config = BotocoreConfig(
            connect_timeout=timeout_seconds,
            read_timeout=timeout_seconds,
            retries={"mode": "standard", "total_max_attempts": 1},
        )
        return BedrockModel(
            boto_client_config=client_config, region_name=region, model_id=model_id, **decoding
        )

    return DeferredModel(build, {"model_id": model_id, **decoding})


def wrap_model(
    model: Model,
    *,
    model_id: str,
    cassette_path: Path | None,
    cassette_mode: Mode | None,
    max_live_calls: int,
) -> _CapturingModel:
    """Build the model stack: capturing, then a cassette when a path is given, then the cap.

    Args:
        model: The caller's strands model.
        model_id: The resolved model id, stamped on every cassette entry.
        cassette_path: The cassette file, or ``None`` to make every call live.
        cassette_mode: The cassette mode; ``None`` reads
            ``CALIBRATED_AGENT_EVALUATION_CASSETTE_MODE`` (default auto).
        max_live_calls: How many live calls the stack may make over its lifetime.

    Returns:
        The capturing wrapper at the top of the stack.

    Raises:
        DetectorConfigError: The cassette mode is not a mode, or the cassette cannot be loaded.
    """
    capped = _LiveCallCap(model, model_id=model_id, max_live_calls=max_live_calls)
    if cassette_path is None:
        return _CapturingModel(capped)
    cassette = _load_cassette(cassette_path, _resolve_mode(cassette_mode))
    return _CapturingModel(CassetteModel(capped, cassette))


class PreparedModel:
    """A checked model stack, and the model identity each call through it is recorded under.

    :func:`prepare_model` builds one for each detector or namer, which makes one call through it at
    a time.

    Attributes:
        model: The capturing wrapper at the top of the stack; arm it before each call.
        model_id: The model id resolved when the stack was built, which the stack stamps on every
            cassette entry and every result records.
    """

    def __init__(
        self,
        stack: _CapturingModel,
        *,
        caller_model: Model,
        model_id: str,
        passed_model_id: str | None,
    ) -> None:
        self.model = stack
        self.model_id = model_id
        self._caller_model = caller_model
        self._passed_model_id = passed_model_id

    def config_sent(self) -> dict[str, Any]:
        """The caller's model configuration as it is now, without ``model_id``, JSON-safe.

        Read before each call, so a result records the configuration its call was made with.

        Returns:
            The configuration the next call is made with.

        Raises:
            DetectorConfigError: The configuration cannot be read or recorded, or the model id it
                holds is no longer the one the stack was built for, so the result and its cassette
                entry would name the wrong model. Nothing is sent.
        """
        config = _read_config(self._caller_model)
        configured = config.get("model_id")
        unchanged = configured == self.model_id or (
            configured is None and self._passed_model_id == self.model_id
        )
        if not unchanged:
            raise DetectorConfigError(
                f"the model's config now holds model_id={configured!r}, but the detector or namer "
                f"was built for {self.model_id!r}, so its result and cassette entry would name the "
                "wrong model; nothing was sent: build a new detector or namer for the model as it "
                "is configured now"
            )
        return _json_safe({key: value for key, value in config.items() if key != "model_id"})


def prepare_model(
    model: object,
    *,
    model_id: str | None = None,
    cassette_path: str | Path | None = None,
    cassette_mode: Mode | None = None,
    max_live_calls: int = 0,
    confidence_threshold: str | None = None,
) -> PreparedModel:
    """Check a detector's or a namer's configuration, then build its model stack.

    Everything is checked before any model call, so a configuration that cannot run is refused
    when the detector or the namer is built.

    Args:
        model: The caller's strands ``Model``. ``None`` and a model id string are refused: the
            detector library would build a Bedrock model of its own from them.
        model_id: The model id, when the model's configuration has none; if both are given they
            must be equal.
        cassette_path: The cassette file that records and replays the calls, or ``None``.
        cassette_mode: The cassette mode; ``None`` reads
            ``CALIBRATED_AGENT_EVALUATION_CASSETTE_MODE`` (default auto).
        max_live_calls: How many live model calls may be made, 0 or more.
        confidence_threshold: The detector's threshold, ``"low"``, ``"medium"`` or ``"high"``;
            ``None`` for the namer, which has none.

    Returns:
        The model stack with the resolved model id.

    Raises:
        DetectorConfigError: Any of the checks above fails,
            ``CALIBRATED_AGENT_EVALUATION_CASSETTE_MODE`` names no cassette mode, the cassette
            cannot be loaded, the cassette does not exist while the mode is replay, or the mode is
            replay and no cassette is given.
    """
    checked = _as_model(model)
    config = _read_config(checked)
    resolved_id = _resolve_model_id(config.get("model_id"), model_id)
    if confidence_threshold is not None and not (
        isinstance(confidence_threshold, str) and confidence_threshold in _CONFIDENCE_LEVELS
    ):
        raise DetectorConfigError(
            f"confidence_threshold={confidence_threshold!r} is not a confidence level: use "
            "'low', 'medium' or 'high', in lower case"
        )
    if (
        isinstance(max_live_calls, bool)
        or not isinstance(max_live_calls, int)
        or max_live_calls < 0
    ):
        raise DetectorConfigError(
            f"max_live_calls={max_live_calls!r} is not a number of live calls: pass a whole "
            "number, 0 or more (0 allows no live call, so only cassette replays run)"
        )
    # Resolved even without a cassette, so a mistyped CALIBRATED_AGENT_EVALUATION_CASSETTE_MODE is
    # never ignored.
    mode = _resolve_mode(cassette_mode)
    path = None if cassette_path is None else Path(cassette_path)
    if path is None and mode is Mode.REPLAY:
        # Replay makes no call to the model; with nothing to replay from, every call would.
        source = (
            "CALIBRATED_AGENT_EVALUATION_CASSETTE_MODE"
            if cassette_mode is None
            else "cassette_mode"
        )
        raise DetectorConfigError(
            f"{source} selects replay, which makes no live call, but no cassette_path was given, "
            "so every call would reach the model: pass the cassette to replay, or choose the auto "
            "or record mode to make live calls"
        )
    if path is not None and mode is Mode.REPLAY and not _exists(path):
        # A missing file loads as an empty cassette, so every replayed call would miss.
        raise DetectorConfigError(
            f"the cassette {path} does not exist, and replay mode makes no live call, so every "
            "call would miss: record the cassette first, or fix the path"
        )
    stack = wrap_model(
        checked,
        model_id=resolved_id,
        cassette_path=path,
        cassette_mode=mode,
        max_live_calls=max_live_calls,
    )
    prepared = PreparedModel(
        stack, caller_model=checked, model_id=resolved_id, passed_model_id=model_id
    )
    # Read once here too, so a configuration that cannot be recorded is refused before any call.
    prepared.config_sent()
    return prepared


def call_error(exc: BaseException, *, trace_id: str | None = None) -> DetectorError:
    """The error to raise for whatever a detection's or a naming's model call raised.

    A given ``trace_id`` means the call was a detection. The returned error already has its
    ``__cause__`` set. Its message names the trace when there is one, and gives only the type name
    of the original exception, whose text can hold credentials or prompt text.

    Args:
        exc: The exception the call raised, from the model stack or the detector library.
        trace_id: The id of the trace being detected; ``None`` for a naming call.

    Returns:
        ``exc`` itself when it is a :class:`~calibrated_agent_evaluation.agentcore.DetectorError`
        the stack raised in its own right, such as the live-call limit; for a detection, a
        :class:`~calibrated_agent_evaluation.agentcore.DetectorSessionTooLarge` when the model
        reported a context overflow or the library asked for a call other than the one direct call;
        otherwise a :class:`~calibrated_agent_evaluation.agentcore.DetectorCallError` chained from
        the original exception.
    """
    if isinstance(exc, _CallRefused):
        if trace_id is not None:
            return _chained(
                DetectorSessionTooLarge(
                    f"trace {trace_id!r} is too large for one detector call: the detector library "
                    "asked for a model call other than the one direct call, and it was refused "
                    "before it was sent; detect a smaller trace, or use a model that reads more"
                ),
                exc,
            )
        return _chained(
            DetectorCallError(
                "a model call other than the one naming call was asked for, and it was refused "
                "before it was sent"
            ),
            exc,
        )
    if isinstance(exc, _CallFailed):
        cause = exc.__cause__ if exc.__cause__ is not None else exc
        if trace_id is not None and exc.origin == "model" and _reports_overflow(cause):
            return _chained(
                DetectorSessionTooLarge(
                    f"trace {trace_id!r} is too large for the detector's model: the model refused "
                    f"it as over its input limit ({type(cause).__name__}); detect a smaller "
                    "trace, or use a model that reads more"
                ),
                cause,
            )
        return _chained(DetectorCallError(_failed_message(cause, trace_id)), cause)
    if isinstance(exc, DetectorError):
        return exc
    return _chained(DetectorCallError(_failed_message(exc, trace_id)), exc)


def run_stream(model: Model, messages: Messages) -> str:
    """Make one ``stream`` call on ``model`` and return its joined text deltas.

    The call runs in a worker thread with its own event loop, so it also works when the caller is
    itself inside a running event loop. Whatever the call raises propagates unchanged.

    Args:
        model: The model to call, usually a :class:`PreparedModel`'s armed stack.
        messages: The messages to send.

    Returns:
        The text of every text delta the stream yielded, joined in order.
    """

    async def drain() -> str:
        chunks: list[str] = []
        async for event in model.stream(messages):
            text = _text_delta(event)
            if text is not None:
                chunks.append(text)
        return "".join(chunks)

    def execute() -> str:
        return asyncio.run(drain())

    with ThreadPoolExecutor(max_workers=1) as executor:
        context = contextvars.copy_context()
        return executor.submit(context.run, execute).result()


def _given(
    tool_choice: ToolChoice | None,
    system_prompt_content: list[SystemContentBlock] | None,
    invocation_state: dict[str, Any] | None,
) -> dict[str, Any]:
    """The keyword arguments the caller passed, so the model below gets exactly those."""
    given = {
        "tool_choice": tool_choice,
        "system_prompt_content": system_prompt_content,
        "invocation_state": invocation_state,
    }
    return {name: value for name, value in given.items() if value is not None}


def _is_one_user_prompt(messages: object, expected: str) -> bool:
    """Whether ``messages`` is one user message whose one content block is the text ``expected``."""
    if not isinstance(messages, Sequence) or isinstance(messages, str) or len(messages) != 1:
        return False
    message = messages[0]
    if not isinstance(message, Mapping) or message.get("role") != "user":
        return False
    content = message.get("content")
    if not isinstance(content, Sequence) or isinstance(content, str) or len(content) != 1:
        return False
    block = content[0]
    return isinstance(block, Mapping) and set(block) == {"text"} and block["text"] == expected


def _text_delta(event: object) -> str | None:
    """The text of a ``contentBlockDelta`` text event, read as the detector library reads it."""
    if not isinstance(event, Mapping) or "contentBlockDelta" not in event:
        return None
    block = event["contentBlockDelta"]
    delta = block.get("delta", {}) if isinstance(block, Mapping) else None
    if not isinstance(delta, Mapping) or "text" not in delta:
        return None
    text = delta["text"]
    if not isinstance(text, str):
        raise TypeError(f"a text delta in the model's stream is {type(text).__name__}, not text")
    return text


def _as_model(model: object) -> Model:
    if model is None:
        raise DetectorConfigError(
            "no model was passed: pass a strands Model, for example "
            "detector_bedrock_model(model_id); the detector library would otherwise build a "
            "Bedrock model of its own and bill calls nobody named"
        )
    if isinstance(model, str):
        raise DetectorConfigError(
            f"a model id string ({model!r}) was passed as the model: pass "
            f"detector_bedrock_model({model!r}) instead, so its calls are capped and recorded"
        )
    if not isinstance(model, Model):
        raise DetectorConfigError(
            f"the model must be a strands Model instance, not {type(model).__name__}; "
            "detector_bedrock_model(model_id) builds one for Bedrock"
        )
    return model


def _read_config(model: Model) -> Mapping[str, Any]:
    try:
        config = model.get_config()
    except Exception as exc:
        raise DetectorConfigError(
            f"the model's configuration cannot be read ({type(exc).__name__}): fix the model's "
            "get_config(), whose error is chained as the cause, or pass another model"
        ) from exc
    if config is None:
        return {}
    if not isinstance(config, Mapping):
        raise DetectorConfigError(
            f"the model's get_config() returned {type(config).__name__}, not a mapping, so its "
            "model id cannot be read or recorded: pass a model whose get_config() returns a "
            "mapping, as strands models do"
        )
    return config


def _resolve_model_id(configured: object, passed: object) -> str:
    if configured is not None and not (isinstance(configured, str) and configured):
        raise DetectorConfigError(
            f"the model's config holds model_id={configured!r}, which is not a model id: "
            "configure the model with its id as a non-empty string"
        )
    if passed is not None and not (isinstance(passed, str) and passed):
        raise DetectorConfigError(
            f"model_id={passed!r} is not a model id: pass the model's id as a non-empty string, "
            "or leave model_id out when the model's config has one"
        )
    if isinstance(configured, str) and isinstance(passed, str) and configured != passed:
        raise DetectorConfigError(
            f"model_id={passed!r} was passed, but the model's config says {configured!r}: pass "
            "the id the model runs with, or leave model_id out"
        )
    for resolved in (passed, configured):
        if isinstance(resolved, str):
            return resolved
    raise DetectorConfigError(
        "the model's config has no model_id and none was passed: pass model_id= so every "
        "detection and cassette entry records the model that produced it"
    )


def _resolve_mode(mode: Mode | None) -> Mode:
    if mode is None:
        try:
            return mode_from_env()
        except ValueError as exc:
            raise DetectorConfigError(
                f"{exc}; fix CALIBRATED_AGENT_EVALUATION_CASSETTE_MODE, or pass cassette_mode="
            ) from exc
    try:
        return Mode(mode)
    except ValueError as exc:
        valid = ", ".join(m.value for m in Mode)
        raise DetectorConfigError(
            f"cassette_mode={mode!r} is not a cassette mode (expected one of: {valid})"
        ) from exc


def _exists(path: Path) -> bool:
    try:
        return path.exists()
    except OSError as exc:
        raise DetectorConfigError(
            f"the cassette path {path} cannot be checked ({type(exc).__name__}: {exc}); check "
            "that the path and the directories above it can be read, or pass another "
            "cassette_path"
        ) from exc


def _load_cassette(path: Path, mode: Mode) -> Cassette:
    try:
        return Cassette(path, mode=mode)
    except CassetteError as exc:
        # Its message names the file and what is wrong with it.
        raise DetectorConfigError(f"{exc}; fix the file or record the cassette again") from exc
    except OSError as exc:
        raise DetectorConfigError(
            f"the cassette {path} cannot be loaded: {exc}; check that the file and its directory "
            "can be read, or pass another cassette_path"
        ) from exc


def _json_safe(config: Mapping[str, Any]) -> dict[str, Any]:
    # The same encoding the cassette applies to the config it stamps, so the two compare equal.
    try:
        return json.loads(json.dumps(dict(config), default=str, sort_keys=True))
    except (TypeError, ValueError) as exc:
        raise DetectorConfigError(
            f"the model's configuration cannot be recorded as JSON ({type(exc).__name__}: {exc}); "
            "configure the model with plain JSON data under string keys"
        ) from exc


def _reports_overflow(cause: BaseException) -> bool:
    """Whether the model's exception says the input was over its limit, as the library reads it."""
    if isinstance(cause, ContextWindowOverflowException):
        return True
    message = str(cause).lower()
    return any(substring in message for substring in CONTEXT_SUBSTRINGS)


def _failed_message(cause: BaseException, trace_id: str | None) -> str:
    call = "the model call" if trace_id is None else f"the detector's call for trace {trace_id!r}"
    return (
        f"{call} failed ({type(cause).__name__}); the original error is chained as the cause and "
        "says what went wrong"
    )


def _chained(error: DetectorError, cause: BaseException) -> DetectorError:
    error.__cause__ = cause
    return error
