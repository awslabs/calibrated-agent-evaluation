# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Record AgentCore Runtime runs into a cassette once, and replay them offline.

The executor reuses the library's :class:`~calibrated_agent_evaluation.execute.cassette.Cassette`;
this module knows what one runtime run looks like in it.

- **The request** an entry is found by (:func:`cassette_request`) holds the payload format, the
  custom builder's name and the prompt, but no session or trace ID, which every live run makes
  anew, so a recorded prompt is found again.
- **The stamp** an entry is recorded under (:func:`cassette_stamp`) names the runtime ARN, the
  qualifier and the configuration bundle. The executor refuses an entry recorded under another
  stamp, or under none, so one runtime's answers are never replayed as another's.
- **The response** (:class:`RecordedRun`, written by :func:`encode_entry` and read back by
  :func:`decode_entry`) holds what the run returned and carried, but not the sampling frame or
  the taxonomy cell, which a replay takes from the executor and the task. It is standard JSON: a
  run holding a NaN or infinite number, or nested more than 256 levels deep counted from the run,
  is refused rather than recorded, so a value or span in it may nest at most 254 levels.

:func:`open_cassette` refuses a cassette missing in replay mode or not readable as one. Every
refusal raises :class:`~calibrated_agent_evaluation.agentcore.runtime_errors.RuntimeInvocationError`
naming the cassette's path, never quoting what the cassette holds.
"""

from __future__ import annotations

import json
import re
from collections.abc import Iterable, Mapping
from datetime import datetime
from pathlib import Path
from typing import Any, ClassVar

from pydantic import ConfigDict, StrictInt, ValidationError, field_validator

from calibrated_agent_evaluation._errors import CalibratedAgentEvaluationDataError
from calibrated_agent_evaluation._finite import (
    MAX_JSON_DEPTH,
    NestedTooDeepError,
    NonFiniteNumberError,
    require_finite,
)
from calibrated_agent_evaluation._model import CalibratedAgentEvaluationModel, replace
from calibrated_agent_evaluation.agentcore.run_manifest import VersionSource
from calibrated_agent_evaluation.agentcore.runtime import (
    CUSTOM_PAYLOAD_FORMAT,
    SESSION_ID_PATTERN,
    W3C_TRACE_ID_PATTERN,
    RuntimeTarget,
)
from calibrated_agent_evaluation.agentcore.runtime_errors import RuntimeInvocationError
from calibrated_agent_evaluation.execute.cassette import Cassette, CassetteError, Mode, model_stamp
from calibrated_agent_evaluation.schema.errors import SchemaError
from calibrated_agent_evaluation.schema.types import Span

__all__ = [
    "EXECUTOR_NAME",
    "RecordedRun",
    "cassette_request",
    "cassette_stamp",
    "decode_entry",
    "encode_entry",
    "open_cassette",
    "rekey_spans",
]

EXECUTOR_NAME = "agentcore-runtime"
"""The ``executor`` every AgentCore Runtime request in a cassette names."""

_SESSION_ID = re.compile(SESSION_ID_PATTERN)
_W3C_TRACE_ID = re.compile(W3C_TRACE_ID_PATTERN)


class RecordedRun(CalibratedAgentEvaluationModel):
    """One AgentCore Runtime run, as a cassette entry's response records it.

    Attributes:
        content_type: The response's content type.
        status_code: The response's HTTP status.
        values: The decoded response values, in the order the body held them.
        spans: The run's span tree, every span keyed to ``trace_id``.
        runtime_version: The runtime version that served the run.
        runtime_version_source: ``endpoint`` when it was the endpoint's live version, ``latest``
            when it was the runtime's latest.
        session_id: The runtime session the run used, in the form of a run's own IDs.
        w3c_trace_id: The W3C trace ID the run's invocations carried, in the same form.
        trace_id: The ID of the run's ``Trace``.
        response_traceparent: The ``traceparent`` the runtime sent back, or ``None``.
        started_at: When the invocation started: ISO-8601 with its UTC offset. The trace's
            ``captured_at``.
        ended_at: When the invocation ended, in the same form.
        attempts: How many invocation attempts the run sent, retries included.
    """

    model_config = ConfigDict(frozen=True, extra="forbid", defer_build=True)
    invalid_error: ClassVar[type[Exception]] = RuntimeInvocationError

    content_type: str
    status_code: StrictInt
    values: list[Any]
    spans: list[Span]
    runtime_version: str
    runtime_version_source: VersionSource
    session_id: str
    w3c_trace_id: str
    trace_id: str
    response_traceparent: str | None
    started_at: str
    ended_at: str
    attempts: StrictInt

    # Checked as a run's own IDs are, since a replay hands them to its run record and trace. The
    # messages name the rule, never the value, which was read from a file.
    @field_validator("session_id")
    @classmethod
    def _is_a_session_id(cls, value: str) -> str:
        if _SESSION_ID.fullmatch(value) is None:
            raise ValueError(
                "not a run's session ID: expected calibrated-agent-evaluation- and 32 lowercase "
                "hex characters"
            )
        if not value.removeprefix("calibrated-agent-evaluation-").strip("0"):
            raise ValueError(
                "an all-zero session ID is invalid; expected calibrated-agent-evaluation- and 32 "
                "lowercase hex "
                "characters with a nonzero digit"
            )
        return value

    @field_validator("w3c_trace_id")
    @classmethod
    def _is_a_w3c_trace_id(cls, value: str) -> str:
        if _W3C_TRACE_ID.fullmatch(value) is None:
            raise ValueError("not a W3C trace ID: expected 32 lowercase hex characters")
        if not value.strip("0"):
            raise ValueError("an all-zero W3C trace ID is invalid; expected a nonzero digit")
        return value

    @field_validator("started_at", "ended_at")
    @classmethod
    def _is_a_timezone_aware_time(cls, value: str) -> str:
        # A replay times its run record from these, so each must read as a moment in time.
        try:
            moment = datetime.fromisoformat(value)
        except ValueError as exc:
            raise ValueError("not an ISO-8601 time") from exc
        if moment.tzinfo is None:
            raise ValueError("an ISO-8601 time without its UTC offset")
        return value


# Every part a replay needs; an entry without one is refused, naming it.
_PARTS: tuple[str, ...] = tuple(RecordedRun.model_fields)


def open_cassette(path: Path, mode: Mode) -> Cassette:
    """Open the cassette at ``path`` in ``mode``.

    In record and auto mode a file that doesn't exist yet is an empty cassette, which the first
    recording creates.

    Args:
        path: The cassette file.
        mode: Whether the cassette records, replays or does both.

    Returns:
        The cassette, its entries loaded.

    Raises:
        RuntimeInvocationError: The mode is replay and the file doesn't exist; or the file can't
            be read, is not UTF-8 JSON, has no ``version``, or holds an entry that is not a
            cassette entry. The message names the path.
    """
    try:
        cassette = Cassette(path, mode=mode)
    except CassetteError as exc:
        # Its message names the file and the key or type at fault, never a recorded value.
        raise RuntimeInvocationError(
            f"cassette {path} can't be read as a cassette ({exc}), so nothing was run. "
            "Re-record it, or pass another cassette"
        ) from exc
    # Asked after the cassette has read the path, so a path it can't reach is already refused.
    if mode is Mode.REPLAY and not path.exists():
        raise RuntimeInvocationError(
            f"cassette {path} does not exist, and replay mode reads every run from it, so "
            "nothing was replayed. Pass the path of a recorded cassette, or record one first"
        )
    return cassette


def cassette_request(
    target: RuntimeTarget,
    builder_name: str | None,
    prompt: str,
    *,
    conversation: tuple[str, int] | None = None,
) -> dict[str, Any]:
    """The request a run is recorded and found under.

    It holds no session or trace ID: those are new on every live run, so a fingerprint that held
    them would make every replay miss. A conversation's turn adds its conversation ID and turn
    number, so a customer message repeated on two turns is two recordings; a single-turn run's
    request, and so every single-turn cassette, is unchanged.

    Args:
        target: The runtime and payload format.
        builder_name: The custom payload builder's name; ignored for the named formats, which
            build their own body.
        prompt: The task's prompt.
        conversation: ``(conversation_id, turn)`` for a conversation's turn, else ``None``.

    Returns:
        ``{"executor", "payload_format", "builder", "prompt"}``, plus ``"conversation_id"`` and
        ``"turn"`` for a conversation's turn.
    """
    custom = target.payload_format == CUSTOM_PAYLOAD_FORMAT
    request: dict[str, Any] = {
        "executor": EXECUTOR_NAME,
        "payload_format": target.payload_format,
        "builder": builder_name if custom else None,
        "prompt": prompt,
    }
    if conversation is not None:
        request["conversation_id"], request["turn"] = conversation
    return request


def cassette_stamp(target: RuntimeTarget) -> dict[str, Any] | None:
    """The stamp a run is recorded under: the runtime ARN and qualifier, and the bundle.

    A lookup refuses an entry recorded under another stamp, so an entry from another runtime,
    qualifier or configuration bundle is never replayed as this one's.

    Args:
        target: The runtime, qualifier and bundle the run uses.

    Returns:
        ``{"id": "<runtime ARN>@<qualifier>", "config": {"config_bundle_arn",
        "config_bundle_version"}}``.
    """
    return model_stamp(
        f"{target.runtime_arn}@{target.qualifier}",
        {
            "config_bundle_arn": target.config_bundle_arn,
            "config_bundle_version": target.config_bundle_version,
        },
    )


def encode_entry(recorded: RecordedRun, *, cassette_path: Path) -> dict[str, Any]:
    """``recorded`` as the JSON-native response a cassette entry stores.

    Args:
        recorded: The run.
        cassette_path: The cassette the entry is for, named in a refusal.

    Returns:
        Plain dicts, lists, strings, numbers, booleans and nulls, exactly as a replay reads them
        back from the cassette file.

    Raises:
        RuntimeInvocationError: The run is not standard JSON: it holds a NaN or infinite number,
            nests more than 256 levels deep counted from the run (a value or span nested more
            than 254 levels inside it), or holds a value JSON has no form for. The message names
            the cassette path and the run's session, never a value.
    """
    refused = (
        f"the run of session {recorded.session_id} finished, but its entry for cassette "
        f"{cassette_path}"
    )
    outcome = "so it is not recorded and its trace is not returned"
    try:
        # Checked on the Python dump: pydantic's JSON dump writes a NaN or an infinity as null.
        require_finite(recorded.model_dump(mode="python"), where="cassette entry")
        # Through JSON text and back, so the live run reads what a later replay will read.
        encoded = json.dumps(recorded.model_dump(mode="json"), allow_nan=False)
    except NonFiniteNumberError as exc:
        raise RuntimeInvocationError(
            f"{refused} holds a number JSON cannot represent (NaN or an infinity), {outcome}. "
            "Check the agent's response and the span reader"
        ) from exc
    except (NestedTooDeepError, RecursionError) as exc:
        raise RuntimeInvocationError(
            f"{refused} nests more than {MAX_JSON_DEPTH} levels deep, so a value or span in the "
            f"run nests more than {MAX_JSON_DEPTH - 2} levels, and {outcome}. Run without a "
            "cassette, or make the agent answer with a flatter response"
        ) from exc
    except (TypeError, ValueError) as exc:
        raise RuntimeInvocationError(
            f"{refused} does not serialize as standard JSON ({type(exc).__name__}), {outcome}"
        ) from exc
    return json.loads(encoded)


def decode_entry(response: object, *, cassette_path: Path, fingerprint: str) -> RecordedRun:
    """Read a cassette entry's response back into the run it records.

    Args:
        response: The entry's response, as the cassette returns it.
        cassette_path: The cassette the entry came from, named in a refusal.
        fingerprint: The entry's fingerprint, named in a refusal.

    Returns:
        The recorded run.

    Raises:
        RuntimeInvocationError: The response is not an object, misses a part a replay needs, or
            holds a part of the wrong shape. The message names the cassette path, the fingerprint
            and the parts, never their content.
    """
    if not isinstance(response, Mapping):
        raise RuntimeInvocationError(
            f"cassette {cassette_path}: the entry for fingerprint {fingerprint} holds no recorded "
            "run, so nothing was replayed. Re-record the cassette"
        )
    missing = [part for part in _PARTS if part not in response]
    if missing:
        raise RuntimeInvocationError(
            f"cassette {cassette_path}: the entry for fingerprint {fingerprint} is missing "
            f"{', '.join(missing)}, which a replay needs, so nothing was replayed. Re-record the "
            "cassette"
        )
    try:
        return RecordedRun.model_validate(dict(response))
    except CalibratedAgentEvaluationDataError as exc:
        raise RuntimeInvocationError(
            f"cassette {cassette_path}: the entry for fingerprint {fingerprint} holds "
            f"{', '.join(_invalid_parts(exc))} in a shape the executor can't replay, so nothing "
            "was replayed. Re-record the cassette"
        ) from exc


def rekey_spans(spans: Iterable[Span], recorded_trace_id: str, trace_id: str) -> list[Span]:
    """A recorded span tree moved from the recorded trace ID to ``trace_id``.

    Every span's ``trace_id`` becomes ``trace_id``. A ``span_id`` or ``parent`` starting with
    ``<recorded_trace_id>/`` has that prefix replaced by ``<trace_id>/``; any other, such as an ID
    an adapter pinned, gets ``<trace_id>/`` in front. IDs and parents share the mapping, so every
    parent edge still resolves, and nothing else changes. Two replays of one entry under different
    trace IDs then share no span ID, so one store can keep both.

    Args:
        spans: The recorded spans.
        recorded_trace_id: The trace ID they were recorded under.
        trace_id: The trace ID to move them to.

    Returns:
        The re-keyed spans, in order; the spans unchanged when ``trace_id`` is the recorded one.

    Raises:
        SchemaError: A re-keyed span is not a valid span.
    """
    if trace_id == recorded_trace_id:
        return list(spans)
    recorded_prefix = f"{recorded_trace_id}/"

    def rekeyed(span_id: str) -> str:
        return f"{trace_id}/{span_id.removeprefix(recorded_prefix)}"

    return [
        replace(
            span,
            trace_id=trace_id,
            span_id=rekeyed(span.span_id),
            parent=None if span.parent is None else rekeyed(span.parent),
        )
        for span in spans
    ]


def _invalid_parts(exc: CalibratedAgentEvaluationDataError) -> list[str]:
    # A span that is not a span refuses itself with a schema error, before the run's model can
    # say which part held it; any other part is named by the model's own validation error.
    if isinstance(exc, SchemaError):
        return ["spans"]
    cause = exc.__cause__
    if isinstance(cause, ValidationError):
        parts = sorted({str(error["loc"][0]) for error in cause.errors() if error["loc"]})
        if parts:
            return parts
    return ["a part"]
