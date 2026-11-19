# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""An invocation's response is decoded, checked for agent errors, and read for its final answer.

Every body is read through a real botocore ``StreamingBody``, line by line, the way the executor
reads an invocation's response off the wire.
"""

from __future__ import annotations

import io
import json
from collections.abc import Iterator, Sequence
from typing import Any

import pytest

pytest.importorskip("botocore")

from botocore.response import StreamingBody

from calibrated_agent_evaluation.agentcore.runtime_errors import (
    AgentReportedError,
    RuntimeInvocationError,
)
from calibrated_agent_evaluation.agentcore.runtime_response import (
    DecodedValue,
    check_errors,
    decode,
    final_text,
)

_SESSION = "calibrated-agent-evaluation-0123456789abcdef0123456789abcdef"
_SSE = "text/event-stream"
_JSON = "application/json"
_TEXT_KEYS = ("response", "content", "message", "text", "output", "answer")
_SENTINEL_TEXT = "SENTINEL-9b2e"


def _lines(body: bytes, *, chunk_size: int) -> Iterator[bytes]:
    return StreamingBody(io.BytesIO(body), len(body)).iter_lines(chunk_size=chunk_size)


def _decode(
    content_type: str, body: bytes | str, *, chunk_size: int = 1024
) -> tuple[DecodedValue, ...]:
    raw = body.encode("utf-8") if isinstance(body, str) else body
    return decode(content_type, _lines(raw, chunk_size=chunk_size), session_id=_SESSION)


def _event(value: dict[str, Any]) -> str:
    return f"data: {json.dumps(value)}\n\n"


def _says(error: BaseException, *parts: str, never: Sequence[str] = ()) -> None:
    """Assert that the message of ``error`` names the session and every one of ``parts``, and
    holds none of ``never``."""
    message = str(error)
    for part in (_SESSION, *parts):
        assert part in message
    for part in never:
        assert part not in message


def _refused(
    content_type: str, body: bytes | str, *parts: str, never: Sequence[str] = ()
) -> RuntimeInvocationError:
    """Decode ``body``, assert it is refused as :func:`_says` checks, and return the refusal."""
    with pytest.raises(RuntimeInvocationError) as refused:
        _decode(content_type, body)
    _says(refused.value, *parts, never=never)
    return refused.value


# Event streams


def test_event_stream_values_keep_their_order_and_non_json_data_is_kept_as_text() -> None:
    body = 'data: {"response": "Hel"}\n\ndata: lo, \n\ndata: "wor"\n\ndata: [1, 2]\n\ndata: ld\n\n'

    values = _decode(_SSE, body)

    assert values == ({"response": "Hel"}, "lo, ", "wor", [1, 2], "ld")
    assert final_text(values, _SSE, session_id=_SESSION) == "Hello, world"


def test_a_data_field_without_a_space_is_read() -> None:
    values = _decode(_SSE, 'data:{"text": "a"}\n\ndata:b\n\ndata:  c\n\n')

    # Only the first space after the colon is syntax; a second one belongs to the value.
    assert values == ({"text": "a"}, "b", " c")


def test_multi_line_data_in_one_event_is_joined_with_a_newline() -> None:
    body = 'data: first\ndata: second\n\ndata: {"text":\ndata: "split"}\n\n'

    assert _decode(_SSE, body) == ("first\nsecond", {"text": "split"})


def test_event_id_retry_and_comment_lines_are_ignored() -> None:
    body = (
        ": keep-alive\nevent: message\nid: 7\nretry: 1000\ndata: answer\n\n"
        "event: ping\nid: 8\n\n"
        ":data: not a field\n\n"
    )

    assert _decode(_SSE, body) == ("answer",)


def test_a_bare_number_or_boolean_stays_text() -> None:
    values = _decode(_SSE, "data: 42\n\ndata: true\n\ndata: null\n\ndata: -1.5\n\n")

    assert values == ("42", "true", "null", "-1.5")
    assert final_text(values, _SSE, session_id=_SESSION) == "42truenull-1.5"


@pytest.mark.parametrize(
    "raw",
    [
        *["{", "[", '"', "{not json}", "[1, 2", '"unterminated', '{"a": 1} trailing'],
        # Python's json reads these tokens, but they are not JSON.
        pytest.param('{"response": "x", "score": NaN}', id="NaN"),
        pytest.param("[1, Infinity]", id="Infinity"),
        pytest.param('{"response": "x", "t": -Infinity}', id="-Infinity"),
        # Decoded, the literal would be an infinity that no standard JSON writer can write back.
        pytest.param('{"response": "x", "score": 1e999}', id="1e999"),
        pytest.param("[1, -1e999]", id="-1e999"),
    ],
)
def test_a_stream_value_that_is_not_standard_json_stays_text(raw: str) -> None:
    assert _decode(_SSE, f"data: {raw}\n\ndata: after\n\n") == (raw, "after")


def _nested_json(levels: int) -> str:
    """A JSON object holding a sentinel and a list, ``levels`` deep in all, built with a loop."""
    inner = levels - 1
    return f'{{"note": "{_SENTINEL_TEXT}", "v": ' + "[" * inner + "]" * inner + "}"


def _nested_lists(levels: int) -> list[Any]:
    value: list[Any] = []
    for _ in range(levels - 1):
        value = [value]
    return value


@pytest.mark.parametrize("levels", [257, 100_000], ids=["257-levels", "100000-levels"])
def test_a_deeply_nested_stream_value_stays_text_and_a_deeply_nested_json_body_is_refused(
    levels: int,
) -> None:
    # 257 levels decode, and the depth check refuses them; 100,000 make the decoder itself
    # raise RecursionError. The answer is the same whichever refuses.
    raw = _nested_json(levels)

    assert _decode(_SSE, f"data: {raw}\n\n") == (raw,)
    _refused(_JSON, raw, _JSON, "nests more than 256 levels", never=[_SENTINEL_TEXT, "[["])


def test_a_value_nested_256_levels_decodes() -> None:
    raw = "[" * 256 + "]" * 256
    deepest = _nested_lists(256)

    assert _decode(_SSE, f"data: {raw}\n\n") == (deepest,)
    assert _decode(_JSON, raw) == (deepest,)


def test_a_done_marker_is_skipped_and_reading_continues() -> None:
    values = _decode(_SSE, 'data: "before "\n\ndata: [DONE]\n\ndata: "after"\n\n')

    assert values == ("before ", "after")
    assert final_text(values, _SSE, session_id=_SESSION) == "before after"


@pytest.mark.parametrize("ending", ["", "\n", "\r\n"])
def test_a_final_event_without_a_closing_blank_line_is_kept(ending: str) -> None:
    assert _decode(_SSE, f"data: one\n\ndata: two{ending}") == ("one", "two")


def test_a_bare_data_line_is_a_data_field_with_an_empty_value() -> None:
    assert _decode(_SSE, "data\n\n") == ("",)
    # Inside an event it adds an empty line to the joined value.
    assert _decode(_SSE, "data: a\ndata\ndata: b\n\n") == ("a\n\nb",)


@pytest.mark.parametrize("chunk_size", [1, 1024])
def test_one_leading_byte_order_mark_is_stripped_from_an_event_stream(chunk_size: int) -> None:
    # UTF-8's byte order mark, EF BB BF, at the start of the first line.
    body = b'\xef\xbb\xbfdata: {"response": "hi"}\n\ndata: there\n\n'

    values = _decode(_SSE, body, chunk_size=chunk_size)

    assert values == ({"response": "hi"}, "there")
    # Only one, and only at the start of the body. Any other U+FEFF is text, so a line it starts
    # is an unknown field, as the browser reads it.
    assert _decode(_SSE, "\ufeff\ufeffdata: hidden\n\ndata: read\n\n") == ("read",)
    assert _decode(_SSE, "data: read\n\n\ufeffdata: hidden\n\n") == ("read",)


def test_a_byte_order_mark_inside_a_data_value_is_kept() -> None:
    body = "\ufeffdata: \ufeffleading\n\ndata: in\ufeffside\n\ndata: \ufeff\n\n"

    assert _decode(_SSE, body) == ("\ufeffleading", "in\ufeffside", "\ufeff")


@pytest.mark.parametrize(
    ("content_type", "body", "expected"),
    [
        ("Text/Event-Stream", 'data: {"data": "hi"}\n\n', {"data": "hi"}),
        ("TEXT/EVENT-STREAM; Charset=UTF-8", 'data: {"data": "hi"}\n\n', {"data": "hi"}),
        ("text/event-stream; charset=utf-8", 'data: {"data": "hi"}\n\n', {"data": "hi"}),
        ("text/event-stream;charset=UTF-8", 'data: {"data": "hi"}\n\n', {"data": "hi"}),
        ("APPLICATION/JSON; charset=utf-8", '{"response": "hi"}', {"response": "hi"}),
        ("Application/Json", '{"response": "hi"}', {"response": "hi"}),
        ("application/json; charset=utf-8", '{"response": "hi"}', {"response": "hi"}),
        ("application/json;charset=UTF-8", '{"response": "hi"}', {"response": "hi"}),
    ],
)
def test_media_types_are_matched_case_insensitively_and_without_their_parameters(
    content_type: str, body: str, expected: dict[str, str]
) -> None:
    values = _decode(content_type, body)

    assert values == (expected,)
    # The ``data`` key counts only for event streams, so this also shows final_text's match.
    assert final_text(values, content_type, session_id=_SESSION) == "hi"


@pytest.mark.parametrize("chunk_size", [1, 2, 1024])
def test_crlf_line_endings_are_read(chunk_size: int) -> None:
    body = "data: one\r\n\r\ndata: two\r\ndata: three\r\n\r\n: note\r\ndata: four\r\n\r\n"

    assert _decode(_SSE, body, chunk_size=chunk_size) == ("one", "two\nthree", "four")


@pytest.mark.parametrize("chunk_size", [1, 3])
def test_a_data_line_split_across_chunks_is_read_whole(chunk_size: int) -> None:
    # Multi-byte characters land across chunk boundaries, and U+2028 is a line break to
    # ``str.splitlines`` but not to the event stream.
    answer = "café \u2013 ünïcode\u2028still the same line"
    body = f"data: {json.dumps({'response': answer}, ensure_ascii=False)}\n\n".encode()

    assert _decode(_SSE, body, chunk_size=chunk_size) == ({"response": answer},)


# JSON bodies


def test_a_json_body_is_decoded_as_one_document() -> None:
    document = {"response": "line one\u2028line two", "steps": [{"tool": "lookup"}], "ok": True}
    body = json.dumps(document, indent=2, ensure_ascii=False).encode()

    values = _decode(_JSON, body)

    assert values == (document,)
    assert final_text(values, _JSON, session_id=_SESSION) == "line one\u2028line two"


# Final text


def test_final_text_joins_strings_and_the_first_string_valued_key_in_order() -> None:
    # Each object also holds every lower-ranked key, listed first, so only the rank order picks
    # the key that is read; values that are neither text nor objects add nothing.
    ranked: list[DecodedValue] = [
        {key: key for key in reversed(_TEXT_KEYS[rank:])} for rank in range(len(_TEXT_KEYS))
    ]
    values: list[DecodedValue] = ["<", *ranked, [1, 2], 3, None, True, ">"]

    text = final_text(values, _SSE, session_id=_SESSION)

    assert text == "<responsecontentmessagetextoutputanswer>"


def test_a_non_string_value_falls_through_to_the_next_key() -> None:
    values: list[DecodedValue] = [
        {"message": {"role": "assistant"}, "text": "hi"},
        {"response": None, "content": 5, "message": ["x"], "text": False, "answer": "!"},
    ]

    assert final_text(values, _JSON, session_id=_SESSION) == "hi!"


def test_the_data_key_counts_only_for_event_streams() -> None:
    values: list[DecodedValue] = [{"data": "streamed"}]

    assert final_text(values, _SSE, session_id=_SESSION) == "streamed"
    assert final_text([{"data": "last", "answer": "first"}], _SSE, session_id=_SESSION) == "first"
    with pytest.raises(RuntimeInvocationError):
        final_text(values, _JSON, session_id=_SESSION)


# Agent errors


@pytest.mark.parametrize(
    ("error", "error_type"),
    [
        *(
            pytest.param("boom: confidential detail", error_type, id=error_type)
            for error_type in [
                "ValueError",
                "botocore.exceptions.ReadTimeoutError",
                "_Custom",
                "E" * 100,
            ]
        ),
        # The AgentCore SDK's own error event can carry empty error text.
        pytest.param("", "TimeoutError", id="sdk-event-with-empty-error-text"),
        pytest.param(None, "TimeoutError", id="sdk-event-with-null-error-text"),
    ],
)
def test_an_in_stream_error_event_is_refused_naming_its_type_without_its_text(
    error: str | None, error_type: str
) -> None:
    body = _event({"response": "partial answer"}) + _event(
        {"error": error, "error_type": error_type, "message": "An error occurred during streaming"}
    )
    values = _decode(_SSE, body)

    with pytest.raises(RuntimeInvocationError) as refused:
        check_errors(values, session_id=_SESSION)

    leaked = ["boom", "confidential", "partial answer", "An error occurred"]
    _says(refused.value, error_type, never=leaked)


_NO_TYPE = object()


@pytest.mark.parametrize("error", [None, False, "", [], {}], ids=repr)
@pytest.mark.parametrize("error_type", [_NO_TYPE, None, ""], ids=["absent", "null", "empty"])
def test_a_null_false_or_empty_error_without_a_type_is_not_refused(
    error: object, error_type: object
) -> None:
    document: dict[str, Any] = {"response": "the answer", "error": error}
    if error_type is not _NO_TYPE:
        document["error_type"] = error_type
    values = _decode(_JSON, json.dumps(document))

    check_errors(values, session_id=_SESSION)

    assert final_text(values, _JSON, session_id=_SESSION) == "the answer"


@pytest.mark.parametrize(
    "value",
    [
        {"error": "boom"},
        {"error": "boom", "error_type": 42},
        {"error": "boom", "error_type": None},
        {"error": "boom", "error_type": ""},
        {"error": {"code": "boom"}},
        {"error": ["boom"]},
        {"error": True},
        # Python counts 0 as equal to False, but a zero is not a false error.
        {"error": 0},
        {"error": 0.0},
    ],
    ids=repr,
)
def test_an_error_without_a_string_type_is_still_refused(value: dict[str, Any]) -> None:
    # Read from a JSON body, after a value that holds no error.
    values = [{"response": "fine"}, *_decode(_JSON, json.dumps(value))]

    with pytest.raises(RuntimeInvocationError) as refused:
        check_errors(values, session_id=_SESSION)

    _says(refused.value, never=["boom"])


@pytest.mark.parametrize(
    "error_type",
    [
        "error text with spaces",
        "1LeadingDigit",
        "Value-Error",
        "ValueError\n",
        "ErrörType",
        "E" * 101,
    ],
    ids=repr,
)
def test_a_non_identifier_error_type_is_not_named(error_type: str) -> None:
    with pytest.raises(RuntimeInvocationError) as refused:
        check_errors([{"error": "boom", "error_type": error_type}], session_id=_SESSION)

    _says(refused.value, never=[error_type])


def test_an_agent_error_is_an_agent_reported_error_coded_by_its_type_with_the_run_trace_id() -> (
    None
):
    with pytest.raises(AgentReportedError) as refused:
        check_errors(
            [{"error": "boom", "error_type": "ToolDenied"}],
            session_id=_SESSION,
            trace_id="probe-abc-0123",
        )

    assert isinstance(refused.value, RuntimeInvocationError)
    assert refused.value.error_code == "ToolDenied"
    assert refused.value.trace_id == "probe-abc-0123"
    assert "boom" not in repr(vars(refused.value))


@pytest.mark.parametrize(
    "value",
    [{"error": "boom"}, {"error": "boom", "error_type": "error text with spaces"}],
    ids=["no-type", "non-identifier-type"],
)
def test_an_agent_error_without_an_identifier_type_is_coded_agent_error(
    value: dict[str, Any],
) -> None:
    with pytest.raises(AgentReportedError) as refused:
        check_errors([value], session_id=_SESSION)

    assert refused.value.error_code == "agent_error"
    assert refused.value.trace_id is None
    assert "boom" not in repr(vars(refused.value))


# Refused bodies


@pytest.mark.parametrize(
    "content_type",
    ["text/html; charset=utf-8", "text/plain", "application/octet-stream", "application/x-ndjson"],
)
def test_an_unsupported_content_type_is_refused_naming_it_and_the_session(
    content_type: str,
) -> None:
    _refused(content_type, b'{"response": "hi"}', content_type)


@pytest.mark.parametrize(
    "body",
    [
        b'{"response": "the confidential answer was cut',
        b'\xef\xbb\xbf{"response": "the confidential answer"}',
    ],
    ids=["truncated", "byte-order-mark"],
)
def test_a_json_body_that_does_not_parse_is_refused_without_quoting_it(body: bytes) -> None:
    refused = _refused(_JSON, body, _JSON, never=["confidential"])

    assert isinstance(refused.__cause__, json.JSONDecodeError)


@pytest.mark.parametrize(
    "body",
    [
        *(
            pytest.param(f'{{"response": "{_SENTINEL_TEXT}", "scores": [1, {token}]}}', id=token)
            for token in ["NaN", "Infinity", "-Infinity"]
        ),
        pytest.param(
            f'{{"response": "{_SENTINEL_TEXT}", "score": 1e999}}', id="1e999-in-an-object"
        ),
        pytest.param(
            f'{{"response": "{_SENTINEL_TEXT}", "score": -1e999}}', id="-1e999-in-an-object"
        ),
        pytest.param(f'["{_SENTINEL_TEXT}", [1e999]]', id="1e999-in-a-list"),
        pytest.param(f'["{_SENTINEL_TEXT}", -1e999]', id="-1e999-in-a-list"),
    ],
)
def test_a_json_body_holding_nan_infinity_or_a_number_too_large_for_a_float_is_refused(
    body: str,
) -> None:
    _refused(_JSON, body, _JSON, "a number JSON cannot represent", never=[_SENTINEL_TEXT, "1e999"])


@pytest.mark.parametrize(
    ("content_type", "body"),
    [
        pytest.param(_JSON, b'{"response": "caf\xe9"}', id="in a json body"),
        pytest.param(_SSE, b"data: caf\xe9\n\n", id="in data"),
        pytest.param(_SSE, b"data: ok\n\ndata: \xff", id="in the last event"),
        # A comment carries no answer, but a body that isn't UTF-8 is refused whole.
        pytest.param(_SSE, b": caf\xe9\ndata: ok\n\n", id="in a comment before the data"),
        pytest.param(_SSE, b"data: ok\n\n: \xff\n", id="in a comment after the data"),
    ],
)
def test_invalid_utf8_anywhere_in_a_body_is_refused(content_type: str, body: bytes) -> None:
    refused = _refused(content_type, body, content_type)

    assert isinstance(refused.__cause__, UnicodeDecodeError)


_LINE_BREAKS = {"one LF": b"\n", "LF only": b"\n\n", "CRLF only": b"\r\n", "CR only": b"\r\r"}


@pytest.mark.parametrize(
    ("content_type", "body"),
    [
        *(pytest.param(content_type, b"", id=content_type) for content_type in [_SSE, _JSON]),
        *(
            pytest.param(content_type, body, id=f"{name}-{content_type}")
            for name, body in {**_LINE_BREAKS, "mixed": b"\r\n\n\r"}.items()
            for content_type in [_SSE, _JSON]
        ),
        # An event stream's byte order mark is dropped before the body is checked for content.
        *(
            pytest.param(_SSE, b"\xef\xbb\xbf" + ending, id=f"byte-order-mark-{name}")
            for name, ending in [("alone", b""), ("then-LF", b"\n\n"), ("then-CRLF", b"\r\n")]
        ),
    ],
)
def test_an_empty_body_is_refused(content_type: str, body: bytes) -> None:
    # Refused as an empty body, not as an event stream with no data or as broken JSON.
    _refused(content_type, body, "empty")


def test_an_event_stream_with_no_data_field_is_refused() -> None:
    _refused(_SSE, ": keep-alive\n\nevent: ping\nid: 1\nretry: 500\n\n")


@pytest.mark.parametrize(
    "values",
    [
        (),
        ({"tool_name": "lookup"},),
        ([1, 2],),
        (42,),
        (None,),
        ({"response": None, "text": 7},),
        ("",),
        ({"response": ""},),
    ],
    ids=repr,
)
def test_values_with_no_extractable_text_are_refused(values: tuple[DecodedValue, ...]) -> None:
    with pytest.raises(RuntimeInvocationError) as refused:
        final_text(values, _SSE, session_id=_SESSION)

    _says(refused.value)
