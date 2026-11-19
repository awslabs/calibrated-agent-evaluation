# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""The failure detector runs the detector library once per trace and returns typed results.

A detection is one model call on the converted trace. Its response is read strictly and checked
against the vocabulary, the trace's spans and the library's own reading of it, so the only empty
result is one the model actually gave: every way the response can fail to be read, and every way
the call can fail, is a typed refusal that names the trace.

The scripted model stands in for the real one, so each of its calls counts as a live call.
"""

from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime
from importlib.metadata import version
from pathlib import Path
from typing import Any

import pytest

pytest.importorskip("strands_evals")

from botocore.exceptions import EndpointConnectionError, ReadTimeoutError
from strands.types.exceptions import ContextWindowOverflowException
from strands_evals.detectors import failure_detector
from strands_evals.detectors.chunking import would_exceed_context
from strands_evals.detectors.prompt_templates.failure_detection import failure_detection_v0

from _detector_fakes import ScriptedModel, spy_connections
from _helpers import FIXTURES, make_span, make_tool_call, make_trace
from calibrated_agent_evaluation.agentcore.detectors import detect
from calibrated_agent_evaluation.agentcore.detectors.detect import FailureDetector
from calibrated_agent_evaluation.agentcore.detectors.session import (
    canonical_json,
    session_json,
    trace_to_session,
)
from calibrated_agent_evaluation.agentcore.detectors.types import DetectedFailure, Detection
from calibrated_agent_evaluation.agentcore.errors import (
    AgentCoreError,
    DetectorCallError,
    DetectorCallLimitExceeded,
    DetectorConfigError,
    DetectorOutputError,
    DetectorSessionTooLarge,
    SessionConversionError,
)
from calibrated_agent_evaluation.execute import CassetteMiss, CassetteModelMismatch, Mode
from calibrated_agent_evaluation.normalize import OTelSpans, normalize
from calibrated_agent_evaluation.schema.types import Trace

NOW = datetime(2026, 10, 27, 9, 30, tzinfo=UTC)
VERIFIED_SHA256 = "30d6baa2559a514b87e9477f58afefb977371cc413f5c841eaa3ad9798c809a9"
MISUNDERSTAND = "hallucination-category-hall-misunderstand"
FABRICATE = "hallucination-category-fabricate-tool-outputs"
PREMATURE = "orchestration-related-errors-category-premature-termination"
TOOL_SELECTION = "incorrect-actions-category-tool-selection"
POLICY_EVIDENCE = "the policy tool says within_window although 51 days exceed the 30-day window"
# A credential-shaped string a provider might echo into an error message.
TOKEN = "AKIAIOSFODNN7EXAMPLE"
BEDROCK_URL = "https://bedrock-runtime.us-east-1.amazonaws.com"
OTEL_ORD = FIXTURES / "raw" / "otel" / "ORD-1001.events.otlp.json"
OTEL_DEGRADED = FIXTURES / "raw" / "otel" / "ORD-DEGRADED.otlp.json"
# How a refusal reads when the model call was made and failed, and when none was made.
CALL_FAILED = "the detector's call for trace 'ORD-1001' failed"
CALL_NOT_MADE = "no model call was made"


def _finding(category: str, confidence: str, evidence: str) -> tuple[str, str, str]:
    return category, confidence, evidence


def _entry(location: str, *findings: tuple[str, str, str]) -> dict[str, Any]:
    return {
        "location": location,
        "category": [category for category, _, _ in findings],
        "evidence": [evidence for _, _, evidence in findings],
        "confidence": [confidence for _, confidence, _ in findings],
    }


def _answer(*entries: dict[str, Any]) -> str:
    return json.dumps({"errors": list(entries)})


def _detector(
    *responses: str | BaseException, config: dict[str, Any] | None = None, **options: Any
) -> tuple[FailureDetector, ScriptedModel]:
    scripted = ScriptedModel(responses, config=config)
    options.setdefault("max_live_calls", 1)
    options.setdefault("now", NOW)
    return FailureDetector(scripted, **options), scripted


def _detect(trace: Trace, response: str, **options: Any) -> Detection:
    detector, _ = _detector(response, **options)
    return detector.detect(trace)


def _record(path: Path, trace: Trace, *, config: dict[str, Any], **options: Any) -> None:
    """Record one detection of ``trace`` into the cassette at ``path``."""
    detector, _ = _detector(
        _answer(), config=config, cassette_path=path, cassette_mode=Mode.RECORD, **options
    )
    detector.detect(trace)


def _entries(path: Path) -> list[dict[str, Any]]:
    return json.loads(path.read_text(encoding="utf-8"))["entries"]


def _one_span_trace(**span: Any) -> Trace:
    return make_trace(
        trace_id="SMALL-TRACE",
        spans=[make_span(span_id="SMALL-TRACE/root", trace_id="SMALL-TRACE", **span)],
    )


def _texts(value: Any) -> list[str]:
    if isinstance(value, str):
        return [value]
    if isinstance(value, dict):
        return [text for item in value.values() for text in _texts(item)]
    if isinstance(value, list):
        return [text for item in value for text in _texts(item)]
    return []


def _causes(error: BaseException) -> list[BaseException]:
    """The exceptions ``error`` is chained from, nearest first."""
    causes: list[BaseException] = []
    while error.__cause__ is not None:
        error = error.__cause__
        causes.append(error)
    return causes


def _assert_no_prompt_text(message: str, trace: Trace) -> None:
    """Neither the detector's instructions nor the trace's content is quoted in ``message``."""
    first_instruction = failure_detection_v0.SYSTEM_PROMPT.splitlines()[0]
    assert first_instruction not in message
    for span in trace.spans:
        for text in _texts([span.input, span.output]):
            assert text not in message


def test_detections_map_back_to_calibrated_agent_evaluation_spans_and_tool_calls(
    corpus_ord: Trace, corpus_clm: Trace
) -> None:
    on_tool = _detect(
        corpus_ord,
        _answer(
            _entry("ORD-1001/orderbot#tool1", _finding(MISUNDERSTAND, "high", POLICY_EVIDENCE))
        ),
    )
    assert on_tool.trace_id == "ORD-1001"
    assert on_tool.failures == (
        DetectedFailure(
            span_id="ORD-1001/orderbot",
            tool_call_index=1,
            tool_name="check_refund_policy",
            category=MISUNDERSTAND,
            parent="hallucination",
            confidence="high",
            evidence=POLICY_EVIDENCE,
        ),
    )
    assert on_tool.below_threshold == 0

    # The response lists the locations out of span order; the detection follows the spans, and
    # keeps the response's order within one location, a repeated category included.
    ordered = _detect(
        corpus_clm,
        _answer(
            _entry("CLM-4471/payout", _finding(PREMATURE, "medium", "paid before the fraud score")),
            _entry(
                "CLM-4471/policy-checker#tool1",
                _finding(TOOL_SELECTION, "low", "looked the exclusions up a second time"),
                _finding(FABRICATE, "high", "cites an exclusion the tool did not return"),
            ),
            _entry(
                "CLM-4471/intake",
                _finding(MISUNDERSTAND, "medium", "read the claim date as the loss date"),
                _finding(MISUNDERSTAND, "low", "read the policy number as the claim number"),
            ),
        ),
    )
    assert [
        (f.span_id, f.tool_call_index, f.tool_name, f.category, f.confidence, f.evidence)
        for f in ordered.failures
    ] == [
        (
            "CLM-4471/intake",
            None,
            None,
            MISUNDERSTAND,
            "medium",
            "read the claim date as the loss date",
        ),
        (
            "CLM-4471/intake",
            None,
            None,
            MISUNDERSTAND,
            "low",
            "read the policy number as the claim number",
        ),
        (
            "CLM-4471/policy-checker",
            1,
            "lookup_policy_exclusions",
            TOOL_SELECTION,
            "low",
            "looked the exclusions up a second time",
        ),
        (
            "CLM-4471/policy-checker",
            1,
            "lookup_policy_exclusions",
            FABRICATE,
            "high",
            "cites an exclusion the tool did not return",
        ),
        ("CLM-4471/payout", None, None, PREMATURE, "medium", "paid before the fraud score"),
    ]
    assert [f.parent for f in ordered.failures] == [
        "hallucination",
        "hallucination",
        "incorrect-actions",
        "hallucination",
        "orchestration-related-errors",
    ]


def test_bare_fenced_and_prose_wrapped_payloads_parse_alike(corpus_ord: Trace) -> None:
    payload = _answer(
        _entry("ORD-1001/orderbot#tool1", _finding(MISUNDERSTAND, "high", POLICY_EVIDENCE)),
        _entry("ORD-1001/orderbot", _finding(PREMATURE, "low", "refunded without checking")),
    )
    shapes = {
        "bare": payload,
        "fenced": f"```json\n{payload}\n```",
        "bare fence": f"```\n{payload}\n```",
        "fenced inside prose": f"Here are the failures.\n```json\n{payload}\n```\nThat is all.",
        "after a preamble": f"Looking at the session, here are the failures:\n{payload}",
        # The prose before the block ends, and the prose after it starts, with half of an errors
        # object. Joined across the cut-out block they would read as a second candidate.
        "fenced between fragments of an object": (
            f'An answer starts {{"errors":\n```json\n{payload}\n```\n[]}} ends an empty one.'
        ),
    }

    detections = {shape: _detect(corpus_ord, text) for shape, text in shapes.items()}

    assert len(detections["bare"].failures) == 2
    assert all(detection == detections["bare"] for detection in detections.values())


def test_only_a_parsed_empty_errors_list_means_nothing_found(corpus_ord: Trace) -> None:
    nothing_found = _detect(corpus_ord, '{"errors": []}')
    assert nothing_found.failures == ()
    assert nothing_found.below_threshold == 0

    with pytest.raises(DetectorOutputError, match="ORD-1001") as refused:
        _detect(corpus_ord, "")
    assert "no candidate" in str(refused.value)


def test_a_higher_threshold_drops_lower_confidence_failures(corpus_ord: Trace) -> None:
    high = _finding(MISUNDERSTAND, "high", POLICY_EVIDENCE)
    low = _finding(PREMATURE, "low", "refunded without checking")
    high_and_low = _answer(
        _entry("ORD-1001/orderbot#tool1", high), _entry("ORD-1001/orderbot", low)
    )

    at_medium = _detect(corpus_ord, high_and_low, confidence_threshold="medium")
    assert [(f.category, f.confidence) for f in at_medium.failures] == [(MISUNDERSTAND, "high")]
    assert at_medium.below_threshold == 1
    assert at_medium.provenance.confidence_threshold == "medium"

    only_low = _detect(
        corpus_ord, _answer(_entry("ORD-1001/orderbot", low)), confidence_threshold="medium"
    )
    assert only_low.failures == ()
    assert only_low.below_threshold == 1

    at_low = _detect(corpus_ord, high_and_low)
    assert [f.confidence for f in at_low.failures] == ["low", "high"]
    assert at_low.below_threshold == 0
    assert at_low.provenance.confidence_threshold == "low"


def test_an_oversized_trace_is_refused_before_any_model_call() -> None:
    # The library's offline estimate counts three characters to a token.
    oversized = _one_span_trace(output={"text": "x" * 600_001})
    detector, scripted = _detector(_answer())

    with pytest.raises(DetectorSessionTooLarge, match="SMALL-TRACE"):
        detector.detect(oversized)
    assert scripted.calls == []


def _longest_prompt_under_the_check() -> int:
    """The longest text the library's own context check lets through, in characters."""
    under, over = 0, 10_000_000
    while over - under > 1:
        middle = (under + over) // 2
        if would_exceed_context("x" * middle):
            over = middle
        else:
            under = middle
    return under


def test_a_prompt_over_the_check_only_with_its_instructions_is_refused_before_any_call() -> None:
    # The library checks the session part of the prompt; the call also carries its instructions.
    def user_prompt(trace: Trace) -> str:
        return failure_detection_v0.build_prompt(session_json=session_json(trace_to_session(trace)))

    base = len(user_prompt(_one_span_trace(output={"text": ""})))
    trace = _one_span_trace(output={"text": "x" * (_longest_prompt_under_the_check() - base)})
    assert not would_exceed_context(user_prompt(trace))
    assert would_exceed_context(f"{failure_detection_v0.SYSTEM_PROMPT}\n\n{user_prompt(trace)}")
    detector, scripted = _detector(_answer())

    with pytest.raises(DetectorSessionTooLarge, match="SMALL-TRACE"):
        detector.detect(trace)
    assert scripted.calls == []


@pytest.mark.parametrize(
    "overflow",
    [
        ContextWindowOverflowException("the request is over the model's input limit"),
        RuntimeError("Input is too long for requested model"),
    ],
    ids=["context window overflow", "too-long message"],
)
def test_a_model_context_overflow_is_refused_without_a_second_call(
    corpus_ord: Trace, overflow: Exception
) -> None:
    # Two live calls are allowed, so only the detector stops the library's chunked retry.
    detector, scripted = _detector(overflow, _answer(), max_live_calls=2)

    with pytest.raises(DetectorSessionTooLarge, match="ORD-1001") as refused:
        detector.detect(corpus_ord)
    assert refused.value.__cause__ is overflow
    assert len(scripted.calls) == 1


def test_the_vendors_chunked_path_is_refused_before_any_call(
    corpus_ord: Trace, monkeypatch: pytest.MonkeyPatch
) -> None:
    # The library chunks when its own check says the prompt is too long; the trace has 4 spans,
    # enough for it to split.
    monkeypatch.setattr(failure_detector, "would_exceed_context", lambda *_a, **_k: True)
    detector, scripted = _detector(_answer(), max_live_calls=2)

    with pytest.raises(DetectorSessionTooLarge, match="ORD-1001"):
        detector.detect(corpus_ord)
    assert scripted.calls == []


def test_a_detection_without_its_model_call_is_refused(monkeypatch: pytest.MonkeyPatch) -> None:
    # Two vendor spans are too few for the library to split, so it returns nothing, uncalled.
    monkeypatch.setattr(failure_detector, "would_exceed_context", lambda *_a, **_k: True)
    detector, scripted = _detector(_answer())

    with pytest.raises(DetectorOutputError, match="SMALL-TRACE") as refused:
        detector.detect(_one_span_trace(tool_calls=[make_tool_call()]))
    assert "without making its model call" in str(refused.value)
    assert scripted.calls == []


def test_a_probed_or_spanless_trace_is_refused_before_any_model_call() -> None:
    probed = normalize(OTelSpans(path=OTEL_DEGRADED))
    spanless = make_trace(trace_id="run-empty", ownership="owned", spans=[])

    for trace, reason in ((probed, "is probed"), (spanless, "has no spans")):
        detector, scripted = _detector(_answer())
        with pytest.raises(SessionConversionError, match=f"trace {trace.trace_id} {reason}"):
            detector.detect(trace)
        assert scripted.calls == []


@pytest.mark.parametrize(
    ("response", "problem"),
    [
        ("not json", "no candidate"),
        (
            _answer(_entry("ORD-1001/orderbot", _finding(PREMATURE, "High", "refunded anyway"))),
            "errors.0.confidence.0",
        ),
        (
            json.dumps(
                {
                    "errors": [
                        {
                            "location": "ORD-1001/orderbot",
                            "category": [PREMATURE, MISUNDERSTAND],
                            "evidence": ["refunded anyway"],
                            "confidence": ["high", "low"],
                        }
                    ]
                }
            ),
            "2 categories, 1 evidence strings and 2 confidence levels",
        ),
        (
            json.dumps(
                {
                    "errors": [
                        {
                            "location": "ORD-1001/orderbot",
                            "category": [],
                            "evidence": [],
                            "confidence": [],
                        }
                    ]
                }
            ),
            "names no category",
        ),
        (
            f"First {_answer()} and then, on reflection, "
            + _answer(_entry("ORD-1001/orderbot", _finding(PREMATURE, "low", "refunded"))),
            "2 candidates",
        ),
        (
            f"```json\n{_answer()}\n```\nor perhaps\n```json\n"
            + _answer(_entry("ORD-1001/orderbot", _finding(PREMATURE, "low", "refunded")))
            + "\n```",
            "2 candidates",
        ),
        (
            json.dumps(
                {
                    "errors": [
                        {
                            "location": "ORD-1001/orderbot",
                            "category": [PREMATURE],
                            "confidence": ["low"],
                        }
                    ]
                }
            ),
            "errors.0.evidence",
        ),
        (
            f"```json\n{_answer()}\n```\nOn a second look, though:\n"
            + _answer(_entry("ORD-1001/orderbot", _finding(PREMATURE, "high", "refunded"))),
            "2 candidates",
        ),
        (
            _answer(_entry("ORD-1001/orderbot", _finding(PREMATURE, "high", "refunded")))
            + f"\nIf that reads wrongly, take this one:\n```json\n{_answer()}\n```",
            "2 candidates",
        ),
    ],
    ids=[
        "not json",
        "capitalized confidence",
        "unequal list lengths",
        "three empty lists",
        "two errors objects in prose",
        "two fenced errors objects",
        "missing evidence",
        "a fenced empty list and a failure in prose",
        "a failure in prose and a fenced empty list",
    ],
)
def test_unreadable_or_malformed_responses_are_refused(
    corpus_ord: Trace, response: str, problem: str
) -> None:
    with pytest.raises(DetectorOutputError, match="ORD-1001") as refused:
        _detect(corpus_ord, response)
    assert problem in str(refused.value)
    assert "Detect the trace again" in str(refused.value)


def test_a_quoted_response_fragment_is_cut_to_200_characters(corpus_ord: Trace) -> None:
    unreadable = "no json here, " + "y" * 400 + " TAIL-OF-THE-RESPONSE"

    with pytest.raises(DetectorOutputError, match="ORD-1001") as refused:
        _detect(corpus_ord, unreadable)
    message = str(refused.value)
    assert unreadable[:200] in message
    assert unreadable[:201] not in message
    assert "TAIL-OF-THE-RESPONSE" not in message


def test_unknown_categories_and_locations_are_refused(corpus_ord: Trace) -> None:
    unknowns = [
        ("ORD-1001/orderbot", "made-up-category", "made-up-category"),
        ("ORD-9999/x", PREMATURE, "ORD-9999/x"),
    ]
    for location, category, unknown in unknowns:
        with pytest.raises(DetectorOutputError, match="ORD-1001") as refused:
            _detect(corpus_ord, _answer(_entry(location, _finding(category, "high", "evidence"))))
        assert repr(unknown) in str(refused.value)

    other = _detect(
        corpus_ord, _answer(_entry("ORD-1001/orderbot", _finding("other", "low", "unclear")))
    )
    assert [(f.category, f.parent) for f in other.failures] == [("other", None)]


def _drop_the_first(failures: list[Any]) -> list[Any]:
    return failures[1:]


def _reword_the_first_evidence(failures: list[Any]) -> list[Any]:
    first, *rest = failures
    return [first.model_copy(update={"evidence": ["read otherwise"]}), *rest]


def _recategorize_the_first(failures: list[Any]) -> list[Any]:
    first, *rest = failures
    return [first.model_copy(update={"category": [PREMATURE]}), *rest]


@pytest.mark.parametrize(
    "change",
    [_drop_the_first, _reword_the_first_evidence, _recategorize_the_first],
    ids=["one failure fewer", "one evidence string changed", "one category changed"],
)
def test_a_disagreement_with_the_vendors_parse_is_refused(
    corpus_ord: Trace, monkeypatch: pytest.MonkeyPatch, change: Any
) -> None:
    response = _answer(
        _entry("ORD-1001/orderbot#tool1", _finding(MISUNDERSTAND, "high", POLICY_EVIDENCE)),
        _entry("ORD-1001/orderbot", _finding(PREMATURE, "low", "refunded without checking")),
    )
    real_detect_failures = failure_detector.detect_failures

    def reads_otherwise(*args: Any, **kwargs: Any) -> Any:
        # The real function still runs, so the model call is made and its answer captured.
        output = real_detect_failures(*args, **kwargs)
        return output.model_copy(update={"failures": change(list(output.failures))})

    monkeypatch.setattr(detect, "detect_failures", reads_otherwise)
    with pytest.raises(DetectorOutputError, match="ORD-1001") as refused:
        _detect(corpus_ord, response)
    assert "disagree" in str(refused.value)

    monkeypatch.setattr(detect, "detect_failures", real_detect_failures)
    assert len(_detect(corpus_ord, response).failures) == 2


def test_model_and_cassette_failures_become_detector_call_errors(
    corpus_ord: Trace, corpus_clm: Trace, tmp_path: Path
) -> None:
    model_failures: list[Exception] = [
        RuntimeError(f"the provider rejected credential {TOKEN} for this request"),
        EndpointConnectionError(endpoint_url=BEDROCK_URL),
        ReadTimeoutError(endpoint_url=BEDROCK_URL),
    ]
    for failure in model_failures:
        detector, _ = _detector(failure)
        with pytest.raises(DetectorCallError, match="ORD-1001") as refused:
            detector.detect(corpus_ord)
        assert refused.value.__cause__ is failure
        assert CALL_FAILED in str(refused.value)
        assert TOKEN not in str(refused.value)
        _assert_no_prompt_text(str(refused.value), corpus_ord)

    other_trace = tmp_path / "other-trace.cassette.json"
    _record(other_trace, corpus_clm, config={"model_id": "m-1"})
    other_model = tmp_path / "other-model.cassette.json"
    _record(other_model, corpus_ord, config={"model_id": "m-1"})
    cassette_failures = [
        (other_trace, {"model_id": "m-1"}, CassetteMiss),
        (other_model, {"model_id": "m-2"}, CassetteModelMismatch),
    ]
    for path, config, cause in cassette_failures:
        detector, scripted = _detector(config=config, cassette_path=path, cassette_mode=Mode.REPLAY)
        with pytest.raises(DetectorCallError, match="ORD-1001") as refused:
            detector.detect(corpus_ord)
        assert isinstance(refused.value.__cause__, cause)
        assert CALL_FAILED in str(refused.value)
        _assert_no_prompt_text(str(refused.value), corpus_ord)
        assert scripted.calls == []


def test_a_failure_inside_the_detector_library_becomes_a_detector_call_error(
    corpus_ord: Trace, monkeypatch: pytest.MonkeyPatch
) -> None:
    during_the_call = ValueError(f"the library broke while reading {TOKEN}")

    def raises(*_args: Any, **_kwargs: Any) -> Any:
        raise during_the_call

    monkeypatch.setattr(detect, "detect_failures", raises)
    detector, scripted = _detector(_answer())
    with pytest.raises(DetectorCallError, match="ORD-1001") as refused:
        detector.detect(corpus_ord)
    assert refused.value.__cause__ is during_the_call
    assert CALL_FAILED in str(refused.value)
    assert TOKEN not in str(refused.value)
    assert scripted.calls == []
    monkeypatch.undo()

    # The library's prompt builder runs before the call. What it raises makes the template
    # unusable, so the error has the same type but says that no call was made.
    before_the_call = TypeError("build_prompt() got an unexpected keyword argument")

    def broken_build_prompt(*_args: Any, **_kwargs: Any) -> str:
        raise before_the_call

    monkeypatch.setattr(failure_detection_v0, "build_prompt", broken_build_prompt)
    detector, scripted = _detector(_answer())
    with pytest.raises(DetectorCallError, match="ORD-1001") as refused:
        detector.detect(corpus_ord)
    assert before_the_call in _causes(refused.value)
    assert CALL_NOT_MADE in str(refused.value)
    assert CALL_FAILED not in str(refused.value)
    _assert_no_prompt_text(str(refused.value), corpus_ord)
    assert scripted.calls == []


@pytest.mark.parametrize("reader", ["failure_vocabulary", "detector_prompt"])
def test_an_unusable_template_is_refused_as_a_call_that_was_never_made(
    corpus_ord: Trace, monkeypatch: pytest.MonkeyPatch, reader: str
) -> None:
    # The cause's text quotes a credential and the detector's instructions; neither is repeated.
    unusable = AgentCoreError(
        f"the template is unusable near {TOKEN}: "
        f"{failure_detection_v0.SYSTEM_PROMPT.splitlines()[0]}"
    )

    def raises(*_args: Any, **_kwargs: Any) -> Any:
        raise unusable

    monkeypatch.setattr(detect, reader, raises)
    detector, scripted = _detector(_answer())
    with pytest.raises(DetectorCallError) as refused:
        detector.detect(corpus_ord)
    message = str(refused.value)
    assert refused.value.__cause__ is unusable
    assert "the detector could not prepare its call for trace 'ORD-1001'" in message
    installed = f"strands-agents-evals {version('strands-agents-evals')}"
    assert f"the installed {installed} template is unusable (AgentCoreError)" in message
    assert CALL_NOT_MADE in message
    assert CALL_FAILED not in message
    assert str(unusable) not in message
    assert TOKEN not in message
    _assert_no_prompt_text(message, corpus_ord)
    assert scripted.calls == []


@pytest.mark.parametrize("step", ["session_json", "would_exceed_context"])
def test_a_failure_while_the_session_is_serialized_or_sized_says_no_call_was_made(
    corpus_ord: Trace, monkeypatch: pytest.MonkeyPatch, step: str
) -> None:
    # Both steps run before the call. The cause's text quotes a credential and the trace's own
    # words; neither is repeated.
    quoted = _texts(corpus_ord.spans[0].output)[0]
    before_the_call = ValueError(f"{step} broke near {TOKEN} while reading {quoted}")

    def raises(*_args: Any, **_kwargs: Any) -> Any:
        raise before_the_call

    monkeypatch.setattr(detect, step, raises)
    detector, scripted = _detector(_answer())
    with pytest.raises(DetectorCallError) as refused:
        detector.detect(corpus_ord)
    message = str(refused.value)
    assert refused.value.__cause__ is before_the_call
    assert "the detector could not prepare its call for trace 'ORD-1001'" in message
    assert "(ValueError)" in message
    assert CALL_NOT_MADE in message
    assert CALL_FAILED not in message
    assert TOKEN not in message
    _assert_no_prompt_text(message, corpus_ord)
    assert scripted.calls == []


def test_a_cassette_error_that_names_max_tokens_is_not_read_as_a_context_overflow(
    corpus_ord: Trace, corpus_clm: Trace, tmp_path: Path
) -> None:
    # The library reads "max_tokens" in an error message as a context overflow, and both of these
    # cassette errors carry it: one in the file name, one in the model config it describes.
    named = tmp_path / "max_tokens.cassette.json"
    _record(named, corpus_clm, config={"model_id": "m-1"})
    configured = tmp_path / "configured.cassette.json"
    _record(configured, corpus_ord, config={"model_id": "m-1", "max_tokens": 4096})
    cases = [
        (named, {"model_id": "m-1"}, CassetteMiss),
        (configured, {"model_id": "m-2", "max_tokens": 4096}, CassetteModelMismatch),
    ]

    for path, config, cause in cases:
        detector, scripted = _detector(config=config, cassette_path=path, cassette_mode=Mode.REPLAY)
        with pytest.raises(DetectorCallError, match="ORD-1001") as refused:
            detector.detect(corpus_ord)
        assert isinstance(refused.value.__cause__, cause)
        assert "max_tokens" in str(refused.value.__cause__)
        _assert_no_prompt_text(str(refused.value), corpus_ord)
        assert scripted.calls == []


def _no_model_id(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    return {"config": {}}


def _mismatched_model_id(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    return {"config": {"model_id": "m-1"}, "model_id": "m-2"}


def _corrupt_cassette(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    path = tmp_path / "corrupt.cassette.json"
    path.write_text("not a cassette", encoding="utf-8")
    return {"cassette_path": path, "cassette_mode": Mode.AUTO}


def _missing_cassette(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    return {"cassette_path": tmp_path / "absent.cassette.json", "cassette_mode": Mode.REPLAY}


def _unknown_mode_variable(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    monkeypatch.setenv("CALIBRATED_AGENT_EVALUATION_CASSETTE_MODE", "sometimes")
    return {}


def _replay_without_cassette(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    return {"cassette_mode": Mode.REPLAY}


def _replay_variable_without_cassette(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> dict[str, Any]:
    monkeypatch.setenv("CALIBRATED_AGENT_EVALUATION_CASSETTE_MODE", "replay")
    return {}


def _capitalized_threshold(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    return {"confidence_threshold": "High"}


def _no_threshold(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    return {"confidence_threshold": None}


def _negative_limit(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    return {"max_live_calls": -1}


def _naive_now(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    return {"now": NOW.replace(tzinfo=None)}


@pytest.mark.parametrize(
    ("configure", "problem"),
    [
        (_no_model_id, "no model_id"),
        (_mismatched_model_id, "'m-2'"),
        (_corrupt_cassette, "corrupt.cassette.json"),
        (_missing_cassette, "does not exist"),
        (_unknown_mode_variable, "CALIBRATED_AGENT_EVALUATION_CASSETTE_MODE"),
        (_replay_without_cassette, "no cassette_path"),
        (_replay_variable_without_cassette, "no cassette_path"),
        (_capitalized_threshold, "confidence_threshold='High'"),
        (_no_threshold, "confidence_threshold=None"),
        (_negative_limit, "max_live_calls=-1"),
        (_naive_now, "timezone"),
    ],
    ids=[
        "no model id anywhere",
        "a passed id that differs",
        "a corrupt cassette file",
        "a missing cassette in replay mode",
        "an unknown cassette mode variable",
        "replay mode with no cassette",
        "a replay mode variable with no cassette",
        "a capitalized threshold",
        "no threshold",
        "a negative live-call limit",
        "a now without a timezone",
    ],
)
def test_an_unusable_configuration_is_refused_at_construction(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, configure: Any, problem: str
) -> None:
    options = configure(tmp_path, monkeypatch)
    scripted = ScriptedModel([_answer()], config=options.pop("config", None))

    with pytest.raises(DetectorConfigError) as refused:
        FailureDetector(scripted, **options)
    assert problem in str(refused.value)
    assert scripted.calls == []


def test_a_passed_model_id_is_the_id_the_cassette_records(
    corpus_ord: Trace, tmp_path: Path
) -> None:
    path = tmp_path / "detector.cassette.json"
    response = _answer(_entry("ORD-1001/orderbot", _finding(PREMATURE, "low", "refunded")))
    recorder, _ = _detector(
        response, config={}, model_id="m-1", cassette_path=path, cassette_mode=Mode.RECORD
    )
    recorded = recorder.detect(corpus_ord)

    assert recorded.provenance.model_id == "m-1"
    assert [entry["model"]["id"] for entry in _entries(path)] == ["m-1"]

    same_model, _ = _detector(
        config={}, model_id="m-1", cassette_path=path, cassette_mode=Mode.REPLAY, max_live_calls=0
    )
    assert same_model.detect(corpus_ord) == recorded

    other_model, scripted = _detector(
        config={}, model_id="m-2", cassette_path=path, cassette_mode=Mode.REPLAY
    )
    with pytest.raises(DetectorCallError, match="ORD-1001") as refused:
        other_model.detect(corpus_ord)
    assert isinstance(refused.value.__cause__, CassetteModelMismatch)
    assert scripted.calls == []


def test_a_model_reconfigured_after_the_detector_is_built_is_recorded_as_it_was_called(
    corpus_ord: Trace, tmp_path: Path
) -> None:
    path = tmp_path / "detector.cassette.json"
    detector, scripted = _detector(
        _answer(),
        _answer(),
        config={"model_id": "m-1", "temperature": 0.0},
        cassette_path=path,
        cassette_mode=Mode.RECORD,
        max_live_calls=2,
    )

    scripted.update_config(temperature=0.9)
    provenance = detector.detect(corpus_ord).provenance

    assert (provenance.model_id, provenance.model_config_sent) == ("m-1", {"temperature": 0.9})
    assert [entry["model"] for entry in _entries(path)] == [
        {"id": "m-1", "config": {"temperature": 0.9}}
    ]

    # A detection under another model would credit that model's answer to the one recorded.
    scripted.update_config(model_id="m-2")
    with pytest.raises(DetectorConfigError, match="'m-2'") as refused:
        detector.detect(corpus_ord)
    assert "'m-1'" in str(refused.value)
    assert len(scripted.calls) == 1
    assert len(_entries(path)) == 1


def test_one_detector_records_and_replays_every_trace_in_one_cassette(
    corpus_ord: Trace, corpus_clm: Trace, tmp_path: Path
) -> None:
    path = tmp_path / "detector.cassette.json"
    recorder, _ = _detector(
        _answer(_entry("ORD-1001/orderbot", _finding(PREMATURE, "low", "refunded"))),
        _answer(_entry("CLM-4471/payout", _finding(PREMATURE, "high", "paid early"))),
        cassette_path=path,
        cassette_mode=Mode.RECORD,
        max_live_calls=2,
    )
    recorded = [recorder.detect(corpus_ord), recorder.detect(corpus_clm)]
    assert len(_entries(path)) == 2

    replayer, scripted = _detector(cassette_path=path, cassette_mode=Mode.REPLAY, max_live_calls=0)
    assert [replayer.detect(corpus_ord), replayer.detect(corpus_clm)] == recorded
    assert scripted.calls == []


@pytest.mark.parametrize(
    "vendor_default",
    [None, "global.anthropic.claude-sonnet-4-6"],
    ids=["no model", "a model id string"],
)
def test_the_vendor_default_model_is_never_used(
    corpus_ord: Trace, monkeypatch: pytest.MonkeyPatch, vendor_default: Any
) -> None:
    connections = spy_connections(monkeypatch)

    with pytest.raises(DetectorConfigError):
        FailureDetector(vendor_default, max_live_calls=1, now=NOW)
    assert connections == []

    detection = _detect(corpus_ord, _answer())
    assert detection.provenance.model_id == "scripted-model"
    assert connections == []


def test_every_detection_records_its_detector_provenance(
    corpus_ord: Trace, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = {"model_id": "scripted-model", "temperature": 0.0, "max_tokens": 2048}

    provenance = _detect(corpus_ord, _answer(), config=config).provenance

    assert provenance.detector_package == "strands-agents-evals"
    assert provenance.detector_version == "1.6.0"
    assert provenance.prompt_template == "v0"
    assert provenance.prompt_sha256 == VERIFIED_SHA256
    assert provenance.model_id == "scripted-model"
    assert provenance.model_config_sent == {"temperature": 0.0, "max_tokens": 2048}
    assert provenance.confidence_threshold == "low"
    assert provenance.stamp.dep_versions["strands-agents-evals"] == "1.6.0"
    assert provenance.stamp.run_at == "2026-10-27T09:30:00+00:00"

    changed = failure_detection_v0.SYSTEM_PROMPT + "\nFlag every span."
    monkeypatch.setattr(failure_detection_v0, "SYSTEM_PROMPT", changed)
    expected = hashlib.sha256(
        f"{changed}\n\n{failure_detection_v0.build_prompt(session_json='[]')}".encode()
    ).hexdigest()

    after_change = _detect(corpus_ord, _answer()).provenance
    assert after_change.prompt_sha256 == expected
    assert after_change.prompt_sha256 != VERIFIED_SHA256


def test_a_detection_identifies_the_trace_and_its_time_source(corpus_ord: Trace) -> None:
    from_fixture = _detect(corpus_ord, _answer())
    recorded = _detect(normalize(OTelSpans(path=OTEL_ORD)), _answer())

    assert from_fixture.times == "placeholder"
    assert recorded.times == "recorded"
    assert (
        from_fixture.trace_sha256
        == hashlib.sha256(canonical_json(corpus_ord.model_dump(mode="json")).encode()).hexdigest()
    )
    assert recorded.trace_sha256 != from_fixture.trace_sha256


def test_the_live_call_limit_reaches_the_caller_through_the_vendor(
    corpus_ord: Trace, corpus_clm: Trace
) -> None:
    nothing_allowed, unsent = _detector(_answer(), max_live_calls=0)
    with pytest.raises(DetectorCallLimitExceeded, match="max_live_calls=0") as refused:
        nothing_allowed.detect(corpus_ord)
    assert refused.value.limit == 0
    assert unsent.calls == []

    # One cap serves every detection the detector makes.
    one_allowed, scripted = _detector(_answer(), _answer(), max_live_calls=1)
    one_allowed.detect(corpus_ord)
    with pytest.raises(DetectorCallLimitExceeded, match="max_live_calls=1"):
        one_allowed.detect(corpus_clm)
    assert len(scripted.calls) == 1
