# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Run the ``strands-agents-evals`` failure detector on a normalized trace.

The detector library returns the same empty result when it finds nothing and when it could not
read its model's answer, logging a warning as the only difference. :class:`FailureDetector` keeps
the two apart. It converts the trace, refuses a prompt the library would split into chunks before
any call, lets the library make its one model call through the capturing wrapper, and then reads
the captured answer itself, strictly: one JSON object with an ``errors`` key, every entry with as
many categories as evidence strings and confidence levels, every category from the vocabulary and
every location a span or tool call of the trace. The library's own reading must agree with the
strict one. Anything else is a typed refusal that names the trace, so an empty detection always
means the model reported no failure, or reported only failures below the confidence threshold.
"""

from __future__ import annotations

import hashlib
import threading
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from pydantic import ValidationError
from strands.models.model import Model
from strands_evals.detectors import (
    ConfidenceLevel,
    FailureDetectionStructuredOutput,
    FailureOutput,
    detect_failures,
)
from strands_evals.detectors.chunking import would_exceed_context
from strands_evals.detectors.constants import CONFIDENCE_MAP

from calibrated_agent_evaluation._provenance import capture_provenance
from calibrated_agent_evaluation.agentcore.detectors._payload import PayloadPickError, pick_one_json
from calibrated_agent_evaluation.agentcore.detectors.models import call_error, prepare_model
from calibrated_agent_evaluation.agentcore.detectors.session import (
    ConvertedSession,
    canonical_json,
    session_json,
    trace_to_session,
)
from calibrated_agent_evaluation.agentcore.detectors.types import (
    Confidence,
    DetectedFailure,
    Detection,
    DetectorProvenance,
    FailureVocabulary,
)
from calibrated_agent_evaluation.agentcore.detectors.vocabulary import (
    detector_prompt,
    failure_vocabulary,
    installed_detector,
)
from calibrated_agent_evaluation.agentcore.errors import (
    DetectorCallError,
    DetectorConfigError,
    DetectorError,
    DetectorOutputError,
    DetectorSessionTooLarge,
)
from calibrated_agent_evaluation.execute.cassette import Mode
from calibrated_agent_evaluation.schema.types import Trace

__all__ = ["FailureDetector"]

_DETECTOR_DISTRIBUTION = "strands-agents-evals"
# Confidence levels in increasing order; a threshold keeps its own level and every higher one.
_RANK: dict[str, int] = {"low": 0, "medium": 1, "high": 2}
# A quoted piece of the model's answer is cut to this many characters.
_FRAGMENT_CHARS = 200
# A schema refusal lists at most this many problems and says how many more there were.
_SHOWN_PROBLEMS = 5
# What to do about an answer in the wrong shape.
_TRY_AGAIN = "Detect the trace again: a model can answer in another shape on a second try"


@dataclass(frozen=True)
class _Finding:
    """One category the model named at one location, in the order the answer gave it."""

    location: str
    category: str
    confidence: Confidence
    evidence: str


class FailureDetector:
    """Detects failures in normalized traces with the ``strands-agents-evals`` failure detector.

    The model stack is built and checked once, when the detector is built, so one live-call cap
    and one cassette serve every detection it makes, and one cassette can hold the calls for many
    traces. Each detection makes one model call, with the library's own prompt; the cassette
    records and replays it under that prompt. Detections on one detector run one at a time.

    Args:
        model: The strands ``Model`` to call, for example
            ``detector_bedrock_model("global.anthropic.claude-sonnet-4-6")``. ``None`` and a
            model id string are refused, because the library would build a Bedrock model of its
            own from them.
        model_id: The model id, when the model's configuration has none; if both are given they
            must be equal. It is recorded on every detection and cassette entry.
        confidence_threshold: The lowest confidence a reported failure keeps: ``"low"``,
            ``"medium"`` or ``"high"``. Failures below it are counted, not returned.
        cassette_path: A cassette file that records and replays the model calls, or ``None`` to
            make every call live.
        cassette_mode: The cassette mode; ``None`` reads
            ``CALIBRATED_AGENT_EVALUATION_CASSETTE_MODE`` (default auto).
        max_live_calls: How many live model calls this detector may make over its lifetime. A
            cassette replay is not a live call.
        now: The time every detection's provenance stamp records, for reproducible results; it
            must be timezone-aware. ``None`` reads the clock at each detection.

    Raises:
        DetectorConfigError: The model, the model id, the cassette, the cassette mode, the
            threshold, the live-call limit or ``now`` cannot be used. Nothing is called.
    """

    def __init__(
        self,
        model: Model,
        *,
        model_id: str | None = None,
        confidence_threshold: Confidence = "low",
        cassette_path: str | Path | None = None,
        cassette_mode: Mode | None = None,
        max_live_calls: int = 0,
        now: datetime | None = None,
    ) -> None:
        if confidence_threshold is None:
            # The shared check reads a missing threshold as "no threshold", which a detector needs.
            raise DetectorConfigError(
                "confidence_threshold=None is not a confidence level: use 'low', 'medium' or "
                "'high', in lower case"
            )
        self._prepared = prepare_model(
            model,
            model_id=model_id,
            cassette_path=cassette_path,
            cassette_mode=cassette_mode,
            max_live_calls=max_live_calls,
            confidence_threshold=confidence_threshold,
        )
        if now is not None and (not isinstance(now, datetime) or now.utcoffset() is None):
            # Checked here, because the stamp is taken after the model call has been paid for.
            raise DetectorConfigError(
                f"now={now!r} is not a timezone-aware datetime, so the detections' stamps could "
                "not record when they ran: pass one with a timezone, such as "
                "datetime.now(tz=UTC), or leave now out"
            )
        self._threshold: Confidence = confidence_threshold
        self._now = now
        # The capturing wrapper holds one detection's prompt and answer at a time.
        self._lock = threading.Lock()

    def detect(self, trace: Trace) -> Detection:
        """Detect the failures in one trace with one model call.

        Args:
            trace: A normalized, owned trace with spans.

        Returns:
            One failure for each category the model named at each span or tool call, at or
            above the confidence threshold, in the converted session's span order and then the
            answer's order; a category named twice at one location is kept twice. The failures
            the threshold dropped are counted in ``below_threshold``.

        Raises:
            SessionConversionError: The trace cannot be converted for the detector.
            DetectorSessionTooLarge: The trace is too large for one model call, by the library's
                own estimate before any call or by the model's own refusal.
            DetectorCallLimitExceeded: The call would exceed ``max_live_calls``; it was not sent.
            DetectorCallError: The model, the cassette or the library failed, before or during
                the call; or the call could not be prepared, because the installed template is
                unusable or the trace could not be serialized or sized, and was not made. The
                original error is chained.
            DetectorConfigError: The model's config now names another model id than the one the
                detector was built for, or cannot be read; nothing was sent.
            DetectorOutputError: The library returned without its model call, or the model's
                answer cannot be read faithfully: not one JSON object with an ``errors`` key, a
                malformed entry, an unknown category or location, or a reading the library's own
                parse disagrees with.
        """
        with self._lock:
            return self._detect(trace)

    def _detect(self, trace: Trace) -> Detection:
        trace_id = trace.trace_id
        converted = trace_to_session(trace)
        try:
            vocabulary, user_prompt, full_prompt = _read_template(session_json(converted), trace_id)
            # The library checks the user prompt; the one call it makes also carries the
            # instructions, so both must pass.
            too_large = would_exceed_context(user_prompt) or would_exceed_context(full_prompt)
        except DetectorError:
            raise
        except Exception as exc:
            # Serializing and sizing run before the call, so no call was made. Only the type name
            # of the cause is given: its text can quote the trace.
            raise _not_prepared(
                trace_id,
                f"serializing the trace or estimating the size of its prompt failed "
                f"({type(exc).__name__})",
            ) from exc
        if too_large:
            raise DetectorSessionTooLarge(
                f"trace {trace_id!r} is too large for one detector call: its prompt is over the "
                "detector library's own size estimate, and the library would split it into "
                "several calls whose silent skips the detector does not allow; nothing was sent. "
                "Detect a smaller trace"
            )
        # Read before the call, so the detection records the configuration it was made with.
        config_sent = self._prepared.config_sent()
        model = self._prepared.model
        model.arm(full_prompt)
        try:
            output = detect_failures(
                converted.session, confidence_threshold=ConfidenceLevel.LOW, model=model
            )
        except Exception as exc:
            error = call_error(exc, trace_id=trace_id)
            if error is exc:
                raise
            raise error from error.__cause__
        if not model.completed:
            raise DetectorOutputError(
                f"the detector library returned for trace {trace_id!r} without making its model "
                "call, so its empty result says nothing about the trace; it does this when it "
                "decides a trace must be split. Detect a smaller trace"
            )
        findings = _strict_findings(model.text, trace_id)
        _check_known(findings, vocabulary, converted, trace_id)
        _check_agreement(output, findings, trace_id)
        failures = _failures(findings, vocabulary, converted)
        kept = tuple(f for f in failures if _RANK[f.confidence] >= _RANK[self._threshold])
        return Detection(
            trace_id=trace_id,
            trace_sha256=_trace_sha256(trace),
            times=converted.times,
            failures=kept,
            below_threshold=len(failures) - len(kept),
            provenance=DetectorProvenance(
                detector_package=_DETECTOR_DISTRIBUTION,
                detector_version=vocabulary.detector_version,
                prompt_template=vocabulary.prompt_template,
                prompt_sha256=vocabulary.prompt_sha256,
                model_id=self._prepared.model_id,
                model_config_sent=config_sent,
                confidence_threshold=self._threshold,
                stamp=capture_provenance(
                    now=self._now, extra_distributions=(_DETECTOR_DISTRIBUTION,)
                ),
            ),
        )


def _read_template(serialized: str, trace_id: str) -> tuple[FailureVocabulary, str, str]:
    """The vocabulary and the user and full prompts, read from the installed template.

    Read on every detection, before its model call, so the vocabulary, the hash and the prompt all
    describe the template sent. A :class:`~calibrated_agent_evaluation.agentcore.DetectorError` the
    readers raise, such as a :class:`~calibrated_agent_evaluation.agentcore.VocabularyParseError`,
    passes through unchanged; anything else becomes a
    :class:`~calibrated_agent_evaluation.agentcore.DetectorCallError` that says no call was made,
    chained from it.
    """
    try:
        vocabulary = failure_vocabulary()
        user_prompt, full_prompt = detector_prompt(serialized)
    except DetectorError:
        raise
    except Exception as exc:
        # Only the type name of the cause is given: its text can quote the template.
        raise _not_prepared(
            trace_id,
            f"the installed {installed_detector()} template is unusable ({type(exc).__name__})",
        ) from exc
    return vocabulary, user_prompt, full_prompt


def _not_prepared(trace_id: str, because: str) -> DetectorCallError:
    """The refusal for a failure before the model call, which says that no call was made."""
    return DetectorCallError(
        f"the detector could not prepare its call for trace {trace_id!r}, because {because}; no "
        "model call was made. The original error is chained as the cause and says what is wrong"
    )


def _strict_findings(text: str, trace_id: str) -> list[_Finding]:
    """The model's answer read strictly, one finding per category in answer order."""
    try:
        payload = pick_one_json(text, opener="{", accept=_is_errors_object)
    except PayloadPickError as exc:
        raise DetectorOutputError(
            f"the detector's answer for trace {trace_id!r} cannot be read as one JSON object "
            f"with an 'errors' key ({exc}); {_answer_start(text)}. {_TRY_AGAIN}"
        ) from exc
    try:
        parsed = FailureDetectionStructuredOutput.model_validate(payload)
    except ValidationError as exc:
        raise DetectorOutputError(
            f"the detector's answer for trace {trace_id!r} does not follow the detector's output "
            f"format: {_problems(exc)}. {_TRY_AGAIN}"
        ) from exc
    findings: list[_Finding] = []
    for index, entry in enumerate(parsed.errors):
        where = f"entry {index} (location {_quoted(entry.location)})"
        counts = (len(entry.category), len(entry.evidence), len(entry.confidence))
        if len(set(counts)) != 1:
            raise DetectorOutputError(
                f"the detector's answer for trace {trace_id!r} cannot be read faithfully: {where} "
                f"lists {counts[0]} categories, {counts[1]} evidence strings and {counts[2]} "
                "confidence levels, and each category needs one evidence string and one "
                f"confidence level. {_TRY_AGAIN}"
            )
        if counts[0] == 0:
            raise DetectorOutputError(
                f"the detector's answer for trace {trace_id!r} cannot be read faithfully: {where} "
                "names no category, so a location the model flagged would read as nothing found. "
                f"{_TRY_AGAIN}"
            )
        findings.extend(
            _Finding(entry.location, category, confidence, evidence)
            for category, evidence, confidence in zip(
                entry.category, entry.evidence, entry.confidence, strict=True
            )
        )
    return findings


def _is_errors_object(value: object) -> bool:
    return isinstance(value, dict) and "errors" in value


def _check_known(
    findings: Sequence[_Finding],
    vocabulary: FailureVocabulary,
    converted: ConvertedSession,
    trace_id: str,
) -> None:
    """Refuse a location the trace does not have, or a category the vocabulary does not hold."""
    names = set(vocabulary.names())
    for finding in findings:
        if finding.location not in converted.locations:
            raise DetectorOutputError(
                f"the detector's answer for trace {trace_id!r} points at the location "
                f"{_quoted(finding.location)}, which is no span or tool call of the trace, so the "
                "failure cannot be placed. Detect the trace again, or use another model"
            )
        if finding.category not in names:
            raise DetectorOutputError(
                f"the detector's answer for trace {trace_id!r} names the category "
                f"{_quoted(finding.category)}, which is not in the detector's failure vocabulary. "
                "Detect the trace again, or use another model"
            )


def _check_agreement(output: FailureOutput, findings: Sequence[_Finding], trace_id: str) -> None:
    """Refuse an answer the library read differently from the strict reading."""
    strict = Counter(
        (f.location, f.category, CONFIDENCE_MAP[f.confidence], f.evidence) for f in findings
    )
    library = _library_findings(output)
    if library != strict:
        read = (
            "a failure whose categories, confidence levels and evidence strings differ in number"
            if library is None
            else _count(library.total(), "failure")
        )
        raise DetectorOutputError(
            f"the detector library and the strict reading disagree on the answer for trace "
            f"{trace_id!r}: the library read {read} and the strict reading read "
            f"{_count(len(findings), 'failure')}, or both read as many but not the same ones. "
            "No detection is returned, because one of the two misreads the answer. The "
            f"installed {_DETECTOR_DISTRIBUTION} may read answers differently from the version "
            "Calibrated Agent Evaluation was checked against: install that version, then detect "
            "the trace again"
        )


def _library_findings(output: FailureOutput) -> Counter[tuple[str, str, float, str]] | None:
    """The library's failures as a multiset, or ``None`` when an item's lists differ in length."""
    found: Counter[tuple[str, str, float, str]] = Counter()
    for item in output.failures:
        if not len(item.category) == len(item.confidence) == len(item.evidence):
            return None
        found.update(
            (item.span_id, category, confidence, evidence)
            for category, confidence, evidence in zip(
                item.category, item.confidence, item.evidence, strict=True
            )
        )
    return found


def _failures(
    findings: Sequence[_Finding], vocabulary: FailureVocabulary, converted: ConvertedSession
) -> list[DetectedFailure]:
    """One failure per finding, in span order and then answer order."""
    span_order = {location: index for index, location in enumerate(converted.locations)}
    parents = {entry.name: entry.parent for entry in vocabulary.entries}
    # sorted is stable, so findings at one location keep the order the answer gave them.
    ordered = sorted(findings, key=lambda finding: span_order[finding.location])
    failures: list[DetectedFailure] = []
    for finding in ordered:
        span_id, tool_call_index = converted.locations[finding.location]
        failures.append(
            DetectedFailure(
                span_id=span_id,
                tool_call_index=tool_call_index,
                tool_name=converted.tool_names.get(finding.location),
                category=finding.category,
                parent=parents[finding.category],
                confidence=finding.confidence,
                evidence=finding.evidence,
            )
        )
    return failures


def _count(number: int, noun: str) -> str:
    return f"{number} {noun}" if number == 1 else f"{number} {noun}s"


def _trace_sha256(trace: Trace) -> str:
    return hashlib.sha256(canonical_json(trace.model_dump(mode="json")).encode("utf-8")).hexdigest()


def _problems(exc: ValidationError) -> str:
    """The schema problems by field path, without quoting any of the answer."""
    problems = [
        f"{'.'.join(str(part) for part in error['loc']) or 'the answer'}: {error['msg']}"
        for error in exc.errors(include_url=False, include_context=False, include_input=False)
    ]
    shown = "; ".join(problems[:_SHOWN_PROBLEMS])
    hidden = len(problems) - _SHOWN_PROBLEMS
    return shown if hidden <= 0 else f"{shown}; and {hidden} more"


def _quoted(value: Any) -> str:
    """``value`` quoted, cut to its first characters when it is long."""
    text = str(value)
    if len(text) <= _FRAGMENT_CHARS:
        return repr(text)
    return f"{text[:_FRAGMENT_CHARS]!r} (cut to its first {_FRAGMENT_CHARS} characters)"


def _answer_start(text: str) -> str:
    stripped = text.strip()
    if not stripped:
        return "the answer is empty"
    return f"the answer starts {_quoted(stripped)}"
