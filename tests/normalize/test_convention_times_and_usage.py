# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""The convention adapters keep span times and token usage.

The messages, OpenInference and OpenLLMetry adapters collapse a session into one span and used to
leave its ``started_at``, ``ended_at`` and ``usage`` empty, even when every source span carried
them. Downstream, ``improve --evaluate`` then found no time in the after store and skipped the
check that the pre-registration came first.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest

from calibrated_agent_evaluation.backends.local import LocalBackend
from calibrated_agent_evaluation.ingest.core import run_ingest
from calibrated_agent_evaluation.normalize import SessionSpans, normalize, otlp_json
from calibrated_agent_evaluation.schema.types import TokenUsage

_CONVENTION_FIXTURES = {
    "messages": "otel/messages/swebench_tool_call.otlp.json",
    "openllmetry": "otel/openllmetry/weather_tool_call.otlp.json",
    "openinference": "otel/openinference/weather_tool_call.otlp.json",
}


def _ingest(path: Path, tmp_path: Path) -> Any:
    store = LocalBackend(root=str(tmp_path / "store"))
    report = run_ingest([path], backend=store)
    assert report.exit_code == 0, report.results
    (result,) = report.results
    assert result.trace_id is not None
    return store.get_trace(result.trace_id)


def _iso(nanos: int) -> str:
    moment = datetime.fromtimestamp(nanos // 1_000_000_000, tz=UTC)
    return moment.replace(microsecond=(nanos % 1_000_000_000) // 1_000).isoformat()


@pytest.mark.parametrize("convention", sorted(_CONVENTION_FIXTURES))
def test_each_convention_fixture_keeps_its_span_times(
    convention: str, raw_dir: Path, tmp_path: Path
) -> None:
    path = raw_dir / _CONVENTION_FIXTURES[convention]
    spans = otlp_json.iter_spans(json.loads(path.read_text(encoding="utf-8")))
    (span,) = _ingest(path, tmp_path).spans
    assert span.started_at == _iso(min(int(s["startTimeUnixNano"]) for s in spans))
    assert span.ended_at == _iso(max(int(s["endTimeUnixNano"]) for s in spans))


def test_the_openinference_fixture_keeps_its_token_counts(raw_dir: Path, tmp_path: Path) -> None:
    """The capture reports ``llm.token_count.prompt/completion/total`` = 97 / 52 / 149."""
    (span,) = _ingest(raw_dir / _CONVENTION_FIXTURES["openinference"], tmp_path).spans
    assert span.usage == TokenUsage(input_tokens=97, output_tokens=52, total_tokens=149)


@pytest.mark.parametrize("convention", ["messages", "openllmetry"])
def test_a_fixture_without_token_counts_reports_none(
    convention: str, raw_dir: Path, tmp_path: Path
) -> None:
    (span,) = _ingest(raw_dir / _CONVENTION_FIXTURES[convention], tmp_path).spans
    assert span.usage is None


def test_a_dataset_row_keeps_its_iso_times_and_summed_usage(raw_dir: Path, tmp_path: Path) -> None:
    """The swebench row's two chat spans carry ISO times and per-call usage, and no total."""
    path = raw_dir / "hf" / "swebench-session.jsonl"
    record = json.loads(path.read_text(encoding="utf-8").splitlines()[0])
    rows = record["spans"]
    (span,) = _ingest(path, tmp_path).spans

    def utc(raw: str) -> str:
        return datetime.fromisoformat(raw).astimezone(UTC).isoformat()

    assert span.started_at == min(utc(r["start_time"]) for r in rows)
    assert span.ended_at == max(utc(r["end_time"]) for r in rows)
    assert span.usage == TokenUsage(
        input_tokens=sum(r["attributes"]["gen_ai.usage.input_tokens"] for r in rows),
        output_tokens=sum(r["attributes"]["gen_ai.usage.output_tokens"] for r in rows),
        total_tokens=None,
    )


def _legacy_openllmetry_span(*, start: int, end: int, usage: dict[str, int]) -> dict[str, Any]:
    return {
        "startTimeUnixNano": str(start),
        "endTimeUnixNano": str(end),
        "attributes": {
            "gen_ai.system": "openai",
            "gen_ai.prompt.0.role": "user",
            "gen_ai.prompt.0.content": "hello",
            "gen_ai.completion.0.role": "assistant",
            "gen_ai.completion.0.content": "hi",
            **usage,
        },
    }


def test_a_multi_span_session_spans_its_earliest_start_to_its_latest_end() -> None:
    first = 1_700_000_000_000_000_000
    spans = [
        _legacy_openllmetry_span(
            start=first + 5_000_000_000,
            end=first + 9_000_000_000,
            usage={"gen_ai.usage.prompt_tokens": 10, "llm.usage.total_tokens": 12},
        ),
        _legacy_openllmetry_span(
            start=first,
            end=first + 2_000_000_000,
            usage={
                "gen_ai.usage.prompt_tokens": 5,
                "gen_ai.usage.completion_tokens": 1,
                "llm.usage.total_tokens": 6,
            },
        ),
    ]
    trace = normalize(SessionSpans(spans=spans, semconv="openllmetry", trace_id="s"))
    (span,) = trace.spans
    assert span.started_at == _iso(first)
    assert span.ended_at == _iso(first + 9_000_000_000)
    # The first span reports no completion count, so a sum of the second's alone is not stored.
    assert span.usage == TokenUsage(input_tokens=15, output_tokens=None, total_tokens=18)


def test_a_session_without_times_keeps_none() -> None:
    span = _legacy_openllmetry_span(start=0, end=0, usage={})
    del span["startTimeUnixNano"], span["endTimeUnixNano"]
    trace = normalize(SessionSpans(spans=[span], semconv="openllmetry", trace_id="s"))
    assert (trace.spans[0].started_at, trace.spans[0].ended_at, trace.spans[0].usage) == (
        None,
        None,
        None,
    )
