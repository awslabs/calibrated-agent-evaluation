# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Per-step token usage survives normalization.

Normalizing used to *discard* `gen_ai.usage.*`: the attributes were on the wire, the schema had
nowhere to put them, and so an ingested `Trace` could describe a run's topology while being silent
about what the run cost — much of why a team ingests traces at all. These tests pin the recovery
end to end against the real five-agent Strands capture, and pin the reader's refusal to invent a
number it was not given.
"""

from __future__ import annotations

from pathlib import Path

from calibrated_agent_evaluation.normalize import StrandsSession, normalize, otlp_json
from calibrated_agent_evaluation.schema.validate import validate


def _capture(raw_dir: Path) -> StrandsSession:
    return StrandsSession(
        raw_dir / "strands" / "CLM-4471.otlp.json", trace_id="CLM-4471", mode="legacy"
    )


def _span(**attrs: object) -> dict[str, object]:
    """A span carrying already-decoded attributes (the dataset-row shape `attributes()` accepts)."""
    return {"attributes": dict(attrs)}


# --- the reader ------------------------------------------------------------


def test_the_current_genai_spelling_is_read() -> None:
    usage = otlp_json.usage(
        _span(
            **{
                "gen_ai.usage.input_tokens": 100,
                "gen_ai.usage.output_tokens": 20,
                "gen_ai.usage.total_tokens": 120,
            }
        )
    )
    assert usage is not None
    assert (usage.input_tokens, usage.output_tokens, usage.total_tokens) == (100, 20, 120)


def test_the_earlier_spelling_is_read_too() -> None:
    """A trace is produced by whatever instrumentation a team already runs, so refusing the older
    spelling would report "no usage" for a span that plainly carries it."""
    usage = otlp_json.usage(
        _span(
            **{
                "gen_ai.usage.prompt_tokens": 7,
                "gen_ai.usage.completion_tokens": 3,
            }
        )
    )
    assert usage is not None
    assert (usage.input_tokens, usage.output_tokens) == (7, 3)


def test_the_current_spelling_wins_when_both_are_present() -> None:
    usage = otlp_json.usage(
        _span(**{"gen_ai.usage.input_tokens": 9, "gen_ai.usage.prompt_tokens": 1})
    )
    assert usage is not None
    assert usage.input_tokens == 9


def test_a_missing_total_is_not_computed_from_the_parts() -> None:
    """Negative path, and the reason the fields are independently optional: a provider that counts
    cached or reasoning tokens reports a total the pair does not add up to, and a derived total is
    indistinguishable from a reported one once it is stored."""
    usage = otlp_json.usage(
        _span(**{"gen_ai.usage.input_tokens": 10, "gen_ai.usage.output_tokens": 5})
    )
    assert usage is not None
    assert usage.total_tokens is None


def test_a_span_reporting_nothing_yields_none_rather_than_zeroes() -> None:
    assert otlp_json.usage(_span(**{"gen_ai.operation.name": "invoke_agent"})) is None
    assert otlp_json.usage({}) is None


def test_a_boolean_attribute_is_not_a_count() -> None:
    """`bool` is an `int` subclass, so a mis-typed attribute would otherwise become `1` token."""
    assert otlp_json.usage(_span(**{"gen_ai.usage.input_tokens": True})) is None


def test_a_numeric_string_is_accepted() -> None:
    """Some exporters stringify every attribute value; the count is still a count."""
    usage = otlp_json.usage(_span(**{"gen_ai.usage.total_tokens": "512"}))
    assert usage is not None
    assert usage.total_tokens == 512


def test_a_non_numeric_string_is_refused() -> None:
    assert otlp_json.usage(_span(**{"gen_ai.usage.total_tokens": "lots"})) is None


# --- end to end, against the real capture ---------------------------------


def test_every_agent_step_of_the_real_capture_carries_its_usage(raw_dir: Path) -> None:
    trace = normalize(_capture(raw_dir))
    validate(trace)
    assert len(trace.spans) == 5
    for span in trace.spans:
        assert span.usage is not None, f"{span.agent} lost its usage"
        assert span.usage.input_tokens, span.agent
        assert span.usage.output_tokens, span.agent
        assert span.usage.total_tokens, span.agent


def test_usage_is_read_per_agent_and_not_summed_from_model_children(raw_dir: Path) -> None:
    """The capture carries usage on both the `invoke_agent` span and each `chat` child beneath it.
    Reading the agent span is the only non-double-counting choice, and it is also the level the
    attribution is about — so the numbers must be that span's own, not a sum over its cycles."""
    trace = normalize(_capture(raw_dir))
    supervisor = next(s for s in trace.spans if s.agent == "supervisor")
    assert supervisor.usage is not None
    # The supervisor's own single turn, not the crew total: every other agent reports more input
    # than it does, so a summed figure could not be the smallest.
    own_input = supervisor.usage.input_tokens
    assert own_input is not None
    others = [s.usage.input_tokens for s in trace.spans if s.agent != "supervisor" and s.usage]
    assert others
    for other in others:
        assert other is not None
        assert other > own_input
