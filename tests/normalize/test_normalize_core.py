# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Core normalizer contract + error-path tests.

Exercises the public ``normalize`` surface and its typed-error boundary against the real
``fixtures/raw/`` corpus: schema-valid output always, no vendor type on the public
surface, and every external failure wrapped as a chained ``NormalizationError``
rather than a raw vendor/stdlib exception.
"""

from __future__ import annotations

import inspect
import re
import typing
from pathlib import Path

import pytest

import calibrated_agent_evaluation.normalize as norm
from calibrated_agent_evaluation._install import install_hint
from calibrated_agent_evaluation.normalize import (
    InspectEvalLog,
    NormalizationError,
    OTelSpans,
    StrandsSession,
    normalize,
)
from calibrated_agent_evaluation.normalize.tree import Intermediate
from calibrated_agent_evaluation.schema.errors import (
    CalibratedAgentEvaluationDataError,
    SchemaError,
)
from calibrated_agent_evaluation.schema.types import Span, Trace
from calibrated_agent_evaluation.schema.validate import validate

# --- normalize output always passes validate; a mis-mapped tree surfaces typed ---------


def _all_sources(raw: Path) -> list[object]:
    return [
        InspectEvalLog(raw / "inspect" / "ORD-1001.eval"),
        InspectEvalLog(raw / "inspect" / "ORD-1002.eval"),
        InspectEvalLog(raw / "inspect" / "CLM-4471.eval"),
        OTelSpans(raw / "otel" / "ORD-1001.events.otlp.json", trace_id="ORD-1001"),
        OTelSpans(raw / "otel" / "ORD-1001.attrs.otlp.json", trace_id="ORD-1001"),
        OTelSpans(raw / "otel" / "ORD-1001.build2.otlp.json", trace_id="ORD-1001b"),
        OTelSpans(raw / "otel" / "ORD-1002.events.otlp.json", trace_id="ORD-1002"),
        OTelSpans(raw / "otel" / "ORD-DEGRADED.otlp.json", trace_id="ORD-1001d"),
        StrandsSession(raw / "strands" / "CLM-4471.otlp.json", trace_id="CLM-4471"),
        StrandsSession(raw / "strands" / "CLM-7720-swarm.otlp.json", trace_id="CLM-7720"),
        StrandsSession(raw / "strands" / "ORD-3310-as-tool.otlp.json", trace_id="ORD-3310"),
    ]


def test_every_corpus_source_normalizes_to_a_schema_valid_trace(raw_dir: Path) -> None:
    # sweep: normalize every frozen input and re-validate — output is always schema-valid,
    # regardless of source format.
    for source in _all_sources(raw_dir):
        trace = normalize(source)  # type: ignore[arg-type]
        assert isinstance(trace, Trace)
        validate(trace)  # raises if any invariant is violated


def test_mismapped_tree_duplicate_seq_surfaces_as_typed_normalization_error(
    monkeypatch: pytest.MonkeyPatch, raw_dir: Path
) -> None:
    # A normalizer bug that produced a duplicate seq must surface as a typed NormalizationError,
    # never a raw SchemaError leaking from the schema layer.
    from calibrated_agent_evaluation.normalize import ownership as ownership_mod
    from calibrated_agent_evaluation.normalize.ownership import Recovery

    def _broken(inter: Intermediate) -> tuple[Trace, Recovery]:
        bad = Trace(
            trace_id=inter.trace_id,
            frame="discovery",
            ownership="owned",
            input={"m": 1},
            output={"a": 2},
            taxonomy_cell={},
            spans=list(_dup_seq_spans(inter.trace_id)),
        )
        return bad, Recovery("owned", None, len(bad.spans), 0)

    monkeypatch.setattr(ownership_mod, "resolve_with_recovery", _broken)
    monkeypatch.setattr(norm, "resolve_with_recovery", _broken)
    with pytest.raises(NormalizationError) as excinfo:
        normalize(InspectEvalLog(raw_dir / "inspect" / "ORD-1001.eval"))
    # chained from the underlying SchemaError (never a raw SchemaError to the caller).
    assert isinstance(excinfo.value.__cause__, SchemaError)


def _dup_seq_spans(trace_id: str) -> list[Span]:
    common = {
        "trace_id": trace_id,
        "agent": "a",
        "role": "specialist",
        "status": "completed",
        "input": {},
    }
    return [
        Span(span_id=f"{trace_id}/a", parent=None, seq=0, **common),  # type: ignore[arg-type]
        Span(span_id=f"{trace_id}/b", parent=None, seq=0, **common),  # type: ignore[arg-type]
    ]


# --- no vendor type on the public surface (introspection) --------------------------------


def test_public_surface_exposes_no_vendor_type() -> None:
    # normalize's signature references only the vendor-free NormalizerSource union → Trace.
    sig = inspect.signature(normalize)
    return_ann = sig.return_annotation
    assert return_ann in (Trace, "Trace")
    params = list(sig.parameters.values())
    assert len(params) == 1
    # The descriptor dataclasses reference only stdlib / normalized-schema types (no inspect_ai /
    # opentelemetry type anywhere in their field annotations).
    for descriptor in (InspectEvalLog, OTelSpans, StrandsSession):
        for ann in typing.get_type_hints(descriptor).values():
            assert "inspect_ai" not in str(ann)
            assert "opentelemetry" not in str(ann)
            assert "langfuse" not in str(ann)


def test_normalization_error_is_in_the_calibrated_agent_evaluation_data_family() -> None:
    # A caller can catch the whole data-plane family at one boundary.
    assert issubclass(NormalizationError, CalibratedAgentEvaluationDataError)


# --- unknown source / unsupported semconv raise, nothing partial -------------------------


def test_unknown_source_descriptor_raises_normalization_error() -> None:
    class BogusSource:
        pass

    with pytest.raises(NormalizationError, match="unrecognized normalizer source"):
        normalize(BogusSource())  # type: ignore[arg-type]


def test_unsupported_semconv_raises_normalization_error(tmp_path: Path) -> None:
    # An unsupported OTel semantic convention is refused with a typed error, nothing partial.
    otlp = tmp_path / "trace.otlp.json"
    otlp.write_text('{"resourceSpans": []}', encoding="utf-8")
    with pytest.raises(NormalizationError):
        normalize(OTelSpans(otlp, semconv="vendor", trace_id="X"))  # type: ignore[arg-type]


def test_an_unknown_descriptor_names_the_otel_route_for_named_frameworks() -> None:
    class LangGraphRun:
        pass

    with pytest.raises(NormalizationError, match="export the framework's OpenTelemetry spans"):
        normalize(LangGraphRun())  # type: ignore[arg-type]


# --- missing extra → typed error naming the installable string ---------------------------


def test_missing_inspect_extra_raises_named_error(
    monkeypatch: pytest.MonkeyPatch, raw_dir: Path
) -> None:
    # Simulate the [inspect] extra being absent: the lazy import inside the adapter raises
    # ImportError, which must surface as a NormalizationError naming
    # calibrated-agent-evaluation[inspect].
    import builtins

    real_import = builtins.__import__

    def _blocked(name: str, *args: object, **kwargs: object) -> object:
        if name.startswith("inspect_ai"):
            raise ImportError("No module named 'inspect_ai'")
        return real_import(name, *args, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(builtins, "__import__", _blocked)
    with pytest.raises(NormalizationError, match=re.escape(install_hint("inspect"))):
        normalize(InspectEvalLog(raw_dir / "inspect" / "ORD-1001.eval"))


def test_missing_otel_extra_raises_named_error_for_binary_otlp(
    monkeypatch: pytest.MonkeyPatch, raw_dir: Path, tmp_path: Path
) -> None:
    # A binary .otlp path needs [otel]; with opentelemetry.proto absent the adapter must raise a
    # NormalizationError naming calibrated-agent-evaluation[otel], not a bare ImportError.
    import builtins

    binary = tmp_path / "trace.otlp"
    binary.write_bytes(b"\x00\x01\x02not-json")
    real_import = builtins.__import__

    def _blocked(name: str, *args: object, **kwargs: object) -> object:
        if name.startswith("opentelemetry"):
            raise ImportError("No module named 'opentelemetry'")
        return real_import(name, *args, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(builtins, "__import__", _blocked)
    with pytest.raises(NormalizationError, match=re.escape(install_hint("otel"))):
        normalize(OTelSpans(binary, trace_id="X"))


# --- corrupt/truncated source → chained typed error, never a raw vendor/stdlib exc -------


def test_truncated_eval_surfaces_as_chained_normalization_error(
    raw_dir: Path, tmp_path: Path
) -> None:
    # Derive the invalid input by truncating a real captured .eval rather than hand-writing one;
    # the corrupt read must surface typed and chained, never as a raw vendor error.
    good = (raw_dir / "inspect" / "ORD-1001.eval").read_bytes()
    truncated = tmp_path / "truncated.eval"
    truncated.write_bytes(good[: len(good) // 2])
    with pytest.raises(NormalizationError) as excinfo:
        normalize(InspectEvalLog(truncated))
    assert excinfo.value.__cause__ is not None  # chained from the original failure


def test_malformed_otlp_json_surfaces_as_chained_normalization_error(tmp_path: Path) -> None:
    # A malformed OTLP-JSON must raise NormalizationError chained from the JSONDecodeError, never
    # let a raw json.JSONDecodeError escape the public surface.
    bad = tmp_path / "bad.otlp.json"
    bad.write_text('{"resourceSpans": [ this is not json', encoding="utf-8")
    with pytest.raises(NormalizationError) as excinfo:
        normalize(OTelSpans(bad, trace_id="X"))
    import json as _json

    assert isinstance(excinfo.value.__cause__, _json.JSONDecodeError)
