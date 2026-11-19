# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Ingest orchestration result types (Phase 2 foundational).

Vendor-free: this module speaks the normalized schema and the stdlib OTLP-JSON reader only;
never a vendor SDK type. It defines the result records and the no-silent-loss gate
(``verify_no_silent_loss`` + ``source_signals``); the orchestration functions
(``run_ingest`` / ``capture_and_ingest`` / ``ingest_langfuse``) are added in the story phases.
"""

from __future__ import annotations

import json
import re
import tempfile
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal, NoReturn

from calibrated_agent_evaluation._install import install_hint
from calibrated_agent_evaluation._model import replace
from calibrated_agent_evaluation.backends.base import SourceTrace, TraceSource, TraceStoreBackend
from calibrated_agent_evaluation.ingest.detect import detect_dataset, detect_source, wire_trace_id
from calibrated_agent_evaluation.ingest.errors import IngestAbortedError, IngestError
from calibrated_agent_evaluation.ingest.sample import SampleRecord, draw_sample
from calibrated_agent_evaluation.normalize import (
    InspectEvalLog,
    NormalizationError,
    NormalizerSource,
    OTelSpans,
    Recovery,
    StrandsSession,
    content_placements,
    normalize_all,
    normalize_with_recovery,
)
from calibrated_agent_evaluation.normalize.otlp_json import (
    ATTR_AGENT_NAME,
    ATTR_OP,
    ATTR_TOOL_NAME,
    EVENT_ASSISTANT_MESSAGE,
    EVENT_CHOICE,
    EVENT_INFERENCE_DETAILS,
    EVENT_TOOL_MESSAGE,
    EVENT_USER_MESSAGE,
    attributes,
    event_attributes,
    iter_spans,
    latest_content,
)
from calibrated_agent_evaluation.normalize.sources import (
    DatasetSource,
    HFDataset,
    JSONLDataset,
    ParquetDataset,
)
from calibrated_agent_evaluation.schema.errors import BackendError, TraceNotFoundError
from calibrated_agent_evaluation.schema.types import Frame, Ownership, Trace

# ingested (persisted a faithful Trace. skipped) unparseable / no adapter.
# refused; the no-silent-loss gate fired. Only "ingested" is a clean pass.
IngestStatus = Literal["ingested", "skipped", "refused"]


@dataclass(frozen=True)
class IngestResult:
    """The per-source outcome the recovery summary reports."""

    source: str
    status: IngestStatus
    adapter: str | None = None
    span_count: int = 0
    tool_call_count: int = 0
    ownership: Ownership | None = None
    trace_id: str | None = None
    reason: str | None = None  # the actionable message for skipped/refused
    # Nodes that came back with no input, no output and no tool content. Reported on every
    # ingested trace, not only when it degrades: partial content loss does NOT degrade (that would
    # fire on a legitimately content-free span, e.g. a purely-delegating agent), so this count is
    # the only thing making the missing half visible. `contentless_node_count == span_count` is
    # the all-empty degrade condition.
    contentless_node_count: int = 0
    # Source tool calls that did not come back as tool calls. Any shortfall is refused by default,
    # since a trace missing one of its calls is not a record of what the agent did; with
    # `allow_partial_tools` the trace ingests and this count is how the loss stays visible.
    unrecovered_tool_spans: int = 0


@dataclass(frozen=True)
class IngestReport:
    """The aggregate of an ingest invocation; its ``exit_code`` is the process code."""

    results: list[IngestResult] = field(default_factory=list)
    dry_run: bool = False
    # Set only by a sampled ingest (`--sample`); the reproducible evidence for the draw.
    sample: SampleRecord | None = None

    @property
    def exit_code(self) -> int:
        """0 iff every source ingested (or this was a ``--dry-run`` preview); else 1.

        A dry run writes nothing and is a preview, so it never fails the process; a real run
        exits non-zero if any source was skipped or refused; never a silent exit-zero on a drop.
        """
        if self.dry_run:
            return 0
        return 0 if all(r.status == "ingested" for r in self.results) else 1

    @property
    def ingested(self) -> int:
        """How many inputs were ingested."""
        return sum(1 for r in self.results if r.status == "ingested")

    @property
    def refused(self) -> int:
        """How many inputs were refused."""
        return sum(1 for r in self.results if r.status == "refused")

    @property
    def skipped(self) -> int:
        """How many inputs were skipped."""
        return sum(1 for r in self.results if r.status == "skipped")

    @property
    def probed(self) -> int:
        """Ingested sources that degraded to outcome-only (``ownership="probed"``).

        A subset of :attr:`ingested`, not a fourth status: the trace parsed, passed every gate, and
        was persisted faithfully; what degraded is how much of it there was to persist. Counted
        here so the aggregate header can say so, because ``ownership`` was previously visible only
        on the per-source lines and in a ``logger.warning`` nobody reading the report sees.
        """
        return sum(1 for r in self.results if r.status == "ingested" and r.ownership == "probed")


def source_signals(otlp: dict[str, Any]) -> tuple[int, list[str]]:
    """From a raw OTLP-JSON payload, return ``(tool_execution_span_count, duplicate_span_labels)``.

    Computed on the SOURCE, independent of normalization, so the gate can distinguish "the source
    had tool spans but normalize recovered none" (the silent-drop failure mode) from "the agent
    genuinely made no tool calls" (a legitimate zero-tool trace). A tool-execution span is one whose
    ``gen_ai.operation.name`` is ``execute_tool`` or that carries a ``gen_ai.tool.name`` attribute.

    The OpenInference and legacy OpenLLMetry conventions emit no tool span at all; their calls are
    indexed attributes on an LLM span. For those the count is a floor: the most distinct calls any
    one span lists (an id counts once, an id-less entry once each), which no faithful
    reconstruction can come back under, whatever history the other spans resend.
    """
    spans = list(iter_spans(otlp))
    # A tool span that directly wraps an agent invocation is a sub-agent run as a tool, which the
    # adapters recover as an agent node rather than a tool call, so it is not counted as a tool.
    wraps_agent = {
        span.get("parentSpanId")
        for span in spans
        if attributes(span).get(ATTR_OP) == "invoke_agent" and span.get("parentSpanId")
    }
    tool_spans = 0
    seen: set[str] = set()
    dup_labels: dict[str, str] = {}
    for span in spans:
        attrs = attributes(span)
        is_tool = attrs.get(ATTR_OP) == "execute_tool" or ATTR_TOOL_NAME in attrs
        if is_tool and span.get("spanId") not in wraps_agent:
            tool_spans += 1
        span_id = span.get("spanId")
        if isinstance(span_id, str):
            if span_id in seen:
                label = attrs.get(ATTR_AGENT_NAME) or span.get("name") or span_id
                dup_labels[span_id] = str(label)
            seen.add(span_id)
    floor = max((_indexed_tool_call_count(attributes(span)) for span in spans), default=0)
    return max(tool_spans, floor), sorted(set(dup_labels.values()))


# The indexed tool-call grammars of OpenInference and legacy OpenLLMetry, matched here on their own
# rather than imported from the adapters, so the source side of the gate does not share the
# reader's assumptions.
_INDEXED_TOOL_CALL = re.compile(
    r"^(llm\.(?:input|output)_messages\.\d+\.message\.tool_calls\.\d+\.tool_call"
    r"|gen_ai\.completion\.\d+\.tool_calls\.\d+)"
    r"\.(?:id|name|arguments|function\.name|function\.arguments)$"
)


def _indexed_tool_call_count(attrs: dict[str, Any]) -> int:
    """Distinct tool calls one span lists in an indexed grammar: by id, else one per entry."""
    prefixes = {m.group(1) for key in attrs if (m := _INDEXED_TOOL_CALL.match(key))}
    ids: set[str] = set()
    idless = 0
    for prefix in prefixes:
        cid = attrs.get(f"{prefix}.id")
        if cid in (None, ""):
            idless += 1
        else:
            ids.add(str(cid))
    return len(ids) + idless


def source_trace_ids(otlp: dict[str, Any]) -> list[str]:
    """The distinct OTel trace ids on a raw OTLP-JSON payload, in first-seen order."""
    seen: dict[str, None] = {}
    for span in iter_spans(otlp):
        trace_id = span.get("traceId")
        if isinstance(trace_id, str) and trace_id:
            seen.setdefault(trace_id, None)
    return list(seen)


def mixed_traces_error(trace_ids: list[str]) -> IngestError:
    """Build the refusal for one source file that holds spans from several traces."""
    shown = ", ".join(trace_ids[:3]) + (", ..." if len(trace_ids) > 3 else "")
    return IngestError(
        f"the file holds spans from {len(trace_ids)} different traces ({shown}); one file is "
        "ingested as one trace, so these would be fused into a single run. Export one trace per "
        "file, or split the file by traceId and ingest each part"
    )


def duplicate_span_error(dup_span_names: list[str]) -> IngestError:
    """Build the refusal for duplicate span ids, naming the agents and the ``name=`` fix."""
    names = ", ".join(dup_span_names)
    return IngestError(
        f"source trace has duplicate span ids for agent(s) {names}; in the Strands SDK the "
        "agent name seeds the span id, so give each Agent(...) a distinct name= and re-capture"
    )


def tool_loss_error(source_tool_span_count: int) -> IngestError:
    """Build the refusal for tool spans present in the source but none recovered."""
    return IngestError(
        f"source carried {source_tool_span_count} tool-execution span(s) but the normalized "
        "trace recovered 0 tool calls; an adapter/parent-linking mismatch (use the Strands "
        "adapter for Strands traces); refusing rather than persisting an empty trace"
    )


def partial_tool_loss_error(source_tool_count: int, recovered: int) -> IngestError:
    """Build the refusal for a trace that recovered some of its source tool calls but not all."""
    return IngestError(
        f"source carried {source_tool_count} tool call(s) but the normalized trace recovered "
        f"{recovered}; a trace missing {source_tool_count - recovered} of its tool calls is not a "
        "record of what the agent did. Report the trace shape as an adapter gap, or pass "
        "--allow-partial-tools to ingest what was recovered and keep the count on the report"
    )


def existing_trace_error(trace_id: str) -> IngestError:
    """Build the refusal for a trace id the store already holds with different content."""
    return IngestError(
        f"the store already holds a different trace under id {trace_id!r}, and replacing it would "
        "re-attach every label on the old trace to this content; ingest into another store, give "
        "the source a distinct calibrated_agent_evaluation.trace_id span attribute or file name, "
        "or pass --overwrite "
        "to replace it"
    )


# What a reader raises on content outside its contract (a field of the wrong type, hostile nesting,
# a number out of range), as opposed to the typed NormalizationError it raises by design.
_UNHANDLED_CONTENT = (
    ValueError,
    TypeError,
    KeyError,
    AttributeError,
    RecursionError,
    OverflowError,
)


def unreadable_source_error(exc: BaseException) -> str:
    """The skip reason for a source whose content made the normalizer fail outside its contract."""
    return (
        f"normalizing the source failed with {type(exc).__name__}: {exc}; the content is outside "
        "what the reader handles, so it was skipped and the rest of the batch went on"
    )


def content_loss_error(source_content_span_count: int) -> IngestError:
    """Build the refusal for content present on the source spans but none recovered.

    The counterpart failure to :func:`tool_loss_error`, and the one a tool-call *count* cannot
    see: every step is recovered, every tool is named, and every ``input``/``output``/``args``/
    ``result`` is empty. It is what a placement mismatch looks like once the tree is right.
    """
    return IngestError(
        f"source carried per-step content on {source_content_span_count} span(s) but the "
        "normalized trace recovered none; the content is in a placement the adapter did not "
        "read (legacy span events vs latest-convention attributes); refusing rather than "
        "persisting a correctly-shaped trace with every step empty"
    )


def probed_refusal_error(trace_id: str, reason: str | None = None) -> IngestError:
    """Build the ``--require-owned`` refusal for a trace that degraded to outcome-only.

    Not a no-silent-loss refusal: a ``probed`` trace is a faithful record of an under-instrumented
    source, and refusing it by default would reject legitimate outcome-only data. It is refused only
    when the caller has declared that step-level data is a requirement; which is the honest place
    for that judgement, because whether outcome-only traces are usable depends on what the corpus is
    for, not on the trace.
    """
    detail = f" ({reason})" if reason else ""
    return IngestError(
        f"trace {trace_id!r} degraded to ownership='probed'{detail} and --require-owned was set; "
        "it carries terminal input/output only, with no per-step spans, so step-level attribution "
        "and per-agent rates have nothing to read; re-capture with tool/agent instrumentation, or "
        "drop --require-owned to accept outcome-only data"
    )


def source_content_span_count(otlp: dict[str, Any]) -> int:
    """Count source spans that carry recoverable per-step content, in ANY known placement.

    The content analogue of :func:`source_signals`' tool-span count, and computed on the source for
    the same reason: it is the only way to tell "the adapter dropped it" from "the run had none".
    Counts a span that carries any content-bearing legacy event, or a non-empty value on any key a
    registered placement family reads
    (:mod:`~calibrated_agent_evaluation.normalize.content_placements`); deliberately
    placement-agnostic, because the whole point is to compare what was on the wire against what came
    back regardless of where it rode. The keys come from the registry rather than a list here so the
    source side of the comparison can never know fewer placements than the reader does.

    An empty list or object (``"[]"``, ``"{}"``) is the key's presence, not content: counting it
    refused a source with nothing in any placement, blaming a placement the adapter did not read.
    """
    keys = tuple(
        dict.fromkeys(
            key for role in _CONTENT_ROLES for key in content_placements.placement_keys(role)
        )
    )
    events = (
        EVENT_USER_MESSAGE,
        EVENT_ASSISTANT_MESSAGE,
        EVENT_TOOL_MESSAGE,
        EVENT_CHOICE,
        EVENT_INFERENCE_DETAILS,
    )
    count = 0
    for span in iter_spans(otlp):
        if any(_has_payload(latest_content(span, key)) for key in keys):
            count += 1
            continue
        if any(event for name in events for event in event_attributes(span, name)):
            count += 1
    return count


_CONTENT_ROLES: tuple[content_placements.ContentRole, ...] = (
    "agent_input",
    "agent_output",
    "tool_args",
    "tool_result",
)


def _has_payload(value: Any) -> bool:
    """Whether an attribute value holds content, reading a JSON string as the value it encodes."""
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return False
        try:
            value = json.loads(text)
        except json.JSONDecodeError:
            return True
    return value not in (None, "", [], {})


def recovered_content_count(trace: Trace) -> int:
    """Count normalized spans that came back carrying content (per-step I/O or tool args/result)."""
    return sum(
        1
        for span in trace.spans
        if span.input or span.output or any(call.args or call.result for call in span.tool_calls)
    )


def contentless_node_count(trace: Trace) -> int:
    """Count normalized spans that came back with nothing a judge or labeler could read.

    The complement of :func:`recovered_content_count`, named and exposed because the *absence* is
    what gets reported: a tree of correct agents and correctly-named tool calls with ``{}`` in every
    content field is structurally valid, passes the tool-count gate, and is worthless to read. This
    figure is the one thing in the recovery summary that says so.
    """
    return len(trace.spans) - recovered_content_count(trace)


def verify_no_silent_loss(
    source_tool_span_count: int,
    dup_span_names: list[str],
    trace: Trace,
    *,
    source_content_spans: int = 0,
    recovery: Recovery | None = None,
) -> None:
    """Refuse a trace that would silently lose data; no-op on a faithful one.

    Raises :class:`IngestError` when the source has duplicate span ids, when the source
    carried tool-execution spans but the normalized ``trace`` recovered zero tool calls,
    or when the source carried per-step content and none of it survived ('s content half;
    a trace whose tools are all present *by name* with empty args and results passes the count
    check while having lost everything a judge would read). A genuinely tool-free or content-free
    source is never refused: both gates compare the source against the result, so "nothing was
    there" and "something was dropped" stay different claims.

    ``source_content_spans`` defaults to 0, i.e. the content gate is off unless the caller measured
    the source; a caller that cannot measure it must not have its trace refused for that.

    The content gate reads the recovered side through the trace's own ownership, because a
    ``probed`` trace has ``spans == []`` by definition and keeps its content in the terminal
    ``input``/``output`` envelope instead. Counting only span content refused every
    legitimately-degraded trace as a placement mismatch; wrong (nothing was lost, and the
    degradation had already been announced with a reason) and unactionable (there is no placement
    to go and fix). Whether an outcome-only trace is usable is the caller's call, taken through
    ``require_owned``.

    The tool gate reads ``recovery`` when the caller supplies it, so it measures what the ADAPTER
    linked rather than what survived the ownership collapse. The loss gate and the degrade rule ask
    different
    questions of the same ``probed``, ``spans == []`` trace; "did the adapter fail to link these
    tool spans?" (refuse) versus "was every node it linked empty?" (degrade, and explicitly do not
    refuse); and the finished Trace cannot tell them apart, because both look identical once the
    spans are gone. Without ``recovery`` the gate falls back to the post-collapse count, which is
    the conservative reading for a caller that did not measure.
    """
    if dup_span_names:
        raise duplicate_span_error(dup_span_names)
    recovered = (
        recovery.recovered_tool_calls
        if recovery is not None
        else sum(len(span.tool_calls) for span in trace.spans)
    )
    if source_tool_span_count > 0 and recovered == 0:
        raise tool_loss_error(source_tool_span_count)
    if source_content_spans > 0 and not _recovered_any_content(trace, source_content_spans):
        raise content_loss_error(source_content_spans)


def _recovered_any_content(trace: Trace, source_content_spans: int) -> bool:
    """True iff content survived normalization, wherever this ownership is entitled to keep it.

    An ``owned`` trace keeps content per-step, so a tree of correctly-named steps with nothing in
    them is a loss.

    A ``probed`` trace keeps content in the terminal envelope, and whether that is faithful depends
    on how much the source had. Content on exactly ONE source span *is* the terminal envelope;
    that is what "only a terminal input/output envelope was instrumented" means, and the envelope
    carries it intact. Content on several source spans was per-step content, and collapsing it into
    one envelope drops all but the terminal step; the silent loss this gate exists to catch.
    """
    if trace.ownership == "probed":
        return source_content_spans <= 1 and bool(trace.input or trace.output)
    return recovered_content_count(trace) > 0


def _adapter_name(descriptor: NormalizerSource) -> str:
    """The reader a source resolved to, before normalization says which convention read it.

    Shown on a line for a source that never reached normalization. Once a trace is normalized its
    line names the resolved convention instead (:func:`_resolved_label`), because ``otel`` alone
    cannot say whether ``gen_ai/messages``, OpenInference or OpenLLMetry interpreted it.
    """
    if isinstance(descriptor, StrandsSession):
        return "strands"
    if isinstance(descriptor, OTelSpans):
        return "otel"
    if isinstance(descriptor, InspectEvalLog):
        return "inspect"
    return type(descriptor).__name__.lower()


def _resolved_label(reader: str, recovery: Recovery, *, dataset: bool = False) -> str:
    """The summary's adapter column: the convention that read the trace, when the adapter knows."""
    if recovery.convention is None:
        return reader
    return f"{reader} ({recovery.convention})" if dataset else recovery.convention


def trace_id_collision_error(trace_id: str, paths: list[str]) -> IngestError:
    """Build the refusal for two sources in one batch that resolve to the same trace id."""
    listed = ", ".join(repr(p) for p in sorted(paths))
    return IngestError(
        f"{len(paths)} sources in this batch resolve to trace id {trace_id!r} ({listed}) and the "
        "store is keyed on it, so persisting them would overwrite one with the other; rename them "
        "so their first dotted filename segments differ, or tag each with a distinct "
        "calibrated_agent_evaluation.trace_id span attribute"
    )


def repeated_trace_id_error(trace_id: str, earlier: str) -> IngestError:
    """Build the refusal for a trace id an earlier source in the same batch already took."""
    return IngestError(
        f"trace id {trace_id!r} was already taken by {earlier!r} in this batch, and the store is "
        "keyed on it, so persisting this one would silently replace that one; ingest the two "
        "into separate stores, or drop the duplicate"
    )


def resolve_batch_trace_ids(
    paths: list[Path],
) -> tuple[dict[Path, str | None], dict[str, list[str]]]:
    """Resolve each path's trace id for the batch, returning ``(overrides, collisions)``.

    The store persists with ``INSERT OR REPLACE`` keyed on ``trace_id``, and the default trace id is
    the file name's first dotted segment; so ``run.json`` and ``run.otlp.json`` both resolve to
    ``"run"`` and the second silently replaced the first, one file at a time, while the report said
    ``2 ingested`` and exit 0.

    Two stages, in this order for a reason:

    1. A colliding stem is re-resolved from the source's own wire id (``wire_trace_id``). Only on
       collision: the stem is what makes ingest reproduce the frozen oracle ids, and preferring the
       wire id unconditionally would turn every fixture id into a hex blob for no gain.
    2. Anything still colliding is genuinely ambiguous; the same wire id, or no wire id to read;
       and is refused by the caller rather than silently overwritten.

    An override of ``None`` means "keep the stem default".
    """
    by_stem: dict[str, list[Path]] = {}
    for path in paths:
        by_stem.setdefault(path.name.split(".")[0], []).append(path)

    overrides: dict[Path, str | None] = dict.fromkeys(paths)
    resolved: dict[str, list[str]] = {}
    for stem, group in by_stem.items():
        if len(group) == 1:
            resolved.setdefault(stem, []).append(str(group[0]))
            continue
        for path in group:
            wire = wire_trace_id(path)
            if wire is not None:
                overrides[path] = wire
            resolved.setdefault(wire or stem, []).append(str(path))

    collisions = {tid: sources for tid, sources in resolved.items() if len(sources) > 1}
    return overrides, collisions


def _ingest_path(
    path: Path,
    *,
    backend: TraceStoreBackend,
    fmt: str,
    dry_run: bool,
    require_owned: bool = False,
    trace_id: str | None = None,
    claimed: dict[str, str] | None = None,
    convention: str = "auto",
    overwrite: bool = False,
    allow_partial_tools: bool = False,
    frame: Frame = "discovery",
    stamp_frame: Frame | None = None,
    session_id: str | None = None,
) -> list[IngestResult]:
    """Ingest one source file: the per-source step of :func:`run_ingest`.

    Detect the adapter, compute source signals from the raw OTLP, normalize, run the no-silent-loss
    gate, then persist (unless ``dry_run``). Any failure becomes a typed ``skipped``/``refused``
    result with an actionable reason; the batch is never aborted and nothing is silently dropped.

    Returns a **list** because one source is not always one trace: an Inspect ``.eval`` holds one
    sample per dataset row, so an N-row eval is N traces, and a JSONL / parquet / HuggingFace
    dataset is one trace per record. It used to return a single result and the
    adapter read only the first sample, so an eval over a dataset ingested as 1/N of itself.

    ``claimed`` maps each trace id this batch has already taken to the source that took it. The
    store persists with ``INSERT OR REPLACE`` keyed on trace id, so a second source resolving to a
    claimed id is refused rather than silently replacing the first.
    """
    src = str(path)
    claimed = {} if claimed is None else claimed
    try:
        dataset = detect_dataset(path, fmt, convention=convention)
    except IngestError as exc:
        return [IngestResult(source=src, status="skipped", reason=str(exc))]
    if dataset is not None:
        return _ingest_dataset(
            src,
            dataset,
            adapter=f"dataset/{_DATASET_LABEL[type(dataset)]}",
            backend=backend,
            dry_run=dry_run,
            require_owned=require_owned,
            claimed=claimed,
            overwrite=overwrite,
            allow_partial_tools=allow_partial_tools,
            frame=frame,
            stamp_frame=stamp_frame,
        )
    try:
        descriptor = detect_source(path, fmt, trace_id=trace_id, convention=convention, frame=frame)
    except IngestError as exc:
        return [IngestResult(source=src, status="skipped", reason=str(exc))]
    adapter = _adapter_name(descriptor)

    tool_spans = 0
    content_spans = 0
    dup_names: list[str] = []
    if isinstance(descriptor, OTelSpans | StrandsSession):
        try:
            loaded = _read_source_otlp(path)
        except (OSError, json.JSONDecodeError, NormalizationError) as exc:
            reason = f"cannot read {path.name!r}: {exc}"
            return [IngestResult(source=src, status="skipped", adapter=adapter, reason=reason)]
        if not isinstance(loaded, dict):
            # The loss gates measure the source, and a source they cannot read must not slip past
            # them with a count of zero.
            reason = (
                f"{path.name!r} is not an OTLP-JSON object (its top level is a "
                f"{type(loaded).__name__}), so it cannot be read as a trace"
            )
            return [IngestResult(source=src, status="skipped", adapter=adapter, reason=reason)]
        try:
            trace_ids = source_trace_ids(loaded)
            tool_spans, dup_names = source_signals(loaded)
            content_spans = source_content_span_count(loaded)
        except _UNHANDLED_CONTENT as exc:
            reason = unreadable_source_error(exc)
            return [IngestResult(source=src, status="skipped", adapter=adapter, reason=reason)]
        if len(trace_ids) > 1:
            reason = str(mixed_traces_error(trace_ids))
            return [IngestResult(source=src, status="refused", adapter=adapter, reason=reason)]

    if dup_names:  # refuse before normalize (which would raise a less actionable error)
        reason = str(duplicate_span_error(dup_names))
        return [IngestResult(source=src, status="refused", adapter=adapter, reason=reason)]
    try:
        # An Inspect .eval is one trace PER SAMPLE; every other descriptor is one trace. Both come
        # back as a list so the gate + persist loop below has one shape.
        normalized = _normalize_source(descriptor)
    except NormalizationError as exc:  # unparseable / no-adapter → skip, keep the batch
        return [IngestResult(source=src, status="skipped", adapter=adapter, reason=str(exc))]
    except _UNHANDLED_CONTENT as exc:
        # The normalizer's contract is NormalizationError; anything else is hostile or odd
        # content it does not handle. Skipped, named, and the batch goes on, instead of aborting
        # the run with the traces before it already written and no summary.
        reason = unreadable_source_error(exc)
        return [IngestResult(source=src, status="skipped", adapter=adapter, reason=reason)]

    return [
        _gate_and_persist(
            src,
            _resolved_label(adapter, recovery),
            trace,
            recovery,
            tool_spans=tool_spans if source_tools is None else source_tools,
            content_spans=content_spans,
            backend=backend,
            dry_run=dry_run,
            require_owned=require_owned,
            claimed=claimed,
            overwrite=overwrite,
            allow_partial_tools=allow_partial_tools,
            stamp_frame=stamp_frame,
            session_id=session_id,
        )
        for trace, recovery, source_tools in normalized
    ]


def _read_source_otlp(path: Path) -> Any:
    """The raw source document the gates measure: binary ``.otlp`` decoded, else parsed JSON."""
    if path.suffix == ".otlp":
        from calibrated_agent_evaluation.normalize.otel_adapter import read_otlp

        return read_otlp(path)
    return json.loads(path.read_text(encoding="utf-8"))


_DATASET_LABEL: dict[type, str] = {
    JSONLDataset: "jsonl",
    ParquetDataset: "parquet",
    HFDataset: "hf",
}


def _ingest_dataset(
    src: str,
    dataset: DatasetSource,
    *,
    adapter: str,
    backend: TraceStoreBackend,
    dry_run: bool,
    require_owned: bool,
    claimed: dict[str, str],
    overwrite: bool,
    allow_partial_tools: bool,
    frame: Frame = "discovery",
    stamp_frame: Frame | None = None,
) -> list[IngestResult]:
    """Ingest a dataset record by record, each through the same gates as a trace file.

    Each record's spans are measured on the source side before normalization, as a file's are, so a
    record whose tool calls or content would be lost is refused like a file would be. A record that
    cannot be mapped or normalized is skipped on its own line. A reader failure (a missing extra,
    an unreadable file, a malformed JSONL line) ends the stream, so it is reported once for the
    records from that point on and the records already read stand.
    """
    from calibrated_agent_evaluation.normalize.datasets import iter_sessions

    results: list[IngestResult] = []
    index = 0
    try:
        for index, session in enumerate(iter_sessions(dataset, frame=frame)):
            label = f"{src}[{index}]"
            if isinstance(session, NormalizationError):
                results.append(
                    IngestResult(
                        source=label, status="skipped", adapter=adapter, reason=str(session)
                    )
                )
                continue
            otlp = {"resourceSpans": [{"scopeSpans": [{"spans": _list_events(session.spans)}]}]}
            tool_spans, dup_names = source_signals(otlp)
            if dup_names:
                reason = str(duplicate_span_error(dup_names))
                results.append(
                    IngestResult(source=label, status="refused", adapter=adapter, reason=reason)
                )
                continue
            try:
                trace, recovery = normalize_with_recovery(session)
            except NormalizationError as exc:
                results.append(
                    IngestResult(source=label, status="skipped", adapter=adapter, reason=str(exc))
                )
                continue
            except _UNHANDLED_CONTENT as exc:
                reason = unreadable_source_error(exc)
                results.append(
                    IngestResult(source=label, status="skipped", adapter=adapter, reason=reason)
                )
                continue
            results.append(
                _gate_and_persist(
                    label,
                    _resolved_label(adapter, recovery, dataset=True),
                    trace,
                    recovery,
                    tool_spans=tool_spans,
                    content_spans=source_content_span_count(otlp),
                    backend=backend,
                    dry_run=dry_run,
                    require_owned=require_owned,
                    claimed=claimed,
                    overwrite=overwrite,
                    allow_partial_tools=allow_partial_tools,
                    stamp_frame=stamp_frame,
                )
            )
    except NormalizationError as exc:
        where = f"{src}[{index + 1 if results else 0}:]"
        results.append(
            IngestResult(source=where, status="skipped", adapter=adapter, reason=str(exc))
        )
    if not results:
        results.append(
            IngestResult(
                source=src, status="skipped", adapter=adapter, reason="dataset has no records"
            )
        )
    return results


def _list_events(spans: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Dataset spans with ``events`` as a list, for the source-side measurement only.

    A dataset row may carry ``events`` as a JSON string (``"[]"``), which the OTLP readers would
    iterate character by character. The normalizer tolerates it already; the gate has to as well.
    """
    out: list[dict[str, Any]] = []
    for span in spans:
        events = span.get("events")
        if isinstance(events, str):
            try:
                decoded = json.loads(events)
            except json.JSONDecodeError:
                decoded = []
            span = {**span, "events": decoded if isinstance(decoded, list) else []}
        out.append(span)
    return out


def _gate_and_persist(
    src: str,
    adapter: str,
    trace: Trace,
    recovery: Recovery,
    *,
    tool_spans: int,
    content_spans: int,
    backend: TraceStoreBackend,
    dry_run: bool,
    require_owned: bool,
    claimed: dict[str, str],
    overwrite: bool = False,
    allow_partial_tools: bool = False,
    stamp_frame: Frame | None = None,
    session_id: str | None = None,
) -> IngestResult:
    """Run the gates on one normalized trace and persist it unless refused or a dry run.

    ``stamp_frame`` overrides the frame the source declared on the wire. A sampled ingest sets it,
    and so does a caller that knows how the source's traces were drawn: there the draw, not
    whoever exported the trace, is what decides the frame.
    ``session_id`` is the session the source knows the trace by (AgentCore's ``session.id``),
    which no normalizer reads; it is stamped on the trace before the gates.
    """
    if stamp_frame is not None and trace.frame != stamp_frame:
        trace = replace(trace, frame=stamp_frame)
    if session_id is not None:
        trace = replace(trace, session_id=session_id)
    try:
        # Tool-count and content gates (duplicates were handled before normalizing).
        # `recovery` carries the pre-collapse counts so a contentless degrade is not misread as a
        # linking failure. Per trace, not per source: one bad sample in an eval must not discard
        # the good ones, and each gets its own reported reason.
        verify_no_silent_loss(
            tool_spans, [], trace, source_content_spans=content_spans, recovery=recovery
        )
    except IngestError as exc:
        return IngestResult(
            source=src, status="refused", adapter=adapter, trace_id=trace.trace_id, reason=str(exc)
        )
    unrecovered = max(0, tool_spans - recovery.recovered_tool_calls)
    if unrecovered and not allow_partial_tools:
        return IngestResult(
            source=src,
            status="refused",
            adapter=adapter,
            trace_id=trace.trace_id,
            unrecovered_tool_spans=unrecovered,
            reason=str(partial_tool_loss_error(tool_spans, recovery.recovered_tool_calls)),
        )

    if require_owned and trace.ownership == "probed":
        # Refused AFTER the no-silent-loss gates deliberately: those refusals describe a defect
        # in the adapter or the source, this one a mismatch between the source and what the
        # caller asked for. Reporting the adapter fault first keeps the better message.
        return IngestResult(
            source=src,
            status="refused",
            adapter=adapter,
            ownership=trace.ownership,
            trace_id=trace.trace_id,
            reason=str(probed_refusal_error(trace.trace_id)),
        )

    earlier = claimed.get(trace.trace_id)
    if earlier is not None:
        return IngestResult(
            source=src,
            status="refused",
            adapter=adapter,
            trace_id=trace.trace_id,
            reason=str(repeated_trace_id_error(trace.trace_id, earlier)),
        )
    if not overwrite and _holds_different_trace(backend, trace):
        return IngestResult(
            source=src,
            status="refused",
            adapter=adapter,
            trace_id=trace.trace_id,
            reason=str(existing_trace_error(trace.trace_id)),
        )
    claimed[trace.trace_id] = src
    if not dry_run:
        backend.put_trace(trace)
    return IngestResult(
        source=src,
        status="ingested",
        adapter=adapter,
        span_count=len(trace.spans),
        tool_call_count=sum(len(span.tool_calls) for span in trace.spans),
        ownership=trace.ownership,
        trace_id=trace.trace_id,
        contentless_node_count=contentless_node_count(trace),
        unrecovered_tool_spans=unrecovered,
    )


def _holds_different_trace(backend: TraceStoreBackend, trace: Trace) -> bool:
    """Whether the store already holds this trace id with other content (read before write).

    Ids default to the file name's first segment, and exporters reuse fixed names
    (``trace.json``), so Tuesday's run replaced Monday's under ``INSERT OR REPLACE`` and Monday's
    labels were re-attached to it. An identical stored trace is not refused, so re-running an
    ingest stays idempotent. Only a typed not-found frees the id; any other read failure
    propagates as the store's own ``BackendError`` and aborts the batch, since treating it as
    "free" would let the write replace what is there.
    """
    try:
        existing = backend.get_trace(trace.trace_id)
    except TraceNotFoundError:
        return False
    return existing != trace


def _normalize_source(
    descriptor: NormalizerSource,
) -> list[tuple[Trace, Recovery, int | None]]:
    """Normalize a source to ``(Trace, Recovery, source tool calls)``: N for an Inspect log, else 1.

    The third item is the trace's own source-side tool count when the source has one per trace,
    and ``None`` when the file-level count measured from the raw OTLP applies.

    The Inspect path goes through ``normalize_all`` rather than ``normalize_with_recovery`` because
    the latter is one-trace-shaped. `Recovery` is synthesized from the finished trace there, and the
    source side is each sample's assistant-message tool calls, a record of the same requests kept
    apart from the tool events the adapter reads.
    """
    if isinstance(descriptor, InspectEvalLog):
        from calibrated_agent_evaluation.normalize.inspect_adapter import message_tool_call_counts

        traces = normalize_all(descriptor)
        counts = message_tool_call_counts(descriptor)
        if len(counts) != len(traces):
            raise NormalizationError(
                f"Inspect log {str(descriptor.path)!r} read as {len(traces)} trace(s) but "
                f"{len(counts)} sample(s) on a second read; refusing rather than pairing them "
                "wrongly"
            )
        return [
            (
                trace,
                Recovery(
                    trace.ownership,
                    None,
                    len(trace.spans),
                    sum(len(span.tool_calls) for span in trace.spans),
                ),
                count,
            )
            for trace, count in zip(traces, counts, strict=True)
        ]
    trace, recovery = normalize_with_recovery(descriptor)
    return [(trace, recovery, None)]


def run_ingest(
    paths: Iterable[Path],
    *,
    backend: TraceStoreBackend,
    fmt: str = "auto",
    dry_run: bool = False,
    require_owned: bool = False,
    trace_id_override: str | None = None,
    convention: str = "auto",
    overwrite: bool = False,
    allow_partial_tools: bool = False,
    frame: Frame = "discovery",
    sample_size: int | None = None,
    sample_seed: int | None = None,
    sample_id: str = "sample",
) -> IngestReport:
    """Ingest each trace file: detect → normalize → gate → persist (006/011;  dry-run).

    A source that can't be parsed or matched is recorded as ``skipped`` (never aborts the batch,
    never a silent drop -); a lossy source is ``refused``. Only a faithfully
    persisted source is ``ingested``; the report's ``exit_code`` is non-zero if any other status.
    Backend-agnostic, so the same call persists to local or Langfuse identically.

    ``trace_id_override`` forces the trace id for a single-source call, bypassing the
    filename-stem default (``capture_and_ingest`` uses it to pass the captured OTel trace id).

    ``convention`` declares the OTel GenAI convention to read OTLP and dataset sources with
    (``calibrated-agent-evaluation ingest --convention``), for a trace whose signals match more than
    one; ``"auto"`` detects it.

    ``require_owned`` refuses a source that degrades to ``ownership="probed"`` instead of persisting
    it; for a corpus that will be analyzed step by step, where an outcome-only trace is not a
    smaller result but an unusable one. Off by default, because an outcome-only trace is a faithful
    record of an under-instrumented agent and refusing it unasked would reject real data.

    A trace id the store already holds with different content is refused unless ``overwrite`` is
    set; re-ingesting an identical trace is a no-op success. A trace that recovered some but not
    all of its source tool calls is refused unless ``allow_partial_tools`` is set, and then reports
    the shortfall in ``unrecovered_tool_spans``.

    ``frame`` is the sampling frame for a source whose spans carry no
    ``calibrated_agent_evaluation.frame`` attribute. It defaults to ``"discovery"``, the frame that
    authorizes no headline rate, because a file says nothing about how its traces were sampled;
    ``"production-representative"`` is a claim the caller makes, never one inherited by omission.

    Args:
        paths: The trace files or directories to ingest.
        backend: The store to write to.
        fmt: The source format, or ``"auto"`` to detect it per file.
        dry_run: Report what would happen without writing.
        require_owned: Refuse a source that degrades to ``probed``.
        trace_id_override: The trace id to use for a single source.
        convention: The OTel GenAI convention to read with, or ``"auto"``.
        overwrite: Replace a stored trace whose id a source resolves to.
        allow_partial_tools: Ingest a trace that recovered only some of its tool calls.
        frame: The sampling frame for sources that do not declare one.
        sample_size: Ingest a seeded random sample of this many sources instead of all of them.
        sample_seed: The seed for that draw; required with ``sample_size``.
        sample_id: The id the written
            :class:`~calibrated_agent_evaluation.ingest.sample.SampleRecord` carries.

    Returns:
        One result per source, and the exit code for the batch.

    Raises:
        IngestError: Two sources in the batch resolve to the same trace id, or the sample
            arguments are incomplete or ask for more sources than the population holds.
    """
    listed = list(paths)
    drawn: list[str] | None = None
    population = len(listed)
    stamp_frame: Frame | None = None
    if sample_size is not None:
        if sample_seed is None:
            raise IngestError(
                "a sample needs a seed so the draw can be repeated; pass the seed alongside the "
                "sample size"
            )
        drawn = draw_sample([str(path) for path in listed], size=sample_size, seed=sample_seed)
        keep = set(drawn)
        listed = [path for path in listed if str(path) in keep]
        # A seeded random sample of the population IS production-representative, whatever the
        # exported traces happen to be tagged.
        stamp_frame = "production-representative"
    if trace_id_override is not None:
        # Single-source path (`capture_and_ingest`): the caller already resolved the id, so the
        # filename-stem default and its collision machinery have nothing to contribute.
        if len(listed) != 1:
            raise IngestError(f"trace_id_override applies to exactly one source, got {len(listed)}")
        return IngestReport(
            results=_ingest_path(
                listed[0],
                backend=backend,
                fmt=fmt,
                dry_run=dry_run,
                require_owned=require_owned,
                trace_id=trace_id_override,
                convention=convention,
                overwrite=overwrite,
                allow_partial_tools=allow_partial_tools,
                frame=frame,
                stamp_frame=stamp_frame,
            ),
            dry_run=dry_run,
        )
    # A dataset's trace ids come from its records, not its file name, so datasets stay out of the
    # stem-collision resolver; `claimed` catches a record id that repeats across the batch.
    single = [path for path in listed if detect_dataset(path, fmt) is None]
    overrides, collisions = resolve_batch_trace_ids(single)
    colliding_paths = {src for sources in collisions.values() for src in sources}
    claimed: dict[str, str] = {}

    results: list[IngestResult] = []
    for path in listed:
        if str(path) in colliding_paths:
            tid = next(t for t, srcs in collisions.items() if str(path) in srcs)
            results.append(
                IngestResult(
                    source=str(path),
                    status="refused",
                    reason=str(trace_id_collision_error(tid, collisions[tid])),
                )
            )
            continue
        try:
            results.extend(
                _ingest_path(
                    path,
                    backend=backend,
                    fmt=fmt,
                    dry_run=dry_run,
                    require_owned=require_owned,
                    trace_id=overrides.get(path),
                    claimed=claimed,
                    convention=convention,
                    overwrite=overwrite,
                    allow_partial_tools=allow_partial_tools,
                    frame=frame,
                    stamp_frame=stamp_frame,
                )
            )
        except BackendError as exc:
            _raise_aborted(str(path), results, exc)
    return IngestReport(
        results=results,
        dry_run=dry_run,
        sample=_sample_record(
            sample_id,
            source="files",
            drawn=drawn,
            population_size=population,
            seed=sample_seed,
            results=results,
        ),
    )


def _sample_record(
    sample_id: str,
    *,
    source: str,
    drawn: list[str] | None,
    population_size: int,
    seed: int | None,
    results: list[IngestResult],
    cell: dict[str, str] | None = None,
    excluded_unknown_cell: int = 0,
) -> SampleRecord | None:
    """The record for a sampled ingest; ``None`` when the ingest was not sampled."""
    if drawn is None or seed is None:
        return None
    from calibrated_agent_evaluation._provenance import capture_provenance

    return SampleRecord(
        sample_id=sample_id,
        source=source,
        population_size=population_size,
        sample_size=len(drawn),
        drawn=tuple(drawn),
        trace_ids=tuple(
            r.trace_id for r in results if r.status == "ingested" and r.trace_id is not None
        ),
        cell=dict(cell or {}),
        excluded_unknown_cell=excluded_unknown_cell,
        provenance=capture_provenance(seed=seed),
    )


def _raise_aborted(source: str, results: list[IngestResult], exc: BackendError) -> NoReturn:
    """Re-raise a store failure, naming what the batch had already written if it wrote anything.

    A failure on the first write leaves the store as it was, so the store's own error is the whole
    story and is re-raised as it is. A failure after some writes is reported as an abort that
    names them, because the store now holds part of the batch.
    """
    written = tuple(r.trace_id for r in results if r.status == "ingested" and r.trace_id)
    if not written:
        raise exc
    shown = ", ".join(written[:5]) + (", ..." if len(written) > 5 else "")
    raise IngestAbortedError(
        f"the store failed while ingesting {source}, so the rest of the batch was not ingested; "
        f"{len(written)} trace(s) were written before it failed ({shown}). Fix the store and "
        f"re-run the batch: {exc}",
        written=written,
    ) from exc


def ingest_langfuse(
    source: TraceStoreBackend | TraceSource,
    *,
    backend: TraceStoreBackend,
    dry_run: bool = False,
    require_owned: bool = False,
    overwrite: bool = False,
    allow_partial_tools: bool = False,
    sample_size: int | None = None,
    sample_seed: int | None = None,
    sample_id: str = "sample",
    cell: dict[str, str] | None = None,
    source_label: str = "langfuse",
    source_kind: str = "langfuse",
    frame: Frame | None = None,
) -> IngestReport:
    """Pull every trace from a Langfuse source into the destination backend.

    A source that implements :class:`~calibrated_agent_evaluation.backends.base.TraceSource`
    (``LangfuseBackend`` does) is read with ``pull``, which returns traces Calibrated Agent
    Evaluation wrote as normalized ``Trace``s and traces anything else wrote as OTLP-JSON. The
    foreign ones go through the same path as a file (detect the convention, normalize, the
    no-silent-loss gates), so they are ingested, refused or skipped exactly as the same spans in a
    file would be. A trace that cannot be read is ``skipped`` with its reason and the pull carries
    on; it used to raise out of ``query`` and abort the pull on the first foreign trace.

    Any other backend is read with ``query``, its own traces only.

    ``overwrite`` and ``allow_partial_tools`` mean what they mean for :func:`run_ingest`.
    ``source_kind`` names where each trace came from in its result's ``source``
    (``<source_kind>:<trace id>``), and is the ``adapter`` of a trace the source could not hand
    over; a source that reuses this function, such as AgentCore, passes its own.

    ``sample_size`` + ``sample_seed`` ingest a seeded random sample of the pulled population
    instead of all of it, tag the sampled traces ``production-representative`` and return the
    draw as the report's :class:`~calibrated_agent_evaluation.ingest.sample.SampleRecord`. ``cell``
    restricts the population to traces in that taxonomy cell, which makes the draw a stratified one;
    a pulled trace whose cell is unknown (a foreign export carries none) cannot be in the stratum,
    so it is left out of the population and counted on the record. ``frame`` stamps every pulled
    trace with that sampling frame, for a caller that knows how the source's traces were drawn (an
    AgentCore online config that scores every session, or a random share of them, reads a random
    sample of the agent's traffic); a sample is always ``production-representative``, so ``frame``
    is refused together with ``sample_size``.

    The source's listing raises ``BackendError`` before any trace is yielded when it is
    unreachable, so nothing is reported ingested. Vendor-free here: this speaks the
    backend protocols and plain OTLP-JSON dicts, never a Langfuse type.
    """
    if frame is not None and sample_size is not None:
        raise IngestError(
            "a sampled ingest is always production-representative, so it takes no frame; pass "
            "either the sample size or the frame"
        )
    pulled: Iterable[SourceTrace]
    if isinstance(source, TraceSource):
        pulled = source.pull()
    else:
        pulled = (SourceTrace(source_id=t.trace_id, trace=t) for t in source.query())

    # Traces Calibrated Agent Evaluation wrote go first, so when a foreign export of the same run
    # resolves to the same trace id, the faithful copy is the one kept and the foreign one is
    # refused as a repeat.
    items = sorted(pulled, key=lambda item: item.trace is None)
    drawn: list[str] | None = None
    stamp_frame: Frame | None = frame
    excluded = 0
    population = len(items)
    if sample_size is not None:
        if sample_seed is None:
            raise IngestError(
                "a sample needs a seed so the draw can be repeated; pass the seed alongside the "
                "sample size"
            )
        if cell:
            eligible = [item for item in items if _item_in_cell(item, cell)]
            excluded = len(items) - len(eligible)
            items = eligible
        population = len(items)
        # Keyed on the normalized trace id where the source has one, so the record's draw names
        # the traces a reader can look up rather than the store's internal id for them.
        keyed = {_item_key(item): item for item in items}
        drawn = draw_sample(list(keyed), size=sample_size, seed=sample_seed)
        items = [keyed[key] for key in drawn]
        stamp_frame = "production-representative"
    claimed: dict[str, str] = {}
    results: list[IngestResult] = []
    with tempfile.TemporaryDirectory(prefix="calibrated-agent-evaluation-langfuse-") as tmp:
        for index, item in enumerate(items):
            label = f"{source_kind}:{item.source_id}"
            try:
                results.extend(
                    _ingest_pulled(
                        item,
                        label,
                        Path(tmp) / f"{index}.otlp.json",
                        backend=backend,
                        dry_run=dry_run,
                        require_owned=require_owned,
                        claimed=claimed,
                        overwrite=overwrite,
                        allow_partial_tools=allow_partial_tools,
                        stamp_frame=stamp_frame,
                        source_kind=source_kind,
                    )
                )
            except BackendError as exc:
                _raise_aborted(label, results, exc)
    return IngestReport(
        results=results,
        dry_run=dry_run,
        sample=_sample_record(
            sample_id,
            source=source_label,
            drawn=drawn,
            population_size=population,
            seed=sample_seed,
            results=results,
            cell=cell,
            excluded_unknown_cell=excluded,
        ),
    )


def _item_key(item: SourceTrace) -> str:
    """The id a pulled item is drawn and recorded under: its normalized trace id if it has one."""
    return item.trace.trace_id if item.trace is not None else item.source_id


def _item_in_cell(item: SourceTrace, cell: dict[str, str]) -> bool:
    """Whether a pulled item's trace is in ``cell``; a trace with no cell never is."""
    if item.trace is None:
        return False
    return all(item.trace.taxonomy_cell.get(key) == value for key, value in cell.items())


def _ingest_pulled(
    item: SourceTrace,
    label: str,
    path: Path,
    *,
    backend: TraceStoreBackend,
    dry_run: bool,
    require_owned: bool,
    claimed: dict[str, str],
    overwrite: bool,
    allow_partial_tools: bool,
    stamp_frame: Frame | None = None,
    source_kind: str = "langfuse",
) -> list[IngestResult]:
    """Ingest one pulled item: a library-authored trace, a foreign export, or an error."""
    if item.trace is not None:
        return [
            _ingest_normalized(
                item.trace,
                label,
                backend=backend,
                dry_run=dry_run,
                require_owned=require_owned,
                claimed=claimed,
                overwrite=overwrite,
                stamp_frame=stamp_frame,
            )
        ]
    if item.otlp is not None:
        # Through a file so the foreign trace takes the file path unchanged: the same detection,
        # the same source-side signals, the same gates and the same report.
        path.write_text(json.dumps(item.otlp), encoding="utf-8")
        return [
            replace(result, source=label)
            for result in _ingest_path(
                path,
                backend=backend,
                fmt="auto",
                dry_run=dry_run,
                require_owned=require_owned,
                trace_id=item.source_id,
                claimed=claimed,
                overwrite=overwrite,
                allow_partial_tools=allow_partial_tools,
                stamp_frame=stamp_frame,
                session_id=item.session_id,
            )
        ]
    reason = item.error or "the source returned neither a trace nor an error"
    # The source could not hand over a trace, so no normalizer ran; the adapter is the source.
    return [IngestResult(source=label, status="skipped", adapter=source_kind, reason=reason)]


def _ingest_normalized(
    trace: Trace,
    label: str,
    *,
    backend: TraceStoreBackend,
    dry_run: bool,
    require_owned: bool,
    claimed: dict[str, str],
    overwrite: bool,
    stamp_frame: Frame | None = None,
) -> IngestResult:
    """Ingest a trace the source already holds normalized (a library-authored Langfuse trace).

    ``stamp_frame`` is the sampled-ingest frame claim; see :func:`_gate_and_persist`.
    """
    if stamp_frame is not None and trace.frame != stamp_frame:
        trace = replace(trace, frame=stamp_frame)
    if require_owned and trace.ownership == "probed":
        return IngestResult(
            source=label,
            status="refused",
            adapter="langfuse",
            ownership=trace.ownership,
            trace_id=trace.trace_id,
            reason=str(probed_refusal_error(trace.trace_id)),
        )
    earlier = claimed.get(trace.trace_id)
    if earlier is not None:
        return IngestResult(
            source=label,
            status="refused",
            adapter="langfuse",
            trace_id=trace.trace_id,
            reason=str(repeated_trace_id_error(trace.trace_id, earlier)),
        )
    if not overwrite and _holds_different_trace(backend, trace):
        return IngestResult(
            source=label,
            status="refused",
            adapter="langfuse",
            trace_id=trace.trace_id,
            reason=str(existing_trace_error(trace.trace_id)),
        )
    claimed[trace.trace_id] = label
    if not dry_run:
        backend.put_trace(trace)
    return IngestResult(
        source=label,
        status="ingested",
        adapter="langfuse",
        span_count=len(trace.spans),
        tool_call_count=sum(len(span.tool_calls) for span in trace.spans),
        ownership=trace.ownership,
        trace_id=trace.trace_id,
        contentless_node_count=contentless_node_count(trace),
    )


def capture_and_ingest(
    agent: Callable[[], object],
    *,
    service_name: str,
    trace_id: str | None = None,
    backend: TraceStoreBackend,
    dry_run: bool = False,
    register_global: bool = True,
    require_owned: bool = False,
    overwrite: bool = False,
    agent_label: str | None = None,
) -> IngestReport:
    """Run an in-process agent under OTel capture, then ingest the captured trace.

    Captures the spans ``agent()`` emits in **this** process (a subprocess cannot share the
    in-memory exporter, so capture is in-process; an instrumented callable, like 's executor),
    encodes canonical OTLP-JSON, and runs it through the same file-ingest path (detect → gate →
    persist) so the no-silent-loss guarantee applies. If the agent emits no span, raises
    :class:`IngestError` rather than persisting an empty trace. Needs the ``[strands]``
    extra for the OTel SDK; its absence is a typed error naming the extra.

    ``trace_id`` is optional and defaults to the **captured OTel trace id**, unique per run by
    construction. It used to default (via the CLI) to ``--service``, which itself defaults to the
    constant ``"agent"``; so two captures with no explicit ids resolved to the same trace id, and
    the second silently replaced the first under ``INSERT OR REPLACE`` while both runs reported
    ``1 ingested`` and exit 0.

    ``overwrite`` must be set to replace a trace id the store already holds. Without it, an explicit
    ``trace_id`` that already exists is refused rather than clobbering the earlier capture; an id
    collision is at least as likely to be a mistake as an intention, and only the caller knows.

    An agent that raises is reported as an :class:`IngestError` naming ``agent_label`` (the CLI
    passes the ``--agent`` spec; it defaults to the callable's qualified name), chained from the
    agent's own exception, and nothing is persisted: a run that died part-way is not a trace of the
    agent's behaviour, and persisting the spans it left would read as a complete run.

    Args:
        agent: A no-argument callable that runs the agent once in this process.
        service_name: The OTel service name the capture records spans under.
        trace_id: The trace id to store; defaults to the captured OTel trace id.
        backend: The store to write to.
        dry_run: Report what would happen without writing.
        register_global: Install the capture's tracer provider as the global one.
        require_owned: Refuse a capture that degrades to ``probed``.
        overwrite: Replace a trace the store already holds under ``trace_id``.
        agent_label: The name to use for the agent in error messages.

    Returns:
        The ingest report for the captured trace.

    Raises:
        IngestError: The agent raised, emitted no spans, the OTel SDK is missing, the trace id is
            already stored without ``overwrite``, or the store can't be read to check it.
    """
    try:
        from calibrated_agent_evaluation.execute.strands_capture import (
            CaptureProviderError,
            capture_spans,
            write_otlp_json,
        )
    except ImportError as exc:  # missing extra → typed, actionable
        # Name the module that failed. The bare "install the extra" hint was all a user saw when
        # the extra itself lacked a dependency, so they reinstalled what they had.
        missing = exc.name or str(exc)
        raise IngestError(
            "calibrated-agent-evaluation capture needs the OTel SDK and OTLP encoder (cannot "
            f"import {missing!r}); "
            f"{install_hint('strands')}"
        ) from exc

    label = agent_label or getattr(agent, "__qualname__", None) or repr(agent)
    with capture_spans(service_name) as cap:
        if register_global:
            try:
                cap.register_global()
            except CaptureProviderError as exc:
                raise IngestError(f"cannot capture {label!r}: {exc}") from exc
        try:
            agent()
        except Exception as exc:  # the user's code; any failure is reported, never swallowed
            raise IngestError(
                f"agent {label!r} raised {type(exc).__name__}: {exc}; the run did not finish, so "
                "nothing was persisted"
            ) from exc
    # Read only after the `with` exits. `SpanCapture.__exit__` force-flushes the provider, and a
    # span that ends during that flush (an instrumentation closing its span from a flush hook, or a
    # buffering processor draining) reaches the exporter only then. Reading inside the block
    # dropped those spans and reported the capture as complete.
    spans = cap.finished_spans
    if not spans:  # a broken/uninstrumented agent that emitted nothing
        raise IngestError(
            f"captured 0 spans from {label!r}: the agent emitted no OpenTelemetry spans "
            "(is it instrumented? did it exit before a span closed?), so nothing was stored"
        )
    with tempfile.TemporaryDirectory(prefix="calibrated-agent-evaluation-capture-") as tmp:
        # The file name seeds the trace id (see `detect._trace_id_for`), so an explicit id rides
        # through the name and an absent one leaves the adapter to read the captured OTel traceId.
        stem = trace_id or "calibrated-agent-evaluation-capture"
        otlp_path = Path(tmp) / f"{stem}.otlp.json"
        write_otlp_json(spans, otlp_path)
        resolved = trace_id or wire_trace_id(otlp_path)
        if resolved is not None and not dry_run and not overwrite:
            _refuse_existing_trace(backend, resolved, explicit=trace_id is not None)
        return run_ingest(
            [otlp_path],
            backend=backend,
            dry_run=dry_run,
            require_owned=require_owned,
            trace_id_override=None if trace_id else resolved,
            # `_refuse_existing_trace` above already ran the stricter capture check (a capture
            # refuses even an identical stored trace unless told to replace it).
            overwrite=True,
        )


def _refuse_existing_trace(backend: TraceStoreBackend, trace_id: str, *, explicit: bool) -> None:
    """Refuse to overwrite a trace id the store already holds (sibling; --overwrite opts in).

    Read before write. The store's ``INSERT OR REPLACE`` makes a repeated id a silent replacement,
    and a capture is not repeatable; the run that produced the earlier trace is gone.

    Only a typed not-found frees the id. Any other read failure (auth, network, a 5xx) refuses
    the capture, since treating it as "free" would let the write replace what is there.
    """
    try:
        backend.get_trace(trace_id)
    except TraceNotFoundError:
        return
    except Exception as exc:
        raise IngestError(
            f"could not check whether the store already holds trace {trace_id!r}, so the capture "
            f"was not written: {type(exc).__name__}: {exc}"
        ) from exc
    hint = (
        "pass a different --trace-id, or --overwrite to replace it"
        if explicit
        else "this should not happen with a derived id; pass --overwrite if it is intended"
    )
    raise IngestError(
        f"the store already holds trace {trace_id!r} and a capture cannot be re-run to recover "
        f"what would be replaced; {hint}"
    )
