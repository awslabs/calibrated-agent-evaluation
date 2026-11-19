# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Which arm of an A/B test each trace ran in, read from the arm stamps on its spans.

Vendor-free. A Gateway routing an A/B test sends the experiment ARN and the variant name as
baggage, and the AgentCore SDK copies each onto every span it starts, under its own ``is not
None`` check. So a span carries both (stamped), one (half-stamped) or neither, and
:func:`span_stamp` is the one rule every count here uses. :func:`read_arms` reads any
``TraceSource``; :class:`RawTraceLog` wraps one of the trace ingest's readers to keep what the
arm read needs that the ingest's source does not pass on.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass, field, replace
from typing import TYPE_CHECKING, Any, Literal, Protocol

from calibrated_agent_evaluation.agentcore.audit_types import (
    ArmAssignment,
    ArmStatus,
    canonical_json,
    sha256_text,
)
from calibrated_agent_evaluation.agentcore.errors import (
    AgentCoreAppliedBundleMismatch,
    AgentCoreArmHalfStamped,
    AgentCoreArmSessionAmbiguous,
    AgentCoreArmSessionSplit,
    AgentCoreArmsMissing,
    AgentCoreArmSourceIdRepeated,
    AgentCoreArmsReadChanged,
    AgentCoreMixedArm,
)
from calibrated_agent_evaluation.normalize.otlp_json import attributes, iter_spans

if TYPE_CHECKING:
    from calibrated_agent_evaluation.agentcore.sources import AgentCoreReader, RawTrace
    from calibrated_agent_evaluation.backends.base import SourceTrace, TraceSource

__all__ = [
    "ARM_ARN_ATTRIBUTE",
    "ARM_VARIANT_ATTRIBUTE",
    "ArmRead",
    "KeptTrace",
    "RawTraceLog",
    "arm_counts",
    "arms_read_id",
    "merge_arms_read",
    "read_arms",
    "runtime_session_id",
    "span_stamp",
]

# The span attributes the AgentCore SDK's baggage span processor writes; a test pins them to the
# installed SDK's own constants.
ARM_ARN_ATTRIBUTE = "aws.agentcore.gateway.routing_experiment_arn"
ARM_VARIANT_ATTRIBUTE = "aws.agentcore.gateway.routing_experiment_variant_name"

_ARMS: frozenset[str] = frozenset({"C", "T1"})
_SERVER_KINDS: frozenset[object] = frozenset({"SERVER", "SPAN_KIND_SERVER", 2})
_INVOCATIONS_ROUTE = "/invocations"
_SPANS_LOG_GROUP = "aws/spans"
_GATEWAY_CAUSE = (
    "Only agents built on BedrockAgentCoreApp and invoked through a Gateway that routes the A/B "
    "test carry them"
)

Stamp = Literal["stamped", "half", "none"]


class _Window(Protocol):
    @property
    def start(self) -> str: ...

    @property
    def end(self) -> str: ...


def span_stamp(attrs: Mapping[str, Any]) -> Stamp:
    """``stamped`` with both arm attributes, ``half`` with one of them, ``none`` with neither."""
    present = sum(attrs.get(key) is not None for key in (ARM_ARN_ATTRIBUTE, ARM_VARIANT_ATTRIBUTE))
    return "stamped" if present == 2 else "half" if present == 1 else "none"


def _session_id(attrs: Mapping[str, Any]) -> str | None:
    value = attrs.get("session.id")
    return value if isinstance(value, str) and value else None


def runtime_session_id(otlp_trace: Mapping[str, Any]) -> str | None:
    """The runtime session a trace ran in: the ``session.id`` of its ``/invocations`` server span.

    An agent's own spans may carry another ``session.id``; only the runtime's request span names
    the session the client sent. ``None`` when the trace has no such span or it carries no id.

    Raises:
        AgentCoreArmSessionAmbiguous: Two ``/invocations`` server spans name different sessions.
    """
    found: set[str] = set()
    for span in iter_spans(dict(otlp_trace)):
        attrs = attributes(span)
        if span.get("kind") in _SERVER_KINDS and attrs.get("http.route") == _INVOCATIONS_ROUTE:
            session = _session_id(attrs)
            if session is not None:
                found.add(session)
    if len(found) > 1:
        raise AgentCoreArmSessionAmbiguous(
            f"the trace's /invocations spans name {len(found)} runtime sessions "
            f"({', '.join(sorted(found))}), so its session is ambiguous"
        )
    return next(iter(found), None)


def arms_read_id(
    *,
    runtime: str,
    window_start: str,
    window_end: str,
    omni: bool,
    rekey_sessions: bool,
    applied_bundle_attribute: str | None,
    applied_bundle_require: Mapping[str, str],
) -> str:
    """The id of one arms read: a digest of what was read and how."""
    doc = {
        "runtime": runtime,
        "window_start": window_start,
        "window_end": window_end,
        "omni": omni,
        "rekey_sessions": rekey_sessions,
        "applied_bundle_attribute": applied_bundle_attribute,
        "applied_bundle_require": dict(applied_bundle_require),
    }
    return "arms-" + sha256_text(canonical_json(doc))[:16]


@dataclass(frozen=True)
class KeptTrace:
    """What :class:`RawTraceLog` keeps of one raw trace the ingest's reader yielded."""

    trace_id: str
    session_ids: tuple[str, ...]
    rekey_target: str
    spans_total: int
    spans_stamped: int
    spans_half_stamped: int
    log_groups: tuple[str, ...] = ()
    content_missing: bool = False
    experiment_arns: tuple[str, ...] = ()


@dataclass(frozen=True)
class ArmRead:
    """One arms read: its id, one assignment per source id, its counts and the ARNs it saw."""

    read_id: str
    assignments: tuple[ArmAssignment, ...]
    counts: dict[str, int] = field(default_factory=dict)
    arns_seen: tuple[str, ...] = ()


@dataclass
class _Span:
    attrs: dict[str, Any]
    stamp: Stamp


@dataclass
class _Context:
    ab_test_arn: str
    runtime: str
    window_start: str
    window_end: str
    omni: bool
    rekey_sessions: bool
    read_id: str
    applied_bundle_attribute: str | None
    applied_bundle_require: dict[str, str]

    def assignment(self, trace_id: str, status: ArmStatus, **fields: Any) -> ArmAssignment:
        return ArmAssignment(
            trace_id=trace_id,
            read_id=self.read_id,
            ab_test_arn=self.ab_test_arn,
            runtime=self.runtime,
            window_start=self.window_start,
            window_end=self.window_end,
            omni=self.omni,
            rekey_sessions=self.rekey_sessions,
            status=status,
            applied_bundle_attribute=self.applied_bundle_attribute,
            applied_bundle_require=dict(self.applied_bundle_require),
            **fields,
        )


def _spans_of(otlp: Mapping[str, Any]) -> list[_Span]:
    out = []
    for span in iter_spans(dict(otlp)):
        attrs = attributes(span)
        out.append(_Span(attrs=attrs, stamp=span_stamp(attrs)))
    return out


def _otlp_trace_ids(otlp: Mapping[str, Any]) -> tuple[str, ...]:
    ids = {s.get("traceId") for s in iter_spans(dict(otlp))}
    return tuple(sorted(i for i in ids if isinstance(i, str) and i))


def _half_detail(spans: Sequence[_Span]) -> str:
    pairs = sorted(
        {
            (key, str(span.attrs[key]))
            for span in spans
            if span.stamp == "half"
            for key in (ARM_ARN_ATTRIBUTE, ARM_VARIANT_ATTRIBUTE)
            if span.attrs.get(key) is not None
        }
    )
    return "; ".join(f"{key}={value!r}" for key, value in pairs)


def _one_value(spans: Sequence[_Span], key: str) -> tuple[str | None, list[str]]:
    values = sorted({str(s.attrs[key]) for s in spans if s.attrs.get(key) is not None})
    return (values[0] if len(values) == 1 else None), values


def _applied_bundle_refusal(
    trace_id: str,
    spans: Sequence[_Span],
    variant: str,
    ctx: _Context,
    bundle_versions: Mapping[str, str] | None,
) -> tuple[str | None, dict[str, str]]:
    carried: dict[str, str] = {}
    key = ctx.applied_bundle_attribute
    if key is not None:
        value, values = _one_value(spans, key)
        if len(values) > 1:
            return (
                f"trace {trace_id!r} carries {len(values)} values for {key!r}: {', '.join(values)}",
                carried,
            )
        expected = (bundle_versions or {}).get(variant)
        if value is None:
            return (
                f"trace {trace_id!r} carries no {key!r}, so the bundle version its arm "
                f"{variant} applied ({expected}) cannot be checked"
            ), carried
        carried[key] = value
        if value != expected:
            return (
                f"trace {trace_id!r} is in arm {variant}, whose bundle version is {expected}, but "
                f"its {key!r} is {value}"
            ), carried
    for req_key, required in sorted(ctx.applied_bundle_require.items()):
        value, values = _one_value(spans, req_key)
        if len(values) > 1:
            return (
                f"trace {trace_id!r} carries {len(values)} values for {req_key!r}: "
                f"{', '.join(values)}"
            ), carried
        if value is None:
            return f"trace {trace_id!r} carries no {req_key!r}; {required!r} is required", carried
        carried[req_key] = value
        if value != required:
            return (
                f"trace {trace_id!r} has {req_key}={value!r}; {req_key}={required!r} is required"
            ), carried
    return None, carried


def _assign_one(
    source_id: str,
    delivered: SourceTrace,
    ctx: _Context,
    bundle_versions: Mapping[str, str] | None,
) -> tuple[ArmAssignment, list[_Span]]:
    if delivered.otlp is None:
        reason = delivered.error or (
            "delivered as a normalized trace, which keeps no span attributes, so its arm "
            "stamps cannot be read"
        )
        return ctx.assignment(
            source_id,
            "unread",
            otlp_trace_ids=(),
            runtime_session_id=None,
            span_session_ids=(),
            experiment_arn=None,
            variant=None,
            spans_total=0,
            spans_stamped=0,
            reason=reason,
        ), []
    otlp = delivered.otlp
    spans = _spans_of(otlp)
    base: dict[str, Any] = {
        "otlp_trace_ids": _otlp_trace_ids(otlp),
        "span_session_ids": tuple(
            sorted({sid for s in spans if (sid := _session_id(s.attrs)) is not None})
        ),
        "spans_total": len(spans),
        "spans_stamped": sum(s.stamp == "stamped" for s in spans),
        "spans_half_stamped": sum(s.stamp == "half" for s in spans),
    }
    if len(base["otlp_trace_ids"]) > 1:
        # One delivery holding spans of several traces is several traces under one source id.
        return _shared(source_id, "repeated-source-id", _kept_by_trace(source_id, otlp), ctx), spans

    def refused(
        reason: str, *, arn: str | None = None, session: str | None = None
    ) -> ArmAssignment:
        return ctx.assignment(
            source_id,
            "refused",
            runtime_session_id=session,
            experiment_arn=arn,
            variant=None,
            reason=reason,
            **base,
        )

    if base["spans_half_stamped"]:
        error = AgentCoreArmHalfStamped(
            f"trace {source_id!r} has {base['spans_half_stamped']} span(s) carrying one arm "
            f"attribute without the other ({_half_detail(spans)}); a request whose baggage named "
            "one of the two keys stamps its spans by halves, so the trace's arm is not read"
        )
        return refused(str(error)), spans
    stamped = [s for s in spans if s.stamp == "stamped"]
    if not stamped:
        return ctx.assignment(
            source_id,
            "unassigned",
            runtime_session_id=None,
            experiment_arn=None,
            variant=None,
            reason="no span carries both arm attributes",
            **base,
        ), spans
    arns = sorted({str(s.attrs[ARM_ARN_ATTRIBUTE]) for s in stamped})
    variants = sorted({str(s.attrs[ARM_VARIANT_ATTRIBUTE]) for s in stamped})
    if len(arns) > 1 or len(variants) > 1:
        error = AgentCoreMixedArm(
            f"trace {source_id!r} is stamped with {len(arns)} experiment(s) "
            f"({', '.join(arns)}) and {len(variants)} variant(s) ({', '.join(variants)})"
        )
        return refused(str(error)), spans
    arn, variant = arns[0], variants[0]
    try:
        session = runtime_session_id(otlp)
    except AgentCoreArmSessionAmbiguous as exc:
        return refused(f"trace {source_id!r}: {exc}"), spans
    if arn != ctx.ab_test_arn:
        return ctx.assignment(
            source_id,
            "other-experiment",
            runtime_session_id=session,
            experiment_arn=arn,
            variant=None,
            reason=f"stamped with experiment {arn}, not the audited {ctx.ab_test_arn}",
            **base,
        ), spans
    if variant not in _ARMS:
        return refused(
            f"trace {source_id!r} is stamped with variant {variant!r}; only C and T1 are arms",
            arn=arn,
            session=session,
        ), spans
    mismatch, carried = _applied_bundle_refusal(source_id, spans, variant, ctx, bundle_versions)
    if mismatch is not None:
        error = AgentCoreAppliedBundleMismatch(mismatch)
        return ctx.assignment(
            source_id,
            "refused",
            runtime_session_id=session,
            experiment_arn=arn,
            variant=None,
            applied_bundle_values=carried,
            reason=str(error),
            **base,
        ), spans
    return ctx.assignment(
        source_id,
        "assigned",
        runtime_session_id=session,
        experiment_arn=arn,
        variant=variant,
        applied_bundle_values=carried,
        **base,
    ), spans


def _shared(
    source_id: str,
    kind: Literal["repeated-source-id", "rekey-collision"],
    traces: Sequence[KeptTrace],
    ctx: _Context,
) -> ArmAssignment:
    ids = sorted({t.trace_id for t in traces})
    error = AgentCoreArmSourceIdRepeated(
        f"{len(ids)} traces arrived under source id {source_id!r} ({', '.join(ids)})"
        + (", re-keyed to one runtime session" if kind == "rekey-collision" else "")
        + "; none of them can be assigned"
    )
    return ctx.assignment(
        source_id,
        "refused",
        otlp_trace_ids=tuple(ids),
        runtime_session_id=None,
        span_session_ids=tuple(sorted({s for t in traces for s in t.session_ids})),
        shared_id=kind,
        experiment_arn=None,
        variant=None,
        spans_total=sum(t.spans_total for t in traces),
        spans_stamped=sum(t.spans_stamped for t in traces),
        spans_half_stamped=sum(t.spans_half_stamped for t in traces),
        reason=str(error),
    )


def _kept_by_trace(source_id: str, otlp: Mapping[str, Any] | None) -> list[KeptTrace]:
    """One kept trace per OTLP ``traceId`` in a delivered document."""
    by_trace: dict[str, list[_Span]] = {}
    for span in iter_spans(dict(otlp or {})):
        trace_id = span.get("traceId")
        attrs = attributes(span)
        key = trace_id if isinstance(trace_id, str) and trace_id else source_id
        by_trace.setdefault(key, []).append(_Span(attrs=attrs, stamp=span_stamp(attrs)))
    return [
        KeptTrace(
            trace_id=trace_id,
            session_ids=tuple(sorted({s for sp in spans if (s := _session_id(sp.attrs))})),
            rekey_target=source_id,
            spans_total=len(spans),
            spans_stamped=sum(s.stamp == "stamped" for s in spans),
            spans_half_stamped=sum(s.stamp == "half" for s in spans),
            experiment_arns=_stamped_arns(s.attrs for s in spans),
        )
        for trace_id, spans in sorted(by_trace.items())
    ]


def _stamped_arns(span_attrs: Iterable[Mapping[str, Any]]) -> tuple[str, ...]:
    """The experiment ARNs on the spans that carry both arm attributes, sorted."""
    return tuple(
        sorted(
            {
                str(attrs[ARM_ARN_ATTRIBUTE])
                for attrs in span_attrs
                if span_stamp(attrs) == "stamped"
            }
        )
    )


def _unread_note(unread: Sequence[ArmAssignment]) -> str:
    by_reason: dict[str, list[str]] = {}
    for a in unread:
        by_reason.setdefault(a.reason or "", []).append(a.trace_id)
    parts = [f"{reason} ({', '.join(sorted(ids))})" for reason, ids in sorted(by_reason.items())]
    return f"{len(unread)} trace(s) came back unread: " + "; ".join(parts)


def _split_sessions(assignments: list[ArmAssignment]) -> list[ArmAssignment]:
    variants_by_session: dict[str, set[str]] = {}
    for a in assignments:
        if a.status == "assigned" and a.runtime_session_id is not None and a.variant:
            variants_by_session.setdefault(a.runtime_session_id, set()).add(a.variant)
    split = {s: v for s, v in variants_by_session.items() if len(v) > 1}
    if not split:
        return assignments
    out = []
    for a in assignments:
        if a.status == "assigned" and a.runtime_session_id in split:
            session = a.runtime_session_id
            error = AgentCoreArmSessionSplit(
                f"runtime session {session!r} has traces in {' and '.join(sorted(split[session]))}"
                ", so the session's arm is ambiguous"
            )
            out.append(
                a.model_copy(update={"status": "refused", "variant": None, "reason": str(error)})
            )
        else:
            out.append(a)
    return out


def arm_counts(assignments: Sequence[ArmAssignment]) -> dict[str, int]:
    """The traffic counts of one read.

    Per arm, its traces, runtime sessions and session-less traces; then each other status.
    """
    counts: dict[str, int] = {}
    for arm in sorted(_ARMS):
        assigned = [a for a in assignments if a.status == "assigned" and a.variant == arm]
        counts[f"{arm}.traces"] = len(assigned)
        counts[f"{arm}.runtime_sessions"] = len(
            {a.runtime_session_id for a in assigned if a.runtime_session_id is not None}
        )
        counts[f"{arm}.without_session"] = sum(a.runtime_session_id is None for a in assigned)
    for status in ("unassigned", "refused", "other-experiment", "unread"):
        counts[status] = sum(a.status == status for a in assignments)
    counts["refused.shared_id"] = sum(a.shared_id is not None for a in assignments)
    counts["refused.half_stamped"] = sum(
        a.status == "refused" and a.shared_id is None and a.spans_half_stamped > 0
        for a in assignments
    )
    counts["content_missing"] = sum(a.content_missing for a in assignments)
    return counts


def read_arms(
    source: TraceSource,
    *,
    ab_test_arn: str,
    runtime: str,
    window: _Window,
    omni: bool = False,
    rekey_sessions: bool = False,
    rekey_collisions: Callable[[], Mapping[str, Sequence[KeptTrace]]] | None = None,
    contentless: Callable[[], Mapping[str, Mapping[str, Any]]] | None = None,
    applied_bundle_attribute: str | None = None,
    applied_bundle_require: Iterable[tuple[str, str]] = (),
    bundle_versions: Mapping[str, str] | None = None,
) -> ArmRead:
    """Read which arm of ``ab_test_arn`` each trace the source delivers ran in.

    The source is pulled to the end. A source id delivered more than once, and each re-key
    collision ``rekey_collisions`` reports, is one refused assignment. Every other trace is
    unread (delivered as an error or a normalized trace), refused for a half-stamped span,
    unassigned (no span carries both attributes), refused (two arms, a variant other than C or
    T1, two runtime sessions, a failed applied-bundle condition), set aside as another
    experiment's, or assigned. A runtime session whose traces sit in two arms is then refused.

    Args:
        source: Any trace source; for the AgentCore ingest's, the read path and re-key setting
            used at ingest.
        ab_test_arn: The audited test's ARN.
        runtime: The runtime read, recorded with the read.
        window: The read window, with ``start`` and ``end``.
        omni: Whether the source read through Omni; recorded, not used.
        rekey_sessions: Whether the source re-keyed mixed-session traces; recorded, not used.
        rekey_collisions: Called once after the pull: per runtime session id, every trace that
            re-keyed to it, as :meth:`RawTraceLog.collisions` gives them.
        contentless: Called once after the pull: per trace id, the spans (as an OTLP-JSON
            document with no content) of each trace the source delivered as an error because its
            message content did not come back, as :meth:`RawTraceLog.contentless` gives them. Its
            arm is read from those spans and its assignment marked ``content_missing``: an arm
            stamp is a span attribute, so a trace the ingest refuses for content still shows
            which arm served it.
        applied_bundle_attribute: A span attribute that must equal the arm's bundle version.
        applied_bundle_require: ``(key, value)`` pairs every assigned trace must carry.
        bundle_versions: Each arm's bundle version id, for ``applied_bundle_attribute``.

    Returns:
        The read: its id, one assignment per source id, and its counts.

    Raises:
        AgentCoreArmsMissing: No span is stamped, or no stamped span names the audited test.
        Exception: Whatever the source raises before its first trace, unchanged.
    """
    require = dict(applied_bundle_require)
    read_id = arms_read_id(
        runtime=runtime,
        window_start=window.start,
        window_end=window.end,
        omni=omni,
        rekey_sessions=rekey_sessions,
        applied_bundle_attribute=applied_bundle_attribute,
        applied_bundle_require=require,
    )
    ctx = _Context(
        ab_test_arn=ab_test_arn,
        runtime=runtime,
        window_start=window.start,
        window_end=window.end,
        omni=omni,
        rekey_sessions=rekey_sessions,
        read_id=read_id,
        applied_bundle_attribute=applied_bundle_attribute,
        applied_bundle_require=require,
    )
    groups: dict[str, list[SourceTrace]] = {}
    for delivered in source.pull():
        groups.setdefault(delivered.source_id, []).append(delivered)
    collisions = dict(rekey_collisions()) if rekey_collisions is not None else {}
    stripped = dict(contentless()) if contentless is not None else {}

    assignments: list[ArmAssignment] = []
    stamped_arns: set[str] = set()
    stamped_total = 0
    half_traces: list[str] = []
    consumed: set[str] = set()

    def note_arns(deliveries: Iterable[SourceTrace]) -> None:
        for d in deliveries:
            for span in _spans_of(d.otlp or {}):
                if span.stamp == "stamped":
                    stamped_arns.add(str(span.attrs[ARM_ARN_ATTRIBUTE]))

    for session, traces in sorted(collisions.items()):
        assignments.append(_shared(session, "rekey-collision", traces, ctx))
        stamped_total += sum(t.spans_stamped for t in traces)
        # Every collided turn counts, not only the one the ingest delivered under the session.
        stamped_arns.update(a for t in traces for a in t.experiment_arns)
        if any(t.spans_half_stamped for t in traces):
            half_traces.append(session)
        note_arns(groups.get(session, []))
        consumed.add(session)
        consumed.update(t.trace_id for t in traces)
    for source_id in sorted(groups):
        if source_id in consumed:
            continue
        deliveries = groups[source_id]
        if len(deliveries) > 1:
            kept = [
                k
                for d in deliveries
                if d.otlp is not None
                for k in _kept_by_trace(source_id, d.otlp)
            ]
            assignments.append(_shared(source_id, "repeated-source-id", kept, ctx))
            stamped_total += sum(k.spans_stamped for k in kept)
            if any(k.spans_half_stamped for k in kept):
                half_traces.append(source_id)
            note_arns(deliveries)
            continue
        delivered = deliveries[0]
        doc = stripped.get(source_id) if delivered.error is not None else None
        if doc is not None:
            from calibrated_agent_evaluation.backends.base import SourceTrace as _Delivered

            delivered = _Delivered(source_id=source_id, otlp=dict(doc))
        assignment, spans = _assign_one(source_id, delivered, ctx, bundle_versions)
        if doc is not None:
            assignment = assignment.model_copy(update={"content_missing": True})
        assignments.append(assignment)
        stamped_total += assignment.spans_stamped
        if assignment.spans_half_stamped:
            half_traces.append(source_id)
        stamped_arns.update(str(s.attrs[ARM_ARN_ATTRIBUTE]) for s in spans if s.stamp == "stamped")
    assignments = _split_sessions(assignments)
    unread = [a for a in assignments if a.status == "unread"]
    where = f"runtime {runtime!r}, window {window.start} to {window.end}"
    if stamped_total == 0:
        notes = []
        if unread:
            notes.append(_unread_note(unread))
        if half_traces:
            notes.append(f"trace(s) with half-stamped spans: {', '.join(sorted(half_traces))}")
        tail = "; ".join(notes) if notes else _GATEWAY_CAUSE
        raise AgentCoreArmsMissing(
            f"no span in {where} carries both {ARM_ARN_ATTRIBUTE} and {ARM_VARIANT_ATTRIBUTE}. "
            f"{tail}"
        )
    if ab_test_arn not in stamped_arns:
        seen = ", ".join(sorted(stamped_arns)) or "none"
        note = f"; {_unread_note(unread)}" if unread else ""
        raise AgentCoreArmsMissing(
            f"no trace in {where} is stamped with the audited test {ab_test_arn}; the "
            f"experiments seen are {seen}{note}"
        )
    return ArmRead(
        read_id=read_id,
        assignments=tuple(sorted(assignments, key=lambda a: a.trace_id)),
        counts=arm_counts(assignments),
        arns_seen=tuple(sorted(stamped_arns)),
    )


def merge_arms_read(stored: Sequence[ArmAssignment], read: ArmRead) -> tuple[ArmAssignment, ...]:
    """The assignments of ``read`` to store, given what is already stored for its test.

    A read under a new read id is stored whole; one that repeats a stored read exactly stores
    nothing. A stored read never changes.

    Raises:
        AgentCoreArmsReadChanged: The read repeats a stored read id and adds, drops or reassigns
            a source id; nothing of it may be stored.
    """
    if not read.assignments:
        return ()
    ab_test_arn = read.assignments[0].ab_test_arn
    before = {
        a.trace_id: a for a in stored if a.ab_test_arn == ab_test_arn and a.read_id == read.read_id
    }
    if not before:
        return read.assignments
    after = {a.trace_id: a for a in read.assignments}
    added = sorted(set(after) - set(before))
    missing = sorted(set(before) - set(after))
    changed = sorted(i for i in set(after) & set(before) if after[i] != before[i])
    if added or missing or changed:
        raise AgentCoreArmsReadChanged(
            f"arms read {read.read_id} is already stored and this re-read differs: added "
            f"{added or 'none'}, missing {missing or 'none'}, changed {changed or 'none'}; a "
            "stored read never changes, so nothing was written. Read another window to keep both"
        )
    return ()


class RawTraceLog:
    """Wraps one of the trace ingest's readers: each raw trace passes through, its counts kept.

    The ingest's source drops a trace it refuses for a re-key collision, so the arm read takes what
    it needs from here: each trace's id, session ids, re-key target and span counts, and the log
    groups its spans name.
    """

    def __init__(self, reader: AgentCoreReader) -> None:
        self._reader = reader
        self.kept: list[KeptTrace] = []
        self._stripped: dict[str, dict[str, Any]] = {}

    def read_traces(self) -> Iterator[RawTrace]:
        """Yield the wrapped reader's traces unchanged, keeping what the arm read needs."""
        from calibrated_agent_evaluation.agentcore.adot import adot_to_otlp
        from calibrated_agent_evaluation.agentcore.errors import AgentCoreContentMissing

        for raw in self._reader.read_traces():
            kept = self._keep(raw)
            try:
                adot_to_otlp(raw.span_records, raw.log_records)
            except AgentCoreContentMissing:
                # The ingest's source refuses this trace for its content; its spans still carry
                # the arm stamps, so they are kept, without content, for the arm read.
                kept = replace(kept, content_missing=True)
                self._stripped[raw.trace_id] = {
                    "resourceSpans": [
                        {
                            "resource": {"attributes": {}},
                            "scopeSpans": [{"spans": [dict(s) for s in raw.span_records]}],
                        }
                    ]
                }
            self.kept.append(kept)
            yield raw

    def read_scores(self) -> Iterator[dict[str, Any]]:
        """The wrapped reader's evaluator results, unchanged."""
        return self._reader.read_scores()

    @staticmethod
    def _keep(raw: RawTrace) -> KeptTrace:
        stamps = [span_stamp(attributes(rec)) for rec in raw.span_records]
        arns = _stamped_arns(attributes(rec) for rec in raw.span_records)
        groups: set[str] = set()
        for rec in raw.span_records:
            names = (attributes(rec.get("resource") or {})).get("aws.log.group.names")
            if isinstance(names, str) and names:
                groups.add(names)
            elif isinstance(names, list):
                groups.update(n for n in names if isinstance(n, str) and n)
        return KeptTrace(
            trace_id=raw.trace_id,
            session_ids=tuple(sorted(set(raw.session_ids))),
            rekey_target=raw.runtime_session_id or raw.trace_id,
            spans_total=len(stamps),
            spans_stamped=stamps.count("stamped"),
            spans_half_stamped=stamps.count("half"),
            log_groups=tuple(sorted(groups)),
            experiment_arns=arns,
        )

    def collisions(self, rekeyed: Sequence[tuple[str, str]]) -> dict[str, tuple[KeptTrace, ...]]:
        """Per runtime session id two or more traces re-keyed to, every trace that did.

        A mixed-session trace absent from ``rekeyed`` was refused by the ingest because an earlier
        trace already took its runtime session id; the trace the ingest delivered under that id is
        the one ``rekeyed`` maps to it. No error text is read.
        """
        rekeyed_ids = {old for old, _ in rekeyed}
        delivered_under = {new: old for old, new in rekeyed}
        by_id = {k.trace_id: k for k in self.kept}
        out: dict[str, list[KeptTrace]] = {}
        for kept in self.kept:
            if len(kept.session_ids) <= 1 or kept.trace_id in rekeyed_ids:
                continue
            target = kept.rekey_target
            winner = delivered_under.get(target)
            entry = out.setdefault(target, [])
            if winner is not None and winner in by_id and by_id[winner] not in entry:
                entry.append(by_id[winner])
            entry.append(kept)
        return {t: tuple(sorted(v, key=lambda k: k.trace_id)) for t, v in out.items()}

    def contentless(self, rekeyed: Sequence[tuple[str, str]] = ()) -> dict[str, dict[str, Any]]:
        """Per trace id, the spans of each trace whose message content did not come back.

        A trace whose spans name two session ids, and which the source did not re-key, is left
        out: the ingest refuses it for its sessions whatever its content, so its arm stays unread.
        """
        rekeyed_ids = {old for old, _ in rekeyed}
        sessions = {k.trace_id: k.session_ids for k in self.kept}
        return {
            trace_id: doc
            for trace_id, doc in self._stripped.items()
            if len(sessions.get(trace_id, ())) <= 1 or trace_id in rekeyed_ids
        }

    def groups_read(self) -> tuple[str, ...]:
        """``aws/spans`` and the log groups the spans read so far name, sorted."""
        return tuple(sorted({_SPANS_LOG_GROUP, *(g for k in self.kept for g in k.log_groups)}))
