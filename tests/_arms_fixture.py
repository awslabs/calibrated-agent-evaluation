# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Synthetic stamped traces for the arm reader tests, as the audit's fixture agent writes them.

Every trace's node spans carry the trace's own OTLP ``traceId`` as ``session.id``; a trace with a
runtime session has one ``POST /invocations`` server span carrying that session id, and its run
span ``nl_sql_agent.run`` carries the session id and the ``config.*`` attributes. The ``config.*``
names are the fixture agent's, passed as operator flags in the tests and never built into Calibrated
Agent Evaluation.
"""

from __future__ import annotations

import hashlib
from collections.abc import Iterator, Sequence
from dataclasses import dataclass, field
from typing import Any

from calibrated_agent_evaluation.agentcore.arms import ARM_ARN_ATTRIBUTE, ARM_VARIANT_ATTRIBUTE
from calibrated_agent_evaluation.backends.base import SourceTrace

AB_TEST_ARN = "arn:aws:bedrock-agentcore:us-west-2:111122223333:ab-test/fixture-abtest-stopped"
OTHER_ARN = "arn:aws:bedrock-agentcore:us-west-2:111122223333:ab-test/fixture-abtest-other"
C_VERSION = "0a1b2c3d-0000-4000-8000-000000000001"
T_VERSION = "0a1b2c3d-0000-4000-8000-000000000002"
VERSIONS = {"C": C_VERSION, "T1": T_VERSION}
START = "2026-10-28T19:00:00+00:00"
END = "2026-10-28T20:00:00+00:00"


@dataclass(frozen=True)
class Window:
    start: str = START
    end: str = END


def otlp_id(label: str) -> str:
    """A 32-hex trace id for a table label; sorts before an ``s-`` session id, as live ids do."""
    return hashlib.sha256(label.encode()).hexdigest()[:32]


def _stamp(arn: str | None, variant: str | None) -> dict[str, str]:
    out = {}
    if arn is not None:
        out[ARM_ARN_ATTRIBUTE] = arn
    if variant is not None:
        out[ARM_VARIANT_ATTRIBUTE] = variant
    return out


@dataclass
class Spec:
    """One fixture trace: its stamps, runtime session(s) and ``config.*`` values."""

    label: str
    arm: str | None = None
    arn: str = AB_TEST_ARN
    sessions: Sequence[str] = ()
    version: str | None = None
    source: str | None = None
    stamp_invocations_only: bool = False
    node_arms: Sequence[str | None] = ()
    half_node: dict[str, str] = field(default_factory=dict)
    every_span: dict[str, str] | None = None

    @property
    def trace_id(self) -> str:
        return otlp_id(self.label)


def build(spec: Spec) -> dict[str, Any]:
    tid = spec.trace_id
    stamp = _stamp(spec.arn, spec.arm) if spec.arm is not None else {}
    if spec.every_span is not None:
        stamp = dict(spec.every_span)
    spans: list[dict[str, Any]] = []
    for i, session in enumerate(spec.sessions):
        spans.append(
            {
                "traceId": tid,
                "spanId": f"{i:016x}",
                "name": "POST /invocations",
                "kind": "SERVER",
                "attributes": {"http.route": "/invocations", "session.id": session, **stamp},
            }
        )
    run_attrs: dict[str, Any] = {}
    if spec.sessions:
        run_attrs["session.id"] = spec.sessions[0]
    if spec.version is not None:
        run_attrs["config.bundle_version"] = spec.version
    if spec.source is not None:
        run_attrs["config.source"] = spec.source
    span_stamp = {} if spec.stamp_invocations_only else stamp
    spans.append(
        {
            "traceId": tid,
            "spanId": "a" * 16,
            "name": "nl_sql_agent.run",
            "attributes": {**run_attrs, **span_stamp},
        }
    )
    node_arms = list(spec.node_arms) or [None, None]
    for i, node_arm in enumerate(node_arms):
        attrs: dict[str, Any] = {"session.id": tid}
        if node_arm is not None:
            attrs.update(_stamp(spec.arn, node_arm))
        elif not spec.stamp_invocations_only:
            attrs.update(stamp)
        spans.append(
            {"traceId": tid, "spanId": f"b{i:015x}", "name": f"agent.node{i}", "attributes": attrs}
        )
    if spec.half_node:
        spans.append(
            {
                "traceId": tid,
                "spanId": "c" * 16,
                "name": "agent.half",
                "attributes": {"session.id": tid, **spec.half_node},
            }
        )
    return {"resourceSpans": [{"resource": {"attributes": []}, "scopeSpans": [{"spans": spans}]}]}


MAIN: tuple[Spec, ...] = (
    Spec("c-1", "C", sessions=["s-c1"], version=C_VERSION, source="bundle"),
    Spec("c-2", "C", sessions=["s-c2"], version=C_VERSION, source="bundle"),
    Spec(
        "c-root",
        "C",
        sessions=["s-c1"],
        version=C_VERSION,
        source="bundle",
        stamp_invocations_only=True,
    ),
    Spec(
        "c-fallback", "C", sessions=["s-c2"], version=C_VERSION, source="builtin_after_bundle_error"
    ),
    Spec("c-nosession", "C", version=C_VERSION, source="bundle"),
    Spec("t-1", "T1", sessions=["s-t1"], version=T_VERSION, source="bundle"),
    Spec("t-2", "T1", sessions=["s-t2"], version=T_VERSION, source="bundle"),
    Spec("t-otherversion", "T1", sessions=["s-t1"], version=C_VERSION, source="bundle"),
    Spec("t-noconfig", "T1", sessions=["s-t2"]),
    Spec("unstamped", None, sessions=["s-u"]),
    Spec("mixed", "C", sessions=["s-m"], node_arms=["C", "T1"]),
    Spec("split-c", "C", sessions=["s-split"], version=C_VERSION, source="bundle"),
    Spec("split-t", "T1", sessions=["s-split"], version=T_VERSION, source="bundle"),
    Spec("other-arn", "C", arn=OTHER_ARN, sessions=["s-o"]),
    Spec("t2", "T2", sessions=["s-x"]),
    Spec("two-sessions", "T1", sessions=["s-a", "s-b"], version=T_VERSION, source="bundle"),
)

HALF: tuple[Spec, ...] = (
    Spec("h-arn", "C", sessions=["s-h1"], half_node={ARM_ARN_ATTRIBUTE: AB_TEST_ARN}),
    Spec("h-variant", sessions=["s-h2"], every_span={ARM_VARIANT_ATTRIBUTE: "T1"}),
    Spec("h-full", "T1", sessions=["s-h3"]),
)


def by_label(specs: Sequence[Spec]) -> dict[str, Spec]:
    return {s.label: s for s in specs}


class Double:
    """A ``TraceSource`` that delivers fixed traces, each under its own id unless told otherwise."""

    def __init__(self, items: Sequence[SourceTrace]) -> None:
        self._items = list(items)

    def pull(self) -> Iterator[SourceTrace]:
        yield from self._items


def deliver(specs: Sequence[Spec], *, under: dict[str, str] | None = None) -> list[SourceTrace]:
    under = under or {}
    return [SourceTrace(source_id=under.get(s.label, s.trace_id), otlp=build(s)) for s in specs]


# --- the same traces as span rows for the AgentCore ingest's two readers -----------------------

RUNTIME = "fixture-runtime"
CONTENT_GROUP = "/aws/bedrock-agentcore/runtimes/fixture-runtime-DEFAULT"
_RESOURCE = {"attributes": {"service.name": RUNTIME, "aws.log.group.names": CONTENT_GROUP}}


def span_records(specs: Sequence[Spec], order: str = "node-first") -> list[dict[str, Any]]:
    """The ``aws/spans`` records of ``specs``, in ``node-first`` or ``invocations-first`` row order.

    Which row a reader meets first decides the session id the Logs Insights reader re-keys a
    mixed-session trace to: a node span's (the trace's own id) or the runtime session.
    """
    out: list[dict[str, Any]] = []
    for spec in specs:
        spans = [
            dict(s, resource=_RESOURCE)
            for s in build(spec)["resourceSpans"][0]["scopeSpans"][0]["spans"]
        ]
        first = "agent." if order == "node-first" else "POST /invocations"
        out.extend(sorted(spans, key=lambda s: not str(s["name"]).startswith(first)))
    return out


def content_records(specs: Sequence[Spec]) -> list[dict[str, Any]]:
    """One ``otel-rt-logs`` user-message record per node span of each trace."""
    return [
        {
            "traceId": spec.trace_id,
            "spanId": s["spanId"],
            "eventName": "gen_ai.user.message",
            "body": {"content": [{"text": f"question {spec.label}"}]},
        }
        for spec in specs
        for s in build(spec)["resourceSpans"][0]["scopeSpans"][0]["spans"]
        if str(s["name"]).startswith("agent.")
    ]


def insights_results(records: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """A completed Logs Insights result holding ``records`` as their ``@message``."""
    import json

    rows = [
        [{"field": "@message", "value": json.dumps(r)}, {"field": "@ptr", "value": f"p{i}"}]
        for i, r in enumerate(records)
    ]
    return {
        "results": rows,
        "statistics": {"recordsMatched": float(len(rows))},
        "status": "Complete",
    }


def omni_span_rows(records: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    """Omni span rows; a span without ``session.id`` has no ``session_id`` column, not a null."""
    import json

    rows = []
    for r in records:
        row = {"traceId": r["traceId"], "spanId": r["spanId"], "@message": json.dumps(r)}
        session = (r.get("attributes") or {}).get("session.id")
        if session is not None:
            row["session_id"] = session
        rows.append(row)
    return rows


def omni_content_rows(records: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    import json

    return [
        {"traceId": r["traceId"], "spanId": r["spanId"], "@message": json.dumps(r)} for r in records
    ]


def stub_insights(
    stub: Any, spans: Sequence[dict[str, Any]], content: Sequence[dict[str, Any]]
) -> None:
    """Queue the Logs Insights reader's two queries: ``aws/spans``, then the content group."""
    stub.add_response("start_query", {"queryId": "q-spans"})
    stub.add_response("get_query_results", insights_results(spans))
    stub.add_response("start_query", {"queryId": "q-content"})
    stub.add_response("get_query_results", insights_results(content))


def _omni_round(stub: Any, rows: Sequence[dict[str, Any]]) -> None:
    stub.add_response("start_telemetry_query_session", {"sessionId": "sess"})
    stub.add_response("start_telemetry_query", {"queryId": "q", "sessionId": "sess"})
    stub.add_response(
        "get_telemetry_query_results",
        {
            "status": "Complete",
            "rows": list(rows),
            "statistics": {"partialResults": {"partialResultsDetected": False}},
        },
    )
    stub.add_response("stop_telemetry_query_session", {})


def stub_omni(
    stub: Any, spans: Sequence[dict[str, Any]], content: Sequence[dict[str, Any]]
) -> None:
    """Queue the Omni reader's two query rounds: spans, then content."""
    _omni_round(stub, omni_span_rows(spans))
    _omni_round(stub, omni_content_rows(content))


STAMPED_DIR = "fixtures/raw/agentcore/abtest/stamped"


def window_files() -> dict[str, list[dict[str, Any]]]:
    """The stamped windows as delivered ``SourceTrace`` documents, as their files hold them."""
    repeated = [
        Spec("r-1", "C", sessions=["s-r"]),
        Spec("r-2", "C", sessions=["s-r"]),
        Spec("r-3", "T1", sessions=["s-r3"]),
    ]
    main = by_label(MAIN)
    changed = [
        Spec("c-nosession", "T1", version=C_VERSION, source="bundle")
        if s.label == "c-nosession"
        else s
        for s in MAIN
    ]
    windows = {
        "main": deliver(MAIN),
        "half-stamped": deliver(HALF),
        "repeated-source-id": deliver(repeated, under={"r-1": "s-r", "r-2": "s-r"}),
        "unstamped": deliver([main["unstamped"], Spec("unstamped-2", None, sessions=["s-u2"])]),
        "other-experiment-only": deliver(
            [main["other-arn"], Spec("other-arn-2", "T1", arn=OTHER_ARN, sessions=["s-o2"])]
        ),
        "second-window": deliver([main[k] for k in ("c-1", "c-2", "t-1", "t-2")]),
        "changed-stamp": deliver(changed),
        "grown": deliver([*MAIN, Spec("c-late", "C", sessions=["s-c3"])]),
        "shrunk": deliver([s for s in MAIN if s.label != "t-2"]),
    }
    return {
        name: [t.model_dump(mode="json", exclude_none=True) for t in items]
        for name, items in windows.items()
    }
