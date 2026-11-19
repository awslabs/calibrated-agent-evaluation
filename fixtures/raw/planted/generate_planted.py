# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Derive the ingest gate's two planted-bad OTLP-JSON inputs from the real captures.

Each file is a real frozen capture with one structural fault introduced, so the no-silent-loss
refusals in the acceptance checklist have a committed input to run against. Nothing else in the
source changes; ``PROVENANCE.md`` names the source and the edit.

  * ``_bad_unlinked_tools.otlp.json``: ``fixtures/raw/otel/ORD-1001.attrs.otlp.json`` with every
    ``execute_tool`` span's ``parentSpanId`` pointed at a span id the trace does not contain. The
    generic adapter links tools to their direct parent, so it recovers 0 of the 3 tool calls and
    the lost-tool-call check must refuse the trace.
  * ``_bad_dup_span.otlp.json``: ``fixtures/raw/strands/CLM-4471.otlp.json`` with the
    ``policy-checker`` agent span given the ``intake`` agent span's id (and its child cycle span
    re-pointed to it). The Strands SDK seeds an agent's span id from its ``name=``, so two agents
    built without distinct names emit exactly this, and the duplicate-span-id check must refuse
    the trace.

These files live outside ``fixtures/raw/otel/`` so a glob or directory walk over the good captures
(as two acceptance steps do) does not pick them up. Regenerate with:

    uv run python fixtures/raw/planted/generate_planted.py
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

_HERE = Path(__file__).resolve().parent
_RAW = _HERE.parent

# A span id the source trace does not contain (base64 of eight zero bytes, the OTLP-JSON form).
_MISSING_PARENT = "AAAAAAAAAAA="


def _spans(otlp: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        span
        for resource_spans in otlp["resourceSpans"]
        for scope_spans in resource_spans["scopeSpans"]
        for span in scope_spans["spans"]
    ]


def _attr(span: dict[str, Any], key: str) -> str | None:
    for attr in span.get("attributes", []):
        if attr["key"] == key:
            return attr["value"].get("stringValue")
    return None


def _agent_span(otlp: dict[str, Any], agent: str) -> dict[str, Any]:
    for span in _spans(otlp):
        if (
            _attr(span, "gen_ai.operation.name") == "invoke_agent"
            and _attr(span, "gen_ai.agent.name") == agent
        ):
            return span
    raise SystemExit(f"no invoke_agent span for {agent!r}; the source capture changed")


def unlinked_tools() -> dict[str, Any]:
    otlp = json.loads((_RAW / "otel" / "ORD-1001.attrs.otlp.json").read_text(encoding="utf-8"))
    tools = [s for s in _spans(otlp) if _attr(s, "gen_ai.operation.name") == "execute_tool"]
    if not tools:
        raise SystemExit("ORD-1001.attrs has no execute_tool span; the source capture changed")
    for span in tools:
        span["parentSpanId"] = _MISSING_PARENT
    return otlp


def dup_span() -> dict[str, Any]:
    otlp = json.loads((_RAW / "strands" / "CLM-4471.otlp.json").read_text(encoding="utf-8"))
    intake = _agent_span(otlp, "intake")
    checker = _agent_span(otlp, "policy-checker")
    old_id = checker["spanId"]
    checker["spanId"] = intake["spanId"]
    for span in _spans(otlp):
        if span.get("parentSpanId") == old_id:
            span["parentSpanId"] = intake["spanId"]
    return otlp


def main() -> None:
    for name, build in (
        ("_bad_unlinked_tools.otlp.json", unlinked_tools),
        ("_bad_dup_span.otlp.json", dup_span),
    ):
        (_HERE / name).write_text(json.dumps(build(), indent=2) + "\n", encoding="utf-8")
        print(f"wrote {name}")


if __name__ == "__main__":
    main()
