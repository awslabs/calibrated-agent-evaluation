#!/usr/bin/env python3
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Validate the shared fixture corpus against the normalized schema.

This is the SCHEMA PRESSURE-TEST for the data plane, the root every other stage builds on. It
encodes the normalized Trace/Span/Label contract (``calibrated_agent_evaluation.schema``) and checks
every ``normalized/*.json`` trace and every ``labels/gold.json`` label against it.

Why it uses only the stdlib: it must not import the package it is meant to pressure-test. If a
faithfully hand-transcribed real trace cannot be represented losslessly under this contract, the
validator fails, and that failure is a gap in the data-plane schema to fix before downstream
stages build on the contract.

Usage:
    python3 fixtures/validate_corpus.py            # validate the whole corpus
    python3 fixtures/validate_corpus.py --selftest # prove the validator catches violations

Exits 0 on PASS, 1 on FAIL.
"""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

# The data-plane design's SCHEMA_VERSION constant; the corpus VERSION must equal it.
SCHEMA_VERSION = "1"

FIXTURES = Path(__file__).resolve().parent

# --- the contract (mirrors the calibrated_agent_evaluation.schema data model)
# ---------------------------------

OWNERSHIP = {"owned", "probed"}
FRAME = {"discovery", "production-representative"}
OUTCOME = {"PASS", "FAIL"}
ROLE = {"specialist", "supervisor", "tool"}
LABEL_KIND = {"human", "llm-judge", "code"}
STATUS = {"completed", "errored"}


def _err(errors: list[str], trace_id: str, msg: str) -> None:
    errors.append(f"[{trace_id}] {msg}")


def _require_nonempty_trace_id(obj: dict, ctx: str, errors: list[str]) -> None:
    # Every artifact carries a non-empty trace_id.
    tid = obj.get("trace_id")
    if not isinstance(tid, str) or not tid.strip():
        _err(errors, ctx, "missing/empty trace_id")


def validate_tool_call(tc: dict, ctx: str, errors: list[str]) -> None:
    # Each tool_call carries name, args and result.
    if not isinstance(tc.get("name"), str) or not tc["name"]:
        _err(errors, ctx, "tool_call missing 'name'")
    if not isinstance(tc.get("args"), dict):
        _err(errors, ctx, f"tool_call {tc.get('name')!r} 'args' must be an object")
    if not isinstance(tc.get("result"), dict):
        _err(errors, ctx, f"tool_call {tc.get('name')!r} 'result' must be an object")


def validate_span(span: dict, trace_id: str, errors: list[str]) -> None:
    ctx = span.get("span_id", f"{trace_id}/<unknown-span>")
    _require_nonempty_trace_id(span, ctx, errors)
    if span.get("trace_id") != trace_id:
        _err(errors, ctx, "span.trace_id does not match its parent trace")
    if not isinstance(span.get("span_id"), str) or not span["span_id"]:
        _err(errors, ctx, "missing span_id")
    if not isinstance(span.get("agent"), str) or not span["agent"]:
        _err(errors, ctx, "missing agent")
    if span.get("role") not in ROLE:
        _err(errors, ctx, f"role {span.get('role')!r} not in {sorted(ROLE)}")
    # parent may be None (root) or a string span_id.
    if span.get("parent") is not None and not isinstance(span["parent"], str):
        _err(errors, ctx, "parent must be a span_id string or null")
    # An explicit seq for ordering (Langfuse has no native seq).
    if not isinstance(span.get("seq"), int) or isinstance(span.get("seq"), bool):
        _err(errors, ctx, "seq must be an integer (explicit ordering)")
    if span.get("status") not in STATUS:
        _err(errors, ctx, f"status {span.get('status')!r} not in {sorted(STATUS)}")
    if not isinstance(span.get("input"), dict):
        _err(errors, ctx, "span.input (handoff received) must be an object")
    if not isinstance(span.get("output"), dict):
        _err(errors, ctx, "span.output (handoff emitted) must be an object")
    tool_calls = span.get("tool_calls", [])
    if not isinstance(tool_calls, list):
        _err(errors, ctx, "tool_calls must be a list")
    else:
        for tc in tool_calls:
            validate_tool_call(tc, ctx, errors)


def validate_trace(trace: dict, errors: list[str]) -> None:
    trace_id = trace.get("trace_id", "<unknown>")
    _require_nonempty_trace_id(trace, trace_id, errors)
    if trace.get("frame") not in FRAME:
        _err(errors, trace_id, f"frame {trace.get('frame')!r} not in {sorted(FRAME)}")
    ownership = trace.get("ownership")
    if ownership not in OWNERSHIP:
        _err(errors, trace_id, f"ownership {ownership!r} not in {sorted(OWNERSHIP)}")
    if not isinstance(trace.get("input"), dict):
        _err(errors, trace_id, "trace.input must be an object")
    if not isinstance(trace.get("output"), dict):
        _err(errors, trace_id, "trace.output must be an object")
    if not isinstance(trace.get("taxonomy_cell"), dict):
        _err(errors, trace_id, "taxonomy_cell must be an object")
    if trace.get("schema_version") != SCHEMA_VERSION:
        _err(errors, trace_id, f"schema_ver must be {SCHEMA_VERSION!r}")

    spans = trace.get("spans", [])
    if not isinstance(spans, list):
        _err(errors, trace_id, "spans must be a list")
        return

    # owned => populated span tree; probed => spans empty (input/output only).
    if ownership == "owned" and not spans:
        _err(errors, trace_id, "owned trace must have a non-empty span tree")
    if ownership == "probed" and spans:
        _err(errors, trace_id, "probed trace must have spans == [] (outcome-only)")

    seqs: list[int] = []
    span_ids: set[str] = set()
    for span in spans:
        validate_span(span, trace_id, errors)
        if isinstance(span.get("span_id"), str):
            span_ids.add(span["span_id"])
        if isinstance(span.get("seq"), int) and not isinstance(span.get("seq"), bool):
            seqs.append(span["seq"])

    # Tree integrity: every non-null parent must reference a span in this trace.
    for span in spans:
        parent = span.get("parent")
        if isinstance(parent, str) and parent not in span_ids:
            _err(errors, span.get("span_id", trace_id), f"parent {parent!r} not a span in trace")
    # Ordering: seq values unique within a trace.
    if len(seqs) != len(set(seqs)):
        _err(errors, trace_id, "duplicate seq values within trace")


def validate_label(label: dict, idx: int, known_trace_ids: set[str], errors: list[str]) -> None:
    ctx = f"label[{idx}] {label.get('trace_id', '<unknown>')}"
    _require_nonempty_trace_id(label, ctx, errors)
    if label.get("trace_id") not in known_trace_ids:
        _err(errors, ctx, f"label references unknown trace_id {label.get('trace_id')!r}")
    # The outcome is binary.
    if label.get("outcome") not in OUTCOME:
        _err(errors, ctx, f"outcome {label.get('outcome')!r} not in {sorted(OUTCOME)}")
    # A non-empty critique (the critique-before-verdict asset the judge rules require).
    critique = label.get("critique")
    if not isinstance(critique, str) or not critique.strip():
        _err(errors, ctx, "critique must be non-empty")
    if label.get("kind") not in LABEL_KIND:
        _err(errors, ctx, f"kind {label.get('kind')!r} not in {sorted(LABEL_KIND)}")
    if not isinstance(label.get("span_defect"), bool):
        _err(errors, ctx, "span_defect must be a boolean")
    # span_id may be null (trace-level gold) or a string (span-level gold).
    if label.get("span_id") is not None and not isinstance(label["span_id"], str):
        _err(errors, ctx, "span_id must be a span_id string or null")
    # Optional expected (correct output for an objective FAIL); object or null.
    expected = label.get("expected")
    if expected is not None and not isinstance(expected, dict):
        _err(errors, ctx, "expected must be an object or null")
    # expected may never substitute for the human critique.
    if expected is not None and (not isinstance(critique, str) or not critique.strip()):
        _err(errors, ctx, "a Label with expected set must still carry a non-empty critique")
    # A gold/calibration label is human-ratified (humans own the gold). kind="human" IS the
    # ratification mechanism (there is no separate adjudication type); a non-human
    # (llm-judge/code) label may enter the gold set only with a recorded human ratification
    # (`ratified_by`). An unratified LLM label in the gold path is a constitution violation, not a
    # shortcut (conventions: "never for gold").
    if label.get("kind") != "human":
        ratified_by = label.get("ratified_by")
        if not isinstance(ratified_by, str) or not ratified_by.strip():
            _err(
                errors,
                ctx,
                f"a {label.get('kind')!r} gold label must carry a non-empty 'ratified_by' "
                "human-ratification record (kind='human' is ratified by definition; an unratified "
                "LLM label may not enter a gold/calibration set)",
            )


def load_json(path: Path, errors: list[str]):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        errors.append(f"[{path.name}] could not parse: {exc}")
        return None


def validate_corpus(root: Path) -> list[str]:
    errors: list[str] = []

    version_file = root / "VERSION"
    if version_file.is_file():
        version = version_file.read_text(encoding="utf-8").strip()
        if version != SCHEMA_VERSION:
            errors.append(
                f"[VERSION] corpus version {version!r} != SCHEMA_VERSION {SCHEMA_VERSION!r}"
            )
    else:
        errors.append("[VERSION] missing fixtures/VERSION")

    norm_dir = root / "normalized"
    trace_files = sorted(norm_dir.glob("*.json")) if norm_dir.is_dir() else []
    if not trace_files:
        errors.append("[normalized] no traces found")

    known_trace_ids: set[str] = set()
    for path in trace_files:
        trace = load_json(path, errors)
        if trace is None:
            continue
        if isinstance(trace.get("trace_id"), str):
            known_trace_ids.add(trace["trace_id"])
        # Filename should match trace_id (corpus convention; aids lookup).
        if trace.get("trace_id") != path.stem:
            errors.append(f"[{path.name}] trace_id {trace.get('trace_id')!r} != filename stem")
        validate_trace(trace, errors)

    gold = root / "labels" / "gold.json"
    if gold.is_file():
        labels = load_json(gold, errors)
        if isinstance(labels, list):
            for idx, label in enumerate(labels):
                validate_label(label, idx, known_trace_ids, errors)
        elif labels is not None:
            errors.append("[gold.json] must be a JSON array of labels")
    else:
        errors.append("[labels/gold.json] missing")

    # The per-stage oracle files under expected/ must be well-formed JSON objects (a
    # malformed oracle is a proof-integrity failure, not a silent skip). Each stage knows its own
    # oracle's shape, so the corpus guard only checks parse-ability + object-ness here.
    expected_dir = root / "expected"
    oracle_files = sorted(expected_dir.rglob("*.json")) if expected_dir.is_dir() else []
    for oracle in oracle_files:
        doc = load_json(oracle, errors)
        if doc is not None and not isinstance(doc, dict):
            errors.append(f"[{oracle.relative_to(root)}] per-stage oracle must be a JSON object")
        elif doc is not None and oracle.name.endswith(".traces.json"):
            # An oracle of stored traces (``{"traces": [...]}``) holds normalized traces, so each
            # one is held to the same schema as normalized/.
            traces = doc.get("traces")
            if not isinstance(traces, list) or not traces:
                errors.append(f"[{oracle.relative_to(root)}] 'traces' must be a non-empty list")
            else:
                for trace in traces:
                    validate_trace(trace, errors)

    return errors


# --- selftest: prove the validator actually catches violations ---------------------------

_GOOD_TRACE = {
    "trace_id": "T-1",
    "frame": "discovery",
    "ownership": "owned",
    "input": {},
    "output": {},
    "taxonomy_cell": {},
    "schema_version": "1",
    "spans": [
        {
            "span_id": "T-1/a",
            "trace_id": "T-1",
            "agent": "a",
            "role": "specialist",
            "parent": None,
            "seq": 0,
            "status": "completed",
            "input": {},
            "output": {},
            "tool_calls": [{"name": "t", "args": {}, "result": {}}],
        }
    ],
}
_GOOD_LABEL = {
    "trace_id": "T-1",
    "span_id": None,
    "kind": "human",
    "outcome": "PASS",
    "critique": "ok",
    "span_defect": False,
}


def _selftest() -> int:
    cases: list[tuple[str, dict, dict | None]] = [
        ("empty trace_id", {**_GOOD_TRACE, "trace_id": ""}, None),
        ("bad ownership", {**_GOOD_TRACE, "ownership": "borrowed"}, None),
        ("owned with no spans", {**_GOOD_TRACE, "spans": []}, None),
        ("schema_version mismatch", {**_GOOD_TRACE, "schema_version": "2"}, None),
        (
            "span missing seq",
            {**_GOOD_TRACE, "spans": [{**_GOOD_TRACE["spans"][0], "seq": None}]},
            None,
        ),
        (
            "tool_call missing result",
            {
                **_GOOD_TRACE,
                "spans": [{**_GOOD_TRACE["spans"][0], "tool_calls": [{"name": "t", "args": {}}]}],
            },
            None,
        ),
        ("bad outcome", _GOOD_TRACE, {**_GOOD_LABEL, "outcome": "MAYBE"}),
        ("empty critique", _GOOD_TRACE, {**_GOOD_LABEL, "critique": "   "}),
        ("label unknown trace", _GOOD_TRACE, {**_GOOD_LABEL, "trace_id": "T-NOPE"}),
        (
            "expected set + empty critique",
            _GOOD_TRACE,
            {**_GOOD_LABEL, "critique": "", "expected": {"action": "DENY"}},
        ),
        ("expected wrong type", _GOOD_TRACE, {**_GOOD_LABEL, "expected": "should be an object"}),
        # An llm-judge gold label with no human ratification is refused.
        ("unratified llm gold label", _GOOD_TRACE, {**_GOOD_LABEL, "kind": "llm-judge"}),
    ]
    passed = failed = 0
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        (root / "VERSION").write_text("1\n", encoding="utf-8")
        (root / "normalized").mkdir()
        (root / "labels").mkdir()
        for name, trace, label in cases:
            for f in (root / "normalized").glob("*.json"):
                f.unlink()
            (root / "normalized" / f"{trace['trace_id'] or 'X'}.json").write_text(
                json.dumps(trace), encoding="utf-8"
            )
            (root / "labels" / "gold.json").write_text(
                json.dumps([label] if label else [_GOOD_LABEL]), encoding="utf-8"
            )
            errs = validate_corpus(root)
            if errs:
                passed += 1
            else:
                failed += 1
                print(f"FAIL selftest: '{name}' should have produced an error", file=sys.stderr)
        # And the all-good corpus must pass clean.
        for f in (root / "normalized").glob("*.json"):
            f.unlink()
        (root / "normalized" / "T-1.json").write_text(json.dumps(_GOOD_TRACE), encoding="utf-8")
        (root / "labels" / "gold.json").write_text(json.dumps([_GOOD_LABEL]), encoding="utf-8")
        clean = validate_corpus(root)
        if clean:
            failed += 1
            print(f"FAIL selftest: clean corpus produced errors: {clean}", file=sys.stderr)
        else:
            passed += 1
    print(f"\nvalidate_corpus selftest: {passed} pass, {failed} fail")
    return 0 if failed == 0 else 1


def main(argv: list[str]) -> int:
    if "--selftest" in argv:
        return _selftest()
    errors = validate_corpus(FIXTURES)
    if errors:
        print(
            f"Corpus validation FAILED ({len(errors)} issue(s)) against schema v{SCHEMA_VERSION}:\n"
        )
        for e in errors:
            print(f"  - {e}")
        print(
            "\nA failure here is a gap in the data-plane schema (a real trace the contract "
            "cannot represent losslessly). Fix the schema before building on it."
        )
        return 1
    print(f"Corpus PASS: all normalized traces + gold labels conform to schema v{SCHEMA_VERSION}.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
