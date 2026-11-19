# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Importable builders + corpus loaders for the data-plane test suite.

Kept in a plain module (not ``conftest.py``) so test files can ``import`` the builder
functions directly; ``conftest.py`` wraps the corpus loaders as pytest fixtures. Reading the
committed corpus is filesystem-only (no network).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Literal, TypedDict

from calibrated_agent_evaluation._provenance import ProvenanceStamp
from calibrated_agent_evaluation.schema.serde import label_from_dict, to_dict, trace_from_dict
from calibrated_agent_evaluation.schema.types import Label, Span, ToolCall, Trace

FIXTURES = Path(__file__).resolve().parent.parent / "fixtures"

# A fixed provenance stamp for tests that build records directly (capture_provenance reads the
# clock and the environment, which a known-answer test must not depend on).
TEST_STAMP = ProvenanceStamp(
    run_at="2026-10-13T00:00:00+00:00",
    calibrated_agent_evaluation_version="0.0.0+test",
    git_sha=None,
    python_version="3.11.0",
    dep_versions={"numpy": "2.0.0", "scipy": "1.15.0", "statsmodels": "0.14.0"},
    seed=0,
    n_resamples=9999,
    thresholds={"gate": 0.80},
)


class _IntervalKind(TypedDict):
    method: str
    level: float
    sided: Literal["two-sided", "lower", "upper"]
    n: int


# The kind fields every Interval carries (method/level/sided/n), for tests that build one by hand
# and do not care which estimator produced it.
TEST_INTERVAL_KIND: _IntervalKind = {
    "method": "test",
    "level": 0.95,
    "sided": "two-sided",
    "n": 100,
}


def make_tool_call(**over: object) -> ToolCall:
    base: dict[str, object] = {"name": "lookup", "args": {"id": "A"}, "result": {"ok": True}}
    base.update(over)
    return ToolCall(**base)  # type: ignore[arg-type]


def make_span(**over: object) -> Span:
    base: dict[str, object] = {
        "span_id": "T-1/root",
        "trace_id": "T-1",
        "agent": "orderbot",
        "role": "specialist",
        "parent": None,
        "seq": 0,
        "status": "completed",
        "input": {"message": "hi"},
        "tool_calls": [make_tool_call()],
        "output": {"action": "DONE"},
    }
    base.update(over)
    return Span(**base)  # type: ignore[arg-type]


def make_trace(**over: object) -> Trace:
    base: dict[str, object] = {
        "trace_id": "T-1",
        "frame": "discovery",
        "ownership": "owned",
        "input": {"message": "hi"},
        "output": {"action": "DONE"},
        "taxonomy_cell": {"feature": "refund"},
        "spans": [make_span()],
    }
    base.update(over)
    return Trace(**base)  # type: ignore[arg-type]


def make_label(**over: object) -> Label:
    base: dict[str, object] = {
        "trace_id": "T-1",
        "span_id": None,
        "kind": "human",
        "outcome": "FAIL",
        "critique": "the agent ignored the policy tool result",
    }
    base.update(over)
    return Label(**base)  # type: ignore[arg-type]


def load_corpus_trace(trace_id: str) -> Trace:
    data = json.loads((FIXTURES / "normalized" / f"{trace_id}.json").read_text())
    return trace_from_dict(data)


def load_quickstart_trace(index: int = 0) -> Trace:
    """One trace of the quickstart calibration set: a real OrderBot run with real span timing."""
    import gzip

    path = (
        FIXTURES.parent
        / "src"
        / "calibrated_agent_evaluation"
        / "_quickstart"
        / "data"
        / "certify"
        / "traces.jsonl.gz"
    )
    with gzip.open(path, "rt", encoding="utf-8") as lines:
        for position, line in enumerate(lines):
            if position == index:
                return trace_from_dict(json.loads(line))
    raise IndexError(index)


def load_corpus_labels() -> list[Label]:
    data = json.loads((FIXTURES / "labels" / "gold.json").read_text())
    return [label_from_dict(item) for item in data]


def reseal_dump(dump_dir: Path, *names: str) -> None:
    """Record the edited files' new sha256 in the dump's manifest, as a dump that wrote them would.

    For a test that edits a dump file to reach a check that runs after the manifest's. A JSON list
    gets its new length as its record count; any other file keeps the count it had.
    """
    import hashlib

    manifest_path = dump_dir / "dump-manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    for name in names:
        data = (dump_dir / name).read_bytes()
        manifest["files"][name]["sha256"] = hashlib.sha256(data).hexdigest()
        try:
            doc = json.loads(data)
        except ValueError:
            continue
        if isinstance(doc, list):
            manifest["files"][name]["records"] = len(doc)
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")


def structural_view(trace: Trace | dict[str, Any]) -> dict[str, Any]:
    """The corpus's structural standard (``fixtures/README.md``) for comparing to an oracle.

    Ownership, the agent tree as (agent, parent agent) edges, and each agent's tool calls by name,
    arguments and result. Prose, timestamps and span-id spelling are left out, which is what lets a
    capture of a different run of the same trajectory equal the hand-authored oracle.
    """
    data = to_dict(trace) if isinstance(trace, Trace) else trace
    by_id = {s["span_id"]: s["agent"] for s in data["spans"]}
    return {
        "ownership": data["ownership"],
        "tree": sorted((s["agent"], by_id.get(s["parent"])) for s in data["spans"]),
        "tools": {
            s["agent"]: [(tc["name"], tc["args"], tc["result"]) for tc in s["tool_calls"]]
            for s in data["spans"]
        },
    }


def rate_ids(verdicts: Any, prefix: str = "rate") -> list[str]:
    """One distinct trace id per rate-sample verdict, none shared with any calibration gold."""
    return [f"{prefix}-{i}" for i in range(len(verdicts))]


def rate_ids_by_mode(samples: Any) -> dict[Any, list[str]]:
    """``rate_ids`` for each mode's rate sample, as ``trend_report`` takes them."""
    return {m: rate_ids(v, f"rate-{i}") for i, (m, v) in enumerate(samples.items())}


def langfuse_reachable(client: Any, *, wait_seconds: float = 30.0) -> bool:
    """Whether a live Langfuse answers ``auth_check`` within ``wait_seconds``.

    One probe is not enough: right after a test sends the server a deliberately oversized body,
    the next request can time out though the stack is up, and a single failed probe turned that
    into a skip that looked like "no Langfuse". A refused connection means nothing is listening
    and returns at once; timeouts and other errors are retried until the deadline.
    """
    import time

    import httpx

    deadline = time.monotonic() + wait_seconds
    while True:
        try:
            if client.auth_check():
                return True
        except httpx.ConnectError:
            return False
        except Exception:
            pass
        if time.monotonic() >= deadline:
            return False
        time.sleep(1)
