# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Write ``arms.json.gz`` (gzipped JSON): the arm reads the live A/B window's span rows must give.

Imports no Calibrated Agent Evaluation code. It reads the redacted live capture
(``fixtures/raw/agentcore/abtest/live/insights_spans.json.gz``) and applies its own copy of the one
stamp rule (a span is stamped when it carries both arm attributes, half-stamped with exactly
one), takes each trace's runtime session from its ``POST /invocations`` server span, and checks
the applied-bundle conditions from the span attributes the fixture agent writes. The live read
delivered every trace without its message content (the agent puts content on span attributes,
which the ingest does not recover), so every assignment is marked ``content_missing`` and keyed by
its own trace id. Read ids are the digest of the read's settings in the canonical JSON form.

Run from the repository root: ``python fixtures/expected/audit/make_arms.py``.
"""

from __future__ import annotations

import gzip
import hashlib
import json
from pathlib import Path

_LIVE = Path("fixtures/raw/agentcore/abtest/live")
_OUT = Path("fixtures/expected/audit/arms.json")
ARN_KEY = "aws.agentcore.gateway.routing_experiment_arn"
VARIANT_KEY = "aws.agentcore.gateway.routing_experiment_variant_name"
RUNTIME = "fixture-runtime.DEFAULT"
WINDOW = ("2026-11-03T06:15:00+00:00", "2026-11-03T06:25:00+00:00")
SETTINGS = (
    ("no-flag", None, {}),
    ("attribute", "config.bundle_version", {}),
    ("attribute-and-require", "config.bundle_version", {"config.source": "bundle"}),
)


def canonical(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def read_id(*, omni: bool, attribute: str | None, require: dict[str, str]) -> str:
    doc = {
        "runtime": RUNTIME,
        "window_start": WINDOW[0],
        "window_end": WINDOW[1],
        "omni": omni,
        "rekey_sessions": True,
        "applied_bundle_attribute": attribute,
        "applied_bundle_require": require,
    }
    return "arms-" + hashlib.sha256(canonical(doc).encode("utf-8")).hexdigest()[:16]


def stamp(attrs: dict) -> str:
    present = sum(attrs.get(k) is not None for k in (ARN_KEY, VARIANT_KEY))
    return {2: "stamped", 1: "half", 0: "none"}[present]


def one(spans: list[dict], key: str) -> str | None:
    values = {str(s["attributes"][key]) for s in spans if s["attributes"].get(key) is not None}
    if len(values) > 1:
        raise SystemExit(f"{key} has two values: {values}")
    return next(iter(values), None)


def assign(trace_id: str, spans: list[dict], versions: dict[str, str], setting: tuple) -> dict:
    _, attribute, require = setting
    stamps = [stamp(s["attributes"]) for s in spans]
    stamped = [s for s in spans if stamp(s["attributes"]) == "stamped"]
    variants = {s["attributes"][VARIANT_KEY] for s in stamped}
    arns = {s["attributes"][ARN_KEY] for s in stamped}
    if len(variants) != 1 or len(arns) != 1 or "half" in stamps:
        raise SystemExit(f"trace {trace_id} is not stamped with one arm")
    variant = next(iter(variants))
    sessions = {
        s["attributes"]["session.id"]
        for s in spans
        if s.get("kind") == "SERVER" and s["attributes"].get("http.route") == "/invocations"
    }
    if len(sessions) != 1:
        raise SystemExit(f"trace {trace_id} has {len(sessions)} runtime sessions")
    status, reason = "assigned", None
    if attribute is not None and one(spans, attribute) != versions[variant]:
        status, reason = "refused", "applied bundle version"
    for key, value in require.items():
        if status == "assigned" and one(spans, key) != value:
            status, reason = "refused", f"{key} is not {value}"
    return {
        "trace_id": trace_id,
        "status": status,
        "variant": variant if status == "assigned" else None,
        "runtime_session_id": next(iter(sessions)),
        "spans_total": len(spans),
        "spans_stamped": stamps.count("stamped"),
        "spans_half_stamped": stamps.count("half"),
        "content_missing": True,
        "reason": reason,
    }


def counts(assignments: list[dict]) -> dict[str, int]:
    out: dict[str, int] = {}
    for arm in ("C", "T1"):
        mine = [a for a in assignments if a["status"] == "assigned" and a["variant"] == arm]
        out[f"{arm}.traces"] = len(mine)
        out[f"{arm}.runtime_sessions"] = len({a["runtime_session_id"] for a in mine})
        out[f"{arm}.without_session"] = 0
    for status in ("unassigned", "refused", "other-experiment", "unread"):
        out[status] = sum(a["status"] == status for a in assignments)
    out["refused.shared_id"] = 0
    out["refused.half_stamped"] = 0
    out["content_missing"] = sum(a["content_missing"] for a in assignments)
    return out


def main() -> None:
    rows = json.loads(gzip.decompress((_LIVE / "insights_spans.json.gz").read_bytes()))
    test = json.loads((_LIVE / "abtest_live.json").read_text(encoding="utf-8"))
    versions = {
        v["name"]: v["variantConfiguration"]["configurationBundle"]["bundleVersion"]
        for v in test["variants"]
    }
    by_trace: dict[str, list[dict]] = {}
    for row in rows:
        by_trace.setdefault(row["traceId"], []).append(row)
    reads = []
    for omni in (False, True):
        for setting in SETTINGS:
            assignments = [assign(t, by_trace[t], versions, setting) for t in sorted(by_trace)]
            reads.append(
                {
                    "setting": setting[0],
                    "omni": omni,
                    "read_id": read_id(omni=omni, attribute=setting[1], require=setting[2]),
                    "assignments": assignments,
                    "counts": counts(assignments),
                }
            )
    doc = {
        "ab_test_arn": test["abTestArn"],
        "runtime": RUNTIME,
        "window_start": WINDOW[0],
        "window_end": WINDOW[1],
        "rekey_sessions": True,
        "reads": reads,
    }
    text = (json.dumps(doc, indent=1, sort_keys=True) + "\n").encode("utf-8")
    _OUT.with_name(_OUT.name + ".gz").write_bytes(gzip.compress(text, compresslevel=9, mtime=0))


if __name__ == "__main__":
    main()
