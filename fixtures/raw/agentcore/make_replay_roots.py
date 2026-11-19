# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Write the A/B audit's replay roots as JSON: two disjoint replay sets and a judged copy.

Imports no Calibrated Agent Evaluation code. Each file describes one local store (its traces, their
trace-level labels, and the run provenance a replay root's executor records), which the tests and
the demo build into a store before they run. The verdicts are planted so each set's paired cells are
fixed in advance:

* H1 (``abtest/replay/``): 40 confirmation traces in a ``control`` and a ``treatment`` root,
  human verdicts, target cells ``(a, b, c, d) = (2, 14, 1, 23)``, guard cells ``(3, 1, 2, 34)``;
  a ``development`` root of 10 other questions in the ingest's ``{"answer": ...}`` shape, one
  with a human FAIL for a third mode; ``development-overlap`` adds confirmation question 7 under
  another id; ``development-messages`` adds a trace whose input has no single prompt text.
* H2 (``recommendations/replay/``): 40 other questions in an ``input`` and a ``result`` root,
  target cells ``(4, 10, 2, 24)``, guard cells ``(3, 1, 2, 34)``, and a ``development`` root
  whose 10 trace ids are the recommendation fixtures' session-span trace ids.
* H1-judged (``abtest/replay-judged/``): H1's roots with judge verdicts from synthetic raters,
  a human adjudication agreeing with the judge on each of the target's 15 discordant pairs, and
  the calibration and bias records a record root holds for those judges.

Cells: ``a`` both fail, ``b`` before fail and after pass, ``c`` before pass and after fail, ``d``
both pass. Run from the repository root: ``python fixtures/raw/agentcore/make_replay_roots.py``.
"""

from __future__ import annotations

import gzip
import hashlib
import json
from pathlib import Path

_HERE = Path("fixtures/raw/agentcore")
_ACCOUNT_ARN = "arn:aws:bedrock-agentcore:us-west-2:111122223333"
BUNDLE_ARN = f"{_ACCOUNT_ARN}:configuration-bundle/fixture-bundle-AbCdEf1234"
CONTROL = "0a1b2c3d-0000-4000-8000-000000000001"
TREATMENT = "0a1b2c3d-0000-4000-8000-000000000002"
TARGET = "missing_date_range"
GUARD = "wrong_join"
THIRD = "unbounded_scan"
TARGET_JUDGE = "target-judge@3f1c0a9e5b7d2468"
GUARD_JUDGE = "guard-judge@8e2d4b6a1c3f5079"
N = 40


def h2_development_ids() -> list[str]:
    """The recommendation fixtures' 10 session-span trace ids (32 lowercase hex digits)."""
    return [hashlib.sha256(f"h2-development-{i}".encode()).hexdigest()[:32] for i in range(10)]


def _cells(a: int, b: int, c: int, d: int) -> list[tuple[bool, bool]]:
    """``(before_fail, after_fail)`` per trace, in trace order, for the planted cells."""
    if a + b + c + d != N:
        raise SystemExit(f"cells {(a, b, c, d)} do not sum to {N}")
    return [(True, True)] * a + [(True, False)] * b + [(False, True)] * c + [(False, False)] * d


def _label(trace_id: str, fail: bool, mode: str, *, kind: str, rater: str | None) -> dict:
    return {
        "trace_id": trace_id,
        "kind": kind,
        "outcome": "FAIL" if fail else "PASS",
        "failure_mode": mode,
        "rater_id": rater,
        "critique": f"{'fails' if fail else 'passes'} {mode}",
    }


def _pair(
    ids: list[str],
    questions: list[str],
    target: list[tuple[bool, bool]],
    guard: list[tuple[bool, bool]],
    *,
    judged: bool,
) -> tuple[dict, dict]:
    """The before and after roots of one replay set."""
    kind = "llm-judge" if judged else "human"
    raters = (TARGET_JUDGE, GUARD_JUDGE) if judged else (None, None)
    roots = []
    for side, version in ((0, CONTROL), (1, TREATMENT)):
        traces, labels = [], []
        for i, trace_id in enumerate(ids):
            traces.append(
                {
                    "trace_id": trace_id,
                    "input": {"prompt": questions[i]},
                    "output": {"answer": f"SELECT 1 -- {'after' if side else 'before'} {i}"},
                }
            )
            labels.append(_label(trace_id, target[i][side], TARGET, kind=kind, rater=raters[0]))
            # The guard's cells run in reverse trace order, so they fall on other traces.
            g = guard[len(ids) - 1 - i]
            labels.append(_label(trace_id, g[side], GUARD, kind=kind, rater=raters[1]))
            if judged and side == 1 and target[i][0] != target[i][1]:
                labels.append(_label(trace_id, target[i][1], TARGET, kind="human", rater="analyst"))
        provenance = [
            {"trace_id": t, "bundle_arn": BUNDLE_ARN, "bundle_version": version} for t in ids
        ]
        roots.append({"traces": traces, "labels": labels, "run_provenance": provenance})
    return roots[0], roots[1]


def _development(ids: list[str], questions: list[str], *, third_on: int | None) -> dict:
    traces = [
        {"trace_id": t, "input": {"answer": q}, "output": {"answer": "draft"}}
        for t, q in zip(ids, questions, strict=True)
    ]
    labels = (
        [] if third_on is None else [_label(ids[third_on], True, THIRD, kind="human", rater=None)]
    )
    return {"traces": traces, "labels": labels, "run_provenance": []}


def _records(judges: list[str]) -> dict:
    calibration, bias = [], []
    for i, rater in enumerate(judges):
        judge, _, fingerprint = rater.rpartition("@")
        stamp = "2026-10-30T09:00:00+00:00"
        calibration.append(
            {
                "calib_id": f"calib-{judge}-{i}",
                "judge_id": judge,
                "judge_fingerprint": fingerprint,
                "result": "CERTIFIED",
                "run_at": stamp,
            }
        )
        bias.append(
            {
                "battery_id": f"bias-{judge}-{i}",
                "judge_id": judge,
                "judge_fingerprint": fingerprint,
                "verdict": "CLEAR",
                "run_at": stamp,
            }
        )
    return {"calibration": calibration, "bias": bias}


# A description larger than this is stored gzipped (``<name>.json.gz``), keeping review diffs small.
GZIP_OVER_BYTES = 8000


def _write(path: Path, doc: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    text = (json.dumps(doc, indent=1, ensure_ascii=False) + "\n").encode("utf-8")
    gzipped = path.with_name(path.name + ".gz")
    if len(text) > GZIP_OVER_BYTES:
        gzipped.write_bytes(gzip.compress(text, compresslevel=9, mtime=0))
        path.unlink(missing_ok=True)
    else:
        path.write_bytes(text)
        gzipped.unlink(missing_ok=True)


def main() -> None:
    h1_ids = [f"h1-c{i:02d}" for i in range(N)]
    h1_q = [f"How many orders did store {i:02d} ship last week?" for i in range(N)]
    dev_ids = [f"h1-d{i:02d}" for i in range(10)]
    dev_q = [f"What was the average basket size in store {i:02d} yesterday?" for i in range(10)]
    h1_target, guard = _cells(2, 14, 1, 23), _cells(3, 1, 2, 34)

    replay = _HERE / "abtest" / "replay"
    control, treatment = _pair(h1_ids, h1_q, h1_target, guard, judged=False)
    _write(replay / "control.json", control)
    _write(replay / "treatment.json", treatment)
    development = _development(dev_ids, dev_q, third_on=3)
    _write(replay / "development.json", development)
    overlap = json.loads(json.dumps(development))
    overlap["traces"].append(
        {"trace_id": "h1-d-overlap", "input": {"answer": h1_q[7]}, "output": {"answer": "draft"}}
    )
    _write(replay / "development-overlap.json", overlap)
    messages = json.loads(json.dumps(development))
    messages["traces"].append(
        {
            "trace_id": "h1-d-messages",
            "input": {"messages": [{"role": "user", "content": dev_q[0]}], "user": "analyst"},
            "output": {"answer": "draft"},
        }
    )
    _write(replay / "development-messages.json", messages)

    judged = _HERE / "abtest" / "replay-judged"
    control_j, treatment_j = _pair(h1_ids, h1_q, h1_target, guard, judged=True)
    _write(judged / "control.json", control_j)
    _write(judged / "treatment.json", treatment_j)
    _write(judged / "records-certified.json", _records([TARGET_JUDGE, GUARD_JUDGE]))
    _write(judged / "records-uncertified-guard.json", _records([TARGET_JUDGE]))

    h2_ids = [f"h2-c{i:02d}" for i in range(N)]
    h2_q = [f"Which products did region {i:02d} return most last month?" for i in range(N)]
    rec = _HERE / "recommendations" / "replay"
    before, after = _pair(h2_ids, h2_q, _cells(4, 10, 2, 24), guard, judged=False)
    _write(rec / "input.json", before)
    _write(rec / "result.json", after)
    h2_dev_q = [f"List the top suppliers for warehouse {i:02d}." for i in range(10)]
    _write(rec / "development.json", _development(h2_development_ids(), h2_dev_q, third_on=None))


if __name__ == "__main__":
    main()
