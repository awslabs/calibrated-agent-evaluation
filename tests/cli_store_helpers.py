# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Builders for the CLI stage tests: a local store of verdicts, judge files, and gate records.

Imported by the ``test_cli_*`` modules (pytest puts ``tests/`` on the path); not a test module.
"""

from __future__ import annotations

import json
from pathlib import Path

from calibrated_agent_evaluation._provenance import capture_provenance
from calibrated_agent_evaluation.backends.local import LocalBackend
from calibrated_agent_evaluation.bias import BiasBatteryRecord
from calibrated_agent_evaluation.calibrate.record import Bound, CalibrationRecord
from calibrated_agent_evaluation.cli.records import append_records
from calibrated_agent_evaluation.schema.types import Label, Trace

JUDGE = {
    "judge_id": "refund-judge",
    "model": "bedrock/test-model",
    "prompt_version": "v1",
    "level": "workflow",
    "prompt_template": "Grade the refund. Critique first, then a pass/fail verdict.",
    "failure_mode": "over_refund",
}


def judge_fingerprint(**overrides: object) -> str:
    """The content fingerprint of the judge a judge file with ``overrides`` would load."""
    import tempfile

    from calibrated_agent_evaluation.cli.judgefile import load_judge_file

    with tempfile.TemporaryDirectory() as tmp:
        return load_judge_file(
            write_judge_file(Path(tmp) / "judge.json", **overrides)
        ).judge.fingerprint


def write_judge_file(path: Path, **overrides: object) -> Path:
    doc = {**JUDGE, **overrides}
    path.write_text(json.dumps(doc), encoding="utf-8")
    return path


def trace(trace_id: str, frame: str = "production-representative") -> Trace:
    return Trace(
        trace_id=trace_id,
        frame=frame,  # type: ignore[arg-type]
        ownership="probed",
        input={"message": f"refund {trace_id}"},
        output={"text": "done"},
        taxonomy_cell={"feature": "refund"},
    )


def seed_verdicts(
    root: Path,
    n_pass: int,
    n_fail: int,
    *,
    kind: str = "llm-judge",
    mode: str = "over_refund",
    frame: str = "production-representative",
    prefix: str = "T",
    rater_id: str | None = "the default judge",
) -> list[str]:
    """Store ``n_pass + n_fail`` traces with one verdict each; return their ids.

    An LLM-judge verdict carries the ``rater_id`` the default judge file's judge stamps, as
    ``calibrated-agent-evaluation judge`` would write it, unless ``rater_id`` names another (or
    ``None``).
    """
    if rater_id == "the default judge":
        rater_id = default_judge_rater_id() if kind == "llm-judge" else None
    ids = []
    with LocalBackend(root=root) as backend:
        for i in range(n_pass + n_fail):
            tid = f"{prefix}{i:03d}"
            backend.put_trace(trace(tid, frame))
            backend.put_score(
                Label(
                    trace_id=tid,
                    span_id=None,
                    kind=kind,  # type: ignore[arg-type]
                    outcome="PASS" if i < n_pass else "FAIL",
                    critique="a critique",
                    failure_mode=mode,
                    rater_id=rater_id,
                )
            )
            ids.append(tid)
    return ids


def default_judge_rater_id(**overrides: object) -> str:
    """The ``rater_id`` on verdicts from the judge a judge file with ``overrides`` loads."""
    import tempfile

    from calibrated_agent_evaluation.cli.judgefile import load_judge_file
    from calibrated_agent_evaluation.judges import verdict_rater_id

    with tempfile.TemporaryDirectory() as tmp:
        jf = load_judge_file(write_judge_file(Path(tmp) / "judge.json", **overrides))
        return verdict_rater_id(jf.judge)


def certified_calibration(
    judge_id: str = JUDGE["judge_id"],
    *,
    model: str = JUDGE["model"],
    prompt_version: str = JUDGE["prompt_version"],
    calib_id: str = "calib-refund",
    result: str = "CERTIFIED",
    fingerprint: str | None = None,
) -> CalibrationRecord:
    return CalibrationRecord(
        provenance=capture_provenance(),
        judge_fingerprint=fingerprint
        or judge_fingerprint(model=model, prompt_version=prompt_version),
        calib_id=calib_id,
        judge_id=judge_id,
        model=model,
        prompt_version=prompt_version,
        hunted_set_id="hunted-refund",
        n_pos=80,
        n_neg=80,
        tpr=Bound(point=0.95, lb95=0.90, n=80),
        tnr=Bound(point=0.95, lb95=0.90, n=80),
        youden_j=0.90,
        result=result,  # type: ignore[arg-type]
    )


def clear_battery(
    judge_id: str = JUDGE["judge_id"],
    *,
    model: str = JUDGE["model"],
    prompt_version: str = JUDGE["prompt_version"],
    verdict: str = "CLEAR",
    fingerprint: str | None = None,
) -> BiasBatteryRecord:
    return BiasBatteryRecord(
        provenance=capture_provenance(),
        judge_fingerprint=fingerprint
        or judge_fingerprint(model=model, prompt_version=prompt_version),
        battery_id=f"bias-{judge_id}-{verdict.lower()}",
        judge_id=judge_id,
        model=model,
        prompt_version=prompt_version,
        is_pairwise=False,
        results=(),
        verdict=verdict,  # type: ignore[arg-type]
        failed_biases=(),
    )


def store_calibration(root: Path, record: CalibrationRecord) -> None:
    append_records(root, "calibration", [record])


def store_battery(root: Path, record: BiasBatteryRecord) -> None:
    append_records(root, "bias", [record])


def owned_trace(
    trace_id: str,
    *,
    chars: int = 20,
    steps: int = 1,
    tool_calls: int = 0,
    cell: dict[str, str] | None = None,
    frame: str = "production-representative",
) -> Trace:
    """An owned trace whose input is ``chars`` long, with ``steps`` spans and ``tool_calls`` calls.

    The tool calls all sit on the root span; ``cell`` defaults to one refund cell.
    """
    from calibrated_agent_evaluation.schema.types import Span, ToolCall

    calls = [ToolCall(name="lookup", args={"i": i}, result={"ok": True}) for i in range(tool_calls)]
    spans = [
        Span(
            span_id=f"{trace_id}/s{i}",
            trace_id=trace_id,
            agent="orderbot" if i == 0 else f"helper{i}",
            role="supervisor" if i == 0 else "specialist",
            parent=None if i == 0 else f"{trace_id}/s0",
            seq=i,
            status="completed",
            input={"step": i},
            tool_calls=calls if i == 0 else [],
            output={"done": True},
        )
        for i in range(steps)
    ]
    return Trace(
        trace_id=trace_id,
        frame=frame,  # type: ignore[arg-type]
        ownership="owned",
        input={"message": "x" * chars},
        output={"text": "done"},
        taxonomy_cell=cell if cell is not None else {"feature": "refund"},
        spans=spans,
    )


def seed_traces(
    root: Path,
    traces: list[Trace],
    *,
    n_fail: int = 0,
    kind: str = "llm-judge",
    mode: str = "over_refund",
    rater_id: str | None = "the default judge",
) -> None:
    """Store ``traces`` with one verdict each: the last ``n_fail`` FAIL, the rest PASS."""
    if rater_id == "the default judge":
        rater_id = default_judge_rater_id() if kind == "llm-judge" else None
    with LocalBackend(root=root) as backend:
        for i, tr in enumerate(traces):
            backend.put_trace(tr)
            backend.put_score(
                Label(
                    trace_id=tr.trace_id,
                    span_id=None,
                    kind=kind,  # type: ignore[arg-type]
                    outcome="FAIL" if i >= len(traces) - n_fail else "PASS",
                    critique="a critique",
                    failure_mode=mode,
                    rater_id=rater_id,
                )
            )
