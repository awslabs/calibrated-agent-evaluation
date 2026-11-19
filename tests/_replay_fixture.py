# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Build the A/B audit's replay roots from their JSON descriptions (not a test module).

Each file under ``fixtures/raw/agentcore/`` that ``make_replay_roots.py`` writes describes one
local store: its traces, their trace-level labels, and, for a replay root, the run provenance its
executor recorded. :func:`build_root` writes one into a fresh directory; :func:`build_records`
writes the calibration and bias records a judged re-test's record root holds.
"""

from __future__ import annotations

import shutil
from collections.abc import Callable
from pathlib import Path
from typing import Any

from _fixture_json import read_json
from calibrated_agent_evaluation.agentcore.audit_types import RunProvenance
from calibrated_agent_evaluation.backends.local import LocalBackend
from calibrated_agent_evaluation.cli.records import append_records
from calibrated_agent_evaluation.schema.types import Label, Trace

REPO = Path(__file__).resolve().parents[1]
RAW = REPO / "fixtures" / "raw" / "agentcore"
TARGET = "missing_date_range"
GUARD = "wrong_join"
THIRD = "unbounded_scan"
TARGET_JUDGE = "target-judge@3f1c0a9e5b7d2468"
GUARD_JUDGE = "guard-judge@8e2d4b6a1c3f5079"
_ACCOUNT_ARN = "arn:aws:bedrock-agentcore:us-west-2:111122223333"
BUNDLE_ARN = f"{_ACCOUNT_ARN}:configuration-bundle/fixture-bundle-AbCdEf1234"
CONTROL = "0a1b2c3d-0000-4000-8000-000000000001"
TREATMENT = "0a1b2c3d-0000-4000-8000-000000000002"

H1 = "abtest/replay"
H1_JUDGED = "abtest/replay-judged"
H2 = "recommendations/replay"


def load(name: str) -> dict[str, Any]:
    """The JSON description ``<name>.json`` under ``fixtures/raw/agentcore``."""
    return read_json(RAW / f"{name}.json")


def build_root(
    name: str, dest: Path, *, edit: Callable[[dict[str, Any]], None] | None = None
) -> Path:
    """Write the store ``name`` describes into ``dest`` (after ``edit`` changes the doc)."""
    doc = load(name)
    if edit is not None:
        edit(doc)
    with LocalBackend(root=dest) as backend:
        for t in doc["traces"]:
            backend.put_trace(
                Trace(
                    trace_id=t["trace_id"],
                    frame="production-representative",
                    ownership="probed",
                    input=t["input"],
                    output=t["output"],
                    taxonomy_cell={"feature": "analytics"},
                )
            )
        for lb in doc["labels"]:
            backend.put_score(
                Label(
                    trace_id=lb["trace_id"],
                    span_id=None,
                    kind=lb["kind"],
                    outcome=lb["outcome"],
                    critique=lb["critique"],
                    failure_mode=lb["failure_mode"],
                    rater_id=lb["rater_id"],
                )
            )
    if doc["run_provenance"]:
        append_records(
            dest,
            "agentcore-run-provenance",
            [RunProvenance.model_validate(p) for p in doc["run_provenance"]],
        )
    return dest


def build_pair(name: str, tmp: Path, before: str, after: str) -> tuple[Path, Path]:
    """The before and after roots of one replay set, built under ``tmp``."""
    return (
        build_root(f"{name}/{before}", tmp / f"{name.replace('/', '-')}-{before}"),
        build_root(f"{name}/{after}", tmp / f"{name.replace('/', '-')}-{after}"),
    )


def copy_root(src: Path, dest: Path) -> Path:
    """A copy of a built store, so one test's edits never reach another's."""
    shutil.copytree(src, dest)
    return dest


def build_records(
    name: str, root: Path, *, edit: Callable[[dict[str, Any]], None] | None = None
) -> None:
    """Append the calibration and bias records ``name`` describes to the record root ``root``."""
    from test_bias_store import _record as bias_record
    from test_calibrate_store import _record as calibration_record

    doc = load(name)
    if edit is not None:
        edit(doc)
    calibration = []
    for c in doc["calibration"]:
        base = calibration_record()
        calibration.append(
            base.model_copy(
                update={
                    "calib_id": c["calib_id"],
                    "hunted_set_id": f"hunted-{c['calib_id']}",
                    # A locked test is single-use, so each record has its own gold traces.
                    "trace_ids": tuple(f"{c['calib_id']}-{t}" for t in base.trace_ids),
                    "judge_id": c["judge_id"],
                    "judge_fingerprint": c["judge_fingerprint"],
                    "result": c["result"],
                    "provenance": base.provenance.model_copy(update={"run_at": c["run_at"]}),
                }
            )
        )
    bias = []
    for b in doc["bias"]:
        base = bias_record()
        bias.append(
            base.model_copy(
                update={
                    "battery_id": b["battery_id"],
                    "judge_id": b["judge_id"],
                    "judge_fingerprint": b["judge_fingerprint"],
                    "verdict": b["verdict"],
                    "provenance": base.provenance.model_copy(update={"run_at": b["run_at"]}),
                }
            )
        )
    if calibration:
        append_records(root, "calibration", calibration)
    if bias:
        append_records(root, "bias", bias)


def run_record(
    trace_id: str,
    bundle: str | None,
    version: str | None,
    *,
    runtime_arn: str = "arn:aws:bedrock-agentcore:us-west-2:111122223333:runtime/fixture-runtime",
    qualifier: str = "DEFAULT",
) -> Any:
    """One run-manifest record of the AgentCore Runtime executor, naming the bundle it ran."""
    from calibrated_agent_evaluation.agentcore.run_manifest import RunRecord

    return RunRecord(
        trace_id=trace_id,
        w3c_trace_id="0" * 32,
        session_id=f"s-{trace_id}",
        runtime_arn=runtime_arn,
        qualifier=qualifier,
        runtime_version="1",
        runtime_version_source="endpoint",
        config_bundle_arn=bundle,
        config_bundle_version=version,
        payload_format="prompt",
        started_at="2026-10-30T10:00:00+00:00",
        ended_at="2026-10-30T10:00:01+00:00",
        duration_ms=1000,
        attempts=1,
        status_code=200,
        content_type="text/event-stream",
        response_traceparent=None,
        replayed=False,
        session_stop="stopped",
    )


def as_run_manifest(root: Path) -> Path:
    """Move a built root's run provenance into the executor's run manifest, where generate puts it.

    The root then holds the executor's manifest under ``records/agentcore`` and no provenance
    records, as a replay run through
    ``calibrated-agent-evaluation generate --run --executor agentcore`` leaves it.
    """
    from calibrated_agent_evaluation.agentcore.replay import ProvenanceStore
    from calibrated_agent_evaluation.agentcore.run_manifest import append_run_record

    log_dir = root / "records" / "agentcore"
    for trace_id, p in sorted(ProvenanceStore(root).provenance().items()):
        append_run_record(
            log_dir,
            run_record(
                trace_id,
                p.bundle_arn,
                p.bundle_version,
                **{
                    k: v
                    for k, v in (("runtime_arn", p.runtime_arn), ("qualifier", p.qualifier))
                    if v
                },
            ),
        )
    shutil.rmtree(root / "records" / "agentcore-run-provenance")
    return log_dir
