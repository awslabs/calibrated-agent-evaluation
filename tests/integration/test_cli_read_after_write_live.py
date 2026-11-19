# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Opt-in live test: `calibrated-agent-evaluation run` right after
`calibrated-agent-evaluation judge` reads every new verdict (SDL/22).

Against the self-hosted Langfuse dev stack. Fresh trace ids each run (a re-put duplicates spans).
The traces are written through the CLI's own backend, then `calibrated-agent-evaluation judge` and
`calibrated-agent-evaluation run` are invoked
back to back with no sleep in between: the judge must read all six traces it was just given, and
the run must gate all six verdicts the judge just wrote. The judge's mode is unique to the run, so
the trend counts exactly this run's verdicts in a project other tests also write to. A manifest
entry for a write that never happened makes the next command refuse, naming it.

Skipped unless the stack is up: ``just langfuse-up`` then ``just test-integration``.
"""

from __future__ import annotations

import json
import os
import uuid
from argparse import Namespace
from pathlib import Path

import pytest

from _helpers import langfuse_reachable
from calibrated_agent_evaluation._model import replace
from calibrated_agent_evaluation._provenance import capture_provenance
from calibrated_agent_evaluation.bias import BiasBatteryRecord
from calibrated_agent_evaluation.calibrate.record import Bound, CalibrationRecord
from calibrated_agent_evaluation.cli.backends import backend_from_args
from calibrated_agent_evaluation.cli.consistency import READ_WAIT_ENV, ConsistentBackend
from calibrated_agent_evaluation.cli.dispatch import EXIT_DOMAIN_ERROR, EXIT_OK, main
from calibrated_agent_evaluation.cli.judgefile import load_judge_file
from calibrated_agent_evaluation.cli.records import append_records, load_records_of
from calibrated_agent_evaluation.execute.cassette import Cassette, Mode, fingerprint
from calibrated_agent_evaluation.judges import build_verdict_schema, render_trace
from calibrated_agent_evaluation.schema.serde import trace_from_dict
from calibrated_agent_evaluation.schema.types import Label, Trace

pytestmark = pytest.mark.integration

_FIXTURES = Path(__file__).resolve().parent.parent.parent / "fixtures"
_N = 6


@pytest.fixture
def live(monkeypatch):  # type: ignore[no-untyped-def]
    langfuse = pytest.importorskip("langfuse")
    for key, value in (
        ("LANGFUSE_HOST", "http://localhost:3000"),
        ("LANGFUSE_PUBLIC_KEY", "pk-lf-calibrated-agent-evaluation-dev"),
        ("LANGFUSE_SECRET_KEY", "sk-lf-calibrated-agent-evaluation-dev"),
    ):
        monkeypatch.setenv(key, os.environ.get(key, value))
    client = langfuse.Langfuse(
        public_key=os.environ["LANGFUSE_PUBLIC_KEY"],
        secret_key=os.environ["LANGFUSE_SECRET_KEY"],
        host=os.environ["LANGFUSE_HOST"],
    )
    reachable = langfuse_reachable(client)
    if not reachable:
        pytest.skip("no reachable Langfuse (run `just langfuse-up`)")
    monkeypatch.delenv(READ_WAIT_ENV, raising=False)


def _corpus(trace_id: str) -> Trace:
    return trace_from_dict(json.loads((_FIXTURES / "normalized" / f"{trace_id}.json").read_text()))


def _fresh(source: Trace, new_id: str) -> Trace:
    old = source.trace_id
    spans = [
        replace(
            s,
            trace_id=new_id,
            span_id=s.span_id.replace(old, new_id, 1),
            parent=s.parent.replace(old, new_id, 1) if s.parent else None,
        )
        for s in source.spans
    ]
    return replace(source, trace_id=new_id, spans=spans)


def _judge_file(tmp_path: Path, mode: str) -> Path:
    doc = json.loads((_FIXTURES / "judge-files" / "orderbot-refund-resolved.json").read_text())
    doc["failure_mode"] = mode
    path = tmp_path / "judge.json"
    path.write_text(json.dumps(doc))
    return path


def _trust(root: Path, judge_file: Path) -> None:
    judge = load_judge_file(judge_file).judge
    pinned = {
        "judge_id": judge.judge_id,
        "model": judge.model,
        "prompt_version": judge.prompt_version,
        "judge_fingerprint": judge.fingerprint,
    }
    append_records(
        root,
        "calibration",
        [
            CalibrationRecord(
                provenance=capture_provenance(),
                calib_id="calib-live",
                hunted_set_id="hunted-live",
                n_pos=80,
                n_neg=80,
                tpr=Bound(point=0.95, lb95=0.90, n=80),
                tnr=Bound(point=0.95, lb95=0.90, n=80),
                youden_j=0.90,
                result="CERTIFIED",
                **pinned,
            )
        ],
    )
    append_records(
        root,
        "bias",
        [
            BiasBatteryRecord(
                provenance=capture_provenance(),
                battery_id="bias-live",
                is_pairwise=False,
                results=(),
                verdict="CLEAR",
                failed_biases=(),
                **pinned,
            )
        ],
    )


def _cassette(path: Path, judge_file: Path, traces: list[Trace]) -> Path:
    judge = load_judge_file(judge_file).judge
    cassette = Cassette(path, mode=Mode.AUTO)
    for t in traces:
        request = {
            "system": judge.prompt_template,
            "user": render_trace(t),
            "schema": build_verdict_schema(judge),
            "via": "response_schema",
        }
        # The rendering carries no trace id, so copies of one corpus trace share a reply: the
        # out-of-window ORD-1001 copies fail, the in-window ORD-1002 copies pass.
        verdict = "fail" if "A-7781" in request["user"] else "pass"
        cassette.record(
            fingerprint(request),
            request,
            {
                "fields": {"critique": f"live {t.trace_id}", "verdict": verdict},
                "via": "response_schema",
            },
        )
    return path


def test_run_right_after_judge_gates_every_new_verdict(live, tmp_path) -> None:  # type: ignore[no-untyped-def]
    suffix = uuid.uuid4().hex[:10]
    mode = f"live_raw_{suffix}"
    root = tmp_path / "root"
    traces = [
        _fresh(_corpus("ORD-1001" if i % 2 else "ORD-1002"), f"RAW-{suffix}-{i}") for i in range(_N)
    ]
    judge_file = _judge_file(tmp_path, mode)
    _trust(root, judge_file)

    args = Namespace(backend="langfuse", root=str(root))
    with backend_from_args(args) as writer:  # type: ignore[attr-defined]
        assert isinstance(writer, ConsistentBackend)
        for t in traces:
            writer.put_trace(t)
    # The cassette is keyed on the trace as Langfuse returns it. Since the read-after-write fix the
    # local copy renders the same, but reading it back here also makes this a second command,
    # which waits on the manifest for the traces just written.
    with backend_from_args(args) as reader:  # type: ignore[attr-defined]
        stored = [reader.get_trace(t.trace_id) for t in traces]
    cassette = _cassette(tmp_path / "judge.cassette.json", judge_file, stored)

    base = ["--backend", "langfuse", "--root", str(root)]
    # No sleep anywhere: each command waits for exactly what the one before it wrote.
    for t in traces:
        judge = [*base, "judge", "--judge", str(judge_file), "--cassette", str(cassette)]
        assert main([*judge, "--trace", t.trace_id]) == EXIT_OK
    run = [*base, "run", "--judge", str(judge_file), "--n-resamples", "199"]
    assert main(run) == EXIT_OK

    (record,) = load_records_of(root, "run")
    (point,) = record.trend
    assert point.mode.failure_type == mode
    assert point.corrected_rate.n == _N
    assert point.raw_rate.point == pytest.approx(3 / _N)


def test_a_write_that_never_landed_is_named_after_the_wait(live, tmp_path, monkeypatch) -> None:  # type: ignore[no-untyped-def]
    root = tmp_path / "root"
    args = Namespace(backend="langfuse", root=str(root))
    backend = backend_from_args(args)
    assert isinstance(backend, ConsistentBackend)
    ghost = Label(
        trace_id=f"RAW-ghost-{uuid.uuid4().hex[:10]}",
        span_id=None,
        kind="llm-judge",
        outcome="FAIL",
        critique="never sent",
    )
    # Recorded as written, but never sent: what a write lost after its flush looks like.
    backend.manifest.record_label(ghost, retracted=False)

    monkeypatch.setenv(READ_WAIT_ENV, "3")
    code = main(["--backend", "langfuse", "--root", str(root), "run", "--trace", ghost.trace_id])
    assert code == EXIT_DOMAIN_ERROR
    assert backend.manifest.path.exists()


def test_ingest_from_langfuse_right_after_a_write_pulls_it(live, tmp_path) -> None:  # type: ignore[no-untyped-def]
    """The pull waits for the listing, not only for each write."""
    suffix = uuid.uuid4().hex[:10]
    root = tmp_path / "root"
    traces = [_fresh(_corpus("ORD-1001"), f"PULL-{suffix}-{i}") for i in range(3)]
    args = Namespace(backend="langfuse", root=str(root))
    with backend_from_args(args) as writer:  # type: ignore[attr-defined]
        for t in traces:
            writer.put_trace(t)
        _, project = writer.project()

    # No sleep: the pull waits for the three traces just written to be listed. The dev project
    # also holds traces other clients wrote, some of which ingest refuses (exit 1).
    assert main(["--root", str(root), "ingest", "--from-langfuse", project]) in (EXIT_OK, 1)
    from calibrated_agent_evaluation.backends.local import LocalBackend

    with LocalBackend(root=root) as local:
        held = {t.trace_id for t in local.query()}
    assert {t.trace_id for t in traces} <= held
