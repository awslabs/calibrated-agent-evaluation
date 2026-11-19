# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""A whole-store read of a Langfuse project holding traces Calibrated Agent Evaluation did not
write.

The store is a real ``LangfuseBackend`` over the stub client that mirrors the verified v3 surface,
holding the OrderBot corpus traces Calibrated Agent Evaluation wrote and the two foreign shapes the
live server returned in an earlier review (a plain-SDK trace, a plain OTel-exporter trace). Every
stage reads the
Calibrated Agent Evaluation traces and names the foreign ones it left out; none aborts, none drops
one unseen.
"""

from __future__ import annotations

from typing import Any

import pytest

from _helpers import load_corpus_labels, load_corpus_trace
from calibrated_agent_evaluation.backends import TraceScan, scan_traces
from calibrated_agent_evaluation.backends.local import LocalBackend
from calibrated_agent_evaluation.cli import main
from calibrated_agent_evaluation.cli.dispatch import EXIT_DOMAIN_ERROR, EXIT_OK
from calibrated_agent_evaluation.cli.scope import skip_lines
from calibrated_agent_evaluation.review.session import WorkScope, build_work_list
from calibrated_agent_evaluation.schema.errors import BackendError
from test_ingest_langfuse_foreign import _otlp_written_trace, _sdk_written_trace, _source

SDK_ID = "a" * 32
OTEL_ID = "b" * 32
FOREIGN = "written by another client (no calibrated_agent_evaluation_trace envelope)"


@pytest.fixture
def shared():
    """ORD-1001/ORD-1002 and their gold written by Calibrated Agent Evaluation, beside two foreign
    traces."""
    backend, _ = _source(_sdk_written_trace(SDK_ID), _otlp_written_trace(OTEL_ID))
    for trace_id in ("ORD-1001", "ORD-1002"):
        backend.put_trace(load_corpus_trace(trace_id))
    for label in load_corpus_labels():
        if label.trace_id in ("ORD-1001", "ORD-1002"):
            backend.put_score(label)
    return backend


@pytest.fixture
def stages_read(shared, monkeypatch):
    """Point every CLI stage at the shared project."""
    monkeypatch.setattr(
        "calibrated_agent_evaluation.cli.backends.backend_from_args", lambda _args: shared
    )
    return shared


def test_query_still_refuses_and_scan_reads_around_the_foreign_traces(shared):
    with pytest.raises(BackendError, match="calibrated_agent_evaluation_trace"):
        shared.query()
    scan = shared.scan()
    assert sorted(t.trace_id for t in scan.traces) == ["ORD-1001", "ORD-1002"]
    assert sorted(s.trace_id for s in scan.skipped) == [SDK_ID, OTEL_ID]
    assert all(FOREIGN in s.reason for s in scan.skipped)
    assert all(
        "calibrated-agent-evaluation ingest --from-langfuse" in s.reason for s in scan.skipped
    )


def test_a_foreign_trace_already_ingested_says_so(shared):
    from calibrated_agent_evaluation._model import replace

    # What `calibrated-agent-evaluation ingest --from-langfuse` writes back: a Calibrated Agent
    # Evaluation trace under the foreign id.
    source = load_corpus_trace("ORD-1001")
    spans = [replace(s, trace_id=SDK_ID, span_id=f"{SDK_ID}/orderbot") for s in source.spans]
    shared.put_trace(replace(source, trace_id=SDK_ID, spans=spans))
    scan = shared.scan()
    (ingested,) = [s for s in scan.skipped if s.trace_id == SDK_ID]
    assert (
        "already ingested as the Calibrated Agent Evaluation trace of the same id"
        in ingested.reason
    )
    assert SDK_ID in {t.trace_id for t in scan.traces}


def test_a_corrupt_calibrated_agent_evaluation_trace_still_raises(shared):

    client: Any = shared._client
    lf_id = client.create_trace_id(seed="ORD-1001")
    client._traces[lf_id].metadata = {"calibrated_agent_evaluation_trace": {"trace_id": "ORD-1001"}}
    with pytest.raises(BackendError, match="scan"):
        shared.scan()


def test_a_backend_without_foreign_traces_skips_nothing(tmp_path):
    with LocalBackend(root=tmp_path / "s") as local:
        local.put_trace(load_corpus_trace("ORD-1001"))
        assert scan_traces(local) == TraceScan(traces=[load_corpus_trace("ORD-1001")])


def test_skip_lines_name_each_reason_once():
    from calibrated_agent_evaluation.backends import SkippedTrace

    skipped = tuple(SkippedTrace(trace_id=f"t{i}", reason="r") for i in range(7))
    scan = TraceScan(traces=[], skipped=skipped)
    assert skip_lines(scan) == ["  skipped 7 trace(s), r: 't0', 't1', 't2', 't3', 't4' and 2 more"]


@pytest.mark.parametrize(
    "argv",
    [
        ["define"],
        ["cluster"],
        ["run", "--verdicts", "human", "--n-resamples", "199"],
    ],
)
def test_whole_store_stages_report_the_traces_they_left_out(stages_read, tmp_path, argv, capsys):
    code = main(["--root", str(tmp_path / "records"), *argv])
    out, err = capsys.readouterr()
    assert code == EXIT_OK, err
    assert f"skipped 2 trace(s), {FOREIGN}" in out
    assert repr(SDK_ID) in out
    assert repr(OTEL_ID) in out


def test_calibrate_reports_them_beside_its_gold(stages_read, tmp_path, capsys):
    from pathlib import Path

    repo = Path(__file__).resolve().parent.parent
    argv = [
        "--root",
        str(tmp_path / "records"),
        "calibrate",
        "--judge",
        str(repo / "fixtures/judge-files/orderbot-refund-resolved.json"),
        "--cassette",
        str(repo / "fixtures/raw/cassettes/judge_orderbot_resolved.cassette.json"),
    ]
    assert main(argv) == EXIT_DOMAIN_ERROR  # one gold label per class: REFUSED
    err = capsys.readouterr().err
    assert "on 2 human gold label(s) from 2 stored trace(s): REFUSED" in err
    assert f"skipped 2 trace(s), {FOREIGN}" in err


def test_the_review_work_list_carries_them(shared):
    work = build_work_list(shared, WorkScope(), rater_id="r1")
    assert sorted(t.trace_id for t in work.traces) == ["ORD-1001", "ORD-1002"]
    assert sorted(s.trace_id for s in work.not_calibrated_agent_evaluation) == [SDK_ID, OTEL_ID]
