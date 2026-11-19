# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""An Inspect eval run with ``epochs > 1`` keeps one trace per sample per epoch."""

from __future__ import annotations

from pathlib import Path

import pytest

from calibrated_agent_evaluation.backends.local import LocalBackend
from calibrated_agent_evaluation.ingest import run_ingest
from calibrated_agent_evaluation.normalize import (
    InspectEvalLog,
    NormalizationError,
    normalize,
    normalize_all,
)

pytest.importorskip("inspect_ai")


@pytest.fixture(scope="module")
def two_epoch_log(tmp_path_factory: pytest.TempPathFactory) -> Path:
    from inspect_ai import Task
    from inspect_ai import eval as inspect_eval
    from inspect_ai.dataset import Sample
    from inspect_ai.solver import generate

    out = tmp_path_factory.mktemp("epochs")
    task = Task(
        dataset=[
            Sample(id="q1", input="hello", metadata={"trace_id": "Q1"}),
            Sample(id="q2", input="bye"),
        ],
        solver=generate(),
    )
    (log,) = inspect_eval(task, model="mockllm/model", epochs=2, log_dir=str(out), display="none")
    return Path(log.location)


def test_every_epoch_of_every_sample_gets_its_own_trace_id(two_epoch_log: Path) -> None:
    ids = [t.trace_id for t in normalize_all(InspectEvalLog(path=two_epoch_log))]
    assert len(ids) == 4
    assert len(set(ids)) == 4
    assert {"Q1#e1", "Q1#e2"} <= set(ids)


def test_a_multi_epoch_log_ingests_in_full(two_epoch_log: Path) -> None:
    report = run_ingest([two_epoch_log], backend=LocalBackend(":memory:"))
    assert [r.status for r in report.results] == ["ingested"] * 4
    assert report.exit_code == 0


def test_selecting_a_sample_by_id_names_its_epochs(two_epoch_log: Path) -> None:
    with pytest.raises(NormalizationError, match=r"ran in 2 epochs.*q1#e1, q1#e2"):
        normalize(InspectEvalLog(path=two_epoch_log, sample_id="q1"))
    trace = normalize(InspectEvalLog(path=two_epoch_log, sample_id="q1#e2"))
    assert trace.trace_id == "Q1#e2"
