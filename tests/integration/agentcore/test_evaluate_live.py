# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""The built-in Correctness evaluator scores real traces as an advisory judge, on synthetic times.

Four billed ``Evaluate`` calls, one per trace: two shared-corpus traces with no span timing and two
quickstart-set traces with real span timing and tool calls. Every trace is sent with some synthetic
time (tool calls never have stored times), so every verdict is marked. Each outcome is recorded
next to its human gold, not asserted: an advisory verdict that disagrees with gold is what
calibration measures.
"""

from __future__ import annotations

import gzip
import json
from pathlib import Path
from typing import TYPE_CHECKING, Any

import pytest

from _helpers import FIXTURES, load_corpus_labels, load_corpus_trace, load_quickstart_trace
from calibrated_agent_evaluation.schema.serde import to_dict

if TYPE_CHECKING:
    from calibrated_agent_evaluation.agentcore.pricing import RunEstimate
    from calibrated_agent_evaluation.schema.types import Trace

pytestmark = [pytest.mark.integration, pytest.mark.agentcore]

_JUDGE_FILE = FIXTURES / "judge-files" / "agentcore-correctness.json"
_QUICKSTART = (
    FIXTURES.parent / "src" / "calibrated_agent_evaluation" / "_quickstart" / "data" / "certify"
)
# Positions in the quickstart set's traces file: Q-10000 (gold FAIL, a planted broken-policy-tool
# bug) and Q-20000 (gold PASS).
_QUICKSTART_POSITIONS = (0, 162)


def _quickstart_gold(trace_ids: set[str]) -> dict[str, str]:
    with gzip.open(_QUICKSTART / "gold.jsonl.gz", "rt", encoding="utf-8") as lines:
        rows = [json.loads(line) for line in lines]
    return {row["trace_id"]: row["outcome"] for row in rows if row["trace_id"] in trace_ids}


def _traces() -> list[Trace]:
    corpus = [load_corpus_trace("ORD-1002"), load_corpus_trace("CLM-4471")]
    return corpus + [load_quickstart_trace(i) for i in _QUICKSTART_POSITIONS]


def test_correctness_scores_four_traces_on_synthetic_times_recorded_next_to_gold(
    live_budget: Any,
    live_region: str | None,
    tmp_path: Path,
    record_property: Any,
) -> None:
    from calibrated_agent_evaluation.agentcore import AwsClients
    from calibrated_agent_evaluation.agentcore.evaluate import run_agentcore_judge
    from calibrated_agent_evaluation.agentcore.judge import (
        build_agentcore_judge,
        load_agentcore_judge_file,
    )
    from calibrated_agent_evaluation.backends.local import LocalBackend

    traces = _traces()
    ids = [trace.trace_id for trace in traces]
    assert ids[2:] == ["Q-10000", "Q-20000"], "the quickstart positions still name these traces"
    sent_as = [to_dict(trace) for trace in traces]
    gold = {
        label.trace_id: label.outcome
        for label in load_corpus_labels()
        if label.trace_id in ids and label.span_id is None
    } | _quickstart_gold(set(ids))
    assert set(gold) == set(ids), "every scored trace has a trace-level gold label"
    estimates: list[RunEstimate] = []

    def approve(estimate: RunEstimate) -> bool:
        estimates.append(estimate)
        return True

    billed_before = live_budget.used
    with (
        LocalBackend(root=str(tmp_path / "store")) as store,
        AwsClients.from_profile(region=live_region, handlers=[live_budget]) as clients,
    ):
        judge = build_agentcore_judge(load_agentcore_judge_file(_JUDGE_FILE), clients=clients)
        labels = run_agentcore_judge(judge, traces, clients=clients, confirm=approve, store=store)
        stored_by_trace = {label.trace_id: store.get_scores(label.trace_id) for label in labels}

    (estimate,) = estimates
    assert estimate.calls == len(traces)
    assert live_budget.used - billed_before == len(traces), "one billed Evaluate call per trace"
    # One advisory verdict per trace, each marked as scored on synthetic times, with the raw
    # verdict and the judge's fingerprint kept in its critique and the same verdict persisted.
    assert sorted(label.trace_id for label in labels) == sorted(ids)
    for label in labels:
        assert label.kind == "llm-judge"
        assert label.outcome in ("PASS", "FAIL")
        assert "timing=synthetic" in label.critique
        assert "value=" in label.critique
        assert f"[fingerprint={judge.fingerprint}]" in label.critique
        (stored,) = stored_by_trace[label.trace_id]
        assert (stored.kind, stored.outcome) == (label.kind, label.outcome)
        record_property(label.trace_id, {"outcome": label.outcome, "gold": gold[label.trace_id]})
        print(f"{label.trace_id}: {label.outcome} (gold {gold[label.trace_id]}) {label.critique}")
    # The synthetic times exist only in the request: the traces themselves are unchanged.
    assert [to_dict(trace) for trace in traces] == sent_as
