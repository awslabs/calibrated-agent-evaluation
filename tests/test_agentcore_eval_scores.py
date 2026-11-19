# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""The AgentCore evaluation-score reader: real record shape, error-skip, and the dedicated group.

Grounded in a live capture: ``fixtures/raw/agentcore/evaluation_results.json`` holds real
``gen_ai.evaluation.result`` records produced by a live ``StartBatchEvaluation`` run. (That run
hit the known AgentCore ``LogEventMissingException`` span-correlation issue, so the captured
records are error events — which is itself the case the reader must skip.) The successful-verdict
record is synthesized from the OpenTelemetry GenAI-evaluation fields AWS documents
(``gen_ai.evaluation.score.value`` / ``.score.label`` / ``.explanation``) on the same real
envelope, so the map-to-``Label`` path is exercised without a live green evaluation.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

pytest.importorskip("boto3")

import boto3
from botocore.stub import Stubber

from calibrated_agent_evaluation.agentcore.readers import LogsInsightsReader, eval_record_to_result
from calibrated_agent_evaluation.agentcore.scores import read_scores as map_results_to_labels
from calibrated_agent_evaluation.agentcore.session import AwsClients
from calibrated_agent_evaluation.agentcore.sources import AgentCoreSourceConfig, TimeWindow

_REAL = (
    Path(__file__).resolve().parent.parent
    / "fixtures"
    / "raw"
    / "agentcore"
    / "evaluation_results.json"
)
_EVAL_GROUP = "/aws/bedrock-agentcore/evaluations/batch-evaluations/results/default"
_WINDOW = TimeWindow(start="2026-10-26T17:43:00+00:00", end="2026-10-26T18:05:00+00:00")


def _success_record() -> dict[str, Any]:
    """A successful verdict on the real record envelope (documented OTel evaluation fields)."""
    return {
        "name": "gen_ai.evaluation.result",
        "traceId": "6adf7dc33c6467c957f1b14b36da0ed3",
        "severityNumber": 9,
        "attributes": {
            "gen_ai.evaluation.name": "Builtin.Correctness",
            "gen_ai.evaluation.score.value": 0.92,
            "gen_ai.evaluation.score.label": "correct",
            "gen_ai.evaluation.explanation": "The response matches the policy facts.",
            "session.id": "calibrated_agent_evaluation_score9fb8d5d337ba44ed9e56349a119ce935",
            "gen_ai.response.id": "6adf7dc33c6467c957f1b14b36da0ed3",
            "aws.bedrock_agentcore.evaluator.arn": (
                "arn:aws:bedrock-agentcore:::evaluator/Builtin.Correctness"
            ),
            "aws.bedrock_agentcore.evaluator.rating_scale": "Numerical",
            "aws.bedrock_agentcore.evaluation_level": "Trace",
        },
    }


def _real_records() -> list[dict[str, Any]]:
    return json.loads(_REAL.read_text(encoding="utf-8"))


def test_real_error_event_records_are_skipped_not_mapped() -> None:
    # A failed-evaluation error event (error.type present) is not a verdict and must be skipped.
    errors = [r for r in _real_records() if (r.get("attributes") or {}).get("error.type")]
    assert errors, "the fixture should include a real LogEventMissing error event"
    assert all(r.get("name") == "gen_ai.evaluation.result" for r in errors)
    assert all(eval_record_to_result(r) is None for r in errors)


def test_the_real_successful_record_maps_to_an_advisory_pass_label() -> None:
    # The real Builtin.Correctness verdict captured live: value 1.0, label "Perfectly Correct".
    successes = [r for r in _real_records() if not (r.get("attributes") or {}).get("error.type")]
    assert successes, "the fixture should include the real successful evaluation record"
    result = eval_record_to_result(successes[0])
    assert result is not None
    (label,) = list(map_results_to_labels([result]))
    assert label.kind == "llm-judge"
    assert label.outcome == "PASS"
    assert "value=1.0" in label.critique


def test_a_successful_record_maps_to_the_evaluate_result_fields() -> None:
    result = eval_record_to_result(_success_record())
    assert result is not None
    assert result["value"] == 0.92
    assert result["label"] == "correct"
    assert result["evaluatorArn"].endswith("Builtin.Correctness")
    assert result["context"]["spanContext"]["traceId"] == "6adf7dc33c6467c957f1b14b36da0ed3"
    assert result["context"]["spanContext"]["sessionId"].startswith(
        "calibrated_agent_evaluation_score"
    )


def test_a_non_evaluation_record_is_ignored() -> None:
    assert eval_record_to_result({"name": "gen_ai.choice", "attributes": {}}) is None


def test_a_successful_record_becomes_an_advisory_llm_judge_label() -> None:
    result = eval_record_to_result(_success_record())
    assert result is not None
    (label,) = list(map_results_to_labels([result]))
    assert label.kind == "llm-judge"
    assert label.outcome == "PASS"  # label "correct" is a pass on the correctness scale
    assert "label='correct'" in label.critique  # the raw verdict is preserved verbatim
    assert label.trace_id == "6adf7dc33c6467c957f1b14b36da0ed3"


def _clients() -> AwsClients:
    session = boto3.Session(
        region_name="us-east-1", aws_access_key_id="testing", aws_secret_access_key="testing"
    )
    return AwsClients(session)


def _msg_rows(*records: dict[str, Any]) -> list[list[dict[str, str]]]:
    out: list[list[dict[str, str]]] = []
    for rec in records:
        msg = json.dumps(rec)
        out.append([{"field": "@message", "value": msg}, {"field": "@ptr", "value": msg}])
    return out


def test_read_scores_discovers_the_dedicated_group_and_yields_only_verdicts() -> None:
    clients = _clients()
    stub = Stubber(clients.client("logs"))
    # 1. discover the dedicated evaluation result group by prefix.
    stub.add_response(
        "describe_log_groups",
        {"logGroups": [{"logGroupName": _EVAL_GROUP}]},
        {"logGroupNamePrefix": "/aws/bedrock-agentcore/evaluations/"},
    )
    # 2. read it: a real error event (skipped) alongside a successful verdict (yielded).
    error_rec = json.loads(_REAL.read_text(encoding="utf-8"))[0]
    stub.add_response("start_query", {"queryId": "q"})
    stub.add_response(
        "get_query_results",
        {
            "results": _msg_rows(error_rec, _success_record()),
            "statistics": {"recordsMatched": 2.0},
            "status": "Complete",
        },
    )
    config = AgentCoreSourceConfig(
        runtime="SupportAgent.DEFAULT",
        window=_WINDOW,
        omni=False,
        with_scores=True,
    )
    with stub:
        results = list(LogsInsightsReader(clients, config).read_scores())
    assert len(results) == 1  # the error event was skipped, the verdict kept
    assert results[0]["label"] == "correct"


def test_a_batch_result_is_stamped_with_its_job_id_from_the_log_stream() -> None:
    # For a batch result group, the per-record log stream name is the batchEvaluationId (the job
    # id). The reader carries it onto the result so the advisory label's provenance records which
    # batch job produced the verdict.
    clients = _clients()
    stub = Stubber(clients.client("logs"))
    # _EVAL_GROUP is a .../evaluations/batch* group, so its records carry a job id.
    stub.add_response(
        "describe_log_groups",
        {"logGroups": [{"logGroupName": _EVAL_GROUP}]},
        {"logGroupNamePrefix": "/aws/bedrock-agentcore/evaluations/"},
    )
    row = [
        {"field": "@message", "value": json.dumps(_success_record())},
        {"field": "@ptr", "value": "p1"},
        {"field": "@logStream", "value": "acme_baseline-a1b2c3d4e5"},
    ]
    stub.add_response("start_query", {"queryId": "q"})
    stub.add_response(
        "get_query_results",
        {"results": [row], "statistics": {"recordsMatched": 1.0}, "status": "Complete"},
    )
    config = AgentCoreSourceConfig(
        runtime="SupportAgent.DEFAULT", window=_WINDOW, omni=False, with_scores=True
    )
    with stub:
        results = list(LogsInsightsReader(clients, config).read_scores())
    assert results[0]["jobId"] == "acme_baseline-a1b2c3d4e5"
    (label,) = list(map_results_to_labels(results))
    assert "job=acme_baseline-a1b2c3d4e5" in label.critique  # the job id rides in the provenance


def test_an_online_result_group_carries_no_job_id() -> None:
    # An online result group's stream is a config id, not a job id, so the reader does not request
    # or stamp it; the label's provenance carries no job tag (per AWS docs: batch only).
    online_group = "/aws/bedrock-agentcore/evaluations/online/results/cfg-1"
    clients = _clients()
    stub = Stubber(clients.client("logs"))
    stub.add_response(
        "describe_log_groups",
        {"logGroups": [{"logGroupName": online_group}]},
        {"logGroupNamePrefix": "/aws/bedrock-agentcore/evaluations/"},
    )
    stub.add_response("start_query", {"queryId": "q"})
    stub.add_response(
        "get_query_results",
        {
            "results": _msg_rows(_success_record()),
            "statistics": {"recordsMatched": 1.0},
            "status": "Complete",
        },
    )
    config = AgentCoreSourceConfig(
        runtime="SupportAgent.DEFAULT", window=_WINDOW, omni=False, with_scores=True
    )
    with stub:
        results = list(LogsInsightsReader(clients, config).read_scores())
    assert "jobId" not in results[0]
    (label,) = list(map_results_to_labels(results))
    assert "job=" not in label.critique


def test_read_scores_yields_nothing_without_the_flag() -> None:
    clients = _clients()
    stub = Stubber(clients.client("logs"))  # no calls expected
    config = AgentCoreSourceConfig(
        runtime="SupportAgent.DEFAULT",
        window=_WINDOW,
        omni=False,
        with_scores=False,
    )
    with stub:
        assert list(LogsInsightsReader(clients, config).read_scores()) == []
    stub.assert_no_pending_responses()
