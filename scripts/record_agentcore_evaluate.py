#!/usr/bin/env python3
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Write the AgentCore judge's offline replay fixtures, with no AWS call.

The demo's ``agentcore-judge`` stage replays these through botocore ``Stubber``, so the walk runs
the shipped judge code with no AWS account. Three files go to ``fixtures/raw/agentcore/evaluate/``:

- ``get_evaluator.json``: a ``GetEvaluator`` document for ``Builtin.Correctness`` in the service's
  shape: a ``TRACE`` evaluator on the three-rung scale 0.0 "Incorrect", 0.5 "Partially Correct",
  1.0 "Perfectly Correct", with AWS's account-less built-in evaluator ARN.
- ``evaluate.json``: for ORD-1002 and CLM-4471, the exact ``Evaluate`` request the serializer
  builds (tool calls as tool spans, synthetic times) and a response holding the value, label and
  explanation ``Builtin.Correctness`` returned live for that trace.
- ``price_list.json``: the ``GetProducts`` request for us-east-1 and the AgentCore Evaluations
  Price List entries returned live.

Provenance: written offline, no live response committed, so no account, runtime or caller id can
reach the repository. What is live and what is a placeholder:

- live (recorded 2026-11-03 in an AWS test account): each trace's ``value`` and ``label``,
  the evaluator's type, level and rung values and labels, and the Price List rates; each
  ``explanation`` is abridged from the live run's;
- placeholder: the evaluator's ``instructions``, rung ``definition`` texts, ``modelId`` and
  timestamps, which the replay does not read for scoring.

The responses carry no ``tokenUsage``: the live runs reported usage per run, not per trace.

Re-running this script rewrites the files from the current serializer. A serializer change that
alters a request needs the live values checked again (the opt-in live test) before the files are
written again, since each fixed live value is paired with whatever request is built now.

Usage:
    uv run --all-extras python scripts/record_agentcore_evaluate.py
"""

from __future__ import annotations

import json
import re
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

_REPO = Path(__file__).resolve().parents[1]
FIXTURE_DIR = _REPO / "fixtures" / "raw" / "agentcore" / "evaluate"
GET_EVALUATOR = FIXTURE_DIR / "get_evaluator.json"
EVALUATE = FIXTURE_DIR / "evaluate.json"
PRICE_LIST = FIXTURE_DIR / "price_list.json"
JUDGE_FILE = _REPO / "fixtures" / "judge-files" / "agentcore-correctness.json"
TRACES = ("ORD-1002", "CLM-4471")
REGION = "us-east-1"

_EVALUATOR_ID = "Builtin.Correctness"
_EVALUATOR_ARN = f"arn:aws:bedrock-agentcore:::evaluator/{_EVALUATOR_ID}"

# What Builtin.Correctness returned live for each trace, with tool calls sent as tool spans.
_LIVE_RESULTS = {
    "ORD-1002": (
        1.0,
        "Perfectly Correct",
        "The assistant's summary accurately reflects the tool outputs: order lookup confirmed "
        "$149.00 for Over-ear headphones delivered 2026-05-04, policy check showed 8 days since "
        "delivery within the 30-day window, and refund was issued with ID RF-30022. These match "
        "the tool results exactly.",
    ),
    "CLM-4471": (
        0.5,
        "Partially Correct",
        "The key fact: W-7 exclusion requires occupancy_status data, which the intake tool "
        "returned as null. The narrative states the claimant was in Florida for 'the last "
        "month' visiting family, not that the dwelling was vacant, so occupancy status could "
        "still be unresolved.",
    ),
}

# The us-east-1 AgentCore Evaluations rates the Price List API listed live.
_LIVE_RATES = (
    ("USE1-Evaluations:Consumption-based:BuiltIn-Input:Tier1", "1M Input Tokens", "2.4000000000"),
    (
        "USE1-Evaluations:Consumption-based:BuiltIn-Output:Tier1",
        "1M Output Tokens",
        "12.0000000000",
    ),
    ("USE1-Evaluations:Consumption-based:CustomEvaluators:Tier1", "Evaluations", "0.0015000000"),
)


def _get_evaluator() -> dict[str, Any]:
    stamp = datetime(2026, 1, 1, tzinfo=UTC).isoformat()
    rungs = (
        (0.0, "Incorrect", "The response contains a factual error or a claim its context refutes."),
        (0.5, "Partially Correct", "The response is partly right but misses or misstates a fact."),
        (1.0, "Perfectly Correct", "Every claim in the response is supported by its context."),
    )
    return {
        "evaluatorArn": _EVALUATOR_ARN,
        "evaluatorId": _EVALUATOR_ID,
        "evaluatorName": _EVALUATOR_ID,
        "evaluatorType": "Builtin",
        "level": "TRACE",
        "status": "ACTIVE",
        "createdAt": stamp,
        "updatedAt": stamp,
        "evaluatorConfig": {
            "llmAsAJudge": {
                "instructions": "Judge whether the agent's final answer is factually correct.",
                "modelConfig": {"bedrockEvaluatorModelConfig": {"modelId": "aws-managed"}},
                "ratingScale": {
                    "numerical": [
                        {"value": value, "label": label, "definition": definition}
                        for value, label, definition in rungs
                    ]
                },
            }
        },
    }


def _evaluate_entries() -> list[dict[str, Any]]:
    from calibrated_agent_evaluation.agentcore.evaluate_payload import (
        aws_trace_id,
        trace_to_evaluation_input,
    )
    from calibrated_agent_evaluation.schema.serde import trace_from_dict

    entries = []
    for trace_id in TRACES:
        path = _REPO / "fixtures" / "normalized" / f"{trace_id}.json"
        trace = trace_from_dict(json.loads(path.read_text(encoding="utf-8")))
        target = aws_trace_id(trace.trace_id)
        value, label, explanation = _LIVE_RESULTS[trace_id]
        request = {
            "evaluatorId": _EVALUATOR_ID,
            "evaluationInput": {"sessionSpans": trace_to_evaluation_input(trace)["sessionSpans"]},
            "evaluationTarget": {"traceIds": [target]},
        }
        response = {
            "evaluationResults": [
                {
                    "evaluatorArn": _EVALUATOR_ARN,
                    "evaluatorId": _EVALUATOR_ID,
                    "evaluatorName": _EVALUATOR_ID,
                    "explanation": explanation,
                    "context": {"spanContext": {"sessionId": target, "traceId": target}},
                    "value": value,
                    "label": label,
                }
            ]
        }
        entries.append({"trace_id": trace_id, "request": request, "response": response})
    return entries


def _price_list() -> dict[str, Any]:
    def entry(usage_type: str, unit: str, usd: str) -> str:
        dimension = {"unit": unit, "pricePerUnit": {"USD": usd}}
        return json.dumps(
            {
                "product": {"attributes": {"usagetype": usage_type, "regionCode": REGION}},
                "terms": {"OnDemand": {"T1": {"priceDimensions": {"D1": dimension}}}},
            },
            sort_keys=True,
        )

    return {
        "request": {
            "ServiceCode": "AmazonBedrockAgentCore",
            "Filters": [{"Type": "TERM_MATCH", "Field": "regionCode", "Value": REGION}],
            "FormatVersion": "aws_v1",
        },
        "response": {
            "PriceList": [entry(*rate) for rate in _LIVE_RATES],
            "FormatVersion": "aws_v1",
        },
    }


# A 12-digit run that is not part of a longer number: an account id, in an ARN or on its own.
_ACCOUNT_ID = re.compile(r"(?<!\d)\d{12}(?!\d)")


def _refuse_account_ids(path: Path, text: str) -> None:
    if _ACCOUNT_ID.search(text):
        raise SystemExit(f"refusing to write {path}: it would hold an AWS account id")


def _write(path: Path, data: Any) -> None:
    text = json.dumps(data, indent=2, sort_keys=True) + "\n"
    _refuse_account_ids(path, text)
    path.write_text(text, encoding="utf-8")


def main() -> int:
    FIXTURE_DIR.mkdir(parents=True, exist_ok=True)
    _write(GET_EVALUATOR, _get_evaluator())
    _write(EVALUATE, _evaluate_entries())
    _write(PRICE_LIST, _price_list())
    for path in (GET_EVALUATOR, EVALUATE, PRICE_LIST):
        print(f"wrote {path.relative_to(_REPO)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
