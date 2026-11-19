# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Write ``abtest_verdict.json`` and ``recommendation.json``: the claim records two fixtures map to.

``abtest_verdict.json`` is the stopped two-evaluator A/B test's record; ``recommendation.json``
the completed system-prompt recommendation's, on the fixture bundle.

Imports no Calibrated Agent Evaluation code. It maps the fixture responses field by field and
computes every digest with its own copy of the canonical JSON form, so a change to the audit's
digest rule, or to its mapping, fails the test that compares against this file. ``record_id``,
``read_at`` and ``provenance`` are left out: a read generates them.

Run from the repository root: ``python fixtures/expected/audit/make_claims.py``.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

_RAW = Path("fixtures/raw/agentcore/abtest")
_OUT = Path("fixtures/expected/audit/abtest_verdict.json")
_REC_RAW = Path("fixtures/raw/agentcore/recommendations")
_REC_OUT = Path("fixtures/expected/audit/recommendation.json")


def canonical(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def sha(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def load(name: str) -> dict:
    return json.loads((_RAW / name).read_text(encoding="utf-8"))


def bundle(doc: dict) -> dict:
    components = doc["components"]
    out = []
    for cid in sorted(components):
        conf = components[cid]["configuration"]
        text = canonical(conf)
        keys = []
        if isinstance(conf, dict):
            for key in sorted(conf):
                value = conf[key]
                value_text = value if isinstance(value, str) else canonical(value)
                keys.append({"key": key, "sha256": sha(value_text), "length": len(value_text)})
        out.append({"component_id": cid, "sha256": sha(text), "length": len(text), "keys": keys})
    lineage = doc.get("lineageMetadata") or {}
    return {
        "bundle_arn": doc["bundleArn"],
        "version_id": doc["versionId"],
        "version_created_at": doc["versionCreatedAt"],
        "parent_version_ids": list(lineage.get("parentVersionIds") or []),
        "branch_name": lineage.get("branchName"),
        "created_by": (lineage.get("createdBy") or {}).get("name"),
        "components": out,
        "components_sha256": sha(canonical(components)),
    }


def stats(doc: dict) -> dict:
    interval = doc.get("confidenceInterval") or {}
    return {
        "variant_name": doc["variantName"],
        "sample_size": doc["sampleSize"],
        "mean": doc["mean"],
        "absolute_change": doc.get("absoluteChange"),
        "percent_change": doc.get("percentChange"),
        "p_value": doc.get("pValue"),
        "ci_lower": interval.get("lower"),
        "ci_upper": interval.get("upper"),
        "is_significant": doc.get("isSignificant"),
    }


def text(tool_name: str | None, value: str) -> dict:
    return {"tool_name": tool_name, "sha256": sha(value), "length": len(value)}


def recommendation() -> dict:
    doc = json.loads((_REC_RAW / "rec_system_prompt.json").read_text(encoding="utf-8"))
    versions = {
        load(name)["versionId"]: load(name)
        for name in ("bundle_version_control.json", "bundle_version_treatment.json")
    }
    config = doc["recommendationConfig"]["systemPromptRecommendationConfig"]
    source = config["systemPrompt"]["configurationBundle"]
    result = doc["recommendationResult"]["systemPromptRecommendationResult"]
    spans = config["agentTraces"]["sessionSpans"]
    return {
        "claim_kind": "vendor-claim",
        "recommendation_arn": doc["recommendationArn"],
        "recommendation_id": doc["recommendationId"],
        "name": doc["name"],
        "type": doc["type"],
        "status": doc["status"],
        "created_at": doc["createdAt"],
        "updated_at": doc["updatedAt"],
        "input_bundle": bundle(versions[source["versionId"]]),
        "input_json_paths": [{"tool_name": None, "json_path": source["systemPromptJsonPath"]}],
        "input_inline": [],
        "traces_source": "sessionSpans",
        "traces_window_start": None,
        "traces_window_end": None,
        "traces_service_name": None,
        "traces_log_group_arns": [],
        "traces_batch_evaluation_arn": None,
        "traces_online_evaluation_config_arn": None,
        "traces_inline_trace_ids": sorted({s["traceId"] for s in spans}),
        "evaluator_arn": config["evaluationConfig"]["evaluators"][0]["evaluatorArn"],
        "result_bundle": bundle(versions[result["configurationBundle"]["versionId"]]),
        "result_texts": [text(None, result["recommendedSystemPrompt"])],
        "explanations": [text(None, result["explanation"])],
        "error_code": None,
        "error_message": None,
        "schema_version": "1",
    }


def main() -> None:
    rec = recommendation()
    _REC_OUT.write_text(json.dumps(rec, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    test = load("abtest_stopped.json")
    versions = {
        load(name)["versionId"]: load(name)
        for name in ("bundle_version_control.json", "bundle_version_treatment.json")
    }
    variants = []
    for v in sorted(test["variants"], key=lambda v: v["name"]):
        ref = v["variantConfiguration"]["configurationBundle"]
        variants.append(
            {
                "name": v["name"],
                "weight": v["weight"],
                "bundle": bundle(versions[ref["bundleVersion"]]),
                "target_name": None,
            }
        )
    results = test.get("results") or {}
    claims = [
        {
            "evaluator_arn": m["evaluatorArn"],
            "control": stats(m["controlStats"]),
            "variants": [stats(v) for v in m["variantResults"]],
        }
        for m in results.get("evaluatorMetrics") or []
    ]
    record = {
        "claim_kind": "vendor-claim",
        "ab_test_arn": test["abTestArn"],
        "ab_test_id": test["abTestId"],
        "name": test["name"],
        "status": test["status"],
        "execution_status": test["executionStatus"],
        "gateway_arn": test["gatewayArn"],
        "variants": variants,
        "evaluation_config_arns": [test["evaluationConfig"]["onlineEvaluationConfigArn"]],
        "current_run_id": test.get("currentRunId"),
        "error_details": list(test.get("errorDetails") or []),
        "started_at": test.get("startedAt"),
        "stopped_at": test.get("stoppedAt"),
        "max_duration_expires_at": test.get("maxDurationExpiresAt"),
        "analysis_timestamp": results.get("analysisTimestamp"),
        "updated_at": test["updatedAt"],
        "evaluator_claims": claims,
        "identical_arms": variants[0]["bundle"]["components_sha256"]
        == variants[1]["bundle"]["components_sha256"],
        "provisional": test["executionStatus"] in ("RUNNING", "PAUSED"),
        "schema_version": "1",
    }
    _OUT.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
