# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""What the audit report says it could not check, and that live traffic is counted, not tested."""

from __future__ import annotations

import re
from typing import Any

from _arms_fixture import AB_TEST_ARN, MAIN, VERSIONS, Double, Window, deliver
from calibrated_agent_evaluation.agentcore.arms import read_arms
from calibrated_agent_evaluation.agentcore.audit_report import read_summary, render_report
from test_agentcore_audit_types import claim_record


def _read(**flags: Any) -> Any:
    return read_arms(
        Double(deliver(MAIN)),
        ab_test_arn=AB_TEST_ARN,
        runtime="fixture-runtime",
        window=Window(),
        bundle_versions=VERSIONS,
        **flags,
    ).assignments


def _text(claim: Any, assignments: Any = ()) -> str:
    return "\n".join(render_report(claim, read_summary(list(assignments))))


def test_the_report_says_what_it_could_not_check() -> None:
    stopped = claim_record()
    empty = stopped.model_copy(update={"evaluator_claims": (), "execution_status": "NOT_STARTED"})
    assert "AgentCore reported no results (status ACTIVE, execution NOT_STARTED)" in _text(empty)

    running = stopped.model_copy(update={"execution_status": "RUNNING", "provisional": True})
    assert f"provisional: the test is RUNNING; analysis time {stopped.analysis_timestamp}" in _text(
        running
    )
    variants = tuple(
        v.model_copy(update={"bundle": stopped.variants[0].bundle}) for v in stopped.variants
    )
    aa = stopped.model_copy(update={"identical_arms": True, "variants": variants})
    assert "A/A comparison" in _text(aa)

    unchecked = "the agent's use of each arm's bundle was not checked"
    named_only = "whether the bundle was applied was not checked"
    plain = _text(stopped, _read())
    assert unchecked in plain
    assert named_only not in plain
    attribute = _text(stopped, _read(applied_bundle_attribute="config.bundle_version"))
    assert named_only in attribute
    assert "shows only that the agent named it" in attribute
    assert unchecked not in attribute
    both = _text(
        stopped,
        _read(
            applied_bundle_attribute="config.bundle_version",
            applied_bundle_require=[("config.source", "bundle")],
        ),
    )
    assert unchecked not in both
    assert named_only not in both


def test_live_arm_traffic_is_counted_not_tested() -> None:
    stopped = claim_record()
    text = _text(stopped, _read())

    assert "C: 5 trace(s) in 2 runtime session(s); AgentCore sampleSize 120, 120" in text
    assert "T1: 4 trace(s) in 2 runtime session(s); AgentCore sampleSize 118, 118" in text
    assert "(the number of sessions evaluated)" in text
    assert "traces without a runtime session: 1" in text
    traffic = text.split("traffic in arms read", 1)[1].split("Units:", 1)[0]
    assert "counted by Calibrated Agent Evaluation, not tested" in traffic
    assert not re.search(r"\bp[ =]|interval|McNemar|Wilson", traffic)
    assert "No re-test is stored" in text
