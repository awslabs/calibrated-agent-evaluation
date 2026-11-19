# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Opt-in live detection against real Bedrock, which keeps the scripted-model unit tests honest.

Skipped by default (marked ``integration``, deselected in the default run). The unit tests drive
the failure detector with a scripted model; this test sends one real detection, so a change in how
the real model answers (a fence, prose around the payload, a capitalized confidence) that the
strict reading refuses shows up here rather than in a reviewer's run. It makes one billed
``ConverseStream`` call, which any profile allowed ``bedrock:InvokeModelWithResponseStream`` can
make:

    CALIBRATED_AGENT_EVALUATION_DETECTOR_LIVE=1 AWS_PROFILE=<your-profile> \
        uv run --extra agentcore pytest -m integration tests/integration/test_detectors_live.py
"""

from __future__ import annotations

import os
from importlib.metadata import version

import pytest
from _bedrock import bedrock_model_id

from _helpers import load_corpus_trace

pytestmark = pytest.mark.integration


def test_live_bedrock_detection_parses_strictly(monkeypatch: pytest.MonkeyPatch) -> None:
    if os.environ.get("CALIBRATED_AGENT_EVALUATION_DETECTOR_LIVE") != "1":
        pytest.skip(
            "CALIBRATED_AGENT_EVALUATION_DETECTOR_LIVE is not 1: set it to run one billed live "
            "detection"
        )
    if not (os.environ.get("AWS_PROFILE") or os.environ.get("AWS_ACCESS_KEY_ID")):
        pytest.skip(
            "AWS_PROFILE and AWS_ACCESS_KEY_ID are unset: set AWS_PROFILE to a profile with "
            "Bedrock access"
        )
    pytest.importorskip("strands_evals")
    from calibrated_agent_evaluation.agentcore import (
        FailureDetector,
        detector_bedrock_model,
        failure_vocabulary,
    )

    # The live call is the point of the test; a replay setting left in the shell would refuse it.
    monkeypatch.delenv("CALIBRATED_AGENT_EVALUATION_CASSETTE_MODE", raising=False)
    model_id = bedrock_model_id()
    detector = FailureDetector(detector_bedrock_model(model_id), max_live_calls=1)

    # A response the strict reading cannot read raises here instead of returning a detection.
    detection = detector.detect(load_corpus_trace("ORD-1001"))

    assert detection.trace_id == "ORD-1001"
    assert detection.provenance.model_id == model_id
    assert detection.provenance.detector_version == version("strands-agents-evals")
    assert detection.below_threshold == 0
    names = set(failure_vocabulary().names())
    assert all(failure.category in names for failure in detection.failures)
    assert all(failure.span_id == "ORD-1001/orderbot" for failure in detection.failures)
