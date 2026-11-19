# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""The online fixture recorder scrubs account ids, access key ids and AWS-internal hosts, only."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from types import ModuleType

import pytest

_SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "record_agentcore_online.py"
_ACCOUNT = "202020202020"


def _recorder() -> ModuleType:
    spec = importlib.util.spec_from_file_location("record_agentcore_online", _SCRIPT)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_the_recorder_replaces_only_account_ids_key_ids_and_internal_hosts() -> None:
    recorder = _recorder()
    recorded = {
        "evaluator": (
            f"arn:aws:bedrock-agentcore:us-east-1:{_ACCOUNT}:evaluator/"
            "calibrated_agent_evaluation_live_x"
        ),
        "role": f"arn:aws:iam::{_ACCOUNT}:role/calibrated-agent-evaluation-live",
        "session.id": f"d165f{_ACCOUNT}c637402f2b4b9b5",
        "durationNano": 123456789012,
        "aws.request_id": "001c5d3e-9bda-483b-a877-123456789012",
        "aws.auth.account.access_key": "ASIAIOSFODNN7EXAMPLE",
        "http.server_name": "runtime.example.aws.dev",
        "http.url": "http://runtime.example.aws.dev/invocations",
        "server.address": "bedrock-runtime.us-east-1.amazonaws.com",
    }

    scrubbed = recorder.scrub(recorded)

    assert scrubbed == {
        "evaluator": (
            "arn:aws:bedrock-agentcore:us-east-1:111122223333:evaluator/"
            "calibrated_agent_evaluation_live_x"
        ),
        "role": "arn:aws:iam::111122223333:role/calibrated-agent-evaluation-live",
        "session.id": recorded["session.id"],
        "durationNano": recorded["durationNano"],
        "aws.request_id": recorded["aws.request_id"],
        "aws.auth.account.access_key": "REDACTED-ACCESS-KEY-ID",
        "http.server_name": "agentcore-runtime.invalid",
        "http.url": "http://agentcore-runtime.invalid/invocations",
        "server.address": recorded["server.address"],
    }


def test_the_recorder_refuses_a_file_that_still_holds_the_account_id(tmp_path: Path) -> None:
    recorder = _recorder()
    out = tmp_path / "record.json"
    leaks = [
        {"note": f"recorded in {_ACCOUNT}"},  # outside an ARN, so the scrub leaves it
        {"url": f"/runtimes/arn%3Aaws%3Abedrock-agentcore%3Aus-east-1%3A{_ACCOUNT}%3Aruntime%2Fx"},
    ]

    for leaking in leaks:
        with pytest.raises(SystemExit, match="account"):
            recorder.write(out, leaking, account_id=_ACCOUNT)
    assert not out.exists()

    clean = {"session.id": "d165f123456789012c637402f2b4b9b5"}
    recorder.write(out, clean, account_id=_ACCOUNT)
    assert json.loads(out.read_text(encoding="utf-8")) == clean
