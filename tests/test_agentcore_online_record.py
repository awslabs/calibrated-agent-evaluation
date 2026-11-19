# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""The online-evaluation record and the session log: their stores and their refusals."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from calibrated_agent_evaluation.agentcore.errors import AgentCoreError
from calibrated_agent_evaluation.agentcore.online_record import (
    AgentCoreSessionLog,
    AttachedEvaluator,
    OnlineEvaluationRecord,
    OnlineReadRecord,
    ReadVerdict,
    dump_online_records,
    dump_read_records,
    dump_session_logs,
    load_online_records,
    load_read_records,
    load_session_logs,
    session_log_id,
)
from calibrated_agent_evaluation.agentcore.online_version import ONLINE_SCHEMA_VERSION

_EVALUATOR_ARN = "arn:aws:bedrock-agentcore:us-east-1:111122223333:evaluator/"


def _evaluator(name: str, **overrides: Any) -> AttachedEvaluator:
    fields: dict[str, Any] = {
        "evaluator_id": f"{name}-AbCdEf1234",
        "evaluator_arn": f"{_EVALUATOR_ARN}{name}-AbCdEf1234",
        "updated_at": "2026-11-04T20:52:50+00:00",
        "fingerprint": f"sha256:{name}",
        "calib_id": f"calib-{name}",
        "battery_id": f"battery-{name}",
    }
    fields.update(overrides)
    return AttachedEvaluator(**fields)


def online_record(**overrides: Any) -> OnlineEvaluationRecord:
    fields: dict[str, Any] = {
        "config_id": "calibrated_agent_evaluation_live_refunds-ZE8rieGKWt",
        "config_name": "calibrated_agent_evaluation_live_refunds",
        "runtime_service_name": "orderbot.DEFAULT",
        "evaluators": (_evaluator("refund_policy_v2"), _evaluator("refund_policy_v1")),
        "excluded_session_ids": (
            "calibrated-agent-evaluation-calibration-0001",
            "calibrated-agent-evaluation-bias-0001",
        ),
        "sampling_percentage": 100.0,
        "session_timeout_minutes": 15,
        "created_at": "2026-11-04T20:52:50+00:00",
    }
    fields.update(overrides)
    return OnlineEvaluationRecord(**fields)


def session_log(**overrides: Any) -> AgentCoreSessionLog:
    fields: dict[str, Any] = {
        "run_kind": "calibration",
        "run_id": "calib-refund_policy_v2",
        "judge_id": "orderbot-refund-policy",
        "evaluator_fingerprint": "sha256:refund_policy_v2",
        "session_ids": (
            "calibrated-agent-evaluation-calibration-0001",
            "calibrated-agent-evaluation-calibration-0002",
        ),
    }
    fields.update(overrides)
    return AgentCoreSessionLog(**fields)


def read_record(**overrides: Any) -> OnlineReadRecord:
    fields: dict[str, Any] = {
        "read_id": "calibrated_agent_evaluation_live_rate-AbCdEf1234:5f1c",
        "config_id": "calibrated_agent_evaluation_live_rate-AbCdEf1234",
        "backend": "local",
        "since": "2026-11-06T18:14:00+00:00",
        "until": "2026-11-06T18:50:00+00:00",
        "verdicts": (ReadVerdict(evaluator_id="v2", trace_id="1" * 32, verdict_digest="ab" * 8),),
        "errors": {"ValidationException": 1},
    }
    fields.update(overrides)
    return OnlineReadRecord(**fields)


def test_a_read_record_round_trips_from_json_and_csv(tmp_path: Path) -> None:
    record = read_record()
    dump_read_records([record], tmp_path, fmt="json")

    assert load_read_records(tmp_path, fmt="json") == [record]
    assert load_read_records(tmp_path, fmt="csv") == [record]


_READ_DOC = read_record().model_dump(mode="json")


@pytest.mark.parametrize(
    ("doc", "says"),
    [
        pytest.param({**_READ_DOC, "surprise": 1}, "surprise", id="unknown-field"),
        pytest.param(
            {k: v for k, v in _READ_DOC.items() if k != "schema_version"},
            "no schema_version",
            id="no-schema-version",
        ),
        pytest.param({**_READ_DOC, "schema_version": "0"}, "schema_version '0'", id="other"),
    ],
)
def test_a_stored_read_record_that_is_not_exactly_this_format_is_refused(
    tmp_path: Path, doc: dict[str, Any], says: str
) -> None:
    (tmp_path / "agentcore_online_read.json").write_text(json.dumps([doc]), encoding="utf-8")

    with pytest.raises(AgentCoreError, match=says):
        load_read_records(tmp_path, fmt="json")


def test_an_online_record_with_two_evaluators_round_trips_from_json_and_csv(tmp_path: Path) -> None:
    record = online_record()
    dump_online_records([record], tmp_path, fmt="json")

    assert load_online_records(tmp_path, fmt="json") == [record]
    assert load_online_records(tmp_path, fmt="csv") == [record]
    loaded = load_online_records(tmp_path, fmt="json")[0]
    assert loaded.evaluators[1].fingerprint == "sha256:refund_policy_v1"
    assert loaded.excluded_session_ids == (
        "calibrated-agent-evaluation-calibration-0001",
        "calibrated-agent-evaluation-bias-0001",
    )


def test_calibration_and_bias_session_logs_round_trip_and_are_keyed_by_kind_and_run(
    tmp_path: Path,
) -> None:
    calibration = session_log()
    bias = session_log(run_kind="bias", run_id="battery-refund_policy_v2", session_ids=("b-1",))
    dump_session_logs([calibration, bias], tmp_path, fmt="json")

    assert load_session_logs(tmp_path, fmt="json") == [calibration, bias]
    assert load_session_logs(tmp_path, fmt="csv") == [calibration, bias]
    assert session_log_id(calibration) == "calibration:calib-refund_policy_v2"
    assert session_log_id(bias) != session_log_id(calibration)


def _stored_online(tmp_path: Path, doc: dict[str, Any]) -> None:
    (tmp_path / "agentcore_online_config.json").write_text(json.dumps([doc]), encoding="utf-8")


_ONLINE_DOC = online_record().model_dump(mode="json")


@pytest.mark.parametrize(
    ("doc", "says"),
    [
        pytest.param({**_ONLINE_DOC, "alarm_arn": "x"}, "alarm_arn", id="unknown-field"),
        pytest.param(
            {
                **_ONLINE_DOC,
                "evaluators": [{**_ONLINE_DOC["evaluators"][0], "threshold": 0.9}],
            },
            "threshold",
            id="unknown-field-on-an-evaluator",
        ),
        pytest.param(
            {k: v for k, v in _ONLINE_DOC.items() if k != "schema_version"},
            "no schema_version",
            id="no-schema-version",
        ),
        pytest.param({**_ONLINE_DOC, "schema_version": "0"}, "schema_version '0'", id="other"),
    ],
)
def test_a_stored_online_record_that_is_not_exactly_this_format_is_refused(
    tmp_path: Path, doc: dict[str, Any], says: str
) -> None:
    _stored_online(tmp_path, doc)

    with pytest.raises(AgentCoreError, match=says):
        load_online_records(tmp_path, fmt="json")


_SESSION_DOC = session_log().model_dump(mode="json")


@pytest.mark.parametrize(
    ("doc", "says"),
    [
        pytest.param({**_SESSION_DOC, "trace_ids": []}, "trace_ids", id="unknown-field"),
        pytest.param(
            {k: v for k, v in _SESSION_DOC.items() if k != "schema_version"},
            "no schema_version",
            id="no-schema-version",
        ),
        pytest.param({**_SESSION_DOC, "schema_version": "0"}, "schema_version '0'", id="other"),
    ],
)
def test_a_stored_session_log_that_is_not_exactly_this_format_is_refused(
    tmp_path: Path, doc: dict[str, Any], says: str
) -> None:
    (tmp_path / "agentcore_session_log.json").write_text(json.dumps([doc]), encoding="utf-8")

    with pytest.raises(AgentCoreError, match=says):
        load_session_logs(tmp_path, fmt="json")


@pytest.mark.parametrize(
    "overrides",
    [
        {"sampling_percentage": 0.01},
        {"sampling_percentage": 100.0},
        {"session_timeout_minutes": 1},
        {"session_timeout_minutes": 1440},
    ],
    ids=["least-sampling", "all-sessions", "one-minute", "one-day"],
)
def test_an_online_record_at_agentcores_bounds_is_accepted(overrides: dict[str, Any]) -> None:
    record = online_record(**overrides)
    key, value = next(iter(overrides.items()))
    assert getattr(record, key) == value


@pytest.mark.parametrize(
    "overrides",
    [
        {"sampling_percentage": 0.0},
        {"sampling_percentage": 100.5},
        {"session_timeout_minutes": 0},
        {"session_timeout_minutes": 1441},
    ],
    ids=["no-sampling", "over-100-percent", "no-timeout", "over-a-day"],
)
def test_an_online_record_outside_agentcores_bounds_is_refused(overrides: dict[str, Any]) -> None:
    with pytest.raises(AgentCoreError, match=next(iter(overrides))):
        online_record(**overrides)


def test_a_session_log_of_an_unknown_run_kind_is_refused() -> None:
    with pytest.raises(AgentCoreError, match="run_kind"):
        session_log(run_kind="online")


def test_records_are_stamped_with_the_running_schema_version(tmp_path: Path) -> None:
    dump_online_records([online_record()], tmp_path, fmt="json")
    dump_session_logs([session_log()], tmp_path, fmt="json")

    for stem in ("agentcore_online_config", "agentcore_session_log"):
        stored = json.loads((tmp_path / f"{stem}.json").read_text(encoding="utf-8"))
        assert {doc["schema_version"] for doc in stored} == {ONLINE_SCHEMA_VERSION}


def test_a_session_log_with_a_gold_source_round_trips_and_an_earlier_one_loads(
    tmp_path: Path,
) -> None:
    recorded = session_log(gold_source="synthetic-rule")
    earlier = session_log(run_id="calib-earlier")
    for fmt in ("json", "csv"):
        out = tmp_path / fmt
        dump_session_logs([recorded, earlier], out, fmt=fmt)

        assert load_session_logs(out, fmt=fmt) == [recorded, earlier]
        assert [log.gold_source for log in load_session_logs(out, fmt=fmt)] == [
            "synthetic-rule",
            None,
        ]
    # A log written before the field existed has no key for it at all.
    doc = earlier.model_dump(mode="json")
    del doc["gold_source"]
    (tmp_path / "old").mkdir()
    (tmp_path / "old" / "agentcore_session_log.json").write_text(
        json.dumps([doc]), encoding="utf-8"
    )

    assert load_session_logs(tmp_path / "old", fmt="json") == [earlier]
