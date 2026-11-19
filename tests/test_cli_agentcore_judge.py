# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""``calibrated-agent-evaluation calibrate`` with an ``agentcore`` judge file, against a stubbed
AgentCore account."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from decimal import Decimal
from pathlib import Path
from typing import Any

import pytest

pytest.importorskip("boto3")

import boto3
from botocore.config import Config
from botocore.stub import Stubber

from calibrated_agent_evaluation.agentcore.evaluate_payload import aws_trace_id
from calibrated_agent_evaluation.agentcore.judge import config_digest
from calibrated_agent_evaluation.agentcore.pricing import EvaluationRates
from calibrated_agent_evaluation.backends.local import LocalBackend
from calibrated_agent_evaluation.cli import main
from calibrated_agent_evaluation.cli.dispatch import EXIT_DOMAIN_ERROR, EXIT_OK, EXIT_USAGE
from calibrated_agent_evaluation.cli.records import load_records_of
from calibrated_agent_evaluation.schema.types import Label, Span, Trace

_ACCOUNT = "111122223333"
_ID = "orderbot_refund_policy_v2-AbCdEf1234"
_ARN = f"arn:aws:bedrock-agentcore:us-east-1:{_ACCOUNT}:evaluator/{_ID}"
_UPDATED = datetime(2026, 11, 5, 16, 19, 46, tzinfo=UTC)
_MODE = "trusted_broken_policy_tool"


def _trace(trace_id: str) -> Trace:
    span = Span(
        span_id=f"{trace_id}/agent",
        trace_id=trace_id,
        agent="orderbot",
        role="specialist",
        parent=None,
        seq=0,
        status="completed",
        input={"message": "refund please"},
        output={"answer": "refunded"},
        started_at="2026-01-01T00:00:00+00:00",
        ended_at="2026-01-01T00:00:01+00:00",
    )
    return Trace(
        trace_id=trace_id,
        frame="discovery",
        ownership="owned",
        input={"message": "refund please"},
        output={"answer": "refunded"},
        taxonomy_cell={"label_round": "2"},
        spans=[span],
    )


def _gold(trace_id: str, outcome: str) -> Label:
    return Label(
        trace_id=trace_id,
        span_id=None,
        kind="human",
        outcome=outcome,  # type: ignore[arg-type]
        rater_id="synthetic-rule-oracle",
        failure_mode=_MODE if outcome == "FAIL" else None,
        critique="Synthetic gold (rule)",
    )


_GOLD = {f"Q-1000{i}": "FAIL" for i in range(3)} | {f"Q-2000{i}": "PASS" for i in range(3)}


def _evaluator(level: str = "TRACE") -> dict[str, Any]:
    return {
        "evaluatorArn": _ARN,
        "evaluatorId": _ID,
        "evaluatorName": "orderbot_refund_policy_v2",
        "evaluatorType": "Custom",
        "level": level,
        "status": "ACTIVE",
        "createdAt": _UPDATED,
        "updatedAt": _UPDATED,
        "evaluatorConfig": {
            "llmAsAJudge": {
                "instructions": "Judge it. {context} {assistant_turn}",
                "ratingScale": {
                    "numerical": [
                        {"value": 0.0, "label": "FAIL", "definition": "fails"},
                        {"value": 1.0, "label": "PASS", "definition": "passes"},
                    ]
                },
                "modelConfig": {
                    "bedrockEvaluatorModelConfig": {
                        "modelId": "global.anthropic.claude-haiku-4-5-20251001-v1:0"
                    }
                },
            }
        },
    }


_BASE = "Builtin.Helpfulness"


def _derived() -> dict[str, Any]:
    """The stubbed evaluator as a derived one, on a pinned model, over a built-in base."""
    doc = _evaluator()
    doc["evaluatorType"] = "CustomDerived"
    doc["evaluatorConfig"] = {
        "derived": {
            "baseEvaluatorId": _BASE,
            "modelConfig": {
                "bedrockEvaluatorModelConfig": {
                    "modelId": "global.anthropic.claude-haiku-4-5-20251001-v1:0"
                }
            },
        }
    }
    return doc


def _base() -> dict[str, Any]:
    doc = _evaluator()
    doc.update(
        evaluatorId=_BASE,
        evaluatorArn=f"arn:aws:bedrock-agentcore:::evaluator/{_BASE}",
        evaluatorName=_BASE,
        evaluatorType="Builtin",
    )
    return doc


@pytest.fixture
def account(request: pytest.FixtureRequest, monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    """A store with gold traces (six unless the test asks for more), a judge file, and a stubbed
    account that grades by gold and records the most calls it ever had open at once."""
    import threading
    import time

    from calibrated_agent_evaluation.agentcore.session import AwsClients

    gold_by_id: dict[str, str] = getattr(request, "param", None) or _GOLD

    for var in (
        "AWS_PROFILE",
        "AWS_REGION",
        "AWS_DEFAULT_REGION",
        "CALIBRATED_AGENT_EVALUATION_LOCAL_ROOT",
    ):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.delenv("CALIBRATED_AGENT_EVALUATION_AGENTCORE_MAX_BILLED_CALLS", raising=False)
    monkeypatch.delenv("CALIBRATED_AGENT_EVALUATION_AGENTCORE_MAX_CONCURRENCY", raising=False)
    monkeypatch.setenv("AWS_CONFIG_FILE", str(tmp_path / "no-aws-config"))
    rates = EvaluationRates(
        region="us-east-1",
        builtin_input_per_million=Decimal("2.40"),
        builtin_output_per_million=Decimal("12.00"),
        custom_per_evaluation=Decimal("0.0015"),
    )
    monkeypatch.setattr(
        "calibrated_agent_evaluation.agentcore.evaluate.lookup_evaluation_rates", lambda _c: rates
    )

    root = tmp_path / "root"
    with LocalBackend(root=str(root)) as store:
        for trace_id, outcome in gold_by_id.items():
            store.put_trace(_trace(trace_id))
            store.put_score(_gold(trace_id, outcome))
    judge_file = tmp_path / "judge-v2.agentcore.json"
    judge_file.write_text(
        json.dumps(
            {
                "backend": "agentcore",
                "judge_id": "orderbot-refund-policy",
                "evaluator_id": _ID,
                "failure_mode": _MODE,
                "level": "workflow",
            }
        ),
        encoding="utf-8",
    )
    by_aws_id = {aws_trace_id(t): t for t in gold_by_id}
    lock = threading.Lock()
    state: dict[str, Any] = {
        "opened": False,
        "missing": False,
        "derived": False,
        "gets": 0,
        "ops": [],
        "handlers": [],
        "calls": 0,
        "open": 0,
        "peak": 0,
        "level": "TRACE",
        "unreadable": None,
        "root": root,
        "file": judge_file,
    }

    def evaluated(target: str) -> dict[str, Any]:
        # A trace with no gold (a bias probe) is answered PASS.
        trace_id = by_aws_id.get(target)
        value = 0.0 if trace_id is not None and gold_by_id[trace_id] == "FAIL" else 1.0
        return {
            "evaluationResults": [
                {
                    "evaluatorArn": _ARN,
                    "evaluatorId": _ID,
                    "evaluatorName": "orderbot_refund_policy_v2",
                    "explanation": "graded",
                    "value": value,
                    "label": "PASS" if value else "FAIL",
                    "context": {"spanContext": {"sessionId": target, "traceId": target}},
                }
            ],
            "ResponseMetadata": {"HTTPStatusCode": 200},
        }

    def from_profile(cls, *, region=None, profile=None, handlers=(), **_kw):
        from botocore.awsrequest import AWSResponse

        state["opened"] = True
        state["handlers"] = list(handlers)

        session = boto3.Session(
            region_name="us-east-1", aws_access_key_id="testing", aws_secret_access_key="testing"
        )
        clients = AwsClients(session, handlers=handlers).with_config(
            Config(retries={"total_max_attempts": 1})
        )
        stub = Stubber(clients.client("bedrock-agentcore-control"))
        if state["missing"]:
            stub.add_client_error(
                "get_evaluator",
                service_error_code="ResourceNotFoundException",
                http_status_code=404,
            )
        elif state["derived"]:
            stub.add_response("get_evaluator", _derived(), {"evaluatorId": _ID})
            stub.add_response("get_evaluator", _base(), {"evaluatorId": _BASE})
        else:
            stub.add_response("get_evaluator", _evaluator(state["level"]), {"evaluatorId": _ID})
        stub.activate()

        def count_get(model: Any, **_: Any) -> None:
            state["ops"].append(model.name)
            if model.name == "GetEvaluator":
                state["gets"] += 1

        # On the wildcard and first: botocore runs wildcard handlers, the Stubber's among them,
        # before handlers for the operation's own name.
        control_events = clients.client("bedrock-agentcore-control").meta.events
        control_events.register_first("before-call.*.*", count_get)

        def strip(parsed: dict[str, Any], **_: Any) -> None:
            # The METADATA_ONLY shape: the configuration without one of its content fields. The
            # stubbed response must match botocore's model, which marks both fields required, so
            # the field is removed from the parsed response, as a service reply without it parses.
            if state["unreadable"]:
                del parsed["evaluatorConfig"]["llmAsAJudge"][state["unreadable"]]

        control_events.register("after-call.bedrock-agentcore-control.GetEvaluator", strip)
        # Answer Evaluate the way Stubber does, from botocore's own hooks, so request validation
        # and the guards the CLI installed (the billed-call budget) run first, for real.
        events = clients.client("bedrock-agentcore").meta.events

        def capture(params: dict[str, Any], context: dict[str, Any], **_: Any) -> None:
            context["calibrated_agent_evaluation_target"] = params["evaluationTarget"]["traceIds"][
                0
            ]

        def respond(context: dict[str, Any], **_: Any) -> Any:
            with lock:
                state["calls"] += 1
                state["open"] += 1
                state["peak"] = max(state["peak"], state["open"])
            # Long enough for calls sent together to overlap, so the peak shows the limit.
            time.sleep(0.02)
            with lock:
                state["open"] -= 1
            return AWSResponse("https://stub", 200, {}, None), evaluated(
                context["calibrated_agent_evaluation_target"]
            )

        events.register("before-parameter-build.bedrock-agentcore.Evaluate", capture)
        events.register("before-call.bedrock-agentcore.Evaluate", respond)
        return clients

    monkeypatch.setattr(
        "calibrated_agent_evaluation.agentcore.session.AwsClients.from_profile",
        classmethod(from_profile),
    )
    return state


def _calibrate(state: dict[str, Any], *extra: str) -> int:
    return main(
        [
            "--root",
            str(state["root"]),
            "calibrate",
            "--judge",
            str(state["file"]),
            "--cell",
            "label_round=2",
            *extra,
        ]
    )


def test_without_yes_calibrate_shows_the_estimate_and_sends_nothing(account, capsys) -> None:
    code = _calibrate(account)

    out = capsys.readouterr().out
    assert code == EXIT_OK
    assert "scores 6 gold trace(s) through AgentCore" in out
    assert "6 billed AgentCore Evaluate call(s)" in out
    assert "Nothing was sent" in out
    assert account["calls"] == 0
    assert load_records_of(account["root"], "calibration") == []


def test_calibrate_scores_the_gold_through_agentcore_and_records_the_pin_and_sessions(
    account, capsys
) -> None:
    code = _calibrate(account, "--yes", "--explain")

    captured = capsys.readouterr()
    text = captured.out + captured.err
    assert account["calls"] == 6
    (record,) = load_records_of(account["root"], "calibration")
    assert record.model == _ARN
    # Pinned to the evaluator's configuration, not to its updatedAt, which a lock moves.
    assert record.prompt_version == config_digest(_evaluator())
    assert record.n_pos == 3
    assert record.n_neg == 3
    # Six pairs is far below the plan, so the judge is refused however well it did.
    assert record.result == "REFUSED"
    assert code == EXIT_DOMAIN_ERROR
    assert "gold source: synthetic-rule" in text.lower()
    (log,) = load_records_of(account["root"], "agentcore-session")
    assert log.run_kind == "calibration"
    assert log.run_id == record.calib_id
    assert log.evaluator_fingerprint == record.judge_fingerprint
    assert sorted(log.session_ids) == sorted(aws_trace_id(t) for t in _GOLD)


def test_a_budget_smaller_than_the_locked_test_is_refused_before_any_billed_call(
    account, monkeypatch, capsys
) -> None:
    monkeypatch.setenv("CALIBRATED_AGENT_EVALUATION_AGENTCORE_MAX_BILLED_CALLS", "5")

    code = _calibrate(account, "--yes")

    assert code == EXIT_DOMAIN_ERROR
    assert "BilledCallBudgetExceeded" in capsys.readouterr().err
    assert account["calls"] == 0
    assert load_records_of(account["root"], "calibration") == []


@pytest.mark.parametrize("level", ["SESSION", "TOOL_CALL"])
def test_an_evaluator_that_is_not_trace_level_is_refused_before_any_billed_call(
    account, level, capsys
) -> None:
    account["level"] = level

    code = _calibrate(account, "--yes")

    assert code == EXIT_DOMAIN_ERROR
    assert level in capsys.readouterr().err
    assert account["calls"] == 0


@pytest.mark.parametrize("missing", ["instructions", "ratingScale"])
def test_an_evaluator_read_without_its_instructions_or_scale_is_refused_before_any_billed_call(
    account, missing, capsys
) -> None:
    account["unreadable"] = missing

    code = _calibrate(account, "--yes")

    err = capsys.readouterr().err
    assert code == EXIT_DOMAIN_ERROR
    assert missing in err
    assert _ID in err
    assert account["calls"] == 0
    assert load_records_of(account["root"], "calibration") == []


def test_aws_flags_are_refused_with_a_prompt_judge_file(account, tmp_path, capsys) -> None:
    prompt = (
        Path(__file__).resolve().parents[1]
        / "src/calibrated_agent_evaluation/_quickstart/data/certify/judge-v2.json"
    )

    code = main(["--root", str(account["root"]), "calibrate", "--judge", str(prompt), "--yes"])

    assert code == EXIT_USAGE
    assert "--yes applies to an AgentCore judge file only" in capsys.readouterr().err


def test_a_cassette_is_refused_with_an_agentcore_judge_file(account, tmp_path, capsys) -> None:
    code = _calibrate(account, "--cassette", str(tmp_path / "c.json"))

    assert code == EXIT_USAGE
    assert "an AgentCore judge is scored by AgentCore" in capsys.readouterr().err
    assert account["calls"] == 0


@pytest.mark.parametrize("flag", [["--profile", "dev"], ["--region", "us-east-1"]])
def test_aws_location_flags_are_refused_with_a_prompt_judge_file(account, flag, capsys) -> None:
    prompt = (
        Path(__file__).resolve().parents[1]
        / "src/calibrated_agent_evaluation/_quickstart/data/certify/judge-v2.json"
    )

    code = main(["--root", str(account["root"]), "calibrate", "--judge", str(prompt), *flag])

    assert code == EXIT_USAGE
    assert f"{flag[0]} applies to an AgentCore judge file only" in capsys.readouterr().err


# --- several calls at a time, progress on standard error, a bad concurrency value -----------------


def _gold_set(count: int) -> dict[str, str]:
    half = count // 2
    return {f"Q-1{i:04d}": "FAIL" for i in range(half)} | {
        f"Q-2{i:04d}": "PASS" for i in range(count - half)
    }


@pytest.mark.parametrize("account", [_gold_set(20)], indirect=True)
def test_calibrate_sends_up_to_the_set_limit_and_prints_progress_to_stderr(
    account, capsys, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("CALIBRATED_AGENT_EVALUATION_AGENTCORE_MAX_CONCURRENCY", "4")

    _calibrate(account, "--yes")

    captured = capsys.readouterr()
    assert account["calls"] == 20
    assert 2 <= account["peak"] <= 4
    progress = [line for line in captured.err.splitlines() if line.startswith("scored ")]
    assert progress == [f"scored {n} of 20 trace(s)" for n in range(2, 21, 2)]
    assert "scored " not in captured.out


@pytest.mark.parametrize("account", [_gold_set(7)], indirect=True)
def test_a_run_under_ten_traces_prints_one_progress_line_per_call(account, capsys) -> None:
    _calibrate(account, "--yes")

    progress = [line for line in capsys.readouterr().err.splitlines() if line.startswith("scored ")]
    assert progress == [f"scored {n} of 7 trace(s)" for n in range(1, 8)]


@pytest.mark.parametrize("value", ["0", "11", "four", "1_0", "+4"])
def test_a_concurrency_outside_one_to_ten_is_refused_before_any_call(
    account, capsys, monkeypatch: pytest.MonkeyPatch, value: str
) -> None:
    monkeypatch.setenv("CALIBRATED_AGENT_EVALUATION_AGENTCORE_MAX_CONCURRENCY", value)

    code = _calibrate(account, "--yes")

    err = capsys.readouterr().err
    assert code != EXIT_OK
    assert "CALIBRATED_AGENT_EVALUATION_AGENTCORE_MAX_CONCURRENCY" in err
    assert "1 to 10" in err
    # Refused before any AWS client was opened, so not even GetEvaluator was sent.
    assert account["opened"] is False
    assert account["calls"] == 0
    assert load_records_of(account["root"], "calibration") == []


# --- calibrated-agent-evaluation judge and calibrated-agent-evaluation calibrate --check with an
# agentcore judge file -----------------------


def _admit(state: dict[str, Any], *, battery: str | None = "CLEAR", calibrated: bool = True) -> str:
    """Store a certifying calibration and a bias battery pinned to the stubbed evaluator.

    Returns the judge's rater id, ``<judge_id>@<fingerprint>``, which its verdicts carry.
    """
    from calibrated_agent_evaluation.agentcore.judge import judge_from_evaluator
    from cli_store_helpers import (
        certified_calibration,
        clear_battery,
        store_battery,
        store_calibration,
    )

    judge = judge_from_evaluator("orderbot-refund-policy", _evaluator(), level="workflow")
    pin = {
        "model": judge.model,
        "prompt_version": judge.prompt_version,
        "fingerprint": judge.fingerprint,
    }
    if calibrated:
        store_calibration(
            state["root"],
            certified_calibration("orderbot-refund-policy", calib_id="calib-ac", **pin),
        )
    if battery is not None:
        store_battery(
            state["root"], clear_battery("orderbot-refund-policy", verdict=battery, **pin)
        )
    return f"{judge.judge_id}@{judge.fingerprint}"


def _judge(state: dict[str, Any], *extra: str) -> int:
    return main(["--root", str(state["root"]), "judge", "--judge", str(state["file"]), *extra])


def _stored(state: dict[str, Any]) -> list[Label]:
    with LocalBackend(root=str(state["root"])) as store:
        return [
            label
            for trace in store.all_traces()
            for label in store.get_scores(trace.trace_id)
            if label.kind == "llm-judge"
        ]


def test_without_yes_judge_shows_the_estimate_and_sends_nothing(account, capsys) -> None:
    _admit(account)

    code = _judge(account)

    out = capsys.readouterr().out
    assert code == EXIT_OK
    assert "6 billed AgentCore Evaluate call(s)" in out
    assert "Nothing was sent" in out
    assert account["calls"] == 0
    assert _stored(account) == []
    assert load_records_of(account["root"], "judge") == []


def test_judge_stores_one_verdict_per_trace_with_its_mode_and_rater_and_records_the_run(
    account, capsys
) -> None:
    rater = _admit(account)

    code = _judge(account, "--yes", "--profile", "dev", "--region", "us-east-1")

    assert code == EXIT_OK, capsys.readouterr().err
    assert account["calls"] == 6
    verdicts = _stored(account)
    assert sorted(v.trace_id for v in verdicts) == sorted(_GOLD)
    assert {v.failure_mode for v in verdicts} == {_MODE}
    assert {v.rater_id for v in verdicts} == {rater}
    # The stubbed account grades by gold, so each verdict matches its trace's gold.
    assert {v.trace_id: v.outcome for v in verdicts} == _GOLD
    (run,) = load_records_of(account["root"], "judge")
    assert run.model == _ARN
    assert run.prompt_version == config_digest(_evaluator())
    assert run.judge_fingerprint == rater.rsplit("@", 1)[1]
    assert run.calib_id == "calib-ac"
    assert run.n_computed == 6
    assert run.level == "workflow"
    assert sorted(run.trace_ids) == sorted(_GOLD)


@pytest.mark.parametrize("account", [_gold_set(7)], indirect=True)
def test_judge_prints_its_progress_to_stderr(account, capsys) -> None:
    _admit(account)

    _judge(account, "--yes")

    progress = [line for line in capsys.readouterr().err.splitlines() if line.startswith("scored ")]
    assert progress == [f"scored {n} of 7 trace(s)" for n in range(1, 8)]


@pytest.mark.parametrize(
    ("seed", "expected"),
    [
        ({"calibrated": False}, "no calibration record"),
        ({"battery": "BIASED"}, "bias gate refused"),
    ],
    ids=["uncalibrated", "biased"],
)
def test_an_unadmitted_agentcore_judge_is_refused_before_the_estimate(
    account, capsys, seed: dict[str, Any], expected: str
) -> None:
    _admit(account, **seed)

    code = _judge(account, "--yes")

    captured = capsys.readouterr()
    assert code == EXIT_DOMAIN_ERROR
    assert expected in captured.out + captured.err
    assert "billed AgentCore Evaluate call" not in captured.out
    assert account["calls"] == 0


def test_a_judge_file_naming_an_unknown_evaluator_is_refused_naming_it(account, capsys) -> None:
    _admit(account)
    account["missing"] = True

    code = _judge(account, "--yes")

    assert code == EXIT_DOMAIN_ERROR
    assert _ID in capsys.readouterr().err
    assert account["calls"] == 0


def test_a_cassette_is_refused_by_judge_with_an_agentcore_judge_file(
    account, tmp_path, capsys
) -> None:
    _admit(account)

    code = _judge(account, "--yes", "--cassette", str(tmp_path / "c.json"))

    assert code == EXIT_USAGE
    assert "an AgentCore judge is scored by AgentCore" in capsys.readouterr().err
    assert account["calls"] == 0


_PROMPT = (
    Path(__file__).resolve().parents[1]
    / "src/calibrated_agent_evaluation/_quickstart/data/certify/judge-v2.json"
)


@pytest.mark.parametrize(
    ("argv", "flag"),
    [
        (["judge", "--judge", str(_PROMPT), "--yes"], "--yes"),
        (["judge", "--check", str(_PROMPT), "--profile", "x"], "--profile"),
        (["calibrate", "--check", "--judge", str(_PROMPT), "--yes"], "--yes"),
        (["calibrate", "--check", "--judge", str(_PROMPT), "--region", "x"], "--region"),
    ],
    ids=["judge", "judge-check", "calibrate-check", "calibrate-check-region"],
)
def test_aws_flags_are_refused_when_no_judge_file_is_an_agentcore_one(
    account, capsys, argv: list[str], flag: str
) -> None:
    code = main(["--root", str(account["root"]), *argv])

    assert code == EXIT_USAGE
    assert f"{flag} applies to an AgentCore judge file only" in capsys.readouterr().err
    assert account["opened"] is False


@pytest.mark.parametrize("account", [_gold_set(50)], indirect=True)
def test_calibrate_check_scores_fresh_gold_through_agentcore_and_records_the_check(
    account, capsys
) -> None:
    _admit(account)

    code = main(
        [
            "--root",
            str(account["root"]),
            "calibrate",
            "--check",
            "--judge",
            str(account["file"]),
            "--yes",
        ]
    )

    captured = capsys.readouterr()
    assert code == EXIT_OK, captured.err
    assert account["calls"] == 50
    (check,) = load_records_of(account["root"], "judge-check")
    assert check.model == _ARN
    assert check.against_calib_id == "calib-ac"
    assert check.n_pos == 25
    assert check.n_neg == 25


def test_judge_check_accepts_the_aws_flags_with_an_agentcore_judge_file(account, capsys) -> None:
    code = main(["judge", "--check", str(account["file"]), "--profile", "x"])

    assert code == EXIT_OK
    assert "judge file ok" in capsys.readouterr().out
    assert account["opened"] is False


@pytest.mark.parametrize("account", [_gold_set(50)], indirect=True)
def test_without_yes_calibrate_check_shows_the_estimate_and_sends_nothing(account, capsys) -> None:
    _admit(account)

    code = main(
        ["--root", str(account["root"]), "calibrate", "--check", "--judge", str(account["file"])]
    )

    out = capsys.readouterr().out
    assert code == EXIT_OK
    assert "Checking" in out
    assert "Nothing was sent" in out
    assert account["calls"] == 0
    assert load_records_of(account["root"], "judge-check") == []


# --- calibrated-agent-evaluation run, dump --judges-only and load with an agentcore judge file
# -------------------------


def test_run_corrects_the_agentcore_judges_verdicts_reading_only_the_evaluator(
    account, capsys
) -> None:
    from calibrated_agent_evaluation.agentcore.guards import ReadOnlyGuard
    from cli_store_helpers import seed_verdicts

    rater = _admit(account)
    seed_verdicts(account["root"], 160, 40, mode=_MODE, rater_id=rater)

    code = main(
        [
            "--root",
            str(account["root"]),
            "run",
            "--n-resamples",
            "199",
            "--judge",
            str(account["file"]),
            "--seed",
            "3",
        ]
    )

    out = capsys.readouterr().out
    assert code == EXIT_OK
    assert "corrected 0." in out
    assert "calib-ac" in out
    assert account["gets"] == 1
    assert account["calls"] == 0
    # The clients run --judge opens allow GetEvaluator and nothing else.
    (guard,) = [h for h in account["handlers"] if isinstance(h, ReadOnlyGuard)]
    assert guard.allowed == {("bedrock-agentcore-control", "GetEvaluator")}


def test_a_derived_judges_identity_reads_the_evaluator_and_its_base(account, capsys) -> None:
    account["derived"] = True

    code = main(["--root", str(account["root"]), "run", "--judge", str(account["file"])])

    capsys.readouterr()
    # The store's six gold traces are too few for the gate to pass; the identity is what's tested.
    assert code == EXIT_DOMAIN_ERROR
    assert account["ops"] == ["GetEvaluator", "GetEvaluator"]
    assert account["calls"] == 0


def _dump_kit(state: dict[str, Any], kit: Path) -> int:
    argv = ["--root", str(state["root"]), "dump", "--judges-only", "--judge", str(state["file"])]
    return main([*argv, str(kit)])


def test_a_judges_only_dump_and_load_round_trip_with_no_aws_call_on_load(
    account, tmp_path, capsys
) -> None:
    _admit(account)
    kit = tmp_path / "kit"
    assert _dump_kit(account, kit) == EXIT_OK, capsys.readouterr().err
    assert account["gets"] == 1
    account["opened"] = False
    fresh = tmp_path / "fresh"

    code = main(["--root", str(fresh), "load", str(kit)])

    assert code == EXIT_OK, capsys.readouterr().err
    assert account["opened"] is False
    restored = fresh / "judges" / account["file"].name
    assert restored.read_bytes() == account["file"].read_bytes()
    (calib,) = load_records_of(fresh, "calibration")
    assert calib.calib_id == "calib-ac"
    assert len(load_records_of(fresh, "bias")) == 1


def test_a_dump_whose_judge_file_names_another_evaluator_is_refused_on_load(
    account, tmp_path, capsys
) -> None:
    _admit(account)
    kit = tmp_path / "kit"
    assert _dump_kit(account, kit) == EXIT_OK
    dumped = kit / "judges" / account["file"].name
    doc = json.loads(dumped.read_text(encoding="utf-8"))
    doc["evaluator_id"] = "another_evaluator-ZzZzZzZzZz"
    dumped.write_text(json.dumps(doc), encoding="utf-8")
    capsys.readouterr()

    code = main(["--root", str(tmp_path / "fresh"), "load", str(kit)])

    assert code == EXIT_DOMAIN_ERROR
    assert "another_evaluator-ZzZzZzZzZz" in capsys.readouterr().err
    assert load_records_of(tmp_path / "fresh", "calibration") == []


@pytest.mark.parametrize(
    "argv",
    [
        ["run", "--judge", str(_PROMPT), "--profile", "x"],
        ["run", "--region", "us-east-1"],
        ["dump", "--judges-only", "--judge", str(_PROMPT), "--profile", "x", "kit"],
        ["dump", "--profile", "x", "kit"],
    ],
    ids=["run-prompt", "run-no-judge", "dump-prompt", "dump-plain"],
)
def test_run_and_dump_refuse_the_aws_flags_when_no_judge_file_is_an_agentcore_one(
    account, tmp_path, capsys, argv: list[str]
) -> None:
    argv = [str(tmp_path / a) if a == "kit" else a for a in argv]

    code = main(["--root", str(account["root"]), *argv])

    assert code == EXIT_USAGE
    err = capsys.readouterr().err
    flag = "--region" if "--region" in argv else "--profile"
    assert f"{flag} applies to an AgentCore judge file only" in err
    assert account["opened"] is False


# --- calibrated-agent-evaluation bias with an agentcore judge file
# -----------------------------------------------------

_CERTIFY = (
    Path(__file__).resolve().parents[1] / "src/calibrated_agent_evaluation/_quickstart/data/certify"
)


def _probes(tmp_path: Path, trials: int = 3) -> Path:
    """The quickstart's round-2 probe set, each probe cut to its first ``trials`` trials."""
    out = tmp_path / "probes"
    out.mkdir()
    for src in sorted((_CERTIFY / "bias-probes-2").glob("*.json")):
        doc = json.loads(src.read_text(encoding="utf-8"))
        doc["trials"] = doc["trials"][:trials]
        (out / src.name).write_text(json.dumps(doc), encoding="utf-8")
    return out


def _probe_calls(probes: Path) -> int:
    """How many inputs, placebo repeats included, the battery asks a judge about."""
    total = 0
    for src in probes.glob("*.json"):
        doc = json.loads(src.read_text(encoding="utf-8"))
        if not doc.get("pairwise"):
            total += sum(2 * len(t["inputs"]) - 1 for t in doc["trials"])
    return total


def _bias(state: dict[str, Any], probes: Path, thresholds: Path, *extra: str) -> int:
    argv = ["--root", str(state["root"]), "bias", "--judge", str(state["file"])]
    return main([*argv, "--probes", str(probes), "--thresholds", str(thresholds), *extra])


_THRESHOLDS = _CERTIFY / "bias-thresholds.json"


def test_without_yes_bias_shows_the_estimate_and_sends_nothing(account, tmp_path, capsys) -> None:
    _admit(account, battery=None)
    probes = _probes(tmp_path)

    code = _bias(account, probes, _THRESHOLDS)

    out = capsys.readouterr().out
    assert code == EXIT_OK
    assert f"{_probe_calls(probes)} billed AgentCore Evaluate call(s)" in out
    assert "Nothing was sent" in out
    assert account["calls"] == 0
    assert load_records_of(account["root"], "bias") == []


def test_bias_records_the_battery_and_its_sessions_and_names_the_thresholds_entry(
    account, tmp_path, capsys
) -> None:
    _admit(account, battery=None)
    probes = _probes(tmp_path)

    _bias(account, probes, _THRESHOLDS, "--yes", "--battery-id", "bias-v2-run")

    err = capsys.readouterr().err
    assert account["calls"] == _probe_calls(probes)
    (battery,) = load_records_of(account["root"], "bias")
    assert battery.battery_id == "bias-v2-run"
    assert battery.model == _ARN
    (log,) = load_records_of(account["root"], "agentcore-session")
    assert log.run_kind == "bias"
    assert log.run_id == battery.battery_id
    assert log.evaluator_fingerprint == battery.judge_fingerprint
    assert len(log.session_ids) == _probe_calls(probes)
    assert "bedrock/global.anthropic.claude-haiku-4-5-20251001-v1:0" in err


def test_bias_on_an_uncalibrated_agentcore_judge_is_refused_before_any_probe(
    account, tmp_path, capsys
) -> None:
    code = _bias(account, _probes(tmp_path), _THRESHOLDS, "--yes")

    assert code == EXIT_DOMAIN_ERROR
    assert "not certified" in capsys.readouterr().err
    assert account["calls"] == 0


def test_bias_sends_up_to_the_set_limit_and_prints_progress_to_stderr(
    account, tmp_path, capsys, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("CALIBRATED_AGENT_EVALUATION_AGENTCORE_MAX_CONCURRENCY", "2")
    _admit(account, battery=None)

    _bias(account, _probes(tmp_path), _THRESHOLDS, "--yes")

    captured = capsys.readouterr()
    assert 1 <= account["peak"] <= 2
    assert any(line.startswith("scored ") for line in captured.err.splitlines())
    assert "scored " not in captured.out


def test_bias_refuses_a_concurrency_outside_one_to_ten_before_any_call(
    account, tmp_path, capsys, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("CALIBRATED_AGENT_EVALUATION_AGENTCORE_MAX_CONCURRENCY", "11")
    _admit(account, battery=None)

    code = _bias(account, _probes(tmp_path), _THRESHOLDS, "--yes")

    err = capsys.readouterr().err
    assert code != EXIT_OK
    assert "CALIBRATED_AGENT_EVALUATION_AGENTCORE_MAX_CONCURRENCY" in err
    assert "1 to 10" in err
    assert account["opened"] is False


def test_bias_with_thresholds_for_neither_evaluator_nor_model_is_refused_naming_both(
    account, tmp_path, capsys
) -> None:
    _admit(account, battery=None)
    other = tmp_path / "thresholds.json"
    doc = json.loads(_THRESHOLDS.read_text(encoding="utf-8"))
    doc["by_model"] = {"bedrock/some-other-model": next(iter(doc["by_model"].values()))}
    other.write_text(json.dumps(doc), encoding="utf-8")

    code = _bias(account, _probes(tmp_path), other, "--yes")

    err = capsys.readouterr().err
    assert code == EXIT_DOMAIN_ERROR
    assert _ARN in err
    assert "bedrock/global.anthropic.claude-haiku-4-5-20251001-v1:0" in err
    assert account["calls"] == 0


def test_calibrate_records_the_gold_source_in_its_session_log(account) -> None:
    _calibrate(account, "--yes")

    (log,) = load_records_of(account["root"], "agentcore-session")
    assert log.gold_source == "synthetic-rule"


def test_a_bias_runs_session_log_records_no_gold_source(tmp_path: Path) -> None:
    from types import SimpleNamespace
    from typing import cast

    from calibrated_agent_evaluation.cli.stages.calibrate_agentcore import AgentCoreCalibrateRun
    from cli_store_helpers import clear_battery

    judge = SimpleNamespace(judge_id="orderbot-refund-policy", fingerprint="0123456789abcdef")
    run = AgentCoreCalibrateRun(
        args=cast("Any", None),
        judge_file=cast("Any", SimpleNamespace(judge=judge)),
        clients=cast("Any", None),
        session_ids=("b-1",),
        gold_source="synthetic-rule",
    )
    run.record_sessions(tmp_path, clear_battery())

    (log,) = load_records_of(tmp_path, "agentcore-session")
    assert log.run_kind == "bias"
    assert log.gold_source is None
