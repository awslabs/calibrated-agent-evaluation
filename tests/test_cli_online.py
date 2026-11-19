# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""``calibrated-agent-evaluation online create-evaluator`` against a stubbed AgentCore account."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

pytest.importorskip("boto3")

import boto3
from botocore.stub import ANY, Stubber

from calibrated_agent_evaluation.cli import main
from calibrated_agent_evaluation.cli.dispatch import EXIT_DOMAIN_ERROR, EXIT_OK, EXIT_USAGE
from test_agentcore_evaluator_create import _ARN, _ID, _NAME, _UPDATED, _judge, _request

_JUDGE_V2 = (
    Path(__file__).resolve().parents[1]
    / "src/calibrated_agent_evaluation/_quickstart/data/certify/judge-v2.json"
)


@pytest.fixture
def account(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    """Route the CLI's AWS clients to a stubbed account, keeping the guards the CLI installs."""
    from calibrated_agent_evaluation.agentcore.session import AwsClients

    for var in ("AWS_PROFILE", "AWS_REGION", "AWS_DEFAULT_REGION"):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setenv("AWS_CONFIG_FILE", str(tmp_path / "no-aws-config"))
    session = boto3.Session(
        region_name="us-east-1", aws_access_key_id="testing", aws_secret_access_key="testing"
    )
    holder: dict[str, Any] = {}

    def from_profile(cls, *, region=None, profile=None, handlers=(), **_kw):
        clients = AwsClients(session, handlers=handlers)
        stub = Stubber(clients.client("bedrock-agentcore-control"))
        for queue in holder["queue"]:
            queue(stub)
        stub.activate()
        holder["stub"] = stub
        return clients

    monkeypatch.setattr(
        "calibrated_agent_evaluation.agentcore.session.AwsClients.from_profile",
        classmethod(from_profile),
    )
    monkeypatch.chdir(tmp_path)
    holder["queue"] = []
    return holder


def _list(*names: str):
    def queue(stub: Stubber) -> None:
        evaluators = [
            {
                "evaluatorArn": f"{_ARN.rsplit('/', 1)[0]}/{n}-AbCdEf1234",
                "evaluatorId": f"{n}-AbCdEf1234",
                "evaluatorName": n,
                "evaluatorType": "Custom",
                "status": "ACTIVE",
                "createdAt": _UPDATED,
                "updatedAt": _UPDATED,
            }
            for n in names
        ]
        stub.add_response("list_evaluators", {"evaluators": evaluators}, {})

    return queue


def _create_and_activate(stub: Stubber) -> None:
    stub.add_response(
        "create_evaluator",
        {"evaluatorArn": _ARN, "evaluatorId": _ID, "createdAt": _UPDATED, "status": "CREATING"},
        {**_request(_judge()), "clientToken": ANY, "description": ANY},
    )
    stub.add_response(
        "get_evaluator",
        {
            "evaluatorArn": _ARN,
            "evaluatorId": _ID,
            "evaluatorName": _NAME,
            "evaluatorConfig": _request()["evaluatorConfig"],
            "evaluatorType": "Custom",
            "level": "TRACE",
            "status": "ACTIVE",
            "createdAt": _UPDATED,
            "updatedAt": _UPDATED,
        },
        {"evaluatorId": _ID},
    )


def test_without_yes_it_shows_the_request_and_sends_nothing(account, tmp_path, capsys) -> None:
    account["queue"] = [_list()]

    code = main(["online", "create-evaluator", "--judge", str(_JUDGE_V2)])

    out = capsys.readouterr().out
    assert code == EXIT_OK
    assert "CreateEvaluator" in out
    assert "{context}" in out
    assert "{assistant_turn}" in out
    assert '"level": "TRACE"' in out
    assert "global.anthropic.claude-haiku-4-5-20251001-v1:0" in out
    assert "Nothing was sent" in out
    account["stub"].assert_no_pending_responses()
    assert not (tmp_path / "judge-v2.agentcore.json").exists()


def test_with_yes_it_creates_the_evaluator_and_writes_its_judge_file(
    account, tmp_path, capsys
) -> None:
    account["queue"] = [_list(), _create_and_activate]

    code = main(["online", "create-evaluator", "--judge", str(_JUDGE_V2), "--yes"])

    out = capsys.readouterr().out
    assert code == EXIT_OK
    assert f"Created the AgentCore evaluator {_NAME} ({_ID})" in out
    assert "CreateEvaluator" in out  # the approved run shows the request it sent, too
    assert "{assistant_turn}" in out
    written = json.loads((tmp_path / "judge-v2.agentcore.json").read_text(encoding="utf-8"))
    assert written["backend"] == "agentcore"
    assert written["evaluator_id"] == _ID
    assert "calibrated-agent-evaluation judge --check" in out
    account["stub"].assert_no_pending_responses()


def test_an_evaluator_name_in_use_is_refused(account, tmp_path, capsys) -> None:
    account["queue"] = [_list(_NAME)]

    code = main(["online", "create-evaluator", "--judge", str(_JUDGE_V2), "--yes"])

    assert code == EXIT_DOMAIN_ERROR
    assert f"evaluator {_NAME} already exists" in capsys.readouterr().err
    assert not (tmp_path / "judge-v2.agentcore.json").exists()


def test_an_agentcore_judge_file_cannot_be_created_again(account, tmp_path, capsys) -> None:
    agentcore_file = tmp_path / "already.json"
    agentcore_file.write_text(
        json.dumps(
            {
                "backend": "agentcore",
                "judge_id": "x",
                "evaluator_id": "Builtin.Correctness",
                "failure_mode": "m",
            }
        ),
        encoding="utf-8",
    )

    code = main(["online", "create-evaluator", "--judge", str(agentcore_file)])

    assert code == EXIT_DOMAIN_ERROR
    assert "AgentCore judge" in capsys.readouterr().err


def test_create_evaluator_needs_a_judge_file(account, capsys) -> None:
    assert main(["online", "create-evaluator"]) == EXIT_USAGE
    assert "needs --judge" in capsys.readouterr().err


def test_a_span_level_judge_is_refused_before_any_call(account, tmp_path, capsys) -> None:
    doc = json.loads(_JUDGE_V2.read_text(encoding="utf-8"))
    span = tmp_path / "span.json"
    span.write_text(json.dumps({**doc, "level": "span", "agent": "orderbot"}), encoding="utf-8")

    code = main(["online", "create-evaluator", "--judge", str(span), "--yes"])

    assert code == EXIT_DOMAIN_ERROR
    assert "span-level judge" in capsys.readouterr().err
    assert "stub" not in account  # no AWS clients were built


def test_a_wait_that_is_not_positive_is_refused(account, capsys) -> None:
    code = main(["online", "create-evaluator", "--judge", str(_JUDGE_V2), "--wait", "0"])

    assert code == EXIT_USAGE
    assert "--wait must be a positive" in capsys.readouterr().err


# --- calibrated-agent-evaluation online attach
# -------------------------------------------------------------------------


@pytest.fixture
def attach(account, monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    """A store holding a certified, bias-cleared judge with both session logs, and its file."""
    from decimal import Decimal

    from calibrated_agent_evaluation.agentcore.pricing import EvaluationRates
    from calibrated_agent_evaluation.cli.records import append_records
    from test_agentcore_online import _EVALUATOR_ID, _online_judge, _session_logs

    for var in (
        "CALIBRATED_AGENT_EVALUATION_AGENTCORE_RUNTIME_ARN",
        "CALIBRATED_AGENT_EVALUATION_AGENTCORE_EVAL_ROLE_ARN",
    ):
        monkeypatch.delenv(var, raising=False)
    rates = EvaluationRates(region="us-east-1", custom_per_evaluation=Decimal("0.0015"))
    monkeypatch.setattr(
        "calibrated_agent_evaluation.agentcore.pricing.lookup_evaluation_rates", lambda _c: rates
    )
    root = tmp_path / "root"
    online = _online_judge()
    append_records(root, "calibration", [online.calib])
    append_records(root, "bias", [online.battery])
    append_records(root, "agentcore-session", _session_logs(online))
    judge_file = tmp_path / "refunds.agentcore.json"
    judge_file.write_text(
        json.dumps(
            {
                "backend": "agentcore",
                "judge_id": "refunds",
                "evaluator_id": _EVALUATOR_ID,
                "failure_mode": "wrong_refund",
                "level": "workflow",
            }
        ),
        encoding="utf-8",
    )
    account["root"] = root
    account["file"] = judge_file
    account["online"] = online
    return account


def _evaluator_read(level: str = "TRACE"):
    from test_agentcore_online import _EVALUATOR_ID, _evaluator_doc

    def queue(stub: Stubber) -> None:
        stub.add_response(
            "get_evaluator", {**_evaluator_doc(), "level": level}, {"evaluatorId": _EVALUATOR_ID}
        )

    return queue


def _attach_argv(state: dict[str, Any], *extra: str) -> list[str]:
    from test_agentcore_online import _ROLE, _RUNTIME_ARN

    return [
        "--root",
        str(state["root"]),
        "online",
        "attach",
        "--judge",
        str(state["file"]),
        "--name",
        "calibrated_agent_evaluation_live_attach",
        "--runtime-arn",
        _RUNTIME_ARN,
        "--role-arn",
        _ROLE,
        *extra,
    ]


def test_without_yes_attach_shows_the_derived_config_and_creates_nothing(attach, capsys) -> None:
    from test_agentcore_online import _RUNTIME_ID, _listed, _runtime

    attach["queue"] = [_evaluator_read(), _runtime, lambda s: _listed(s, [])]

    code = main(
        _attach_argv(attach, "--sampling", "100", "--filter", "gen_ai.agent.name", "Equals", "x")
    )

    out = capsys.readouterr().out
    assert code == EXIT_OK
    assert "calibrated_agent_evaluation_orderbot.DEFAULT" in out
    assert f"/aws/bedrock-agentcore/runtimes/{_RUNTIME_ID}-DEFAULT" in out
    assert '"samplingPercentage": 100.0' in out
    assert '"sessionTimeoutMinutes": 15' in out
    assert '"stringValue": "x"' in out
    assert "$0.0015" in out
    assert "Nothing was sent" in out
    attach["stub"].assert_no_pending_responses()


def test_attach_creates_the_config_and_stores_its_record(attach, capsys) -> None:
    from calibrated_agent_evaluation.cli.records import load_records_of
    from test_agentcore_online import _CONFIG_ID, _create, _listed, _runtime, _status

    attach["queue"] = [
        _evaluator_read(),
        _runtime,
        lambda s: _listed(s, []),
        _create,
        lambda s: _status(s, "ACTIVE"),
    ]

    code = main(_attach_argv(attach, "--sampling", "100", "--yes"))

    assert code == EXIT_OK, capsys.readouterr().err
    (record,) = load_records_of(attach["root"], "agentcore-online")
    assert record.config_id == _CONFIG_ID
    (attached,) = record.evaluators
    assert attached.fingerprint == attach["online"].judge.fingerprint
    assert record.excluded_session_ids == ("c1", "c2", "b1")


@pytest.mark.parametrize("level", ["SESSION", "TOOL_CALL"])
def test_attach_refuses_an_evaluator_that_is_not_trace_level_before_any_write(
    attach, capsys, level: str
) -> None:
    attach["queue"] = [_evaluator_read(level)]

    code = main(_attach_argv(attach, "--sampling", "100", "--yes"))

    assert code == EXIT_DOMAIN_ERROR
    assert level in capsys.readouterr().err
    attach["stub"].assert_no_pending_responses()


@pytest.mark.parametrize(
    ("drop", "says"),
    [
        ("--runtime-arn", "CALIBRATED_AGENT_EVALUATION_AGENTCORE_RUNTIME_ARN"),
        ("--role-arn", "CALIBRATED_AGENT_EVALUATION_AGENTCORE_EVAL_ROLE_ARN"),
    ],
)
def test_a_missing_runtime_or_role_is_refused_naming_its_flag_and_variable(
    attach, capsys, drop: str, says: str
) -> None:
    argv = _attach_argv(attach, "--sampling", "100")
    index = argv.index(drop)
    del argv[index : index + 2]

    code = main(argv)

    err = capsys.readouterr().err
    assert code == EXIT_USAGE
    assert drop in err
    assert says in err
    assert "stub" not in attach


@pytest.mark.parametrize(
    ("extra", "says", "expected"),
    [
        # A missing flag is a usage error; a value the config builder refuses is a refusal.
        ([], "--sampling", EXIT_USAGE),
        (["--sampling", "0"], "0.0", EXIT_DOMAIN_ERROR),
        (["--sampling", "150"], "150", EXIT_DOMAIN_ERROR),
        (["--sampling", "100", "--session-timeout", "2000"], "2000", EXIT_DOMAIN_ERROR),
        (["--sampling", "100", "--filter", "k", "Like", "v"], "Like", EXIT_DOMAIN_ERROR),
        (["--sampling", "100", "--filter", "k", "Equals", "nan"], "nan", EXIT_DOMAIN_ERROR),
        (
            ["--sampling", "100", *(["--filter", "k", "Equals", "v"] * 6)],
            "five",
            EXIT_DOMAIN_ERROR,
        ),
    ],
    ids=["no-sampling", "sampling-0", "sampling-150", "timeout", "operator", "nan", "six-filters"],
)
def test_a_bad_attach_value_is_refused_before_any_aws_call(
    attach, capsys, extra: list[str], says: str, expected: int
) -> None:
    code = main(_attach_argv(attach, *extra))

    captured = capsys.readouterr()
    assert code == expected
    assert says in captured.err
    assert "CreateOnlineEvaluationConfig" not in captured.out
    assert "stub" not in attach


def test_a_name_agentcore_would_refuse_is_refused_before_any_aws_call(attach, capsys) -> None:
    argv = _attach_argv(attach, "--sampling", "100")
    argv[argv.index("calibrated_agent_evaluation_live_attach")] = "1st_config"

    code = main(argv)

    assert code == EXIT_DOMAIN_ERROR
    assert "1st_config" in capsys.readouterr().err
    assert "stub" not in attach


@pytest.mark.parametrize(
    ("value", "typed"),
    [("3", '"doubleValue": 3.0'), ("true", '"booleanValue": true'), ("x", '"stringValue": "x"')],
    ids=["number", "boolean", "string"],
)
def test_a_filter_value_is_sent_as_a_number_a_boolean_or_a_string(
    attach, capsys, value: str, typed: str
) -> None:
    from test_agentcore_online import _listed, _runtime

    attach["queue"] = [_evaluator_read(), _runtime, lambda s: _listed(s, [])]

    main(_attach_argv(attach, "--sampling", "100", "--filter", "k", "GreaterThan", value))

    assert typed in capsys.readouterr().out


@pytest.mark.parametrize("kind", ["missing", "prompt"])
def test_attach_reports_a_judge_file_it_cannot_use_before_any_aws_call(
    attach, tmp_path, capsys, kind: str
) -> None:
    path = tmp_path / "absent.json" if kind == "missing" else _JUDGE_V2
    argv = _attach_argv(attach, "--sampling", "100")
    argv[argv.index(str(attach["file"]))] = str(path)

    code = main(argv)

    err = capsys.readouterr().err
    assert code == EXIT_DOMAIN_ERROR
    assert str(path) in err
    assert ("does not exist" if kind == "missing" else "not an agentcore judge file") in err
    assert "stub" not in attach


# --- calibrated-agent-evaluation online read / publish / cost
# --------------------------------------------------------

_RESULTS_GROUP = (
    "/aws/bedrock-agentcore/evaluations/results/calibrated_agent_evaluation_live_attach-AbCdEf1234"
)
_WINDOW = ("--since", "2026-11-05T18:00:00+00:00", "--until", "2026-11-05T19:00:00+00:00")


@pytest.fixture
def reading(attach, monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    """The attach store plus the config's record, with each AWS service stubbed from a queue.

    ``state["traces"]`` is what the span reader returns; ``state["results"]`` the records the
    config's result group holds; ``state["control"]`` and ``state["cloudwatch"]`` the stubbed
    replies of those services, in order.
    """
    from decimal import Decimal

    from calibrated_agent_evaluation.agentcore.online_record import (
        AttachedEvaluator,
        OnlineEvaluationRecord,
    )
    from calibrated_agent_evaluation.agentcore.pricing import EvaluationRates
    from calibrated_agent_evaluation.agentcore.session import AwsClients
    from calibrated_agent_evaluation.cli.records import append_records
    from test_agentcore_online import _CONFIG_ID, _EVALUATOR_ARN, _EVALUATOR_ID, _WHEN
    from test_agentcore_readback import _Reader, _rows

    online = attach["online"]
    record = OnlineEvaluationRecord(
        config_id=_CONFIG_ID,
        config_name="calibrated_agent_evaluation_live_attach",
        runtime_service_name="calibrated_agent_evaluation_orderbot.DEFAULT",
        evaluators=(
            AttachedEvaluator(
                evaluator_id=_EVALUATOR_ID,
                evaluator_arn=_EVALUATOR_ARN,
                updated_at=online.judge.updated_at,
                fingerprint=online.judge.fingerprint,
                calib_id=online.calib.calib_id,
                battery_id=online.battery.battery_id,
            ),
        ),
        excluded_session_ids=("c1", "c2", "b1"),
        sampling_percentage=100.0,
        session_timeout_minutes=2,
        created_at=_WHEN,
    )
    append_records(attach["root"], "agentcore-online", [record])
    monkeypatch.setenv("CALIBRATED_AGENT_EVALUATION_AGENTCORE_SETTLE_SECONDS", "0")
    rates = EvaluationRates(region="us-east-1", custom_per_evaluation=Decimal("0.0015"))
    monkeypatch.setattr(
        "calibrated_agent_evaluation.agentcore.pricing.lookup_evaluation_rates", lambda _c: rates
    )
    state: dict[str, Any] = {
        **attach,
        "record": record,
        "traces": [],
        "results": [],
        "control": [],
        "cloudwatch": [],
        "opened": False,
        "handlers": [],
    }
    session = boto3.Session(
        region_name="us-east-1", aws_access_key_id="testing", aws_secret_access_key="testing"
    )

    def from_profile(cls, *, region=None, profile=None, handlers=(), **_kw):
        state["opened"] = True
        state["handlers"] = list(handlers)
        clients = AwsClients(session, handlers=handlers)
        control = Stubber(clients.client("bedrock-agentcore-control"))
        for queue in state["control"]:
            queue(control)
        control.activate()
        logs = Stubber(clients.client("logs"))
        logs.add_response("start_query", {"queryId": "q-1"})
        logs.add_response(
            "get_query_results",
            {
                "status": "Complete",
                "results": _rows(state["results"]),
                "statistics": {"recordsMatched": float(len(state["results"]))},
            },
        )
        if not state["results"]:
            logs.add_response("filter_log_events", {"events": []})
        logs.activate()
        cloudwatch = Stubber(clients.client("cloudwatch"))
        for queue in state["cloudwatch"]:
            queue(cloudwatch)
        cloudwatch.activate()
        return clients

    monkeypatch.setattr(
        "calibrated_agent_evaluation.agentcore.session.AwsClients.from_profile",
        classmethod(from_profile),
    )
    monkeypatch.setattr(
        "calibrated_agent_evaluation.agentcore.readers.LogsInsightsReader",
        lambda clients, config: _Reader(state["traces"]),
    )
    return state


def _evaluator_doc_queue(document: dict[str, Any] | None = None, *, times: int = 2):
    from test_agentcore_online import _EVALUATOR_ID, _evaluator_doc

    def queue(stub: Stubber) -> None:
        for _ in range(times):
            stub.add_response(
                "get_evaluator", document or _evaluator_doc(), {"evaluatorId": _EVALUATOR_ID}
            )

    return queue


def _online_argv(state: dict[str, Any], action: str, *extra: str) -> list[str]:
    return ["--root", str(state["root"]), "online", action, "--judge", str(state["file"]), *extra]


def _traffic(state: dict[str, Any], count: int, *, message: str = "a refund for Q-30000") -> None:
    """``count`` scored OrderBot sessions: their traces, and a FAIL then PASS result for each."""
    from test_agentcore_online import _EVALUATOR_ARN
    from test_agentcore_readback import _raw, _result

    for i in range(count):
        trace_id = f"{i + 1:032x}"
        raw = _raw(trace_id, f"s-{i}")
        for record in raw.log_records:
            if record["eventName"] == "gen_ai.user.message":
                record["body"] = {"content": [{"text": message}]}
        state["traces"].append(raw)
        result = _result(trace_id, f"s-{i}", "FAIL" if i % 5 == 0 else "PASS")
        result["attributes"]["aws.bedrock_agentcore.evaluator.arn"] = _EVALUATOR_ARN
        state["results"].append(result)


def test_read_stores_the_verdicts_and_a_read_record_naming_each_one(reading, capsys) -> None:
    from calibrated_agent_evaluation.backends.local import LocalBackend
    from calibrated_agent_evaluation.cli.records import load_records_of

    _traffic(reading, 3)
    reading["control"] = [_evaluator_doc_queue()]

    code = main(_online_argv(reading, "read", *_WINDOW))

    out = capsys.readouterr()
    assert code == EXIT_OK, out.err
    assert "3 verdict(s)" in out.out
    (read,) = load_records_of(reading["root"], "agentcore-read")
    assert read.config_id == reading["record"].config_id
    assert read.backend == "local"
    assert (read.since, read.until) == (_WINDOW[1], _WINDOW[3])
    with LocalBackend(root=str(reading["root"])) as store:
        stored = {t: store.get_scores(t) for t in (f"{i + 1:032x}" for i in range(3))}
    assert {v.trace_id for v in read.verdicts} == set(stored)
    assert all(len(labels) == 1 for labels in stored.values())


def test_a_second_read_of_the_same_window_names_the_verdicts_it_found_already_stored(
    reading, capsys
) -> None:
    from calibrated_agent_evaluation.cli.records import load_records_of

    _traffic(reading, 2)
    reading["control"] = [_evaluator_doc_queue()]
    assert main(_online_argv(reading, "read", *_WINDOW)) == EXIT_OK
    reading["control"] = [_evaluator_doc_queue()]

    assert main(_online_argv(reading, "read", *_WINDOW)) == EXIT_OK

    first, second = load_records_of(reading["root"], "agentcore-read")
    assert first.read_id != second.read_id
    assert {v.verdict_digest for v in second.verdicts} == {v.verdict_digest for v in first.verdicts}
    assert "2 already stored" in capsys.readouterr().out


def test_read_names_no_error_result_and_no_session_calibrated_agent_evaluation_sent(
    reading,
) -> None:
    from calibrated_agent_evaluation.cli.records import load_records_of
    from test_agentcore_readback import _error, _raw, _result

    _traffic(reading, 1)
    reading["traces"].append(_raw("e" * 32, "s-err"))
    reading["results"].append(_error("e" * 32, "s-err", "ValidationException"))
    reading["traces"].append(_raw("c" * 32, "c1"))  # a calibration session
    reading["results"].append(_result("c" * 32, "c1"))
    reading["control"] = [_evaluator_doc_queue()]

    assert main(_online_argv(reading, "read", *_WINDOW)) == EXIT_OK

    (read,) = load_records_of(reading["root"], "agentcore-read")
    assert [v.trace_id for v in read.verdicts] == [f"{1:032x}"]
    assert read.errors == {"ValidationException": 1}


def test_read_without_the_drift_flags_says_drift_was_not_measured(reading, capsys) -> None:
    _traffic(reading, 1)
    reading["control"] = [_evaluator_doc_queue()]

    assert main(_online_argv(reading, "read", *_WINDOW)) == EXIT_OK

    out = capsys.readouterr().out
    assert "drift was not measured" in out
    assert "--drift-baseline" in out
    assert "--drift-feature" in out


def test_one_drift_flag_alone_is_refused_before_any_aws_call(reading, capsys) -> None:
    code = main(_online_argv(reading, "read", *_WINDOW, "--drift-feature", "input-chars"))

    assert code != EXIT_OK
    assert "--drift-baseline" in capsys.readouterr().err
    assert not reading["opened"]


def _baseline(tmp_path: Path, reading: dict[str, Any], *, message: str) -> Path:
    """A store whose production-representative traces were read back like the fresh ones."""
    import copy

    saved = (reading["traces"], reading["results"], reading["root"])
    baseline = tmp_path / "baseline"
    import shutil

    shutil.copytree(reading["root"], baseline)
    reading["traces"], reading["results"], reading["root"] = [], [], baseline
    _traffic(reading, 60, message=message)
    reading["control"] = [_evaluator_doc_queue()]
    assert main(_online_argv(reading, "read", *_WINDOW)) == EXIT_OK
    reading["traces"], reading["results"], reading["root"] = copy.copy(saved)
    return baseline


def test_read_flags_drift_against_a_baseline_whose_inputs_differ_and_states_m(
    reading, tmp_path, capsys
) -> None:
    baseline = _baseline(tmp_path, reading, message="a refund for Q-30000")
    capsys.readouterr()
    _traffic(reading, 60, message="a refund for Q-30000 " + "because it never arrived " * 20)
    reading["control"] = [_evaluator_doc_queue()]

    code = main(
        _online_argv(
            reading,
            "read",
            *_WINDOW,
            "--drift-baseline",
            str(baseline),
            "--drift-feature",
            "input-chars",
            "--explain",
        )
    )

    out = capsys.readouterr().out
    assert code == EXIT_OK
    assert "the current traces differ from the baseline" in out
    assert "m = 2" in out
    # The flagged drift is a staleness trigger, as on calibrated-agent-evaluation run.
    assert "may need recalibrating: its inputs have drifted" in out


def test_read_flags_no_drift_against_a_baseline_like_the_traffic(reading, tmp_path, capsys) -> None:
    baseline = _baseline(tmp_path, reading, message="a refund for Q-30000")
    capsys.readouterr()
    _traffic(reading, 60, message="a refund for Q-30000")
    reading["control"] = [_evaluator_doc_queue()]

    code = main(
        _online_argv(
            reading,
            "read",
            *_WINDOW,
            "--drift-baseline",
            str(baseline),
            "--drift-feature",
            "input-chars",
        )
    )

    out = capsys.readouterr().out
    assert code == EXIT_OK
    assert "Input drift on input-chars: none found" in out
    assert "may need recalibrating" not in out


def test_a_read_back_too_small_for_drift_says_so_and_how_many_traces_each_side_has(
    reading, tmp_path, capsys
) -> None:
    baseline = _baseline(tmp_path, reading, message="a refund for Q-30000")
    capsys.readouterr()
    _traffic(reading, 3)
    reading["control"] = [_evaluator_doc_queue()]

    code = main(
        _online_argv(
            reading,
            "read",
            *_WINDOW,
            "--drift-baseline",
            str(baseline),
            "--drift-feature",
            "input-chars",
        )
    )

    out = capsys.readouterr().out
    assert code == EXIT_OK
    assert "not measured" in out
    assert "3 now" in out


def test_a_changed_evaluator_gets_the_staleness_warning_and_its_verdicts_are_not_stored(
    reading, capsys
) -> None:
    from test_agentcore_online import _evaluator_doc

    edited = _evaluator_doc()
    llm = dict(edited["evaluatorConfig"]["llmAsAJudge"])
    llm["instructions"] = "Judge the refund strictly. {context} {assistant_turn}"
    edited["evaluatorConfig"] = {"llmAsAJudge": llm}
    _traffic(reading, 2)
    reading["control"] = [_evaluator_doc_queue(edited)]

    assert main(_online_argv(reading, "read", *_WINDOW)) == EXIT_OK

    out = capsys.readouterr().out
    assert "changed since it was certified" in out
    assert "may need recalibrating: it no longer matches its calibration record" in out
    assert "0 verdict(s)" in out


def test_a_missing_evaluator_gets_a_not_checkable_staleness_line(reading, capsys) -> None:
    def gone(stub: Stubber) -> None:
        stub.add_client_error(
            "get_evaluator", service_error_code="ResourceNotFoundException", http_status_code=404
        )

    _traffic(reading, 1)
    reading["control"] = [gone]

    assert main(_online_argv(reading, "read", *_WINDOW)) == EXIT_OK

    out = capsys.readouterr().out
    assert "no longer exists" in out
    assert "not checkable" in out


@pytest.mark.parametrize("action", ["read", "publish"])
def test_judge_files_that_do_not_match_the_attached_evaluators_are_refused(
    reading, tmp_path, action, capsys
) -> None:
    other = tmp_path / "other.agentcore.json"
    other.write_text(
        json.dumps(
            {
                "backend": "agentcore",
                "judge_id": "refunds",
                "evaluator_id": "calibrated_agent_evaluation_live_other-AbCdEf1234",
                "failure_mode": "wrong_refund",
            }
        ),
        encoding="utf-8",
    )
    renamed = tmp_path / "renamed.agentcore.json"
    doc = json.loads(Path(reading["file"]).read_text(encoding="utf-8"))
    renamed.write_text(json.dumps({**doc, "judge_id": "someone-else"}), encoding="utf-8")
    base = ["--root", str(reading["root"]), "online", action]

    # An attached evaluator left unnamed.
    assert main([*base, "--judge", str(other)]) != EXIT_OK
    assert "calibrated_agent_evaluation_live_refunds-AbCdEf1234" in capsys.readouterr().err
    # An evaluator the config did not attach.
    assert main([*base, "--judge", str(reading["file"]), "--judge", str(other)]) != EXIT_OK
    assert "calibrated_agent_evaluation_live_other-AbCdEf1234" in capsys.readouterr().err
    # The attached evaluator under another judge id than its calibration record's.
    assert main([*base, "--judge", str(renamed)]) != EXIT_OK
    assert "someone-else" in capsys.readouterr().err
    assert not reading["opened"]


def test_publish_with_no_read_record_is_refused_naming_online_read(reading, capsys) -> None:
    code = main(_online_argv(reading, "publish"))

    assert code != EXIT_OK
    assert "calibrated-agent-evaluation online read" in capsys.readouterr().err
    assert not reading["opened"]


def _put(stub: Stubber) -> None:
    stub.add_response("put_metric_data", {})


def test_publish_corrects_over_the_named_verdicts_only_and_publishes_three_metrics(
    reading, capsys
) -> None:
    from calibrated_agent_evaluation.backends.local import LocalBackend

    _traffic(reading, 160)
    reading["control"] = [_evaluator_doc_queue()]
    assert main(_online_argv(reading, "read", *_WINDOW)) == EXIT_OK
    # The same judge's on-demand verdicts, with its rater id: on a pulled trace, and elsewhere.
    from calibrated_agent_evaluation.agentcore.online_record import (
        OnlineReadRecord,
        ReadVerdict,
        verdict_digest,
    )
    from calibrated_agent_evaluation.cli.records import append_records
    from test_agentcore_online import _CONFIG_ID
    from test_cli_agentcore_judge import _trace as _plain_trace

    with LocalBackend(root=str(reading["root"])) as store:
        (online,) = store.get_scores(f"{1:032x}")
        on_demand = online.model_copy(update={"outcome": "PASS", "critique": "on-demand"})
        store.put_score(on_demand)  # on a trace the read pulled
        elsewhere = on_demand.model_copy(update={"trace_id": "a" * 32})
        store.put_trace(_plain_trace("a" * 32))
        store.put_score(elsewhere)  # on a trace no read pulled
    # Another evaluator's verdict on that trace, named by a read record: still not this judge's.
    append_records(
        reading["root"],
        "agentcore-read",
        [
            OnlineReadRecord(
                read_id=f"{_CONFIG_ID}:other",
                config_id=_CONFIG_ID,
                backend="local",
                since=_WINDOW[1],
                until=_WINDOW[3],
                verdicts=(
                    ReadVerdict(
                        evaluator_id="calibrated_agent_evaluation_live_other-AbCdEf1234",
                        trace_id="a" * 32,
                        verdict_digest=verdict_digest(elsewhere),
                    ),
                ),
            )
        ],
    )
    capsys.readouterr()
    reading["control"] = [_evaluator_doc_queue()]
    reading["cloudwatch"] = [_put]

    code = main(_online_argv(reading, "publish"))

    out = capsys.readouterr()
    assert code == EXIT_OK, out.err
    assert "160 online verdict(s)" in out.out
    # Corrected over the 32 FAIL and 128 PASS online verdicts: the on-demand PASS on the first
    # (FAIL) trace would have moved the rate had it been counted.
    from calibrated_agent_evaluation.calibrate.engine import corrected_rate

    scored = corrected_rate(
        reading["online"].calib,
        [1 if i % 5 == 0 else 0 for i in range(160)],
        f"online-{reading['record'].config_id}",
        rate_trace_ids=[f"{i + 1:032x}" for i in range(160)],
    ).corrected_rate
    assert scored is not None
    assert scored.point is not None
    assert f"CorrectedPassRate {1 - scored.point:.3f}" in out.out
    for name in ("CorrectedPassRate", "CorrectedPassRateLower95", "CorrectedPassRateUpper95"):
        assert name in out.out


def test_publish_refuses_a_named_verdict_the_store_no_longer_holds(reading, capsys) -> None:
    from calibrated_agent_evaluation.agentcore.online_record import OnlineReadRecord, ReadVerdict
    from calibrated_agent_evaluation.cli.records import append_records
    from test_agentcore_online import _CONFIG_ID, _EVALUATOR_ID

    append_records(
        reading["root"],
        "agentcore-read",
        [
            OnlineReadRecord(
                read_id=f"{_CONFIG_ID}:1",
                config_id=_CONFIG_ID,
                backend="local",
                since=_WINDOW[1],
                until=_WINDOW[3],
                verdicts=(
                    ReadVerdict(
                        evaluator_id=_EVALUATOR_ID, trace_id="f" * 32, verdict_digest="0" * 16
                    ),
                ),
            )
        ],
    )
    reading["control"] = [_evaluator_doc_queue(times=1)]

    code = main(_online_argv(reading, "publish"))

    assert code != EXIT_OK
    assert "f" * 32 in capsys.readouterr().err


def test_publish_uses_only_the_read_records_for_the_backend_it_is_given(reading, capsys) -> None:
    from calibrated_agent_evaluation.agentcore.online_record import OnlineReadRecord
    from calibrated_agent_evaluation.cli.records import append_records
    from test_agentcore_online import _CONFIG_ID

    append_records(
        reading["root"],
        "agentcore-read",
        [
            OnlineReadRecord(
                read_id=f"{_CONFIG_ID}:lf",
                config_id=_CONFIG_ID,
                backend="langfuse",
                since=_WINDOW[1],
                until=_WINDOW[3],
                verdicts=(),
            )
        ],
    )

    code = main(_online_argv(reading, "publish"))

    assert code != EXIT_OK
    assert "calibrated-agent-evaluation online read" in capsys.readouterr().err


def test_cost_counts_the_runs_and_the_results_and_its_clients_allow_only_its_calls(
    reading, capsys
) -> None:
    from test_agentcore_readback import _error

    _traffic(reading, 2)
    reading["results"].append(_error("e" * 32, "s-err", "ValidationException"))

    code = main(["--root", str(reading["root"]), "online", "cost"])

    out = capsys.readouterr().out
    assert code == EXIT_OK
    assert "2 scored online result(s)" in out
    assert "1 error result(s), not priced" in out
    assert "not included" in out
    (guard,) = [h for h in reading["handlers"] if hasattr(h, "allowed")]
    assert guard.allowed == {
        ("logs", "StartQuery"),
        ("logs", "GetQueryResults"),
        ("logs", "FilterLogEvents"),
        ("pricing", "GetProducts"),
    }


def _filtered(reading: dict[str, Any]) -> None:
    import shutil

    from calibrated_agent_evaluation.cli.records import append_records, records_dir

    shutil.rmtree(records_dir(reading["root"], "agentcore-online"))
    append_records(
        reading["root"],
        "agentcore-online",
        [reading["record"].model_copy(update={"filtered": True})],
    )


def test_a_filtered_config_read_with_the_drift_flags_is_refused_before_any_aws_call(
    reading, tmp_path, capsys
) -> None:
    _filtered(reading)

    code = main(
        _online_argv(
            reading,
            "read",
            *_WINDOW,
            "--drift-baseline",
            str(tmp_path),
            "--drift-feature",
            "input-chars",
        )
    )

    assert code != EXIT_OK
    assert "not a random sample" in capsys.readouterr().err
    assert not reading["opened"]


def test_a_filtered_config_read_without_the_drift_flags_says_drift_is_not_measured_on_it(
    reading, capsys
) -> None:
    _filtered(reading)
    _traffic(reading, 1)
    reading["control"] = [_evaluator_doc_queue()]

    assert main(_online_argv(reading, "read", *_WINDOW)) == EXIT_OK

    out = capsys.readouterr().out
    assert "not measured on a filtered config's traffic" in out
    assert "pass --drift-baseline" not in out


def test_a_baseline_with_no_random_sample_is_refused_before_any_aws_call(
    reading, tmp_path, capsys
) -> None:
    empty = tmp_path / "empty-baseline"
    empty.mkdir()

    code = main(
        _online_argv(
            reading,
            "read",
            *_WINDOW,
            "--drift-baseline",
            str(empty),
            "--drift-feature",
            "input-chars",
        )
    )

    assert code != EXIT_OK
    assert "production-representative" in capsys.readouterr().err
    assert not reading["opened"]


def test_a_changed_evaluator_read_with_the_drift_flags_still_gets_its_staleness_warning(
    reading, tmp_path, capsys
) -> None:
    from test_agentcore_online import _evaluator_doc

    baseline = _baseline(tmp_path, reading, message="a refund for Q-30000")
    capsys.readouterr()
    edited = _evaluator_doc()
    llm = dict(edited["evaluatorConfig"]["llmAsAJudge"])
    llm["instructions"] = "Judge the refund strictly. {context} {assistant_turn}"
    edited["evaluatorConfig"] = {"llmAsAJudge": llm}
    _traffic(reading, 60, message="a refund for Q-30000")
    reading["control"] = [_evaluator_doc_queue(edited)]

    code = main(
        _online_argv(
            reading,
            "read",
            *_WINDOW,
            "--drift-baseline",
            str(baseline),
            "--drift-feature",
            "input-chars",
        )
    )

    out = capsys.readouterr().out
    assert code == EXIT_OK
    assert "may need recalibrating: it no longer matches its calibration record" in out


def test_a_lock_alone_leaves_the_evaluator_certified_and_its_verdicts_stored(
    reading, capsys
) -> None:
    from test_agentcore_online import _evaluator_doc

    locked = {**_evaluator_doc(), "updatedAt": "2026-11-06T10:00:00+00:00"}
    locked["lockedForModification"] = True
    _traffic(reading, 2)
    reading["control"] = [_evaluator_doc_queue(locked)]

    assert main(_online_argv(reading, "read", *_WINDOW)) == EXIT_OK

    out = capsys.readouterr().out
    assert "Stored 2 verdict(s)" in out
    assert "may need recalibrating" not in out


def test_an_evaluator_read_that_fails_is_reported_not_skipped(reading, capsys) -> None:
    from test_agentcore_online import _EVALUATOR_ID, _evaluator_doc

    def then_throttled(stub: Stubber) -> None:
        stub.add_response("get_evaluator", _evaluator_doc(), {"evaluatorId": _EVALUATOR_ID})
        stub.add_client_error(
            "get_evaluator", service_error_code="ThrottlingException", http_status_code=400
        )

    _traffic(reading, 1)
    reading["control"] = [then_throttled]

    code = main(_online_argv(reading, "read", *_WINDOW))

    assert code != EXIT_OK
    assert "ThrottlingException" in capsys.readouterr().err


def test_a_read_names_a_verdict_identical_to_an_on_demand_one_as_already_stored(
    reading, capsys
) -> None:
    from calibrated_agent_evaluation.agentcore.online_record import verdict_digest
    from calibrated_agent_evaluation.backends.local import LocalBackend
    from calibrated_agent_evaluation.cli.records import load_records_of

    _traffic(reading, 1)
    reading["control"] = [_evaluator_doc_queue()]
    assert main(_online_argv(reading, "read", *_WINDOW)) == EXIT_OK
    with LocalBackend(root=str(reading["root"])) as store:
        (label,) = store.get_scores(f"{1:032x}")
    # Start again from a store holding only that verdict, as an earlier on-demand run left it.
    import shutil

    from calibrated_agent_evaluation.cli.records import records_dir

    shutil.rmtree(records_dir(reading["root"], "agentcore-read"))
    capsys.readouterr()
    reading["control"] = [_evaluator_doc_queue()]

    assert main(_online_argv(reading, "read", *_WINDOW)) == EXIT_OK

    (read,) = load_records_of(reading["root"], "agentcore-read")
    assert [v.verdict_digest for v in read.verdicts] == [verdict_digest(label)]
    assert "1 already stored" in capsys.readouterr().out


def test_publish_refuses_two_different_verdicts_named_for_one_trace(reading, capsys) -> None:
    from calibrated_agent_evaluation.agentcore.online_record import OnlineReadRecord, ReadVerdict
    from calibrated_agent_evaluation.cli.records import append_records
    from test_agentcore_online import _CONFIG_ID, _EVALUATOR_ID

    trace = "d" * 32
    append_records(
        reading["root"],
        "agentcore-read",
        [
            OnlineReadRecord(
                read_id=f"{_CONFIG_ID}:{digest[0]}",
                config_id=_CONFIG_ID,
                backend="local",
                since=_WINDOW[1],
                until=_WINDOW[3],
                verdicts=(
                    ReadVerdict(evaluator_id=_EVALUATOR_ID, trace_id=trace, verdict_digest=digest),
                ),
            )
            for digest in ("1" * 16, "2" * 16)
        ],
    )

    code = main(_online_argv(reading, "publish"))

    assert code != EXIT_OK
    assert trace in capsys.readouterr().err
    assert not reading["opened"]


def test_cost_reads_the_result_group_from_the_configs_creation(reading, monkeypatch) -> None:
    seen: list[Any] = []
    real = __import__(
        "calibrated_agent_evaluation.agentcore.readers", fromlist=["run_logs_insights"]
    )

    def recording(client: Any, group: str, config: Any, **kw: Any) -> Any:
        seen.append((group, config.window.start))
        return real_run(client, group, config, **kw)

    real_run = real.run_logs_insights
    monkeypatch.setattr(
        "calibrated_agent_evaluation.agentcore.readers.run_logs_insights", recording
    )

    assert main(["--root", str(reading["root"]), "online", "cost"]) == EXIT_OK

    assert seen == [(_RESULTS_GROUP, reading["record"].created_at)]


def test_cost_clients_refuse_a_call_outside_its_allowlist(reading) -> None:
    from calibrated_agent_evaluation.agentcore.errors import ReadOnlyViolation
    from calibrated_agent_evaluation.agentcore.session import AwsClients

    assert main(["--root", str(reading["root"]), "online", "cost"]) == EXIT_OK
    session = boto3.Session(
        region_name="us-east-1", aws_access_key_id="testing", aws_secret_access_key="testing"
    )
    clients = AwsClients(session, handlers=reading["handlers"])
    with Stubber(clients.client("cloudwatch")) as stub:
        stub.add_response("put_metric_data", {})
        with pytest.raises(ReadOnlyViolation):
            clients.client("cloudwatch").put_metric_data(
                Namespace="CalibratedAgentEvaluation/Judges",
                MetricData=[{"MetricName": "x", "Value": 1.0}],
            )


def test_a_changed_evaluator_that_can_no_longer_be_a_judge_is_not_checkable(
    reading, capsys
) -> None:
    from test_agentcore_online import _EVALUATOR_ID, _evaluator_doc

    session_level = {**_evaluator_doc(), "level": "SESSION"}

    def queue(stub: Stubber) -> None:
        for _ in range(2):
            stub.add_response("get_evaluator", session_level, {"evaluatorId": _EVALUATOR_ID})

    _traffic(reading, 1)
    reading["control"] = [queue]

    assert main(_online_argv(reading, "read", *_WINDOW)) == EXIT_OK

    out = capsys.readouterr().out
    assert "not checkable" in out
    assert "Stored 0 verdict(s)" in out


def test_seed_without_the_drift_flags_is_refused_before_any_aws_call(reading, capsys) -> None:
    assert main(_online_argv(reading, "read", *_WINDOW, "--seed", "3")) != EXIT_OK
    assert "--drift-baseline" in capsys.readouterr().err
    assert not reading["opened"]


def test_publish_counts_verdicts_on_traces_of_one_session_as_one_cluster(
    reading, monkeypatch, capsys
) -> None:
    from test_agentcore_online import _EVALUATOR_ARN
    from test_agentcore_readback import _raw, _result

    # Two traces (turns) of one session, and one trace of another.
    for trace_id, session in (("1" * 32, "s-a"), ("2" * 32, "s-a"), ("3" * 32, "s-b")):
        reading["traces"].append(_raw(trace_id, session))
        result = _result(trace_id, session)
        result["attributes"]["aws.bedrock_agentcore.evaluator.arn"] = _EVALUATOR_ARN
        reading["results"].append(result)
    reading["control"] = [_evaluator_doc_queue()]
    assert main(_online_argv(reading, "read", *_WINDOW)) == EXIT_OK
    seen: dict[str, Any] = {}

    def recording(*args: Any, **kwargs: Any) -> Any:
        seen["clusters"] = list(kwargs["rate_cluster_ids"])
        seen["traces"] = [label.trace_id for label in kwargs["rate_labels"]]
        raise SystemExit(0)

    monkeypatch.setattr(
        "calibrated_agent_evaluation.agentcore.metrics.publish_pass_rate", recording
    )
    reading["control"] = [_evaluator_doc_queue()]
    with pytest.raises(SystemExit):
        main(_online_argv(reading, "publish"))

    by_trace = dict(zip(seen["traces"], seen["clusters"], strict=True))
    assert by_trace == {"1" * 32: "s-a", "2" * 32: "s-a", "3" * 32: "s-b"}


def test_read_with_omni_makes_the_omni_calls_through_its_own_clients(
    reading, monkeypatch, capsys
) -> None:
    from calibrated_agent_evaluation.agentcore.readers import _OMNI_SERVICE, run_omni_query
    from test_agentcore_readback import _Reader

    _traffic(reading, 1)
    reading["control"] = [_evaluator_doc_queue()]
    used: list[str] = []

    class _OmniLike(_Reader):
        """Runs Omni's real call sequence on the command's clients, then yields the traces."""

        def __init__(self, clients: Any, config: Any) -> None:
            super().__init__(reading["traces"])
            self._clients = clients
            used.append("omni")

        def read_traces(self) -> Any:
            omni = Stubber(self._clients.client(_OMNI_SERVICE))
            omni.add_response("start_telemetry_query_session", {"sessionId": "s"})
            omni.add_response("start_telemetry_query", {"queryId": "q", "sessionId": "s"})
            omni.add_response(
                "get_telemetry_query_results",
                {"status": "Complete", "rows": [], "statistics": {}},
            )
            omni.add_response("stop_telemetry_query_session", {})
            with omni:
                run_omni_query(self._clients.client(_OMNI_SERVICE), "SELECT 1")
                omni.assert_no_pending_responses()
            yield from super().read_traces()

    monkeypatch.setattr("calibrated_agent_evaluation.agentcore.readers.OmniReader", _OmniLike)

    code = main(_online_argv(reading, "read", *_WINDOW, "--omni"))

    assert code == EXIT_OK, capsys.readouterr().err
    assert used == ["omni"]


def test_every_call_an_online_command_allows_is_a_call_aws_has() -> None:
    from botocore.loaders import Loader

    from calibrated_agent_evaluation.agentcore.evaluator_create import CREATE_EVALUATOR_OPERATIONS
    from calibrated_agent_evaluation.cli.stages.online import _ATTACH_OPERATIONS
    from calibrated_agent_evaluation.cli.stages.online_read import (
        _COST_OPERATIONS,
        _OMNI_OPERATIONS,
        _PUBLISH_OPERATIONS,
        _READ_OPERATIONS,
    )

    loader = Loader()

    def operations(service: str) -> set[str]:
        model: dict[str, Any] = loader.load_service_model(service, "service-2")
        return set(model["operations"])

    allowed = (
        CREATE_EVALUATOR_OPERATIONS
        | _ATTACH_OPERATIONS
        | _READ_OPERATIONS
        | _OMNI_OPERATIONS
        | _PUBLISH_OPERATIONS
        | _COST_OPERATIONS
    )
    unknown = sorted(
        (service, name) for service, name in allowed if name not in operations(service)
    )

    assert unknown == []


def _published_dimensions(seen: list[dict[str, str]]):
    """A cloudwatch queue that accepts one PutMetricData and keeps its metrics' dimensions."""

    def queue(stub: Stubber) -> None:
        def capture(params: dict[str, Any], **_kw: Any) -> None:
            for metric in params["MetricData"]:
                seen.append({d["Name"]: d["Value"] for d in metric["Dimensions"]})

        stub.client.meta.events.register("provide-client-params.cloudwatch.PutMetricData", capture)
        stub.add_response("put_metric_data", {})

    return queue


def _without_gold_source(reading: dict[str, Any]) -> None:
    """Replace the store's session logs with ones written before the gold source was recorded."""
    import shutil

    from calibrated_agent_evaluation.cli.records import append_records, records_dir
    from test_agentcore_online import _session_logs

    shutil.rmtree(records_dir(reading["root"], "agentcore-session"))
    append_records(
        reading["root"], "agentcore-session", _session_logs(reading["online"], gold_source=None)
    )


def _store_gold(reading: dict[str, Any]) -> None:
    """The calibration's gold, rule-derived, on two traces it names, in the reading store."""
    import shutil

    from calibrated_agent_evaluation.backends.local import LocalBackend
    from calibrated_agent_evaluation.cli.records import append_records, records_dir
    from calibrated_agent_evaluation.schema.types import Label
    from test_cli_agentcore_judge import _trace as _plain_trace

    calib = reading["online"].calib.model_copy(update={"trace_ids": ("gold-1", "gold-2")})
    shutil.rmtree(records_dir(reading["root"], "calibration"))
    append_records(reading["root"], "calibration", [calib])
    with LocalBackend(root=str(reading["root"])) as store:
        for trace_id in calib.trace_ids:
            store.put_trace(_plain_trace(trace_id))
            store.put_score(
                Label(
                    trace_id=trace_id,
                    span_id=None,
                    kind="human",
                    outcome="PASS",
                    critique="rule",
                    rater_id="synthetic-rule-oracle",
                    codebook_version="v1",
                )
            )


def test_publish_from_a_store_without_the_gold_sends_the_recorded_gold_source(
    reading, capsys
) -> None:
    _traffic(reading, 160)
    reading["control"] = [_evaluator_doc_queue()]
    assert main(_online_argv(reading, "read", *_WINDOW)) == EXIT_OK
    seen: list[dict[str, str]] = []
    reading["control"] = [_evaluator_doc_queue()]
    reading["cloudwatch"] = [_published_dimensions(seen)]

    code = main(_online_argv(reading, "publish"))

    assert code == EXIT_OK, capsys.readouterr().err
    assert len(seen) == 3
    assert {dims["GoldSource"] for dims in seen} == {"synthetic-rule"}


def test_read_and_cost_name_the_recorded_gold_source(reading, capsys) -> None:
    _traffic(reading, 1)
    reading["control"] = [_evaluator_doc_queue()]

    assert main(_online_argv(reading, "read", *_WINDOW)) == EXIT_OK
    assert "Gold source: synthetic-rule." in capsys.readouterr().out
    assert main(["--root", str(reading["root"]), "online", "cost"]) == EXIT_OK
    assert "Gold source: synthetic-rule." in capsys.readouterr().out


def test_a_log_without_a_gold_source_falls_back_to_the_stores_gold_and_says_so(
    reading, capsys
) -> None:
    _without_gold_source(reading)
    _traffic(reading, 160)
    reading["control"] = [_evaluator_doc_queue()]
    # With none of the gold in the store, read and cost name it unknown.
    assert main(_online_argv(reading, "read", *_WINDOW)) == EXIT_OK
    assert "Gold source: unknown" in capsys.readouterr().out
    assert main(["--root", str(reading["root"]), "online", "cost"]) == EXIT_OK
    assert "Gold source: unknown" in capsys.readouterr().out
    _store_gold(reading)
    fallback = (
        "Gold source: synthetic-rule (read from this store's gold, not recorded at calibration)."
    )
    reading["control"] = [_evaluator_doc_queue()]
    assert main(_online_argv(reading, "read", *_WINDOW)) == EXIT_OK
    assert fallback in capsys.readouterr().out
    assert main(["--root", str(reading["root"]), "online", "cost"]) == EXIT_OK
    assert fallback in capsys.readouterr().out
    seen: list[dict[str, str]] = []
    reading["control"] = [_evaluator_doc_queue()]
    reading["cloudwatch"] = [_published_dimensions(seen)]

    assert main(_online_argv(reading, "publish")) == EXIT_OK
    assert fallback in capsys.readouterr().out
    assert {dims["GoldSource"] for dims in seen} == {"synthetic-rule"}


def test_publish_refuses_an_unknown_gold_source_before_any_aws_call(reading, capsys) -> None:
    _without_gold_source(reading)
    _traffic(reading, 160)
    reading["control"] = [_evaluator_doc_queue()]
    assert main(_online_argv(reading, "read", *_WINDOW)) == EXIT_OK
    capsys.readouterr()
    reading["opened"] = False
    seen: list[dict[str, str]] = []
    reading["control"] = [_evaluator_doc_queue()]
    reading["cloudwatch"] = [_published_dimensions(seen)]

    code = main(_online_argv(reading, "publish"))

    err = capsys.readouterr().err
    assert code == EXIT_DOMAIN_ERROR
    assert "gold source" in err
    assert "the store `calibrated-agent-evaluation calibrate` graded the judge in" in err
    assert "calibrate the judge again" in err
    assert not reading["opened"]
    assert seen == []
