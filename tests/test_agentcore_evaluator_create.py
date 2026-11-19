# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Creating a Calibrated Agent Evaluation prompt judge as an AgentCore evaluator."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest

pytest.importorskip("boto3")

import boto3
from botocore.stub import ANY, Stubber

from calibrated_agent_evaluation.agentcore.errors import (
    AgentCoreError,
    AgentCoreResourceExists,
    AgentCoreResourceNotActive,
    AgentCoreRunDeclined,
)
from calibrated_agent_evaluation.agentcore.evaluator_create import (
    PromptJudge,
    build_evaluator_request,
    create_evaluator,
    default_evaluator_name,
)
from calibrated_agent_evaluation.agentcore.judge import (
    judge_from_evaluator,
    load_agentcore_judge_file,
    outcome_for_value,
)
from calibrated_agent_evaluation.agentcore.session import AwsClients

_QUICKSTART = (
    Path(__file__).resolve().parents[1] / "src/calibrated_agent_evaluation/_quickstart/data/certify"
)
_ACCOUNT = "111122223333"
_NAME = "orderbot_refund_policy_v2"
_ID = f"{_NAME}-AbCdEf1234"
_ARN = f"arn:aws:bedrock-agentcore:us-east-1:{_ACCOUNT}:evaluator/{_ID}"
_UPDATED = datetime(2026, 11, 5, 10, 0, 5, tzinfo=UTC)


def _judge(version: int = 2) -> PromptJudge:
    doc = json.loads((_QUICKSTART / f"judge-v{version}.json").read_text(encoding="utf-8"))
    return PromptJudge(
        judge_id=doc["judge_id"],
        model=doc["model"],
        prompt_version=doc["prompt_version"],
        prompt_template=doc["prompt_template"],
        failure_mode=doc["failure_mode"],
        agent=doc.get("agent"),
    )


def _request(judge: PromptJudge | None = None) -> dict[str, Any]:
    j = judge or _judge()
    return build_evaluator_request(
        j.judge_id, j.model, j.prompt_template, j.failure_mode, name=default_evaluator_name(j)
    )


# --- the request ------------------------------------------------------------------------------


def test_the_quickstart_judge_builds_a_trace_level_pass_fail_request() -> None:
    judge = _judge()
    request = _request(judge)
    llm = request["evaluatorConfig"]["llmAsAJudge"]

    assert request["level"] == "TRACE"
    assert request["evaluatorName"] == _NAME
    assert judge.prompt_template in llm["instructions"]
    assert "{context}" in llm["instructions"]
    assert "{assistant_turn}" in llm["instructions"]
    assert "expected_response" not in llm["instructions"]
    assert [(r["value"], r["label"]) for r in llm["ratingScale"]["numerical"]] == [
        (0.0, "FAIL"),
        (1.0, "PASS"),
    ]
    model = llm["modelConfig"]["bedrockEvaluatorModelConfig"]
    assert model["modelId"] == "global.anthropic.claude-haiku-4-5-20251001-v1:0"
    assert model["inferenceConfig"]["temperature"] == 0.0


def test_the_judges_outcome_rule_reads_the_created_scale_as_pass_and_fail() -> None:
    request = _request()
    evaluator = {
        "evaluatorId": _ID,
        "evaluatorArn": _ARN,
        "evaluatorType": "Custom",
        "level": "TRACE",
        "updatedAt": _UPDATED,
        "evaluatorConfig": request["evaluatorConfig"],
    }
    judge = judge_from_evaluator("orderbot-refund-policy", evaluator)

    assert outcome_for_value(1.0, judge.rating_scale) == "PASS"
    assert outcome_for_value(0.0, judge.rating_scale) == "FAIL"
    assert judge.pinned_model == "global.anthropic.claude-haiku-4-5-20251001-v1:0"


def test_two_versions_of_one_judge_get_different_default_names() -> None:
    assert default_evaluator_name(_judge(1)) == "orderbot_refund_policy_v1"
    assert default_evaluator_name(_judge(2)) == _NAME


def test_a_name_aws_would_refuse_is_refused_before_any_call() -> None:
    judge = _judge()
    with pytest.raises(AgentCoreError, match="evaluator name"):
        build_evaluator_request(
            judge.judge_id, judge.model, judge.prompt_template, judge.failure_mode, name="9-bad"
        )


@pytest.mark.parametrize("model", ["openai/gpt-5", "anthropic/claude-haiku-4-5", ""])
def test_a_model_amazon_bedrock_cannot_run_is_refused(model: str) -> None:
    judge = _judge()
    with pytest.raises(AgentCoreError, match="Bedrock"):
        build_evaluator_request(judge.judge_id, model, judge.prompt_template, judge.failure_mode)


# --- creating it ------------------------------------------------------------------------------


def _clients() -> AwsClients:
    session = boto3.Session(
        region_name="us-east-1", aws_access_key_id="testing", aws_secret_access_key="testing"
    )
    return AwsClients(session)


def _listed(stub: Stubber, *names: str) -> None:
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


def _created(stub: Stubber) -> None:
    stub.add_response(
        "create_evaluator",
        {"evaluatorArn": _ARN, "evaluatorId": _ID, "createdAt": _UPDATED, "status": "CREATING"},
        {**_request(), "clientToken": ANY, "description": ANY},
    )


def _got(stub: Stubber, status: str) -> None:
    stub.add_response(
        "get_evaluator",
        {
            "evaluatorArn": _ARN,
            "evaluatorId": _ID,
            "evaluatorName": _NAME,
            "evaluatorConfig": _request()["evaluatorConfig"],
            "evaluatorType": "Custom",
            "level": "TRACE",
            "status": status,
            "createdAt": _UPDATED,
            "updatedAt": _UPDATED,
        },
        {"evaluatorId": _ID},
    )


class _Clock:
    """A clock the wait reads, moved on by each sleep."""

    def __init__(self) -> None:
        self.now = 0.0

    def __call__(self) -> float:
        return self.now

    def sleep(self, seconds: float) -> None:
        self.now += seconds


def _create(clients: AwsClients, out: Path, **kwargs: Any) -> Any:
    clock = _Clock()
    return create_evaluator(
        _judge(),
        out=out,
        clients=clients,
        confirm=kwargs.pop("confirm", lambda _: True),
        poll_seconds=1.0,
        sleep=clock.sleep,
        clock=clock,
        **kwargs,
    )


def test_an_approved_create_waits_for_active_and_writes_the_agentcore_judge_file(
    tmp_path: Path,
) -> None:
    clients = _clients()
    out = tmp_path / "judge-v2.agentcore.json"
    with Stubber(clients.client("bedrock-agentcore-control")) as stub:
        _listed(stub, "someone_else")
        _created(stub)
        _got(stub, "CREATING")
        _got(stub, "ACTIVE")
        created = _create(clients, out)
        stub.assert_no_pending_responses()

    assert created.evaluator_arn == _ARN
    assert created.updated_at == _UPDATED.isoformat()
    loaded = load_agentcore_judge_file(out)
    assert loaded.evaluator_id == _ID
    assert loaded.judge_id == "orderbot-refund-policy"
    assert loaded.failure_mode == "trusted_broken_policy_tool"


def test_a_declined_create_sends_no_write_and_writes_no_file(tmp_path: Path) -> None:
    clients = _clients()
    out = tmp_path / "judge-v2.agentcore.json"
    with Stubber(clients.client("bedrock-agentcore-control")) as stub:
        _listed(stub)
        with pytest.raises(AgentCoreRunDeclined, match="CreateEvaluator"):
            _create(clients, out, confirm=lambda _: False)
        stub.assert_no_pending_responses()

    assert not out.exists()


@pytest.mark.parametrize(
    ("statuses", "says"),
    [
        (["CREATING"] * 200, "was still CREATING when the wait ran out"),
        (["CREATING", "CREATE_FAILED"], "reported CREATE_FAILED"),
    ],
    ids=["timed-out", "failed"],
)
def test_an_evaluator_that_does_not_become_active_is_not_recorded(
    tmp_path: Path, statuses: list[str], says: str
) -> None:
    clients = _clients()
    out = tmp_path / "judge-v2.agentcore.json"
    with Stubber(clients.client("bedrock-agentcore-control")) as stub:
        _listed(stub)
        _created(stub)
        for status in statuses:
            _got(stub, status)
        with pytest.raises(AgentCoreResourceNotActive, match=says) as refused:
            _create(clients, out, wait_seconds=10.0)

    assert _ID in str(refused.value)
    assert not out.exists()


def test_an_existing_name_is_refused_and_nothing_is_created(tmp_path: Path) -> None:
    clients = _clients()
    out = tmp_path / "judge-v2.agentcore.json"
    with Stubber(clients.client("bedrock-agentcore-control")) as stub:
        _listed(stub, _NAME)
        # Nothing else is queued: a CreateEvaluator call would fail as unstubbed.
        with pytest.raises(AgentCoreResourceExists, match=f"evaluator {_NAME} already exists"):
            _create(clients, out)
        stub.assert_no_pending_responses()

    assert not out.exists()


def test_a_name_taken_between_the_check_and_the_create_is_refused_as_existing(
    tmp_path: Path,
) -> None:
    clients = _clients()
    with Stubber(clients.client("bedrock-agentcore-control")) as stub:
        _listed(stub)
        stub.add_client_error(
            "create_evaluator",
            service_error_code="ConflictException",
            service_message="An evaluator with this name already exists",
        )
        with pytest.raises(AgentCoreResourceExists, match=_NAME):
            _create(clients, tmp_path / "out.json")


def test_a_name_in_use_on_a_later_page_of_the_listing_is_still_refused(tmp_path: Path) -> None:
    clients = _clients()
    first = {
        "evaluators": [
            {
                "evaluatorArn": f"{_ARN.rsplit('/', 1)[0]}/other-AbCdEf1234",
                "evaluatorId": "other-AbCdEf1234",
                "evaluatorName": "other",
                "evaluatorType": "Custom",
                "status": "ACTIVE",
                "createdAt": _UPDATED,
                "updatedAt": _UPDATED,
            }
        ],
        "nextToken": "page-2",
    }
    with Stubber(clients.client("bedrock-agentcore-control")) as stub:
        stub.add_response("list_evaluators", first, {})
        second = {
            "evaluators": [{**first["evaluators"][0], "evaluatorName": _NAME, "evaluatorId": _ID}]
        }
        stub.add_response("list_evaluators", second, {"nextToken": "page-2"})
        with pytest.raises(AgentCoreResourceExists, match=_NAME):
            _create(clients, tmp_path / "out.json")
        stub.assert_no_pending_responses()


def test_an_evaluator_being_deleted_ends_the_wait_at_once(tmp_path: Path) -> None:
    clients = _clients()
    with Stubber(clients.client("bedrock-agentcore-control")) as stub:
        _listed(stub)
        _created(stub)
        _got(stub, "DELETING")
        with pytest.raises(AgentCoreResourceNotActive, match="reported DELETING"):
            _create(clients, tmp_path / "out.json", wait_seconds=600.0)
        stub.assert_no_pending_responses()


@pytest.mark.parametrize("temperature", [-0.1, 1.5])
def test_a_temperature_agentcore_does_not_take_is_refused(temperature: float) -> None:
    judge = _judge()
    with pytest.raises(AgentCoreError, match="outside 0 to 1"):
        build_evaluator_request(
            judge.judge_id,
            judge.model,
            judge.prompt_template,
            judge.failure_mode,
            temperature=temperature,
        )


def test_an_existing_output_file_is_refused_before_any_call(tmp_path: Path) -> None:
    clients = _clients()
    out = tmp_path / "judge-v2.agentcore.json"
    out.write_text("{}", encoding="utf-8")
    with Stubber(clients.client("bedrock-agentcore-control")):
        with pytest.raises(AgentCoreError, match="already exists"):
            _create(clients, out)

    assert out.read_text(encoding="utf-8") == "{}"
