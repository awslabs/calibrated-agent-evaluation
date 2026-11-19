# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""AwsClients caches a client per service, runs its handlers on every call, and wraps SDK errors."""

from __future__ import annotations

from typing import Any

import pytest

pytest.importorskip("boto3")

import boto3
from botocore.config import Config
from botocore.stub import Stubber

from calibrated_agent_evaluation.agentcore import (
    AgentCoreError,
    AwsClients,
    AwsCredentialsMissing,
    BilledCallBudget,
    ReadOnlyGuard,
)
from calibrated_agent_evaluation.agentcore.errors import BilledCallBudgetExceeded, ReadOnlyViolation


def _session(region: str = "us-west-2") -> Any:
    return boto3.Session(
        region_name=region, aws_access_key_id="testing", aws_secret_access_key="testing"
    )


_DEFAULT_RULE: dict[str, Any] = {
    "Name": "Default",
    "Rule": {"Probabilistic": {"DesiredSamplingPercentage": 1.0}},
}
_EVALUATE_PARAMS: dict[str, Any] = {
    "evaluatorId": "Builtin.Helpfulness",
    "evaluationInput": {"sessionSpans": [{}]},
}


def test_one_client_per_service_is_built_and_reused() -> None:
    clients = AwsClients(_session())

    xray = clients.client("xray")

    assert clients.client("xray") is xray
    assert clients.client("logs") is not xray
    assert clients.region == "us-west-2"


def test_every_handler_sees_each_call_before_the_stub_answers() -> None:
    seen: list[str] = []

    def record(**kwargs: Any) -> None:
        seen.append(kwargs["model"].name)

    clients = AwsClients(_session(), handlers=[record])
    xray = clients.client("xray")
    stub = Stubber(xray)
    stub.add_response(
        "get_trace_segment_destination", {"Destination": "CloudWatchLogs", "Status": "ACTIVE"}
    )
    with stub:
        xray.get_trace_segment_destination()

    assert seen == ["GetTraceSegmentDestination"]


def test_a_handler_that_raises_stops_the_call_before_it_is_sent() -> None:
    class Refused(Exception):
        pass

    def refuse(**_kwargs: Any) -> None:
        raise Refused

    clients = AwsClients(_session(), handlers=[refuse])
    xray = clients.client("xray")
    stub = Stubber(xray)
    stub.add_response(
        "get_trace_segment_destination", {"Destination": "CloudWatchLogs", "Status": "ACTIVE"}
    )
    with stub, pytest.raises(Refused):
        xray.get_trace_segment_destination()

    with pytest.raises(AssertionError):
        stub.assert_no_pending_responses()


def test_an_unknown_service_is_a_typed_error() -> None:
    with pytest.raises(AgentCoreError, match="no-such-service"):
        AwsClients(_session()).client("no-such-service")


def test_an_invalid_region_name_is_a_typed_error() -> None:
    with pytest.raises(AgentCoreError, match="us west 2"):
        AwsClients(_session(region="us west 2")).client("xray")


def test_a_missing_profile_is_a_typed_error() -> None:
    with pytest.raises(AgentCoreError, match="calibrated-agent-evaluation-no-such-profile"):
        AwsClients.from_profile(
            region="us-west-2", profile="calibrated-agent-evaluation-no-such-profile"
        )


def test_clients_get_explicit_timeouts_and_standard_retries() -> None:
    xray = AwsClients(_session()).client("xray")

    assert xray.meta.config.connect_timeout == 10
    assert xray.meta.config.read_timeout == 30
    assert xray.meta.config.retries["mode"] == "standard"


def test_closing_releases_the_clients_and_later_calls_build_new_ones() -> None:
    with AwsClients(_session()) as clients:
        first = clients.client("xray")
    reopened = clients.client("xray")

    assert reopened is not first


@pytest.mark.parametrize("code", ["ExpiredToken", "ExpiredTokenException", "InvalidClientTokenId"])
def test_an_expired_or_invalid_token_is_a_credentials_error_naming_the_refresh(code: str) -> None:
    clients = AwsClients(_session())
    stub = Stubber(clients.client("sts"))
    stub.add_client_error("get_caller_identity", service_error_code=code, http_status_code=403)
    stub.activate()

    with pytest.raises(AwsCredentialsMissing, match="aws sso login"):
        clients.caller_identity()


def test_another_sts_service_error_is_still_a_typed_agentcore_error() -> None:
    clients = AwsClients(_session())
    stub = Stubber(clients.client("sts"))
    stub.add_client_error("get_caller_identity", service_error_code="Throttling")
    stub.activate()

    with pytest.raises(AgentCoreError, match="could not confirm") as excinfo:
        clients.caller_identity()
    assert not isinstance(excinfo.value, AwsCredentialsMissing)


def test_with_config_clients_keep_the_installed_guards_and_caller_settings() -> None:
    allowed = {("xray", "GetIndexingRules"), ("bedrock-agentcore", "Evaluate")}
    budget = BilledCallBudget(limit=1)
    caller_settings = Config(
        user_agent_extra="calibrated-agent-evaluation-test/1",
        connect_timeout=7,
        read_timeout=30,
        retries={"mode": "standard", "max_attempts": 3},
    )
    base = AwsClients(_session(), handlers=[ReadOnlyGuard(allowed), budget], config=caller_settings)
    base_data_plane = base.client("bedrock-agentcore")
    base_stub = Stubber(base_data_plane)
    base_stub.add_response("evaluate", {"evaluationResults": []})
    with base_stub:
        base_data_plane.evaluate(**_EVALUATE_PARAMS)

    view = base.with_config(Config(read_timeout=300))
    xray = view.client("xray")
    data_plane = view.client("bedrock-agentcore")
    xray_stub, data_stub = Stubber(xray), Stubber(data_plane)
    xray_stub.add_response("get_indexing_rules", {"IndexingRules": [_DEFAULT_RULE]})
    xray_stub.add_response("update_indexing_rule", {"IndexingRule": _DEFAULT_RULE})
    data_stub.add_response("evaluate", {"evaluationResults": []})
    with xray_stub, data_stub:
        assert xray.get_indexing_rules()["IndexingRules"][0]["Name"] == "Default"
        with pytest.raises(ReadOnlyViolation, match="UpdateIndexingRule"):
            xray.update_indexing_rule(
                Name="Default", Rule={"Probabilistic": {"DesiredSamplingPercentage": 100.0}}
            )
        # The budget is the same one the base clients spent: the view's billed call is the second.
        with pytest.raises(BilledCallBudgetExceeded):
            data_plane.evaluate(**_EVALUATE_PARAMS)

    assert type(view) is AwsClients
    assert view.region == base.region == "us-west-2"
    assert data_plane is not base_data_plane
    assert xray.meta.config.user_agent_extra == "calibrated-agent-evaluation-test/1"
    assert xray.meta.config.connect_timeout == 7
    assert xray.meta.config.read_timeout == 300


def test_with_config_leaves_the_base_clients_retry_settings_unchanged() -> None:
    base = AwsClients(_session())
    built_before = base.client("bedrock-agentcore")

    view = base.with_config(Config(retries={"mode": "standard", "total_max_attempts": 1}))
    single_attempt = view.client("bedrock-agentcore")
    built_after = base.client("bedrock-agentcore-control")

    # botocore turns the base's three retries into four total attempts on the clients it builds.
    assert single_attempt.meta.config.retries == {"mode": "standard", "total_max_attempts": 1}
    assert built_before.meta.config.retries == {"mode": "standard", "total_max_attempts": 4}
    assert built_after.meta.config.retries == {"mode": "standard", "total_max_attempts": 4}
    assert base.client("bedrock-agentcore") is built_before
    assert built_before.meta.config.read_timeout == 30
