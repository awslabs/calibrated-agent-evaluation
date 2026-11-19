# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""The live-test role grants exact ARNs, and "*" only for actions that take no resource ARN."""

from __future__ import annotations

import copy
import re
from pathlib import Path
from typing import Any

import pytest

yaml = pytest.importorskip("yaml")

_TEMPLATE = Path(__file__).resolve().parents[1] / "infra" / "agentcore-live-test-role.yaml"

# Actions the AWS Service Authorization Reference lists with no resource type: "*" is the only
# grant possible for them. Anything else must name its resources.
_NO_RESOURCE_LEVEL = frozenset(
    {
        "bedrock-agentcore:CreateDataset",
        "bedrock-agentcore:CreateEvaluator",
        "bedrock-agentcore:CreateOnlineEvaluationConfig",
        "bedrock-agentcore:ListAgentRuntimes",
        "bedrock-agentcore:ListDatasets",
        "bedrock-agentcore:ListEvaluators",
        "bedrock-agentcore:ListOnlineEvaluationConfigs",
        "bedrock-agentcore:StartBatchEvaluation",
        # GetMetricData and ListMetrics take an optional dataset ARN that the Omni query's
        # dataset does not give in advance, so they are granted on "*" too; so does
        # PutMetricData, whose publish names no dataset.
        "cloudwatch:GetMetricData",
        "cloudwatch:PutMetricData",
        "cloudwatch:GetRecords",
        "cloudwatch:GetTelemetryQueryResults",
        "cloudwatch:ListDomains",
        "cloudwatch:ListMetrics",
        "cloudwatch:ListSpaces",
        "cloudwatch:StartTelemetryQuery",
        "cloudwatch:StartTelemetryQuerySession",
        "cloudwatch:StopTelemetryQuerySession",
        "logs:DescribeLogGroups",
        "pricing:GetProducts",
        "xray:BatchGetTraces",
        "xray:GetIndexingRules",
        "xray:GetTraceSegmentDestination",
    }
)


class _CfnLoader(yaml.SafeLoader):
    """A SafeLoader that keeps CloudFormation short-form tags (``!Sub``, ``!Ref``) as values."""


def _tagged(loader: Any, tag_suffix: str, node: Any) -> dict[str, Any]:
    if isinstance(node, yaml.ScalarNode):
        value: Any = loader.construct_scalar(node)
    elif isinstance(node, yaml.SequenceNode):
        value = loader.construct_sequence(node)
    else:
        value = loader.construct_mapping(node)
    return {f"!{tag_suffix}": value}


_CfnLoader.add_multi_constructor("!", _tagged)


def _template() -> dict[str, Any]:
    return yaml.load(_TEMPLATE.read_text(), Loader=_CfnLoader)


def _unwrap(statement: dict[str, Any]) -> dict[str, Any]:
    """A statement held in ``!If [condition, statement, AWS::NoValue]``, as granted when set."""
    if "!If" in statement:
        _condition, granted, _otherwise = statement["!If"]
        return granted
    return statement


def _statements(template: dict[str, Any]) -> list[dict[str, Any]]:
    policies = template["Resources"]["LiveTestRole"]["Properties"]["Policies"]
    return [_unwrap(s) for p in policies for s in p["PolicyDocument"]["Statement"]]


def _as_list(value: Any) -> list[Any]:
    return value if isinstance(value, list) else [value]


# AgentCore generates a dataset's id when it is created, so the live test's datasets can only be
# named by the prefix it gives them. This trailing wildcard, on dataset ARNs in the stack's own
# account and region, is one of the two kinds of wildcard a resource may carry.
_LIVE_DATASET_ARN = (
    "arn:${AWS::Partition}:bedrock-agentcore:${AWS::Region}:${AWS::AccountId}:dataset/"
    "calibrated_agent_evaluation_live_*"
)

# The span reader reads the shared aws/spans group and the invoked endpoint's own log group.
_LOG_GROUP = "arn:${AWS::Partition}:logs:${AWS::Region}:${AWS::AccountId}:log-group:"
_SPANS_GROUP_ARN = _LOG_GROUP + "aws/spans"
_RUNTIME_GROUP_ARN = (
    _LOG_GROUP + "/aws/bedrock-agentcore/runtimes/${AgentRuntimeId}-${AgentRuntimeEndpointName}"
)
# CloudWatch Logs checks some calls against a group's ARN with ":*" appended, which covers that
# group's own streams and nothing outside it. That suffix on exactly these two groups is the other
# kind of wildcard a resource may carry.
_SPAN_READER_GROUP_STREAMS = frozenset({_SPANS_GROUP_ARN + ":*", _RUNTIME_GROUP_ARN + ":*"})


# The online judge's evaluators and online configs are named
# calibrated_agent_evaluation_live_<random>; AgentCore generates each id as that name plus a suffix,
# so they too can only be named by that prefix, as can the log group each online config writes its
# results to.
_AGENTCORE = "arn:${AWS::Partition}:bedrock-agentcore:${AWS::Region}:${AWS::AccountId}:"
_LIVE_EVALUATOR_ARN = _AGENTCORE + "evaluator/calibrated_agent_evaluation_live_*"
_LIVE_ONLINE_CONFIG_ARN = _AGENTCORE + "online-evaluation-config/calibrated_agent_evaluation_live_*"
_LIVE_RESULTS_GROUP_ARN = (
    _LOG_GROUP + "/aws/bedrock-agentcore/evaluations/results/calibrated_agent_evaluation_live_*"
)
_LIVE_PREFIXES = frozenset(
    {
        _LIVE_DATASET_ARN,
        _LIVE_EVALUATOR_ARN,
        _LIVE_ONLINE_CONFIG_ARN,
        _LIVE_RESULTS_GROUP_ARN,
        _LIVE_RESULTS_GROUP_ARN + ":*",
    }
)


# The judge model's foundation model, in every region its inference profile routes to. The model is
# named by the stack's parameter; only the region is open.
_JUDGE_MODEL_IN_ANY_REGION = (
    "arn:${AWS::Partition}:bedrock:*::foundation-model/${JudgeFoundationModelId}"
)


def _is_allowed_wildcard(arn: str) -> bool:
    return (
        arn in _LIVE_PREFIXES
        or arn in _SPAN_READER_GROUP_STREAMS
        or arn == _JUDGE_MODEL_IN_ANY_REGION
    )


def _violations(template: dict[str, Any]) -> list[str]:
    """Every statement that grants more than exact ARNs allow, named by its Sid."""
    problems: list[str] = []
    for statement in _statements(template):
        sid = statement.get("Sid", "<no Sid>")
        actions = _as_list(statement["Action"])
        resources = _as_list(statement["Resource"])
        if statement.get("Effect") != "Allow":
            continue
        wild_actions = [a for a in actions if "*" in a]
        if wild_actions:
            problems.append(f"{sid}: wildcard action {wild_actions}")
        if "*" in resources:
            extra = sorted(set(actions) - _NO_RESOURCE_LEVEL)
            if extra:
                problems.append(f"{sid}: Resource '*' for {extra}")
            continue
        for resource in resources:
            if resource == {"!Ref": "EvaluationRoleArn"} and actions == ["iam:PassRole"]:
                continue  # the one role the online config passes, named by the stack's parameter
            if not (isinstance(resource, dict) and "!Sub" in resource):
                problems.append(f"{sid}: resource {resource!r} is not built from parameters")
            elif "*" in str(resource["!Sub"]) and not _is_allowed_wildcard(str(resource["!Sub"])):
                problems.append(f"{sid}: resource {resource['!Sub']!r} contains a wildcard")
    return problems


def test_the_shipped_template_grants_exact_arns_or_only_no_resource_actions_on_star() -> None:
    assert _violations(_template()) == []


def test_the_role_trusts_only_the_named_principal() -> None:
    trust = _template()["Resources"]["LiveTestRole"]["Properties"]["AssumeRolePolicyDocument"]

    (statement,) = trust["Statement"]
    assert statement["Principal"] == {"AWS": {"!Ref": "TrustedPrincipalArn"}}
    assert statement["Action"] == "sts:AssumeRole"


# The Price List API answers only from a few endpoint regions, so its call is made to us-east-1
# whatever the stack's region; it is the one "*" statement pinned to that endpoint instead. Keep
# this in step with _PRICING_REGION in src/calibrated_agent_evaluation/agentcore/pricing.py.
_PRICE_LIST_REGION = "us-east-1"


def test_the_star_statement_is_pinned_to_the_stack_region() -> None:
    star = [s for s in _statements(_template()) if s["Resource"] == "*"]

    assert star
    for statement in star:
        pinned = statement["Condition"]["StringEquals"]["aws:RequestedRegion"]
        if _as_list(statement["Action"]) == ["pricing:GetProducts"]:
            assert pinned == _PRICE_LIST_REGION
        else:
            assert pinned == {"!Ref": "AWS::Region"}


def _with_statement(statement: dict[str, Any]) -> dict[str, Any]:
    template = copy.deepcopy(_template())
    policies = template["Resources"]["LiveTestRole"]["Properties"]["Policies"]
    policies[0]["PolicyDocument"]["Statement"].append(statement)
    return template


def test_a_wildcard_action_is_flagged_by_sid() -> None:
    template = _with_statement(
        {"Sid": "TooBroad", "Effect": "Allow", "Action": "bedrock-agentcore:*", "Resource": "*"}
    )

    assert any(v.startswith("TooBroad: wildcard action") for v in _violations(template))


def test_star_for_an_action_that_supports_resource_arns_is_flagged_by_sid() -> None:
    template = _with_statement(
        {
            "Sid": "InvokeAnything",
            "Effect": "Allow",
            "Action": ["bedrock-agentcore:InvokeAgentRuntime"],
            "Resource": "*",
        }
    )

    assert _violations(template) == [
        "InvokeAnything: Resource '*' for ['bedrock-agentcore:InvokeAgentRuntime']"
    ]


def test_a_hardcoded_or_wildcard_resource_arn_is_flagged() -> None:
    hardcoded = _with_statement(
        {
            "Sid": "Hardcoded",
            "Effect": "Allow",
            "Action": "bedrock-agentcore:GetAgentRuntime",
            "Resource": "arn:aws:bedrock-agentcore:us-west-2:111122223333:runtime/x-AbCdEf1234",
        }
    )
    wildcard = _with_statement(
        {
            "Sid": "AnyRuntime",
            "Effect": "Allow",
            "Action": "bedrock-agentcore:GetAgentRuntime",
            "Resource": {"!Sub": "arn:${AWS::Partition}:bedrock-agentcore:*:*:runtime/*"},
        }
    )

    assert any(v.startswith("Hardcoded:") for v in _violations(hardcoded))
    assert any(v.startswith("AnyRuntime:") for v in _violations(wildcard))


_RUNTIME_ARN = (
    "arn:${AWS::Partition}:bedrock-agentcore:${AWS::Region}:${AWS::AccountId}"
    ":runtime/${AgentRuntimeId}"
)
_ENDPOINT_ARN = _RUNTIME_ARN + "/runtime-endpoint/${AgentRuntimeEndpointName}"


def _granted_resources(template: dict[str, Any], action: str) -> set[str]:
    """Every resource an Allow statement grants ``action`` on, as its ``!Sub`` text or as is."""
    found: set[str] = set()
    for statement in _statements(template):
        if statement.get("Effect") != "Allow" or action not in _as_list(statement["Action"]):
            continue
        for resource in _as_list(statement["Resource"]):
            is_sub = isinstance(resource, dict) and "!Sub" in resource
            found.add(resource["!Sub"] if is_sub else str(resource))
    return found


@pytest.mark.parametrize(
    "action",
    [
        "bedrock-agentcore:InvokeAgentRuntime",
        "bedrock-agentcore:StopRuntimeSession",
        "bedrock-agentcore:GetAgentRuntimeEndpoint",
    ],
)
def test_the_role_grants_invoke_stop_and_endpoint_reads_on_their_scoped_arns(action: str) -> None:
    """Each action is granted on the named runtime and its named endpoint, and nothing else."""
    template = _template()

    assert _granted_resources(template, action) == {_RUNTIME_ARN, _ENDPOINT_ARN}
    # The endpoint name is a deploy-time parameter; a "*" there would widen the grant.
    pattern = template["Parameters"]["AgentRuntimeEndpointName"]["AllowedPattern"]
    assert re.fullmatch(pattern, "DEFAULT")
    assert re.fullmatch(pattern, "live")
    assert re.fullmatch(pattern, "*") is None
    assert re.fullmatch(pattern, "live/*") is None


def test_stop_and_endpoint_reads_are_not_listed_as_actions_without_resource_scope() -> None:
    """Granting the stop or the endpoint read on "*" is flagged, since both take resource ARNs."""
    template = _with_statement(
        {
            "Sid": "StopAndReadAnything",
            "Effect": "Allow",
            "Action": [
                "bedrock-agentcore:StopRuntimeSession",
                "bedrock-agentcore:GetAgentRuntimeEndpoint",
            ],
            "Resource": "*",
        }
    )

    assert _violations(template) == [
        "StopAndReadAnything: Resource '*' for "
        "['bedrock-agentcore:GetAgentRuntimeEndpoint', 'bedrock-agentcore:StopRuntimeSession']"
    ]


def test_dataset_statement_names_actions_without_wildcards() -> None:
    statements = {s["Sid"]: s for s in _statements(_template())}

    datasets = statements["ManageLiveTestDatasets"]
    assert datasets["Resource"] == [{"!Sub": _LIVE_DATASET_ARN}]
    assert "bedrock-agentcore:DeleteDataset" in datasets["Action"]
    # The one write granted on "*" sits alone, not among the read-only account checks.
    assert statements["CreateDatasetsWithoutResourceLevelPermissions"]["Action"] == [
        "bedrock-agentcore:CreateDataset"
    ]
    read_only = statements["ReadAccountSettingsWithoutResourceLevelPermissions"]["Action"]
    assert "bedrock-agentcore:CreateDataset" not in read_only
    # Any other dataset wildcard, and a wildcard dataset action, are still flagged.
    any_dataset = _with_statement(
        {
            "Sid": "AnyDataset",
            "Effect": "Allow",
            "Action": "bedrock-agentcore:GetDataset",
            "Resource": {
                "!Sub": "arn:${AWS::Partition}:bedrock-agentcore:${AWS::Region}:"
                "${AWS::AccountId}:dataset/*"
            },
        }
    )
    wild_action = _with_statement(
        {
            "Sid": "AllDatasetActions",
            "Effect": "Allow",
            "Action": "bedrock-agentcore:*Dataset*",
            "Resource": {"!Sub": _LIVE_DATASET_ARN},
        }
    )
    assert any(v.startswith("AnyDataset:") for v in _violations(any_dataset))
    assert any(v.startswith("AllDatasetActions:") for v in _violations(wild_action))


@pytest.mark.parametrize(
    "action", ["logs:StartQuery", "logs:GetQueryResults", "logs:FilterLogEvents"]
)
def test_logs_reads_reach_only_the_spans_group_the_endpoints_group_and_live_result_groups(
    action: str,
) -> None:
    """The Logs reads reach aws/spans, the endpoint's group and calibrated_agent_evaluation_live_*
    result groups only."""
    assert _granted_resources(_template(), action) == {
        _SPANS_GROUP_ARN,
        _SPANS_GROUP_ARN + ":*",
        _RUNTIME_GROUP_ARN,
        _RUNTIME_GROUP_ARN + ":*",
        _LIVE_RESULTS_GROUP_ARN,
        _LIVE_RESULTS_GROUP_ARN + ":*",
    }


def test_the_only_logs_actions_granted_are_reads() -> None:
    """The span reader's three reads and the doctor's group listing: no write, no StopQuery."""
    granted = {
        action
        for statement in _statements(_template())
        if statement.get("Effect") == "Allow"
        for action in _as_list(statement["Action"])
        if action.startswith("logs:")
    }

    assert granted == {
        "logs:DescribeLogGroups",
        "logs:FilterLogEvents",
        "logs:GetQueryResults",
        "logs:StartQuery",
    }


@pytest.mark.parametrize(
    "resource",
    [
        _LOG_GROUP + "*",
        _LOG_GROUP + "/aws/bedrock-agentcore/runtimes/*",
        _LOG_GROUP + "aws/spans*",
        _LOG_GROUP + "/aws/bedrock-agentcore/runtimes/${AgentRuntimeId}-*",
    ],
    ids=["any-group", "any-runtime-group", "spans-prefix", "any-endpoint-group"],
)
def test_a_log_group_wildcard_beyond_the_span_reader_groups_streams_is_flagged(
    resource: str,
) -> None:
    template = _with_statement(
        {
            "Sid": "QueryAnyGroup",
            "Effect": "Allow",
            "Action": ["logs:StartQuery"],
            "Resource": {"!Sub": resource},
        }
    )

    assert _violations(template) == [f"QueryAnyGroup: resource {resource!r} contains a wildcard"]


def test_a_logs_query_on_star_is_flagged() -> None:
    """StartQuery takes a log group ARN, so "*" for it is refused like any scoped action."""
    template = _with_statement(
        {"Sid": "QueryEverything", "Effect": "Allow", "Action": "logs:StartQuery", "Resource": "*"}
    )

    assert _violations(template) == ["QueryEverything: Resource '*' for ['logs:StartQuery']"]


@pytest.mark.parametrize("where", ["stack", "role"])
def test_the_descriptions_name_the_log_group_reads_and_that_aws_spans_holds_every_runtimes_spans(
    where: str,
) -> None:
    """The stack and role descriptions, which console and IAM viewers see, state the Logs grant."""
    template = _template()
    description = (
        template["Description"]
        if where == "stack"
        else template["Resources"]["LiveTestRole"]["Properties"]["Description"]
    )
    group_reads = {
        action.removeprefix("logs:")
        for statement in _statements(template)
        if statement.get("Effect") == "Allow" and statement["Resource"] != "*"
        for action in _as_list(statement["Action"])
        if action.startswith("logs:")
    }

    assert group_reads
    assert sorted(a for a in group_reads if a not in description) == []
    assert "aws/spans" in description
    assert "every runtime's spans" in description
    assert "endpoint's own log group" in description


_OMNI_QUERY_ACTIONS = {
    "cloudwatch:StartTelemetryQuerySession",
    "cloudwatch:StartTelemetryQuery",
    "cloudwatch:GetTelemetryQueryResults",
    "cloudwatch:StopTelemetryQuerySession",
}


def test_the_span_reader_may_query_telemetry_through_omni_in_the_stack_region_only() -> None:
    """The Omni path's calls and the reads StartTelemetryQuery needs, on "*" in region only."""
    granted = {
        action: statement
        for statement in _statements(_template())
        if statement.get("Effect") == "Allow"
        for action in _as_list(statement["Action"])
        if action.startswith("cloudwatch:")
    }

    for action in (*_OMNI_QUERY_ACTIONS, "cloudwatch:GetMetricData", "cloudwatch:GetRecords"):
        statement = granted[action]
        assert statement["Resource"] == "*"
        assert statement["Condition"]["StringEquals"]["aws:RequestedRegion"] == {
            "!Ref": "AWS::Region"
        }
    # Nothing that writes or deletes CloudWatch data or resources, except the one metric write,
    # held to the CalibratedAgentEvaluation/Judges namespace (checked below).
    writes = ("Put", "Delete", "Create", "Update", "Tag", "Untag", "Set", "Enable", "Disable")
    written = [a for a in granted if a.split(":", 1)[1].startswith(writes)]
    assert written == ["cloudwatch:PutMetricData"]


# The judge's live test scores with the built-in Correctness evaluator, and the simulated-run live
# test with the session-level GoalSuccessRate one, each named in the one form IAM was seen to check
# on a deployed role: the stack's own region and account.
_EVALUATOR_ARNS = {
    "arn:${AWS::Partition}:bedrock-agentcore:${AWS::Region}:${AWS::AccountId}:evaluator/Builtin.Correctness",
    "arn:${AWS::Partition}:bedrock-agentcore:${AWS::Region}:${AWS::AccountId}:evaluator/Builtin.GoalSuccessRate",
}


@pytest.mark.parametrize("action", ["bedrock-agentcore:Evaluate", "bedrock-agentcore:GetEvaluator"])
def test_the_judges_may_read_and_run_only_their_built_in_and_the_live_tests_own_evaluators(
    action: str,
) -> None:
    assert _granted_resources(_template(), action) == {*_EVALUATOR_ARNS, _LIVE_EVALUATOR_ARN}


def test_only_the_live_tests_own_evaluators_and_online_configs_may_be_deleted() -> None:
    assert _granted_resources(_template(), "bedrock-agentcore:DeleteEvaluator") == {
        _LIVE_EVALUATOR_ARN
    }
    assert _granted_resources(_template(), "bedrock-agentcore:DeleteOnlineEvaluationConfig") == {
        _LIVE_ONLINE_CONFIG_ARN
    }


def _granted_actions(template: dict[str, Any]) -> set[str]:
    return {
        action
        for statement in _statements(template)
        if statement.get("Effect") == "Allow"
        for action in _as_list(statement["Action"])
    }


def test_the_role_grants_no_alarm_dashboard_or_notification_action() -> None:
    granted = _granted_actions(_template())

    assert not [a for a in granted if a.startswith(("sns:", "events:", "sqs:"))]
    assert not [a for a in granted if "Alarm" in a or "Dashboard" in a]
    assert "cloudwatch:PutMetricAlarm" not in granted
    assert "cloudwatch:DescribeAlarms" not in granted


def test_bedrock_may_invoke_only_the_judge_model() -> None:
    granted = {
        action
        for statement in _statements(_template())
        for action in _as_list(statement["Action"])
        if action.startswith("bedrock:")
    }
    assert granted == {"bedrock:InvokeModel", "bedrock:InvokeModelWithResponseStream"}
    for action in granted:
        assert _granted_resources(_template(), action) == {
            "arn:${AWS::Partition}:bedrock:${AWS::Region}:${AWS::AccountId}:"
            "inference-profile/${JudgeInferenceProfileId}",
            _JUDGE_MODEL_IN_ANY_REGION,
            "arn:${AWS::Partition}:bedrock:::foundation-model/${JudgeFoundationModelId}",
        }


def test_invoking_any_model_is_flagged() -> None:
    template = _with_statement(
        {
            "Sid": "InvokeAnyModel",
            "Effect": "Allow",
            "Action": ["bedrock:InvokeModel"],
            "Resource": {"!Sub": "arn:${AWS::Partition}:bedrock:*::foundation-model/*"},
        }
    )
    assert _violations(template) == [
        "InvokeAnyModel: resource 'arn:${AWS::Partition}:bedrock:*::foundation-model/*' contains "
        "a wildcard"
    ]


def test_metrics_may_be_published_to_the_calibrated_agent_evaluation_judges_namespace_only() -> (
    None
):
    (statement,) = [
        s for s in _statements(_template()) if "cloudwatch:PutMetricData" in _as_list(s["Action"])
    ]
    assert _as_list(statement["Action"]) == ["cloudwatch:PutMetricData"]
    assert (
        statement["Condition"]["StringEquals"]["cloudwatch:namespace"]
        == "CalibratedAgentEvaluation/Judges"
    )


def test_the_role_may_pass_only_the_evaluation_role_and_only_to_agentcore() -> None:
    (statement,) = [s for s in _statements(_template()) if "iam:PassRole" in _as_list(s["Action"])]
    assert _as_list(statement["Action"]) == ["iam:PassRole"]
    assert _as_list(statement["Resource"]) == [{"!Ref": "EvaluationRoleArn"}]
    assert statement["Condition"]["StringEquals"]["iam:PassedToService"] == (
        "bedrock-agentcore.amazonaws.com"
    )


def test_passing_any_role_is_flagged() -> None:
    template = _with_statement(
        {"Sid": "PassAnyRole", "Effect": "Allow", "Action": ["iam:PassRole"], "Resource": "*"}
    )
    assert _violations(template) == ["PassAnyRole: Resource '*' for ['iam:PassRole']"]


def test_the_price_lookup_is_the_only_pricing_action_and_only_at_the_price_list_endpoint() -> None:
    pricing = [
        statement
        for statement in _statements(_template())
        if any(a.startswith("pricing:") for a in _as_list(statement["Action"]))
    ]
    (statement,) = pricing
    assert _as_list(statement["Action"]) == ["pricing:GetProducts"]
    assert statement["Resource"] == "*"
    assert statement["Condition"]["StringEquals"]["aws:RequestedRegion"] == _PRICE_LIST_REGION


def test_an_evaluate_grant_on_every_evaluator_is_flagged() -> None:
    template = _with_statement(
        {
            "Sid": "EvaluateAnything",
            "Effect": "Allow",
            "Action": ["bedrock-agentcore:Evaluate"],
            "Resource": "*",
        }
    )
    assert any("EvaluateAnything" in problem for problem in _violations(template))


_AUDIT_READS = {
    "AbTestArn": ("ReadTheAbTest", "bedrock-agentcore:GetABTest", "ab-test"),
    "RecommendationArn": (
        "ReadTheRecommendation",
        "bedrock-agentcore:GetRecommendation",
        "recommendation",
    ),
    "ConfigurationBundleArn": (
        "ReadTheConfigurationBundle",
        "bedrock-agentcore:GetConfigurationBundleVersion",
        "configuration-bundle",
    ),
    "OnlineEvaluationConfigArn": (
        "ReadTheOnlineEvaluationConfig",
        "bedrock-agentcore:GetOnlineEvaluationConfig",
        "online-evaluation-config",
    ),
}


def test_the_audit_reads_exist_only_when_their_arns_are_given_and_grant_one_get_each() -> None:
    template = _template()
    policies = template["Resources"]["LiveTestRole"]["Properties"]["Policies"]
    conditional = [s for p in policies for s in p["PolicyDocument"]["Statement"] if "!If" in s]

    assert len(conditional) == len(_AUDIT_READS)
    for parameter, (condition, action, kind) in _AUDIT_READS.items():
        spec = template["Parameters"][parameter]
        # Empty by default, so a redeploy that names none of them adds no statement.
        assert spec["Default"] == ""
        assert template["Conditions"][condition] == {
            "!Not": [{"!Equals": [{"!Ref": parameter}, ""]}]
        }
        [wrapped] = [s for s in conditional if s["!If"][0] == condition]
        _cond, granted, otherwise = wrapped["!If"]
        assert otherwise == {"!Ref": "AWS::NoValue"}
        assert granted["Effect"] == "Allow"
        assert granted["Action"] == [action]
        assert granted["Resource"] == [{"!Sub": f"${{{parameter}}}"}]
        pattern = spec["AllowedPattern"]
        assert re.fullmatch(pattern, "")
        good = f"arn:aws:bedrock-agentcore:us-west-2:111122223333:{kind}/fixture-AbCdEf1234"
        assert re.fullmatch(pattern, good)
        for bad in (f"{good}*", "*", good.replace(kind, "runtime"), f"{good}/versions/x"):
            assert re.fullmatch(pattern, bad) is None, bad


def test_the_audit_statements_add_no_list_write_or_trace_read() -> None:
    granted = {
        a
        for s in _statements(_template())
        if s.get("Sid", "").startswith("Audit")
        for a in _as_list(s["Action"])
    }
    assert granted == {action for _c, action, _k in _AUDIT_READS.values()}
    assert not any(
        a.split(":")[1].startswith(("List", "Create", "Update", "Delete", "Start")) for a in granted
    )
