# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""``online``: put a calibrated Calibrated Agent Evaluation judge into AgentCore online evaluation.

``create-evaluator`` creates a prompt judge file's judge as an AgentCore LLM-as-judge evaluator
and writes an ``agentcore`` judge file naming it. ``attach`` attaches certified, bias-cleared
``agentcore`` judges to a new AgentCore online evaluation config on a deployed runtime, and stores
the config's record. ``read``, ``publish`` and ``cost``
(:mod:`calibrated_agent_evaluation.cli.stages.online_read`) work with the config once it has scored
sessions. Without ``--yes``, ``create-evaluator`` and ``attach`` show the exact request and send
nothing. They need the ``[agentcore]`` extra.
"""

from __future__ import annotations

import argparse
import contextlib
import os
from pathlib import Path
from typing import TYPE_CHECKING

from calibrated_agent_evaluation.cli.dispatch import EXIT_OK, StageOutcome
from calibrated_agent_evaluation.cli.errors import CliError, UsageError

if TYPE_CHECKING:
    from calibrated_agent_evaluation.agentcore.online import OnlineFilter

# Which flags each action reads; any other flag given is refused rather than silently ignored.
_FLAGS: dict[str, frozenset[str]] = {
    "create-evaluator": frozenset({"judge", "name", "out", "wait", "yes", "profile", "region"}),
    "attach": frozenset(
        {
            "judge",
            "name",
            "runtime_arn",
            "endpoint",
            "role_arn",
            "sampling",
            "session_timeout",
            "filters",
            "wait",
            "yes",
            "profile",
            "region",
        }
    ),
    "read": frozenset(
        {
            "judge",
            "config",
            "since",
            "until",
            "omni",
            "wait_minutes",
            "drift_baseline",
            "drift_feature",
            "seed",
            "profile",
            "region",
        }
    ),
    "publish": frozenset({"judge", "config", "profile", "region"}),
    "cost": frozenset({"config", "profile", "region"}),
}
_REQUIRED: dict[str, tuple[str, ...]] = {
    "create-evaluator": ("judge",),
    "attach": ("judge", "name", "sampling"),
    "read": ("judge",),
    "publish": ("judge",),
    "cost": (),
}
_FLAG_NAMES = {
    "judge": "--judge",
    "name": "--name",
    "out": "--out",
    "runtime_arn": "--runtime-arn",
    "endpoint": "--endpoint",
    "role_arn": "--role-arn",
    "sampling": "--sampling",
    "session_timeout": "--session-timeout",
    "filters": "--filter",
    "wait": "--wait",
    "yes": "--yes",
    "profile": "--profile",
    "region": "--region",
    "config": "--config",
    "since": "--since",
    "until": "--until",
    "omni": "--omni",
    "wait_minutes": "--wait-minutes",
    "drift_baseline": "--drift-baseline",
    "drift_feature": "--drift-feature",
    "seed": "--seed",
}
RUNTIME_ARN_ENV_VAR = "CALIBRATED_AGENT_EVALUATION_AGENTCORE_RUNTIME_ARN"
EVAL_ROLE_ENV_VAR = "CALIBRATED_AGENT_EVALUATION_AGENTCORE_EVAL_ROLE_ARN"
# The calls attach makes; its clients refuse any other.
_ATTACH_OPERATIONS = frozenset(
    {
        ("bedrock-agentcore-control", "GetEvaluator"),
        ("bedrock-agentcore-control", "GetAgentRuntime"),
        ("bedrock-agentcore-control", "ListOnlineEvaluationConfigs"),
        ("bedrock-agentcore-control", "CreateOnlineEvaluationConfig"),
        ("bedrock-agentcore-control", "GetOnlineEvaluationConfig"),
        ("pricing", "GetProducts"),
    }
)


def online_handler(args: argparse.Namespace) -> StageOutcome:
    """Run the ``calibrated-agent-evaluation online`` action ``args.action`` names."""
    action = str(args.action)
    _check_flags(args, action)
    if action == "attach":
        return _attach(args)
    if action in ("read", "publish", "cost"):
        from calibrated_agent_evaluation.cli.stages import online_read

        handler = {
            "read": online_read.online_read,
            "publish": online_read.online_publish,
            "cost": online_read.online_cost,
        }[action]
        return handler(args)
    return _create_evaluator(args)


def _check_flags(args: argparse.Namespace, action: str) -> None:
    # By identity: a value of 0 is given, though 0 == False.
    given = {
        name
        for name in _FLAG_NAMES
        if (value := getattr(args, name, None)) is not None and value is not False
    }
    stray = sorted(_FLAG_NAMES[n] for n in given - _FLAGS[action])
    if stray:
        raise UsageError(
            f"`calibrated-agent-evaluation online {action}` does not take {', '.join(stray)}"
        )
    missing = [_FLAG_NAMES[n] for n in _REQUIRED[action] if n not in given]
    if missing:
        raise UsageError(
            f"`calibrated-agent-evaluation online {action}` needs {', '.join(missing)}"
        )


def _create_evaluator(args: argparse.Namespace) -> StageOutcome:
    from calibrated_agent_evaluation.cli.stages.audit import require_agentcore

    require_agentcore("online create-evaluator")
    from calibrated_agent_evaluation import agentcore
    from calibrated_agent_evaluation.agentcore.errors import AgentCoreRunDeclined
    from calibrated_agent_evaluation.agentcore.evaluator_create import (
        PromptJudge,
        create_evaluator,
        create_evaluator_guard,
    )
    from calibrated_agent_evaluation.agentcore.online import WritePreview
    from calibrated_agent_evaluation.cli.judgefile import load_judge_file

    if len(args.judge) != 1:
        raise UsageError(
            "`calibrated-agent-evaluation online create-evaluator` takes one --judge FILE"
        )
    judge_path = Path(args.judge[0])
    jf = load_judge_file(judge_path)
    if jf.judge.level != "workflow":
        raise CliError(
            f"{judge_path} is a span-level judge; an AgentCore evaluator at level TRACE grades "
            "whole traces, so only a workflow-level judge can be created as one"
        )
    judge = PromptJudge(
        judge_id=jf.judge.judge_id,
        model=jf.judge.model,
        prompt_version=jf.judge.prompt_version,
        prompt_template=jf.judge.prompt_template,
        failure_mode=jf.failure_mode,
        agent=jf.agent,
        temperature=jf.judge.temperature,
    )
    out = Path(args.out) if args.out else Path.cwd() / f"{judge_path.stem}.agentcore.json"
    shown: list[WritePreview] = []

    def confirm(preview: WritePreview) -> bool:
        shown.append(preview)
        return bool(args.yes)

    wait = 120.0 if args.wait is None else float(args.wait)
    if wait <= 0:
        raise UsageError("--wait must be a positive number of seconds")
    with agentcore.AwsClients.from_profile(
        region=args.region, profile=args.profile, handlers=[create_evaluator_guard()]
    ) as clients:
        try:
            created = create_evaluator(
                judge, out=out, clients=clients, confirm=confirm, name=args.name, wait_seconds=wait
            )
        except AgentCoreRunDeclined:
            if args.yes or not shown:
                raise
            summary = [
                shown[0].describe(),
                "Nothing was sent. Run the same command with --yes to create it.",
            ]
            return _outcome(summary, [])
    summary = [
        shown[0].describe(),
        f"Created the AgentCore evaluator {created.evaluator_name} ({created.evaluator_id}); it "
        f"is ACTIVE, updated {created.updated_at}.",
        f"Wrote the agentcore judge file {created.judge_file}.",
        f"Next: check it with `calibrated-agent-evaluation judge --check {created.judge_file}`.",
    ]
    return _outcome(summary, [f"  {created.evaluator_arn}"])


def _outcome(summary: list[str], detail: list[str]) -> StageOutcome:
    return StageOutcome(
        stage="online", exit_code=EXIT_OK, summary="\n".join(summary), detail="\n".join(detail)
    )


def _attach(args: argparse.Namespace) -> StageOutcome:
    from calibrated_agent_evaluation import agentcore
    from calibrated_agent_evaluation.agentcore.errors import AgentCoreError, AgentCoreRunDeclined
    from calibrated_agent_evaluation.agentcore.judge import (
        build_agentcore_judge,
        load_agentcore_judge_file,
    )
    from calibrated_agent_evaluation.agentcore.online import (
        OnlineJudge,
        WritePreview,
        check_config_request,
        create_online_config,
    )
    from calibrated_agent_evaluation.cli.backends import local_root
    from calibrated_agent_evaluation.cli.judgefile import declares_agentcore_backend
    from calibrated_agent_evaluation.cli.records import (
        append_records,
        latest_for_judge,
        load_records_of,
    )

    runtime_arn = args.runtime_arn or os.environ.get(RUNTIME_ARN_ENV_VAR, "").strip()
    role_arn = args.role_arn or os.environ.get(EVAL_ROLE_ENV_VAR, "").strip()
    absent = [
        f"{flag} (or {env})"
        for flag, env, value in (
            ("--runtime-arn", RUNTIME_ARN_ENV_VAR, runtime_arn),
            ("--role-arn", EVAL_ROLE_ENV_VAR, role_arn),
        )
        if not value
    ]
    if absent:
        raise UsageError(
            f"`calibrated-agent-evaluation online attach` needs {' and '.join(absent)}; nothing "
            "was sent"
        )
    filters = [_filter(key, operator, value) for key, operator, value in args.filters or []]
    timeout = 15 if args.session_timeout is None else int(args.session_timeout)
    wait = 300.0 if args.wait is None else float(args.wait)
    if wait <= 0:
        raise UsageError("--wait must be a positive number of seconds")
    try:
        # Before any client is opened, so a value AgentCore would refuse sends nothing.
        check_config_request(args.name, float(args.sampling), timeout, filters)
    except AgentCoreError as exc:
        raise CliError(str(exc)) from exc
    files = []
    for path in args.judge:
        # Read before any client is opened, so a missing or malformed file is reported as such.
        if not Path(path).is_file():
            raise CliError(f"cannot read judge file {path}: it does not exist")
        if not declares_agentcore_backend(path):
            raise CliError(
                f"{path} is not an agentcore judge file; attach takes the judge file "
                "`calibrated-agent-evaluation online create-evaluator` wrote"
            )
        try:
            files.append(load_agentcore_judge_file(path))
        except AgentCoreError as exc:
            raise CliError(str(exc)) from exc
    root = local_root(args)
    calibrations = load_records_of(root, "calibration")
    batteries = load_records_of(root, "bias")
    shown: list[WritePreview] = []

    def confirm(preview: WritePreview) -> bool:
        shown.append(preview)
        return bool(args.yes)

    with contextlib.ExitStack() as held:
        clients = held.enter_context(
            agentcore.AwsClients.from_profile(
                region=args.region,
                profile=args.profile,
                handlers=[agentcore.ReadOnlyGuard(_ATTACH_OPERATIONS)],
            )
        )
        judges: list[OnlineJudge] = []
        for loaded in files:
            judge = build_agentcore_judge(loaded, clients=clients)
            judges.append(
                OnlineJudge(
                    judge=judge,
                    calib=latest_for_judge(calibrations, judge),
                    battery=latest_for_judge(batteries, judge),
                )
            )
        try:
            record = create_online_config(
                judges,
                name=args.name,
                runtime_arn=runtime_arn,
                endpoint=args.endpoint or "DEFAULT",
                execution_role_arn=role_arn,
                sampling_percentage=float(args.sampling),
                session_timeout_minutes=timeout,
                filters=filters,
                session_logs=load_records_of(root, "agentcore-session"),
                clients=clients,
                confirm=confirm,
                wait_seconds=wait,
            )
        except AgentCoreRunDeclined:
            if args.yes or not shown:
                raise
            summary = [
                shown[0].describe(),
                "Nothing was sent. Run the same command with --yes to create it.",
            ]
            return _outcome(summary, [])
    append_records(root, "agentcore-online", [record])
    summary = [
        shown[0].describe(),
        f"Created the online evaluation config {record.config_name} ({record.config_id}); it is "
        f"ACTIVE and scores {record.sampling_percentage:g}% of {record.runtime_service_name}'s "
        f"sessions once each has been idle {record.session_timeout_minutes} minute(s).",
        f"Recorded it with {len(record.evaluators)} judge(s) and "
        f"{len(record.excluded_session_ids)} calibration and bias session id(s) to skip on "
        "read-back.",
    ]
    detail = [f"  {e.evaluator_id}: fingerprint {e.fingerprint}" for e in record.evaluators]
    return _outcome(summary, detail)


def _filter(key: str, operator: str, raw: str) -> OnlineFilter:
    """A ``--filter KEY OPERATOR VALUE``, its value read as a boolean, a number or a string."""
    from calibrated_agent_evaluation.agentcore.online import OnlineFilter

    value: str | float | bool
    if raw.lower() in ("true", "false"):
        value = raw.lower() == "true"
    else:
        try:
            value = float(raw)
        except ValueError:
            value = raw
    return OnlineFilter(key=key, operator=operator, value=value)
