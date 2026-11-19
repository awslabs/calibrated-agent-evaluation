# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""The online judge against a live AgentCore account, under the live-test role.

Each test creates what it needs under a ``calibrated_agent_evaluation_live_<random>`` name, which
is all the role may read, run or delete, and deletes it before it ends. The online-config test
attaches a certified judge the caller supplies (certifying one is 324 billed calls) and deletes only
the config. Creating an evaluator is not billed; scoring a trace through it is one billed
``Evaluate`` call.
"""

from __future__ import annotations

import json
import secrets
from pathlib import Path
from typing import TYPE_CHECKING, Any

import pytest

from _helpers import FIXTURES, load_quickstart_trace

if TYPE_CHECKING:
    from calibrated_agent_evaluation.agentcore.pricing import RunEstimate

pytestmark = [pytest.mark.integration, pytest.mark.agentcore]

_QUICKSTART = (
    FIXTURES.parent / "src" / "calibrated_agent_evaluation" / "_quickstart" / "data" / "certify"
)
# Q-20000 in the quickstart set's traces file (gold PASS).
_Q_20000 = 162


def _prompt_judge(version: int) -> Any:
    from calibrated_agent_evaluation.agentcore.evaluator_create import PromptJudge

    doc = json.loads((_QUICKSTART / f"judge-v{version}.json").read_text(encoding="utf-8"))
    return PromptJudge(
        judge_id=doc["judge_id"],
        model=doc["model"],
        prompt_version=doc["prompt_version"],
        prompt_template=doc["prompt_template"],
        failure_mode=doc["failure_mode"],
        agent=doc.get("agent"),
    )


def test_the_quickstart_judge_is_created_read_back_scores_a_trace_and_is_deleted(
    live_budget: Any, live_region: str | None, tmp_path: Path, record_property: Any
) -> None:
    from calibrated_agent_evaluation.agentcore import AwsClients
    from calibrated_agent_evaluation.agentcore.errors import AgentCoreResourceNotActive
    from calibrated_agent_evaluation.agentcore.evaluate import run_agentcore_judge
    from calibrated_agent_evaluation.agentcore.evaluator_create import create_evaluator
    from calibrated_agent_evaluation.agentcore.judge import (
        build_agentcore_judge,
        load_agentcore_judge_file,
    )
    from calibrated_agent_evaluation.agentcore.online import call_aws

    name = f"calibrated_agent_evaluation_live_{secrets.token_hex(4)}"
    out = tmp_path / "judge-v2.agentcore.json"
    estimates: list[RunEstimate] = []

    def approve(estimate: RunEstimate) -> bool:
        estimates.append(estimate)
        return True

    with AwsClients.from_profile(region=live_region, handlers=[live_budget]) as clients:
        try:
            created = create_evaluator(
                _prompt_judge(2), out=out, clients=clients, confirm=lambda _: True, name=name
            )
        except AgentCoreResourceNotActive as exc:
            # It was created but never became active: delete it before failing.
            call_aws(
                clients, "bedrock-agentcore-control", "DeleteEvaluator", {"evaluatorId": exc.name}
            )
            raise
        try:
            got = call_aws(
                clients,
                "bedrock-agentcore-control",
                "GetEvaluator",
                {"evaluatorId": created.evaluator_id},
            )
            assert got["evaluatorType"] == "Custom"
            assert got["level"] == "TRACE"
            scale = got["evaluatorConfig"]["llmAsAJudge"]["ratingScale"]["numerical"]
            assert [(r["value"], r["label"]) for r in scale] == [(0.0, "FAIL"), (1.0, "PASS")]

            judge = build_agentcore_judge(load_agentcore_judge_file(out), clients=clients)
            assert judge.calibratability.value == "calibratable"
            assert judge.updated_at == created.updated_at
            trace = load_quickstart_trace(_Q_20000)
            assert trace.trace_id == "Q-20000"
            (label,) = run_agentcore_judge(judge, [trace], clients=clients, confirm=approve)
        finally:
            call_aws(
                clients,
                "bedrock-agentcore-control",
                "DeleteEvaluator",
                {"evaluatorId": created.evaluator_id},
            )

    assert label.kind == "llm-judge"
    assert label.outcome in ("PASS", "FAIL")
    assert f"[fingerprint={judge.fingerprint}]" in label.critique
    # The verdict is recorded next to the gold (PASS), not asserted: agreement is what calibration
    # measures.
    record_property("q20000_verdict", label.outcome)
    record_property("q20000_gold", "PASS")


def test_the_created_quickstart_judge_is_calibrated_on_round_two_and_deleted(
    live_budget: Any, live_region: str | None, tmp_path: Path, record_property: Any
) -> None:
    """Calibrates the AgentCore-run v2 judge on the quickstart's round-2 gold: 324 billed calls.

    The result (CERTIFIED or REFUSED, with both lower bounds) is recorded next to the gold source,
    not asserted: whether the AgentCore-run prompt clears the gate is what calibration measures.
    """
    import gzip

    from calibrated_agent_evaluation.agentcore import AwsClients
    from calibrated_agent_evaluation.agentcore.calibrate import (
        agentcore_locked_test_set,
        gold_source,
    )
    from calibrated_agent_evaluation.agentcore.errors import AgentCoreResourceNotActive
    from calibrated_agent_evaluation.agentcore.evaluator_create import create_evaluator
    from calibrated_agent_evaluation.agentcore.judge import (
        build_agentcore_judge,
        load_agentcore_judge_file,
    )
    from calibrated_agent_evaluation.agentcore.online import call_aws
    from calibrated_agent_evaluation.calibrate import InMemoryLockedTestLedger
    from calibrated_agent_evaluation.calibrate.engine import certify
    from calibrated_agent_evaluation.schema.serde import label_from_dict, trace_from_dict

    def rows(name: str) -> list[dict[str, Any]]:
        with gzip.open(_QUICKSTART / name, "rt", encoding="utf-8") as lines:
            return [json.loads(line) for line in lines]

    traces = {
        t.trace_id: t
        for t in map(trace_from_dict, rows("traces.jsonl.gz"))
        if t.taxonomy_cell.get("label_round") == "2"
    }
    gold = [
        (traces[label.trace_id], label)
        for label in map(label_from_dict, rows("gold.jsonl.gz"))
        if label.trace_id in traces
    ]
    assert len(gold) == 324, "round 2 holds 162 FAIL and 162 PASS gold traces"

    name = f"calibrated_agent_evaluation_live_{secrets.token_hex(4)}"
    out = tmp_path / "judge-v2.agentcore.json"
    with AwsClients.from_profile(region=live_region, handlers=[live_budget]) as clients:
        try:
            created = create_evaluator(
                _prompt_judge(2), out=out, clients=clients, confirm=lambda _: True, name=name
            )
        except AgentCoreResourceNotActive as exc:
            call_aws(
                clients, "bedrock-agentcore-control", "DeleteEvaluator", {"evaluatorId": exc.name}
            )
            raise
        try:
            judge = build_agentcore_judge(load_agentcore_judge_file(out), clients=clients)
            run = agentcore_locked_test_set(
                judge, gold, clients=clients, confirm=lambda _: True, set_id="live-round-2"
            )
        finally:
            call_aws(
                clients,
                "bedrock-agentcore-control",
                "DeleteEvaluator",
                {"evaluatorId": created.evaluator_id},
            )

    record = certify(judge, run.calibration_set, ledger=InMemoryLockedTestLedger())
    assert record.model == created.evaluator_arn
    assert record.judge_fingerprint == judge.fingerprint
    assert len(run.session_ids) == 324
    record_property("round2_result", record.result)
    record_property("round2_tpr_lb95", f"{record.tpr.lb95:.3f}")
    record_property("round2_tnr_lb95", f"{record.tnr.lb95:.3f}")
    record_property("round2_gold_source", gold_source(label for _, label in gold))


# How long after a session goes idle its online result may take to land.
_RESULT_WAIT_SECONDS = 30 * 60
_RESULT_POLL_SECONDS = 30.0


def _online_result(clients: Any, config_id: str, session_id: str) -> dict[str, Any] | None:
    """The config's result record for ``session_id``, or ``None`` while there is none yet."""
    from botocore.exceptions import ClientError

    group = f"/aws/bedrock-agentcore/evaluations/results/{config_id}"
    pages = clients.client("logs").get_paginator("filter_log_events")
    try:
        # A page can come back empty with a token for the next one, so every page is read.
        events = [
            event
            for page in pages.paginate(logGroupName=group, filterPattern=f'"{session_id}"')
            for event in page["events"]
        ]
    except ClientError as exc:
        # The group is created with the config's first result.
        if exc.response.get("Error", {}).get("Code") == "ResourceNotFoundException":
            return None
        raise
    for event in events:
        record = json.loads(event["message"])
        if record.get("attributes", {}).get("session.id") == session_id:
            return dict(record)
    return None


def test_a_certified_judge_attached_online_scores_a_new_runtime_session(
    live_budget: Any, live_region: str | None, record_property: Any, tmp_path: Path
) -> None:
    """Attach the certified live judge, run OrderBot once, and find that session's result.

    The judge is the caller's (calibrating a new one is 324 billed calls), so only the config is
    created here and deleted at the end; the judge file and store it was certified in are named by
    ``CALIBRATED_AGENT_EVALUATION_AGENTCORE_LIVE_JUDGE`` and
    ``CALIBRATED_AGENT_EVALUATION_AGENTCORE_LIVE_STORE``.
    """
    import contextlib
    import os
    import time

    names = (
        "CALIBRATED_AGENT_EVALUATION_AGENTCORE_RUNTIME_ARN",
        "CALIBRATED_AGENT_EVALUATION_AGENTCORE_PAYLOAD_FORMAT",
        "CALIBRATED_AGENT_EVALUATION_AGENTCORE_QUESTION",
        "CALIBRATED_AGENT_EVALUATION_AGENTCORE_EVAL_ROLE_ARN",
        "CALIBRATED_AGENT_EVALUATION_AGENTCORE_LIVE_JUDGE",
        "CALIBRATED_AGENT_EVALUATION_AGENTCORE_LIVE_STORE",
    )
    values = [os.environ.get(name, "").strip() for name in names]
    if missing := [name for name, value in zip(names, values, strict=True) if not value]:
        pytest.skip(f"set {', '.join(missing)} to run this live test")
    runtime_arn, payload_format, question, role_arn, judge_file, store = values

    from calibrated_agent_evaluation.agentcore import (
        AgentCoreRuntimeExecutor,
        AgentCoreRuntimeTask,
        AgentCoreSpanReader,
        AwsClients,
        RuntimeTarget,
    )
    from calibrated_agent_evaluation.agentcore.errors import AgentCoreResourceNotActive
    from calibrated_agent_evaluation.agentcore.judge import (
        build_agentcore_judge,
        load_agentcore_judge_file,
    )
    from calibrated_agent_evaluation.agentcore.online import (
        OnlineJudge,
        call_aws,
        create_online_config,
    )
    from calibrated_agent_evaluation.calibrate.engine import certifies, pin_mismatch
    from calibrated_agent_evaluation.cli.records import latest_for_judge, load_records_of

    root = Path(store)
    target = RuntimeTarget(
        runtime_arn=runtime_arn, qualifier="DEFAULT", payload_format=payload_format
    )
    name = f"calibrated_agent_evaluation_live_{secrets.token_hex(4)}"
    with contextlib.ExitStack() as held:
        clients = held.enter_context(
            AwsClients.from_profile(region=live_region, handlers=[live_budget])
        )
        judge = build_agentcore_judge(load_agentcore_judge_file(judge_file), clients=clients)
        online = OnlineJudge(
            judge=judge,
            calib=latest_for_judge(load_records_of(root, "calibration"), judge),
            battery=latest_for_judge(load_records_of(root, "bias"), judge),
        )
        try:
            record = create_online_config(
                [online],
                name=name,
                runtime_arn=runtime_arn,
                execution_role_arn=role_arn,
                sampling_percentage=100.0,
                session_timeout_minutes=1,
                session_logs=load_records_of(root, "agentcore-session"),
                clients=clients,
                confirm=lambda _: True,
            )
        except AgentCoreResourceNotActive as exc:
            # It was created but never became active: delete it before failing.
            call_aws(
                clients,
                "bedrock-agentcore-control",
                "DeleteOnlineEvaluationConfig",
                {"onlineEvaluationConfigId": exc.name},
            )
            raise
        held.callback(
            call_aws,
            clients,
            "bedrock-agentcore-control",
            "DeleteOnlineEvaluationConfig",
            {"onlineEvaluationConfigId": record.config_id},
        )
        # Attaching locks the evaluator and moves its updatedAt; its records still certify it.
        evaluator = call_aws(
            clients,
            "bedrock-agentcore-control",
            "GetEvaluator",
            {"evaluatorId": judge.evaluator_id},
        )
        assert evaluator["lockedForModification"] is True
        rebuilt = build_agentcore_judge(load_agentcore_judge_file(judge_file), clients=clients)
        assert rebuilt.fingerprint == judge.fingerprint
        assert online.calib is not None
        assert certifies(online.calib, rebuilt)
        assert online.battery is not None
        assert online.battery.verdict == "CLEAR"
        assert pin_mismatch(online.battery, rebuilt) is None

        # Through the Runtime executor, as a user's run goes: it reads the run's spans before it
        # stops the session, so the agent's root span is exported first. A session stopped before
        # that is scored as an error ("none of the spans contain the required agent invocation").
        with AgentCoreRuntimeExecutor(
            target,
            span_reader=AgentCoreSpanReader(clients),
            clients=clients,
            log_dir=tmp_path,
        ) as executor:
            executor.run(AgentCoreRuntimeTask(prompt=question))
        (run,) = executor.run_records
        assert run.error_code is None
        assert run.session_stop == "stopped"
        session_id = run.session_id
        stopped = run.session_stop
        idle_at = time.monotonic()

        result = None
        while result is None and time.monotonic() - idle_at < _RESULT_WAIT_SECONDS:
            time.sleep(_RESULT_POLL_SECONDS)
            result = _online_result(clients, record.config_id, session_id)
        waited = time.monotonic() - idle_at

    assert result is not None, (
        f"no online result for session {session_id} within "
        f"{_RESULT_WAIT_SECONDS // 60} minutes of it going idle"
    )
    attributes = result["attributes"]
    # A result AgentCore could not produce carries error.type instead of a score.
    assert "error.type" not in attributes, attributes.get("error.message")
    assert attributes["aws.bedrock_agentcore.evaluator.arn"] == judge.model
    assert attributes["aws.bedrock_agentcore.online_evaluation_config.name"] == name
    assert attributes["gen_ai.evaluation.score.label"] in ("PASS", "FAIL")
    record_property("online_session_stop", stopped)
    record_property("online_result_seconds", f"{waited:.0f}")
    record_property("online_verdict", attributes["gen_ai.evaluation.score.label"])


_EVALUATIONS_PREFIX = "/aws/bedrock-agentcore/evaluations/"


def _evaluation_groups(logs: Any) -> set[str]:
    pages = logs.get_paginator("describe_log_groups")
    return {
        group["logGroupName"]
        for page in pages.paginate(logGroupNamePrefix=_EVALUATIONS_PREFIX)
        for group in page["logGroups"]
    }


def _events_naming(logs: Any, group: str, needle: str, **window: int) -> int | None:
    """How many events in ``group`` name ``needle`` (in ``window``); ``None`` if denied."""
    from botocore.exceptions import ClientError

    pages = logs.get_paginator("filter_log_events")
    try:
        return sum(
            len(page["events"])
            for page in pages.paginate(logGroupName=group, filterPattern=f'"{needle}"', **window)
        )
    except ClientError as exc:
        code = exc.response.get("Error", {}).get("Code")
        if code == "ResourceNotFoundException":
            return 0  # deleted since it was listed: it holds nothing
        if code == "AccessDeniedException":
            return None
        raise


def test_an_on_demand_evaluate_leaves_no_record_in_the_evaluation_log_groups(
    live_budget: Any, live_region: str | None, record_property: Any
) -> None:
    """Score one trace on demand with a judge attached online and find no record of it logged.

    The judge (``CALIBRATED_AGENT_EVALUATION_AGENTCORE_LIVE_JUDGE``) is attached to the active
    online config ``CALIBRATED_AGENT_EVALUATION_AGENTCORE_LIVE_CONFIG``, whose result group
    ``calibrated-agent-evaluation online read`` reads: an on-demand result landing there would be
    read back as an online verdict. Every other group under the evaluations prefix is searched too.
    The live-test role may read only ``calibrated_agent_evaluation_live_*`` result groups, so the
    others are read through ``CALIBRATED_AGENT_EVALUATION_AGENTCORE_LIVE_READ_PROFILE`` (read-only
    calls); a group neither can read fails the test before the billed call. The trace is a
    quickstart trace under a fresh id, so no earlier call can match it, and each group is searched
    over its whole retention: a result may be stamped with the trace's own (old) span times. The
    wait is the trace reader's settle lag for results to reach CloudWatch.
    """
    import os
    import time

    from calibrated_agent_evaluation.agentcore import AwsClients, ReadOnlyGuard
    from calibrated_agent_evaluation.agentcore.evaluate import run_agentcore_judge_with_sessions
    from calibrated_agent_evaluation.agentcore.evaluate_payload import aws_trace_id
    from calibrated_agent_evaluation.agentcore.judge import (
        build_agentcore_judge,
        load_agentcore_judge_file,
    )
    from calibrated_agent_evaluation.agentcore.online import call_aws
    from calibrated_agent_evaluation.agentcore.settle import DEFAULT_LAG_SECONDS
    from calibrated_agent_evaluation.schema.serde import to_dict, trace_from_dict

    judge_file = os.environ.get("CALIBRATED_AGENT_EVALUATION_AGENTCORE_LIVE_JUDGE", "").strip()
    config_id = os.environ.get("CALIBRATED_AGENT_EVALUATION_AGENTCORE_LIVE_CONFIG", "").strip()
    if not judge_file or not config_id:
        pytest.skip(
            "set CALIBRATED_AGENT_EVALUATION_AGENTCORE_LIVE_JUDGE and "
            "CALIBRATED_AGENT_EVALUATION_AGENTCORE_LIVE_CONFIG to run this"
        )
    read_profile = (
        os.environ.get("CALIBRATED_AGENT_EVALUATION_AGENTCORE_LIVE_READ_PROFILE", "").strip()
        or None
    )
    source = load_quickstart_trace(_Q_20000)
    fresh = f"{source.trace_id}-ondemand-{secrets.token_hex(4)}"
    # Only the id fields: the order id in the customer's message is also Q-20000.
    doc = to_dict(source)
    doc["trace_id"] = fresh
    for span in doc["spans"]:
        span["trace_id"] = fresh
    trace = trace_from_dict(doc)
    assert trace.input == source.input
    config_group = f"{_EVALUATIONS_PREFIX}results/{config_id}"

    with (
        AwsClients.from_profile(region=live_region, handlers=[live_budget]) as clients,
        AwsClients.from_profile(
            region=live_region,
            profile=read_profile,
            handlers=[ReadOnlyGuard({("logs", "DescribeLogGroups"), ("logs", "FilterLogEvents")})],
        ) as reader,
    ):
        judge = build_agentcore_judge(load_agentcore_judge_file(judge_file), clients=clients)
        config = call_aws(
            clients,
            "bedrock-agentcore-control",
            "GetOnlineEvaluationConfig",
            {"onlineEvaluationConfigId": config_id},
        )
        assert config["status"] == "ACTIVE"
        assert config["executionStatus"] == "ENABLED"
        # Its results go to its own group, under the prefix searched below.
        assert config["outputConfig"]["cloudWatchConfig"]["logGroupName"] == config_group
        assert {e["evaluatorId"] for e in config["evaluators"]} == {judge.evaluator_id}
        role_logs = clients.client("logs")
        read_logs = reader.client("logs")
        before = _evaluation_groups(role_logs)
        assert config_group in before, f"{config_group} does not exist yet"
        # Every group must be searchable before any billed call is made.
        probe_ms = int(time.time() * 1000)
        denied = {
            group
            for group in before
            if _events_naming(role_logs, group, "probe", startTime=probe_ms) is None
            and _events_naming(read_logs, group, "probe", startTime=probe_ms) is None
        }
        if denied:
            pytest.fail(
                f"cannot search {sorted(denied)}; set "
                "CALIBRATED_AGENT_EVALUATION_AGENTCORE_LIVE_READ_PROFILE to a "
                "profile that may call logs:FilterLogEvents on them"
            )

        run = run_agentcore_judge_with_sessions(
            judge, [trace], clients=clients, confirm=lambda _: True
        )
        (label,) = run.labels
        assert label.trace_id == fresh
        needles = {aws_trace_id(trace.trace_id), *run.session_ids}
        time.sleep(DEFAULT_LAG_SECONDS)
        after = _evaluation_groups(role_logs)
        assert after - before == set(), f"the call created log group(s) {sorted(after - before)}"
        found: dict[str, int] = {}
        for group in sorted(after):
            for needle in sorted(needles):
                count = _events_naming(role_logs, group, needle)
                if count is None:
                    count = _events_naming(read_logs, group, needle)
                assert count is not None, f"{group} became unreadable during the test"
                if count:
                    found[group] = found.get(group, 0) + count

    assert found == {}, f"the on-demand result was logged: {found}"
    record_property("ondemand_groups_searched", ",".join(sorted(after)))
    record_property("ondemand_config_group", config_group)
    record_property("ondemand_ids_searched", ",".join(sorted(needles)))
    record_property("ondemand_verdict", label.outcome)
