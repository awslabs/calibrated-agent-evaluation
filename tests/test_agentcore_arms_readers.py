# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""The arm read through the AgentCore ingest's own source and both its readers, over ``Stubber``.

The span rows are served as each reader parses them, through the audit's read-only clients, so
the ingest's session re-keying and its collision refusal are its own and not a double's.
"""

from __future__ import annotations

from typing import Any

import pytest

from _agentcore_capture import captured_run

pytest.importorskip("boto3")

from botocore.stub import Stubber

from _arms_fixture import (
    AB_TEST_ARN,
    MAIN,
    RUNTIME,
    Spec,
    content_records,
    span_records,
    stub_insights,
    stub_omni,
)
from calibrated_agent_evaluation.agentcore.arms import ARM_ARN_ATTRIBUTE, RawTraceLog, read_arms
from calibrated_agent_evaluation.agentcore.errors import AgentCoreArmsMissing
from calibrated_agent_evaluation.agentcore.optimization import audit_clients
from calibrated_agent_evaluation.agentcore.readers import LogsInsightsReader, OmniReader
from calibrated_agent_evaluation.agentcore.session import AwsClients
from calibrated_agent_evaluation.agentcore.sources import (
    AgentCoreSourceConfig,
    AgentCoreTraceSource,
    TimeWindow,
)

_WINDOW = TimeWindow(start="2026-10-28T19:00:00+00:00", end="2026-10-28T20:00:00+00:00")
# Every main-window trace but the one without a runtime session, which carries one session id
# only and so is never re-keyed.
_MIXED = [s for s in MAIN if s.label != "c-nosession"]


@pytest.fixture
def clients(monkeypatch: pytest.MonkeyPatch) -> AwsClients:
    for key, value in {
        "AWS_ACCESS_KEY_ID": "testing",
        "AWS_SECRET_ACCESS_KEY": "testing",
        "CALIBRATED_AGENT_EVALUATION_AGENTCORE_SETTLE_SECONDS": "0",
    }.items():
        monkeypatch.setenv(key, value)
    monkeypatch.delenv("AWS_PROFILE", raising=False)
    return audit_clients(region="us-west-2")


def _read(
    clients: AwsClients,
    specs: list[Spec],
    *,
    omni: bool,
    rekey: bool,
    order: str = "node-first",
    runtime: str = RUNTIME,
) -> Any:
    config = AgentCoreSourceConfig(runtime=runtime, window=_WINDOW, omni=omni, rekey_sessions=rekey)
    service = "cloudwatchomni" if omni else "logs"
    stub = Stubber(clients.client(service))
    spans, content = span_records(specs, order), content_records(specs)
    (stub_omni if omni else stub_insights)(stub, spans, content)
    reader = OmniReader(clients, config) if omni else LogsInsightsReader(clients, config)
    log = RawTraceLog(reader)
    source = AgentCoreTraceSource(log, config)
    with stub:
        read = read_arms(
            source,
            ab_test_arn=AB_TEST_ARN,
            runtime=runtime,
            window=_WINDOW,
            omni=omni,
            rekey_sessions=rekey,
            rekey_collisions=(lambda: log.collisions(source.rekeyed)) if rekey else None,
            contentless=lambda: log.contentless(source.rekeyed),
        )
    stub.assert_no_pending_responses()
    return read


def _by_label(read: Any, specs: list[Spec]) -> dict[str, tuple[str, str | None]]:
    labels = {s.trace_id: s.label for s in specs}
    return {labels.get(a.trace_id) or a.trace_id: (a.status, a.variant) for a in read.assignments}


_EXPECTED = {
    "c-1": ("assigned", "C"),
    "c-2": ("assigned", "C"),
    "c-root": ("assigned", "C"),
    "c-fallback": ("assigned", "C"),
    "t-1": ("assigned", "T1"),
    "t-2": ("assigned", "T1"),
    "t-otherversion": ("assigned", "T1"),
    "t-noconfig": ("assigned", "T1"),
    "unstamped": ("unassigned", None),
    "mixed": ("refused", None),
    "split-c": ("refused", None),
    "split-t": ("refused", None),
    "other-arn": ("other-experiment", None),
    "t2": ("refused", None),
    "two-sessions": ("refused", None),
}


def test_logs_insights_node_first_rows_keep_each_trace_under_its_own_id(
    clients: AwsClients,
) -> None:
    read = _read(clients, _MIXED, omni=False, rekey=True)

    assert _by_label(read, _MIXED) == _EXPECTED
    assert all(a.rekey_sessions and not a.omni for a in read.assignments)


@pytest.mark.parametrize("order", ["node-first", "invocations-first"])
def test_omni_keeps_each_trace_under_its_own_id_in_either_row_order(
    clients: AwsClients, order: str
) -> None:
    omni = _read(clients, _MIXED, omni=True, rekey=True, order=order)
    insights = _read(clients, _MIXED, omni=False, rekey=True)

    assert _by_label(omni, _MIXED) == _EXPECTED
    assert all(a.omni for a in omni.assignments)
    assert omni.read_id != insights.read_id


@pytest.mark.parametrize("omni", [False, True], ids=["logs-insights", "omni"])
def test_without_rekeying_every_mixed_session_trace_is_unread_and_named(
    clients: AwsClients, omni: bool
) -> None:
    with pytest.raises(AgentCoreArmsMissing) as refused:
        _read(clients, _MIXED, omni=omni, rekey=False)

    message = str(refused.value)
    assert f"{len(_MIXED)} trace(s) came back unread" in message
    assert "--rekey-sessions" in message
    assert all(s.trace_id in message for s in _MIXED)
    assert "BedrockAgentCoreApp" not in message


_COLLIDING = [
    Spec("r-1", "C", sessions=["s-r"]),
    Spec("r-2", "C", sessions=["s-r"]),
    Spec("r-3", "T1", sessions=["s-r3"]),
]


def test_half_stamped_spans_on_collided_traces_name_the_half_stamped_cause(
    clients: AwsClients,
) -> None:
    # The only half-stamped spans sit on two turns re-keyed to one runtime session, and no span
    # carries both attributes, so the refusal must say half-stamped, not blame the agent's build.
    half = {ARM_ARN_ATTRIBUTE: AB_TEST_ARN}
    specs = [
        Spec("r-1", sessions=["s-r"], half_node=half),
        Spec("r-2", sessions=["s-r"], half_node=half),
    ]

    with pytest.raises(AgentCoreArmsMissing) as refused:
        _read(clients, specs, omni=False, rekey=True, order="invocations-first")

    message = str(refused.value)
    assert "half-stamped spans: s-r" in message
    assert "BedrockAgentCoreApp" not in message


@pytest.mark.parametrize("first", ["r-1", "r-2"])
def test_a_logs_insights_rekey_collision_is_one_refused_assignment_whichever_turn_was_kept(
    clients: AwsClients, first: str
) -> None:
    order = sorted(_COLLIDING, key=lambda s: s.label != first)
    read = _read(clients, order, omni=False, rekey=True, order="invocations-first")
    by_id = {a.trace_id: a for a in read.assignments}

    shared = by_id["s-r"]
    assert shared.shared_id == "rekey-collision"
    assert shared.status == "refused"
    assert shared.otlp_trace_ids == tuple(sorted(s.trace_id for s in _COLLIDING[:2]))
    assert not any(a.status == "unread" for a in read.assignments)
    assert by_id["s-r3"].variant == "T1"


def test_node_first_rows_collide_nowhere(clients: AwsClients) -> None:
    for omni in (False, True):
        read = _read(clients, _COLLIDING, omni=omni, rekey=True)
        assert _by_label(read, _COLLIDING) == {
            "r-1": ("assigned", "C"),
            "r-2": ("assigned", "C"),
            "r-3": ("assigned", "T1"),
        }


def test_a_half_stamped_collision_counts_both_turns_whichever_was_delivered(
    clients: AwsClients,
) -> None:
    from calibrated_agent_evaluation.agentcore.arms import ARM_ARN_ATTRIBUTE

    hc1 = Spec("hc-1", "C", sessions=["s-hc"])
    hc2 = Spec("hc-2", "C", sessions=["s-hc"], half_node={ARM_ARN_ATTRIBUTE: AB_TEST_ARN})
    reads = [
        _read(clients, pair, omni=False, rekey=True, order="invocations-first")
        for pair in ([hc1, hc2], [hc2, hc1])
    ]
    shared = [next(a for a in r.assignments if a.trace_id == "s-hc") for r in reads]

    assert shared[0] == shared[1]
    assert shared[0].shared_id == "rekey-collision"
    assert shared[0].spans_half_stamped == 1
    assert shared[0].spans_total == sum(len(span_records([s])) for s in (hc1, hc2))


def test_a_real_trace_without_a_gateway_carries_no_arm(clients: AwsClients) -> None:
    captured = captured_run()
    spans, content = captured["span_records"], captured["log_records"]
    runtime = spans[0]["resource"]["attributes"]["service.name"]
    config = AgentCoreSourceConfig(runtime=runtime, window=_WINDOW, rekey_sessions=True)
    stub = Stubber(clients.client("logs"))
    stub_insights(stub, spans, content)
    log = RawTraceLog(LogsInsightsReader(clients, config))
    source = AgentCoreTraceSource(log, config)

    with stub, pytest.raises(AgentCoreArmsMissing):
        read_arms(
            source,
            ab_test_arn=AB_TEST_ARN,
            runtime=runtime,
            window=_WINDOW,
            rekey_sessions=True,
            rekey_collisions=lambda: log.collisions(source.rekeyed),
        )

    invocation = next(s for s in spans if s["name"] == "POST /invocations")
    assert log.kept[0].spans_stamped == 0
    assert invocation["attributes"]["session.id"] in log.kept[0].session_ids
