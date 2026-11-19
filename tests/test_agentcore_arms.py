# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Reading each trace's A/B arm from its span stamps, through a ``TraceSource`` double."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from _agentcore_capture import captured_run
from _arms_fixture import (
    AB_TEST_ARN,
    HALF,
    MAIN,
    OTHER_ARN,
    VERSIONS,
    Double,
    Spec,
    Window,
    build,
    by_label,
    deliver,
    otlp_id,
)
from calibrated_agent_evaluation.agentcore.arms import (
    ARM_ARN_ATTRIBUTE,
    ARM_VARIANT_ATTRIBUTE,
    RawTraceLog,
    merge_arms_read,
    read_arms,
    runtime_session_id,
    span_stamp,
)
from calibrated_agent_evaluation.agentcore.errors import (
    AgentCoreArmsMissing,
    AgentCoreArmsReadChanged,
    AgentCoreTruncatedResult,
)
from calibrated_agent_evaluation.agentcore.sources import (
    AgentCoreSourceConfig,
    AgentCoreTraceSource,
    RawTrace,
    TimeWindow,
)
from calibrated_agent_evaluation.backends.base import SourceTrace

_MAIN = by_label(MAIN)


def _read(items: list[SourceTrace], **kwargs: Any) -> Any:
    return read_arms(
        Double(items),
        ab_test_arn=AB_TEST_ARN,
        runtime="fixture-runtime",
        window=Window(),
        bundle_versions=VERSIONS,
        **kwargs,
    )


def _status(read: Any) -> dict[str, tuple[str, str | None]]:
    labels = {s.trace_id: s.label for s in [*MAIN, *HALF]}
    return {labels.get(a.trace_id) or a.trace_id: (a.status, a.variant) for a in read.assignments}


_NO_FLAG = {
    "c-1": ("assigned", "C"),
    "c-2": ("assigned", "C"),
    "c-root": ("assigned", "C"),
    "c-fallback": ("assigned", "C"),
    "c-nosession": ("assigned", "C"),
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


@pytest.mark.parametrize(
    ("flags", "changes", "counts"),
    [
        ({}, {}, {"C": (5, 2, 1), "T1": (4, 2, 0), "refused": 5}),
        (
            {"applied_bundle_attribute": "config.bundle_version"},
            {"t-otherversion": ("refused", None), "t-noconfig": ("refused", None)},
            {"C": (5, 2, 1), "T1": (2, 2, 0), "refused": 7},
        ),
        (
            {
                "applied_bundle_attribute": "config.bundle_version",
                "applied_bundle_require": [("config.source", "bundle")],
            },
            {
                "t-otherversion": ("refused", None),
                "t-noconfig": ("refused", None),
                "c-fallback": ("refused", None),
            },
            {"C": (4, 2, 1), "T1": (2, 2, 0), "refused": 8},
        ),
    ],
    ids=["no-flag", "version", "version-and-value"],
)
def test_stamped_traces_are_assigned_to_their_variant(
    flags: dict[str, Any], changes: dict[str, Any], counts: dict[str, Any]
) -> None:
    read = _read(deliver(MAIN), **flags)

    assert _status(read) == {**_NO_FLAG, **changes}
    for arm in ("C", "T1"):
        traces, sessions, without = counts[arm]
        assert read.counts[f"{arm}.traces"] == traces
        assert read.counts[f"{arm}.runtime_sessions"] == sessions
        assert read.counts[f"{arm}.without_session"] == without
    assert read.counts["refused"] == counts["refused"]
    assert read.counts["unassigned"] == 1
    assert read.counts["other-experiment"] == 1
    assert read.read_id.startswith("arms-")
    assert {a.read_id for a in read.assignments} == {read.read_id}


def test_the_read_id_changes_with_the_read_path_rekey_and_conditions() -> None:
    ids = {
        _read(deliver(MAIN), **flags).read_id
        for flags in (
            {},
            {"omni": True},
            {"rekey_sessions": True},
            {"applied_bundle_attribute": "config.bundle_version"},
        )
    }
    assert len(ids) == 4


def test_a_partly_stamped_trace_is_assigned_and_counted() -> None:
    read = _read(deliver(MAIN))
    by_id = {a.trace_id: a for a in read.assignments}

    root = by_id[otlp_id("c-root")]
    assert root.status == "assigned"
    assert root.spans_stamped == 1
    assert root.spans_half_stamped == 0
    assert root.spans_total == 4
    unstamped = by_id[otlp_id("unstamped")]
    assert unstamped.status == "unassigned"
    assert unstamped.reason


def test_a_span_with_one_arm_attribute_refuses_its_trace() -> None:
    read = _read(deliver(HALF))
    by_id = {a.trace_id: a for a in read.assignments}

    h_arn = by_id[otlp_id("h-arn")]
    assert h_arn.status == "refused"
    assert h_arn.spans_half_stamped == 1
    assert h_arn.spans_stamped == h_arn.spans_total - 1
    assert ARM_ARN_ATTRIBUTE in (h_arn.reason or "")
    assert AB_TEST_ARN in (h_arn.reason or "")
    h_variant = by_id[otlp_id("h-variant")]
    assert h_variant.status == "refused"
    assert h_variant.spans_half_stamped == h_variant.spans_total
    assert "'T1'" in (h_variant.reason or "")
    assert by_id[otlp_id("h-full")].status == "assigned"
    assert by_id[otlp_id("h-full")].variant == "T1"
    assert read.counts["refused.half_stamped"] == 2
    assert read.counts["T1.traces"] == 1
    assert read.counts["C.traces"] == 0


def test_traces_the_source_could_not_deliver_as_otlp_are_unread() -> None:
    mixed_session = (
        "trace 'x' carries 2 session ids (a, b); AgentCore Evaluate rejects a mixed-session "
        "trace. Pass --rekey-sessions to collapse them to the runtime session id"
    )
    items = [
        *deliver(MAIN),
        SourceTrace(source_id="err-1", error=mixed_session),
        SourceTrace(source_id="norm-1", trace=_normalized_trace()),
    ]
    read = _read(items)
    by_id = {a.trace_id: a for a in read.assignments}

    assert by_id["err-1"].status == "unread"
    assert by_id["err-1"].reason == mixed_session
    assert by_id["norm-1"].status == "unread"
    assert read.counts["unread"] == 2
    assert read.counts["C.traces"] == 5


def _normalized_trace() -> Any:
    from calibrated_agent_evaluation.schema.serde import trace_from_dict

    fixture = Path(__file__).resolve().parents[1] / "fixtures" / "normalized" / "CLM-4471.json"
    return trace_from_dict(json.loads(fixture.read_text(encoding="utf-8")))


def test_a_source_refusal_propagates_and_stores_nothing() -> None:
    class Refusing:
        def pull(self) -> Any:
            raise AgentCoreTruncatedResult("the window matched more than the read returned")
            yield  # pragma: no cover

    with pytest.raises(AgentCoreTruncatedResult):
        read_arms(Refusing(), ab_test_arn=AB_TEST_ARN, runtime="fixture-runtime", window=Window())


def test_a_window_with_no_arm_stamps_is_refused() -> None:
    with pytest.raises(AgentCoreArmsMissing) as refused:
        _read(deliver([_MAIN["unstamped"]]))

    message = str(refused.value)
    assert ARM_ARN_ATTRIBUTE in message
    assert ARM_VARIANT_ATTRIBUTE in message
    assert "fixture-runtime" in message
    assert "BedrockAgentCoreApp" in message


def test_a_window_whose_traces_came_back_unread_names_them_instead_of_the_gateway() -> None:
    reason = "trace content did not come back"
    items = [
        *deliver([_MAIN["unstamped"]]),
        SourceTrace(source_id="err-1", error=reason),
        SourceTrace(source_id="err-2", error=reason),
    ]
    with pytest.raises(AgentCoreArmsMissing) as refused:
        _read(items)

    message = str(refused.value)
    assert "2 trace(s) came back unread" in message
    assert message.count(reason) == 1
    assert "err-1" in message
    assert "err-2" in message
    assert "BedrockAgentCoreApp" not in message


def test_a_window_stamped_only_by_halves_names_those_traces() -> None:
    half = by_label(HALF)["h-variant"]
    with pytest.raises(AgentCoreArmsMissing) as refused:
        _read(deliver([half]))

    message = str(refused.value)
    assert half.trace_id in message
    assert "BedrockAgentCoreApp" not in message


def test_a_trace_stamped_with_two_arms_is_refused() -> None:
    read = _read(deliver(MAIN))
    mixed = next(a for a in read.assignments if a.trace_id == otlp_id("mixed"))

    assert mixed.status == "refused"
    assert "C" in (mixed.reason or "")
    assert "T1" in (mixed.reason or "")
    assert read.counts["C.traces"] == 5


def test_a_runtime_session_in_two_arms_is_refused() -> None:
    read = _read(deliver(MAIN))
    by_id = {a.trace_id: a for a in read.assignments}

    for label in ("split-c", "split-t"):
        assignment = by_id[otlp_id(label)]
        assert assignment.status == "refused"
        assert "s-split" in (assignment.reason or "")
        assert "C and T1" in (assignment.reason or "")
    # The node spans carry each trace's own id as session.id; those split nothing.
    assert by_id[otlp_id("c-1")].status == "assigned"
    assert otlp_id("c-1") in by_id[otlp_id("c-1")].span_session_ids


def test_traces_delivered_under_one_source_id_are_refused_together() -> None:
    specs = [
        Spec("r-1", "C", sessions=["s-r"]),
        Spec("r-2", "C", sessions=["s-r"]),
        Spec("r-3", "T1", sessions=["s-r3"]),
    ]
    for order in (specs, list(reversed(specs))):
        read = _read(deliver(order, under={"r-1": "s-r", "r-2": "s-r"}))
        by_id = {a.trace_id: a for a in read.assignments}

        shared = by_id["s-r"]
        assert shared.status == "refused"
        assert shared.shared_id == "repeated-source-id"
        assert shared.otlp_trace_ids == tuple(sorted([otlp_id("r-1"), otlp_id("r-2")]))
        assert otlp_id("r-1") in (shared.reason or "")
        assert by_id[otlp_id("r-3")].variant == "T1"
        assert read.counts["C.traces"] == 0


def test_a_variant_other_than_c_or_t1_is_refused() -> None:
    read = _read(deliver(MAIN))
    t2 = next(a for a in read.assignments if a.trace_id == otlp_id("t2"))

    assert t2.status == "refused"
    assert "'T2'" in (t2.reason or "")


def test_traces_of_another_experiment_are_set_aside_and_named() -> None:
    read = _read(deliver(MAIN))
    other = next(a for a in read.assignments if a.trace_id == otlp_id("other-arn"))
    assert other.status == "other-experiment"
    assert other.experiment_arn == OTHER_ARN
    assert OTHER_ARN in read.arns_seen

    only_other = [_MAIN["other-arn"]]
    with pytest.raises(AgentCoreArmsMissing) as refused:
        _read([*deliver(only_other), SourceTrace(source_id="err-1", error="no content")])
    message = str(refused.value)
    assert AB_TEST_ARN in message
    assert OTHER_ARN in message
    assert "1 trace(s) came back unread" in message
    assert "err-1" in message


def test_the_named_bundle_attribute_and_value_must_match_the_arm() -> None:
    by_id_flag = {
        a.trace_id: a
        for a in _read(deliver(MAIN), applied_bundle_attribute="config.bundle_version").assignments
    }
    other = by_id_flag[otlp_id("t-otherversion")]
    assert other.status == "refused"
    assert VERSIONS["C"] in (other.reason or "")
    assert VERSIONS["T1"] in (other.reason or "")
    assert "config.bundle_version" in (by_id_flag[otlp_id("t-noconfig")].reason or "")
    assert by_id_flag[otlp_id("c-fallback")].status == "assigned"

    with_value = {
        a.trace_id: a
        for a in _read(
            deliver(MAIN),
            applied_bundle_attribute="config.bundle_version",
            applied_bundle_require=[("config.source", "bundle")],
        ).assignments
    }
    fallback = with_value[otlp_id("c-fallback")]
    assert fallback.status == "refused"
    assert "config.source" in (fallback.reason or "")
    assert "builtin_after_bundle_error" in (fallback.reason or "")
    assert "'bundle'" in (fallback.reason or "")

    plain = _read(deliver(MAIN))
    assert all(a.applied_bundle_attribute is None for a in plain.assignments)
    assert all(a.applied_bundle_require == {} for a in plain.assignments)


def test_a_traces_session_is_the_runtime_session_on_its_invocation_span() -> None:
    read = _read(deliver(MAIN))
    by_id = {a.trace_id: a for a in read.assignments}

    assert by_id[otlp_id("c-1")].runtime_session_id == "s-c1"
    assert by_id[otlp_id("c-nosession")].runtime_session_id is None
    assert by_id[otlp_id("c-nosession")].status == "assigned"
    ambiguous = by_id[otlp_id("two-sessions")]
    assert ambiguous.status == "refused"
    assert "s-a" in (ambiguous.reason or "")
    assert "s-b" in (ambiguous.reason or "")
    assert runtime_session_id(build(_MAIN["t-1"])) == "s-t1"


def test_the_one_stamp_rule() -> None:
    assert span_stamp({ARM_ARN_ATTRIBUTE: "a", ARM_VARIANT_ATTRIBUTE: "C"}) == "stamped"
    assert span_stamp({ARM_ARN_ATTRIBUTE: "a"}) == "half"
    assert span_stamp({ARM_VARIANT_ATTRIBUTE: "C"}) == "half"
    assert span_stamp({}) == "none"


def test_every_arms_read_is_kept_and_a_changed_reread_is_refused() -> None:
    first = _read(deliver(MAIN))
    assert merge_arms_read([], first) == first.assignments
    stored = list(first.assignments)
    assert merge_arms_read(stored, _read(deliver(MAIN))) == ()
    other_window = read_arms(
        Double(deliver(MAIN)),
        ab_test_arn=AB_TEST_ARN,
        runtime="fixture-runtime",
        window=Window(start="2026-10-28T19:30:00+00:00"),
    )
    assert merge_arms_read(stored, other_window) == other_window.assignments

    changed = [
        Spec("c-nosession", "T1", version="x") if s.label == "c-nosession" else s for s in MAIN
    ]
    grown = [*MAIN, Spec("c-late", "C", sessions=["s-c3"])]
    shrunk = [s for s in MAIN if s.label != "t-2"]
    for specs, expected in (
        (changed, otlp_id("c-nosession")),
        (grown, otlp_id("c-late")),
        (shrunk, otlp_id("t-2")),
    ):
        with pytest.raises(AgentCoreArmsReadChanged) as refused:
            merge_arms_read(stored, _read(deliver(specs)))
        assert first.read_id in str(refused.value)
        assert expected in str(refused.value)


# --- the AgentCore ingest's own source, with re-keying --------------------------------------


class _Reader:
    """An AgentCore ingest reader double: yields fixed raw traces in the order given."""

    def __init__(self, traces: list[RawTrace]) -> None:
        self._traces = traces

    def read_traces(self) -> Any:
        yield from self._traces

    def read_scores(self) -> Any:
        return iter(())


def _raw(spec: Spec) -> RawTrace:
    doc = build(spec)
    records = [
        {
            **span,
            "resource": {
                "attributes": {
                    "aws.log.group.names": "/aws/bedrock-agentcore/runtimes/fixture-runtime-DEFAULT"
                }
            },
        }
        for span in doc["resourceSpans"][0]["scopeSpans"][0]["spans"]
    ]
    sessions = tuple(
        dict.fromkeys(
            s["attributes"]["session.id"] for s in records if "session.id" in s["attributes"]
        )
    )
    return RawTrace(
        trace_id=spec.trace_id,
        span_records=records,
        log_records=[
            {
                "spanId": r["spanId"],
                "eventName": "gen_ai.user.message",
                "body": {"content": [{"text": "q"}]},
            }
            for r in records
            if r["name"].startswith("agent.")
        ],
        session_ids=sessions,
        runtime_session_id=spec.sessions[0] if spec.sessions else None,
    )


def _through_f029(specs: list[Spec], *, rekey: bool, monkeypatch: pytest.MonkeyPatch) -> Any:
    monkeypatch.setenv("CALIBRATED_AGENT_EVALUATION_AGENTCORE_SETTLE_SECONDS", "0")
    config = AgentCoreSourceConfig(
        runtime="fixture-runtime",
        window=TimeWindow(start="2026-10-28T19:00:00+00:00", end="2026-10-28T20:00:00+00:00"),
        rekey_sessions=rekey,
    )
    log = RawTraceLog(_Reader([_raw(s) for s in specs]))
    source = AgentCoreTraceSource(log, config)
    read = read_arms(
        source,
        ab_test_arn=AB_TEST_ARN,
        runtime="fixture-runtime",
        window=config.window,
        rekey_sessions=rekey,
        rekey_collisions=(lambda: log.collisions(source.rekeyed)) if rekey else None,
    )
    return read, log


def test_a_rekey_collision_is_refused_as_one_assignment_whichever_turn_the_ingest_kept(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    r1 = Spec("r-1", "C", sessions=["s-r"])
    r2 = Spec("r-2", "C", sessions=["s-r"], half_node={ARM_ARN_ATTRIBUTE: AB_TEST_ARN})
    r3 = Spec("r-3", "T1", sessions=["s-r3"])
    results = []
    for order in ([r1, r2, r3], [r2, r1, r3]):
        read, log = _through_f029(order, rekey=True, monkeypatch=monkeypatch)
        by_id = {a.trace_id: a for a in read.assignments}
        shared = by_id["s-r"]
        assert shared.shared_id == "rekey-collision"
        assert shared.otlp_trace_ids == tuple(sorted([r1.trace_id, r2.trace_id]))
        assert not any(a.status == "unread" for a in read.assignments)
        assert by_id["s-r3"].variant == "T1"
        assert "/aws/bedrock-agentcore/runtimes/fixture-runtime-DEFAULT" in log.groups_read()
        results.append(shared)
    assert results[0] == results[1]
    assert results[0].spans_half_stamped == 1


def test_without_rekeying_f029s_mixed_session_traces_come_back_unread(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    with pytest.raises(AgentCoreArmsMissing) as refused:
        _through_f029(
            [Spec("c-1", "C", sessions=["s-c1"]), Spec("t-1", "T1", sessions=["s-t1"])],
            rekey=False,
            monkeypatch=monkeypatch,
        )
    message = str(refused.value)
    assert "2 trace(s) came back unread" in message
    assert "--rekey-sessions" in message
    assert "BedrockAgentCoreApp" not in message


def test_a_real_trace_without_a_gateway_carries_no_arm(monkeypatch: pytest.MonkeyPatch) -> None:
    captured = captured_run()
    run, spans, content = captured["trace_id"], captured["span_records"], captured["log_records"]
    sessions = tuple(
        dict.fromkeys(
            s["attributes"]["session.id"]
            for s in spans
            if "session.id" in (s.get("attributes") or {})
        )
    )
    raw = RawTrace(trace_id=run, span_records=spans, log_records=content, session_ids=sessions)
    monkeypatch.setenv("CALIBRATED_AGENT_EVALUATION_AGENTCORE_SETTLE_SECONDS", "0")
    config = AgentCoreSourceConfig(
        runtime="fixture-runtime",
        window=TimeWindow(start="2026-11-06T18:00:00+00:00", end="2026-11-06T19:00:00+00:00"),
        rekey_sessions=True,
    )
    log = RawTraceLog(_Reader([raw]))
    source = AgentCoreTraceSource(log, config)

    with pytest.raises(AgentCoreArmsMissing):
        read_arms(
            source,
            ab_test_arn=AB_TEST_ARN,
            runtime="fixture-runtime",
            window=config.window,
            rekey_sessions=True,
            rekey_collisions=lambda: log.collisions(source.rekeyed),
        )
    invocation = next(s for s in spans if s.get("name") == "POST /invocations")
    assert log.kept[0].spans_stamped == 0
    assert invocation["attributes"]["session.id"] in log.kept[0].session_ids


def test_the_stamped_fixture_files_hold_the_builders_windows() -> None:
    from _arms_fixture import STAMPED_DIR, window_files
    from _fixture_json import read_json

    root = Path(__file__).resolve().parents[1] / STAMPED_DIR
    for name, docs in window_files().items():
        assert read_json(root / f"{name}.json") == docs, name
