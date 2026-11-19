# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""The re-test's replay checks: one prompt text per trace, held out, under each arm's bundle.

The replay roots are the JSON-described stores ``make_replay_roots.py`` writes, built fresh per
test. Recommendation claim records are read from their fixtures through the audit's own
read-only clients over ``Stubber``.
"""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path
from typing import Any

import pytest

from _agentcore_capture import CAPTURED_QUESTION, captured_run
from _replay_fixture import (
    BUNDLE_ARN,
    CONTROL,
    GUARD,
    H1,
    H2,
    TARGET,
    THIRD,
    TREATMENT,
    as_run_manifest,
    build_root,
    copy_root,
    run_record,
)
from calibrated_agent_evaluation.agentcore.audit_types import BundleVersionRef
from calibrated_agent_evaluation.agentcore.errors import (
    AgentCoreDevelopmentInputMissing,
    AgentCoreDevelopmentSourceChanged,
    AgentCoreDevelopmentSourceUnread,
    AgentCoreError,
    AgentCoreNotReplayable,
    AgentCoreReplayProvenanceError,
)
from calibrated_agent_evaluation.agentcore.replay import (
    DevelopmentSet,
    OnlineEvalConfig,
    ProvenanceStore,
    check_replay,
    prompt_digest,
    prompt_text,
    provenance_digest,
    recommendation_development,
    root_development,
    unguarded_modes,
)
from calibrated_agent_evaluation.improve import ConfirmationContaminationError
from calibrated_agent_evaluation.schema.types import Trace

_REPO = Path(__file__).resolve().parents[1]


def _trace(input_: dict[str, Any], trace_id: str = "t") -> Trace:
    return Trace(
        trace_id=trace_id,
        frame="production-representative",
        ownership="probed",
        input=input_,
        output={"answer": "x"},
        taxonomy_cell={"feature": "analytics"},
    )


def _arm(version: str) -> BundleVersionRef:
    return BundleVersionRef(
        bundle_arn=BUNDLE_ARN,
        version_id=version,
        version_created_at="2026-10-28T18:00:00+00:00",
        parent_version_ids=(),
        branch_name=None,
        created_by=None,
        components=(),
        components_sha256="x",
    )


# --- the prompt text ----------------------------------------------------------------------------


def test_the_prompt_text_is_the_inputs_only_string_whatever_its_key() -> None:
    s = "How many orders did store 07 ship last week?"
    assert prompt_text(_trace({"prompt": s})) == prompt_text(_trace({"answer": s})) == s
    assert prompt_digest(s) == prompt_digest(str(prompt_text(_trace({"answer": s}))))
    for shape in ({"prompt": s, "history": []}, {"prompt": 3}, {}, {"messages": [s]}):
        assert prompt_text(_trace(shape)) is None
    assert prompt_digest(s) != prompt_digest(s + " ")


def _captured_trace(clients: Any, *, question: str | None = None) -> Trace:
    """The ingest's captured live trace, read through its source over ``Stubber`` and normalized."""
    from botocore.stub import Stubber

    from _arms_fixture import stub_insights
    from calibrated_agent_evaluation.agentcore.readers import LogsInsightsReader
    from calibrated_agent_evaluation.agentcore.replay import _normalize
    from calibrated_agent_evaluation.agentcore.sources import (
        AgentCoreSourceConfig,
        AgentCoreTraceSource,
        TimeWindow,
    )

    captured = captured_run()
    spans, content = captured["span_records"], captured["log_records"]
    if question is not None:
        swapped = json.dumps([spans, content]).replace(CAPTURED_QUESTION, question)
        spans, content = json.loads(swapped)
    runtime = spans[0]["resource"]["attributes"]["service.name"]
    window = TimeWindow(start="2026-11-06T18:00:00+00:00", end="2026-11-06T19:00:00+00:00")
    config = AgentCoreSourceConfig(runtime=runtime, window=window, rekey_sessions=True)
    stub = Stubber(clients.client("logs"))
    stub_insights(stub, spans, content)
    with stub:
        [delivered] = list(AgentCoreTraceSource(LogsInsightsReader(clients, config), config).pull())
    assert delivered.otlp is not None
    return _normalize(delivered)


@pytest.fixture
def clients(monkeypatch: pytest.MonkeyPatch) -> Any:
    pytest.importorskip("boto3")
    from calibrated_agent_evaluation.agentcore.optimization import audit_clients

    for key, value in {
        "AWS_ACCESS_KEY_ID": "testing",
        "AWS_SECRET_ACCESS_KEY": "testing",
        "CALIBRATED_AGENT_EVALUATION_AGENTCORE_SETTLE_SECONDS": "0",
    }.items():
        monkeypatch.setenv(key, value)
    monkeypatch.delenv("AWS_PROFILE", raising=False)
    return audit_clients(region="us-west-2")


def test_a_normalized_otlp_trace_and_an_executor_trace_of_one_question_match(
    clients: Any,
) -> None:
    question = "Which of your stores are open on public holidays?"
    normalized = _captured_trace(clients, question=question)

    assert prompt_text(normalized) == question
    assert prompt_digest(question) == prompt_digest(str(prompt_text(_trace({"prompt": question}))))


def test_a_real_agentcore_trace_and_an_executor_trace_of_one_question_match(clients: Any) -> None:
    normalized = _captured_trace(clients)
    executor = _trace({"prompt": CAPTURED_QUESTION})

    assert prompt_digest(str(prompt_text(normalized))) == prompt_digest(str(prompt_text(executor)))
    assert prompt_digest(CAPTURED_QUESTION + "\n") != prompt_digest(CAPTURED_QUESTION)


# --- check_replay --------------------------------------------------------------------------------


@pytest.fixture
def h1(tmp_path: Path) -> dict[str, Path]:
    return {
        name: build_root(f"{H1}/{name}", tmp_path / name)
        for name in ("control", "treatment", "development")
    }


def _check(before: Path, after: Path, development: Path | DevelopmentSet, **over: Any) -> Any:
    dev = development if isinstance(development, DevelopmentSet) else root_development(development)
    kwargs: dict[str, Any] = {
        "before_arm": _arm(CONTROL),
        "after_arm": _arm(TREATMENT),
        "before_provenance": ProvenanceStore(before),
        "after_provenance": ProvenanceStore(after),
        "development": dev,
        "prior_audits": (),
        "prior_improves": (),
        "verdict_source": "human",
        **over,
    }
    return check_replay(before, after, **kwargs)


def test_a_replay_under_the_wrong_bundle_is_refused(h1: dict[str, Path], tmp_path: Path) -> None:
    check = _check(h1["control"], h1["treatment"], h1["development"])
    assert len(check.trace_ids) == 40
    assert len(check.confirmation_digests) == 40

    def as_control(doc: dict[str, Any]) -> None:
        for p in doc["run_provenance"]:
            p["bundle_version"] = CONTROL

    wrong = build_root(f"{H1}/treatment", tmp_path / "wrong", edit=as_control)
    with pytest.raises(AgentCoreReplayProvenanceError) as refused:
        _check(h1["control"], wrong, h1["development"])
    message = str(refused.value)
    assert "h1-c00" in message
    assert str(wrong) in message
    assert CONTROL in message
    assert TREATMENT in message

    def drop_one(doc: dict[str, Any]) -> None:
        doc["run_provenance"] = [p for p in doc["run_provenance"] if p["trace_id"] != "h1-c05"]

    missing = build_root(f"{H1}/treatment", tmp_path / "missing", edit=drop_one)
    with pytest.raises(AgentCoreReplayProvenanceError) as refused:
        _check(h1["control"], missing, h1["development"])
    assert "h1-c05" in str(refused.value)
    assert str(missing) in str(refused.value)


def test_a_held_out_slice_that_shares_a_question_in_another_shape_is_refused(
    h1: dict[str, Path], tmp_path: Path
) -> None:
    overlap = build_root(f"{H1}/development-overlap", tmp_path / "overlap")

    with pytest.raises(ConfirmationContaminationError) as refused:
        _check(h1["control"], h1["treatment"], overlap)

    assert "h1-c07" in str(refused.value)
    assert "h1-d-overlap" in str(refused.value)


def test_one_trace_id_with_two_prompts_is_refused(h1: dict[str, Path], tmp_path: Path) -> None:
    def reword(doc: dict[str, Any]) -> None:
        doc["traces"][3]["input"] = {"prompt": "How many orders did store 03 ship this week?"}

    after = build_root(f"{H1}/treatment", tmp_path / "reworded", edit=reword)
    before_text = "How many orders did store 03 ship last week?"
    after_text = "How many orders did store 03 ship this week?"

    with pytest.raises(AgentCoreReplayProvenanceError) as refused:
        _check(h1["control"], after, h1["development"])

    message = str(refused.value)
    assert "h1-c03" in message
    assert prompt_digest(before_text)[:12] in message
    assert prompt_digest(after_text)[:12] in message


def test_an_input_without_one_prompt_text_cannot_be_compared(
    h1: dict[str, Path], tmp_path: Path
) -> None:
    messages = build_root(f"{H1}/development-messages", tmp_path / "messages")
    with pytest.raises(AgentCoreReplayProvenanceError) as refused:
        _check(h1["control"], h1["treatment"], messages)
    message = str(refused.value)
    assert "h1-d-messages" in message
    assert str(messages) in message
    assert "messages, user" in message

    def with_history(doc: dict[str, Any]) -> None:
        doc["traces"][0]["input"] = {**doc["traces"][0]["input"], "history": []}

    before = build_root(f"{H1}/control", tmp_path / "history", edit=with_history)
    with pytest.raises(AgentCoreReplayProvenanceError) as refused:
        _check(before, h1["treatment"], h1["development"])
    assert "h1-c00" in str(refused.value)
    assert "history, prompt" in str(refused.value)

    assert _check(h1["control"], h1["treatment"], h1["development"]).trace_ids


def test_a_reloaded_replay_root_keeps_its_run_provenance(
    h1: dict[str, Path], tmp_path: Path
) -> None:
    from calibrated_agent_evaluation.cli import main
    from calibrated_agent_evaluation.cli.dispatch import EXIT_OK

    dumped, reloaded = tmp_path / "dump", tmp_path / "reloaded"
    assert main(["--root", str(h1["treatment"]), "dump", str(dumped)]) == EXIT_OK
    assert main(["--root", str(reloaded), "load", str(dumped)]) == EXIT_OK

    first = _check(h1["control"], h1["treatment"], h1["development"])
    again = _check(h1["control"], reloaded, h1["development"])
    assert again.after_provenance_sha256 == first.after_provenance_sha256
    assert provenance_digest(ProvenanceStore(reloaded).provenance()) == (
        first.after_provenance_sha256
    )


def test_modes_no_slice_guards_are_listed(h1: dict[str, Path], tmp_path: Path) -> None:
    common = {"target": TARGET, "panel_modes": [GUARD]}
    assert unguarded_modes(h1["control"], h1["treatment"], h1["development"], **common) == (THIRD,)

    def names_target(doc: dict[str, Any]) -> None:
        doc["labels"].append({**doc["labels"][0], "failure_mode": TARGET})
        doc["labels"].append({**doc["labels"][0], "failure_mode": GUARD})

    dev = build_root(f"{H1}/development", tmp_path / "dev-target", edit=names_target)
    assert unguarded_modes(h1["control"], h1["treatment"], dev, **common) == (THIRD,)
    assert unguarded_modes(h1["control"], h1["treatment"], None, **common) == ()


def test_a_runtime_executor_manifest_supplies_run_provenance(
    h1: dict[str, Path], tmp_path: Path
) -> None:
    from calibrated_agent_evaluation.agentcore.replay import RuntimeRunManifest
    from calibrated_agent_evaluation.agentcore.run_manifest import append_run_record

    stored = ProvenanceStore(h1["treatment"]).provenance()
    log_dir = tmp_path / "logs"
    for trace_id, p in sorted(stored.items()):
        append_run_record(log_dir, run_record(trace_id, p.bundle_arn, p.bundle_version))

    manifest = RuntimeRunManifest(log_dir).provenance()
    assert {t: (p.bundle_arn, p.bundle_version) for t, p in manifest.items()} == {
        t: (p.bundle_arn, p.bundle_version) for t, p in stored.items()
    }
    assert {p.qualifier for p in manifest.values()} == {"DEFAULT"}

    append_run_record(log_dir, run_record("h1-unbundled", None, None))
    with pytest.raises(AgentCoreReplayProvenanceError) as refused:
        RuntimeRunManifest(log_dir).provenance()
    assert "h1-unbundled" in str(refused.value)


def test_a_trace_run_again_under_another_bundle_version_is_refused(
    h1: dict[str, Path], tmp_path: Path
) -> None:
    from calibrated_agent_evaluation.agentcore.replay import RuntimeRunManifest
    from calibrated_agent_evaluation.agentcore.run_manifest import append_run_record

    stored = ProvenanceStore(h1["treatment"]).provenance()
    trace_id, p = sorted(stored.items())[0]
    log_dir = tmp_path / "logs"
    append_run_record(log_dir, run_record(trace_id, p.bundle_arn, p.bundle_version))
    append_run_record(log_dir, run_record(trace_id, p.bundle_arn, p.bundle_version))
    assert set(RuntimeRunManifest(log_dir).provenance()) == {trace_id}

    other = "0" * 8 + p.bundle_version[8:]
    append_run_record(log_dir, run_record(trace_id, p.bundle_arn, other))
    with pytest.raises(AgentCoreReplayProvenanceError) as refused:
        RuntimeRunManifest(log_dir).provenance()
    assert trace_id in str(refused.value)
    assert p.bundle_version in str(refused.value)
    assert other in str(refused.value)


def test_a_replay_root_reads_its_run_manifest_when_it_holds_no_provenance_records(
    h1: dict[str, Path], tmp_path: Path
) -> None:
    from calibrated_agent_evaluation.agentcore.replay import RuntimeRunManifest, replay_provenance

    root = copy_root(h1["treatment"], tmp_path / "treatment-root")
    stored = ProvenanceStore(root).provenance()
    assert isinstance(replay_provenance(root), ProvenanceStore)

    as_run_manifest(root)
    source = replay_provenance(root)

    assert isinstance(source, RuntimeRunManifest)
    assert {t: (p.bundle_arn, p.bundle_version) for t, p in source.provenance().items()} == {
        t: (p.bundle_arn, p.bundle_version) for t, p in stored.items()
    }
    empty = tmp_path / "empty"
    empty.mkdir()
    assert replay_provenance(empty).provenance() == {}


# --- a recommendation's development set ---------------------------------------------------------


def _record(name: str, **overrides: Any) -> Any:
    pytest.importorskip("boto3")
    from _rec_fixture import read_fixture

    return read_fixture(name, **overrides)


class _Window:
    """A ``WindowRead`` double: records the window it is asked for and serves fixed traces."""

    def __init__(self, delivered: list[Any], groups: tuple[str, ...]) -> None:
        self.delivered = delivered
        self.groups = groups
        self.asked: list[tuple[str, tuple[str, str]]] = []

    def factory(self, service: str, window: tuple[str, str]) -> _Window:
        self.asked.append((service, window))
        return self

    def pull(self) -> Any:
        yield from self.delivered

    def groups_read(self) -> tuple[str, ...]:
        return self.groups


_GROUPS = ("/aws/bedrock-agentcore/runtimes/fixture-runtime-DEFAULT", "aws/spans")


def _served(*questions: str) -> list[Any]:
    from calibrated_agent_evaluation.backends.base import SourceTrace

    return [
        SourceTrace(source_id=f"w-{i}", trace=_trace({"answer": q}, f"w-{i}"))
        for i, q in enumerate(questions)
    ]


def _online(name: str) -> OnlineEvalConfig:
    from _rec_fixture import online_config

    doc = online_config(name)
    session = (doc["rule"].get("sessionConfig") or {}).get("sessionTimeoutMinutes")
    logs = doc["dataSourceConfig"]["cloudWatchLogs"]
    return OnlineEvalConfig(
        config_arn=doc["onlineEvaluationConfigArn"],
        service_name=logs["serviceNames"][0],
        log_group_names=tuple(logs.get("logGroupNames") or ()),
        log_group_name_prefixes=tuple(logs.get("logGroupNamePrefixes") or ()),
        session_timeout_minutes=session,
        updated_at=doc["updatedAt"],
    )


def test_a_recommendations_development_set_comes_from_its_own_input(tmp_path: Path) -> None:
    dev = build_root(f"{H2}/development", tmp_path / "h2-dev")
    record = _record("rec_system_prompt.json")
    window = _Window([], _GROUPS)

    spans = recommendation_development(
        record, online_config=None, source_factory=window.factory, development_root=dev
    )
    assert spans.trace_ids == tuple(sorted(record.traces_inline_trace_ids))
    assert len(spans.traces) == 10

    batch = _record("rec_batch.json")
    with pytest.raises(AgentCoreNotReplayable) as refused:
        recommendation_development(
            batch, online_config=None, source_factory=window.factory, development_root=None
        )
    assert str(batch.traces_batch_evaluation_arn) in str(refused.value)
    whole = recommendation_development(
        batch, online_config=None, source_factory=window.factory, development_root=dev
    )
    assert len(whole.traces) == 10
    empty = tmp_path / "empty"
    empty.mkdir()
    with pytest.raises(AgentCoreDevelopmentInputMissing) as refused:
        recommendation_development(
            batch, online_config=None, source_factory=window.factory, development_root=empty
        )
    assert str(empty) in str(refused.value)
    assert window.asked == []


def test_a_development_trace_id_without_an_input_is_refused_in_the_root(tmp_path: Path) -> None:
    dev = build_root(f"{H2}/development", tmp_path / "h2-dev")
    record = _record("rec_spans_missing.json")

    with pytest.raises(AgentCoreDevelopmentInputMissing) as refused:
        recommendation_development(
            record, online_config=None, source_factory=_Window([], ()).factory, development_root=dev
        )
    message = str(refused.value)
    [extra] = set(record.traces_inline_trace_ids) - set(root_development(dev).traces)
    assert extra in message
    assert "sessionSpans" in message
    assert str(dev) in message


@pytest.mark.parametrize("updated_at", ["", "yesterday", "2026-11-03T10:00:00"])
def test_an_online_evaluation_without_a_readable_update_time_is_refused_naming_it(
    updated_at: str,
) -> None:
    good = _online("online_eval_timeout5.json")

    with pytest.raises(AgentCoreError) as refused:
        dataclasses.replace(good, updated_at=updated_at)

    assert good.config_arn in str(refused.value)
    assert "development window can't be checked" in str(refused.value)


def test_an_online_evaluation_window_starts_before_its_start_time() -> None:
    record = _record("rec_online.json")
    window = _Window(_served("Which stores closed early?"), _GROUPS)

    dev = recommendation_development(
        record,
        online_config=_online("online_eval_timeout5.json"),
        source_factory=window.factory,
        development_root=None,
    )
    assert window.asked == [
        ("fixture-runtime", ("2026-10-30T08:55:00+00:00", "2026-10-30T11:00:00+00:00"))
    ]
    assert dev.margin_minutes == 65
    assert dev.updated_at == "2026-10-30T08:54:00+00:00"

    unset = _Window(_served("Which stores closed early?"), _GROUPS)
    recommendation_development(
        record,
        online_config=_online("online_eval_no_session.json"),
        source_factory=unset.factory,
        development_root=None,
    )
    assert unset.asked[0][1][0] == "2026-10-30T08:45:00+00:00"

    cloudwatch = _record("rec_cloudwatch.json")
    plain = _Window(_served("Which stores closed early?"), _GROUPS)
    recommendation_development(
        cloudwatch, online_config=None, source_factory=plain.factory, development_root=None
    )
    assert plain.asked == [
        ("fixture-runtime", ("2026-10-30T10:00:00+00:00", "2026-10-30T11:00:00+00:00"))
    ]


def test_an_online_evaluation_changed_after_the_window_start_is_refused(tmp_path: Path) -> None:
    record = _record("rec_online.json")
    window = _Window(_served("q"), _GROUPS)

    with pytest.raises(AgentCoreDevelopmentSourceChanged) as refused:
        recommendation_development(
            record,
            online_config=_online("online_eval_late_update.json"),
            source_factory=window.factory,
            development_root=None,
        )
    message = str(refused.value)
    assert "2026-10-30T08:56:00+00:00" in message
    assert "2026-10-30T10:00:00+00:00" in message
    assert "2026-10-30T08:55:00+00:00" in message
    assert window.asked == []

    # A confirmation question served in the widened window, as a trace that ran one minute before
    # startTime, is caught by the contamination check.
    before, after = (
        build_root(f"{H2}/input", tmp_path / "in"),
        build_root(f"{H2}/result", tmp_path / "out"),
    )
    early = _Window(_served("Which products did region 05 return most last month?"), _GROUPS)
    dev = recommendation_development(
        record,
        online_config=_online("online_eval_timeout5.json"),
        source_factory=early.factory,
        development_root=None,
    )
    with pytest.raises(ConfirmationContaminationError):
        _check(before, after, dev)


def test_a_development_window_must_cover_the_recommendations_log_groups() -> None:
    from calibrated_agent_evaluation.backends.base import SourceTrace

    cloudwatch = _record("rec_cloudwatch.json")
    read = _Window(_served("Which stores closed early?"), _GROUPS)
    dev = recommendation_development(
        cloudwatch, online_config=None, source_factory=read.factory, development_root=None
    )
    assert dev.log_groups == _GROUPS
    assert dev.groups_read == _GROUPS

    for name, named in (
        ("rec_cloudwatch_extra_group.json", "/aws/fixture-other-group"),
        ("rec_cloudwatch_bad_arn.json", "arn:aws:logs:us-west-2:111122223333:aws/spans"),
    ):
        with pytest.raises(AgentCoreDevelopmentSourceUnread) as refused:
            recommendation_development(
                _record(name),
                online_config=None,
                source_factory=_Window(_served("q"), _GROUPS).factory,
                development_root=None,
            )
        assert named in str(refused.value)
        assert "aws/spans" in str(refused.value)

    online = _record("rec_online.json")
    dev = recommendation_development(
        online,
        online_config=_online("online_eval_timeout5.json"),
        source_factory=_Window(_served("q"), _GROUPS).factory,
        development_root=None,
    )
    assert dev.updated_at == "2026-10-30T08:54:00+00:00"
    with pytest.raises(AgentCoreDevelopmentSourceUnread) as refused:
        recommendation_development(
            online,
            online_config=_online("online_eval_prefix.json"),
            source_factory=_Window(_served("q"), _GROUPS).factory,
            development_root=None,
        )
    assert "/aws/bedrock-agentcore/runtimes/fixture-" in str(refused.value)

    errored = _Window([SourceTrace(source_id="w-err", error="no content")], _GROUPS)
    with pytest.raises(AgentCoreDevelopmentInputMissing) as refused:
        recommendation_development(
            cloudwatch, online_config=None, source_factory=errored.factory, development_root=None
        )
    assert "w-err" in str(refused.value)
    assert "2026-10-30T10:00:00+00:00" in str(refused.value)
    with pytest.raises(AgentCoreDevelopmentInputMissing) as refused:
        recommendation_development(
            cloudwatch,
            online_config=None,
            source_factory=_Window([], _GROUPS).factory,
            development_root=None,
        )
    assert "fixture-runtime" in str(refused.value)
