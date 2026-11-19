# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""``calibrated-agent-evaluation dataset run SLICE``: one stored conversation per scenario, and what
it refuses."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import pytest

pytest.importorskip("boto3")
pytest.importorskip("strands_evals")

from _agentcore_runtime_fakes import (
    REGION,
    RUNTIME_ARN,
    StubbedAwsClients,
    add_reader_response,
    register_fake_agentcore_span_reader,
    sse_body,
    stop_params,
)
from _agentcore_runtime_queue import queue_preflight
from calibrated_agent_evaluation.agentcore import AwsClients
from calibrated_agent_evaluation.backends.local import LocalBackend
from calibrated_agent_evaluation.cli import main
from calibrated_agent_evaluation.cli.dispatch import EXIT_DOMAIN_ERROR, EXIT_OK, EXIT_USAGE
from calibrated_agent_evaluation.scenarios.slicefile import read_slice, write_slice
from calibrated_agent_evaluation.scenarios.types import PublishedVersion
from calibrated_agent_evaluation.schema.types import Trace
from test_synth_helpers import FakeActorSimulator

_REPO = Path(__file__).resolve().parent.parent
_SLICE = _REPO / "fixtures" / "scenarios" / "orderbot-simulated.json"
_PREDEFINED = _REPO / "fixtures" / "expected" / "datasets" / "slice.json"
_ANSWER = "Your order A-7781 is outside the 30-day window."
_AGENTCORE = (
    "--executor",
    "agentcore",
    "--runtime-arn",
    RUNTIME_ARN,
    "--payload-format",
    "agentcore_sse",
)


@pytest.fixture(autouse=True)
def _no_ambient_aws(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    for name in list(os.environ):
        if name.startswith("AWS_"):
            monkeypatch.delenv(name)
    for name in (
        "CALIBRATED_AGENT_EVALUATION_AGENTCORE_MAX_BILLED_CALLS",
        "CALIBRATED_AGENT_EVALUATION_CASSETTE_MODE",
        "CALIBRATED_AGENT_EVALUATION_LOCAL_ROOT",
    ):
        monkeypatch.delenv(name, raising=False)
    empty = tmp_path / "empty-aws-config"
    empty.write_text("")
    monkeypatch.setenv("AWS_CONFIG_FILE", str(empty))
    monkeypatch.setenv("AWS_SHARED_CREDENTIALS_FILE", str(empty))


class _Runtime:
    """Stands in for ``AwsClients.from_profile``: answers ``turns`` per conversation, each with a
    span read, and stops each conversation's session once. ``fail_on`` fails that invocation."""

    def __init__(self, conversations: list[int], *, fail_on: int | None = None) -> None:
        self.conversations = conversations
        self.fail_on = fail_on
        self.built: list[StubbedAwsClients] = []

    def __call__(self, *, region: str | None = None, **options: Any) -> StubbedAwsClients:
        options.pop("profile", None)
        clients = StubbedAwsClients(region=region or REGION, **options)
        queue_preflight(clients)
        sent = 0
        for turns in self.conversations:
            for _ in range(turns):
                sent += 1
                view = clients.stub("bedrock-agentcore", view=True)
                if sent == self.fail_on:
                    view.add_client_error(
                        "invoke_agent_runtime",
                        service_error_code="ValidationException",
                        service_message="bad payload",
                        http_status_code=400,
                    )
                    break
                view.add_response(
                    "invoke_agent_runtime",
                    {
                        "response": sse_body({"response": _ANSWER}),
                        "contentType": "text/event-stream",
                        "statusCode": 200,
                    },
                )
                add_reader_response(clients)
            clients.stub("bedrock-agentcore").add_response(
                "stop_runtime_session", {"statusCode": 200}, stop_params()
            )
            if self.fail_on is not None and sent >= self.fail_on:
                break
        self.built.append(clients)
        return clients

    @property
    def invocations(self) -> int:
        return sum(len(c.calls_to("InvokeAgentRuntime")) for c in self.built)


def _actor(monkeypatch: pytest.MonkeyPatch, follow_ups: int) -> None:
    """Each conversation's strands actor sends ``follow_ups`` messages, then stops."""
    import calibrated_agent_evaluation.synth.borrow.simulator as simulator

    monkeypatch.setattr(
        simulator,
        "_load_actor_simulator",
        lambda: lambda **kw: FakeActorSimulator(n_turns=follow_ups, **kw),
    )


def _runtime(monkeypatch: pytest.MonkeyPatch, runtime: _Runtime) -> _Runtime:
    monkeypatch.setattr(AwsClients, "from_profile", runtime)
    register_fake_agentcore_span_reader(monkeypatch)
    return runtime


def _run(root: Path, slice_path: Path, *flags: str) -> int:
    return main(["--root", str(root), "dataset", "run", str(slice_path), *flags])


def _stored(root: Path) -> list[Trace]:
    if not root.exists():
        return []
    with LocalBackend(root=root) as backend:
        return backend.query()


def test_dataset_run_stores_one_conversation_per_scenario(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    runtime = _runtime(monkeypatch, _Runtime([2, 2]))
    _actor(monkeypatch, follow_ups=1)
    root = tmp_path / "store"

    rc = _run(root, _SLICE, *_AGENTCORE, "--yes")

    out = capsys.readouterr().out
    assert rc == EXIT_OK, out
    assert runtime.invocations == 4
    stored = {t.input["scenario_id"]: t for t in _stored(root)}
    assert set(stored) == {"ORD-1001-sim", "ORD-1002-sim"}
    first = stored["ORD-1001-sim"]
    assert first.input["goal"] == (
        "Get a refund for order A-7781, and confirm which card it goes back to and when it "
        "will arrive."
    )
    assert first.input["assertions"] == ["Agent denies a refund outside the return window"]
    assert first.input["conversation"][0]["text"] == (
        "I want a refund for order A-7781, the blender just isn't for me."
    )
    assert first.input["turns"] == 2
    assert first.input["stop_reason"] == "actor_stopped"
    assert "dataset" not in first.input
    assert {s.span_id.split(":", 1)[0] for s in first.spans} == {"t1", "t2"}
    assert "ORD-1001-sim   2 agent turn(s)   actor_stopped" in out
    assert "ORD-1002-sim   2 agent turn(s)   actor_stopped" in out
    # Two conversations, two sessions, each stopped once.
    [clients] = runtime.built
    assert len({c.params["runtimeSessionId"] for c in clients.calls_to("InvokeAgentRuntime")}) == 2
    assert len(clients.calls_to("StopRuntimeSession")) == 2


def test_a_published_slice_is_named_on_its_conversations(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    published = tmp_path / "published.json"
    slice_ = read_slice(_SLICE)
    pointer = PublishedVersion(
        dataset_arn="arn:aws:bedrock-agentcore:us-west-2:111122223333:dataset/orderbot_sim-abc",
        dataset_id="orderbot_sim-abc",
        dataset_name="orderbot_sim",
        version="2",
        region="us-west-2",
        content_digest="0" * 64,
        published_at="2026-11-05T12:00:00+00:00",
    )
    write_slice(published, slice_.model_copy(update={"published": pointer}))
    _runtime(monkeypatch, _Runtime([1, 1]))
    _actor(monkeypatch, follow_ups=0)
    root = tmp_path / "store"

    assert _run(root, published, *_AGENTCORE, "--yes") == EXIT_OK, capsys.readouterr().err
    assert {json.dumps(t.input["dataset"], sort_keys=True) for t in _stored(root)} == {
        json.dumps({"dataset_id": "orderbot_sim-abc", "version": "2"}, sort_keys=True)
    }


def test_a_predefined_slice_cannot_be_run(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    rc = _run(tmp_path / "store", _PREDEFINED, "--agent", "corpus.agents.orderbot_strands:build")
    assert rc == EXIT_DOMAIN_ERROR
    assert "is a predefined slice" in capsys.readouterr().err
    assert _stored(tmp_path / "store") == []


def test_an_unreachable_store_is_refused_before_any_call(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    runtime = _runtime(monkeypatch, _Runtime([1, 1]))
    monkeypatch.setenv("LANGFUSE_HOST", "http://127.0.0.1:9")
    monkeypatch.setenv("LANGFUSE_PUBLIC_KEY", "pk-lf-test")
    monkeypatch.setenv("LANGFUSE_SECRET_KEY", "sk-lf-test")
    rc = main(["--backend", "langfuse", "dataset", "run", str(_SLICE), *_AGENTCORE, "--yes"])
    assert rc != EXIT_OK
    assert "Langfuse reachability check failed" in capsys.readouterr().err
    assert runtime.built == []


@pytest.mark.parametrize(
    ("flags", "complaint"),
    [
        ((), "needs --agent module:callable"),
        ((*_AGENTCORE, "--agent", "corpus.agents.orderbot_strands:build"), "--agent doesn't apply"),
        (
            ("--agent", "corpus.agents.orderbot_strands:build", "--runtime-arn", RUNTIME_ARN),
            "--runtime-arn is read only with --executor agentcore",
        ),
    ],
)
def test_agent_flags_follow_the_generate_rules(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], flags: tuple[str, ...], complaint: str
) -> None:
    assert _run(tmp_path / "store", _SLICE, *flags) == EXIT_USAGE
    assert complaint in capsys.readouterr().err


def test_a_live_run_needs_yes(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    runtime = _runtime(monkeypatch, _Runtime([1, 1]))
    rc = _run(tmp_path / "store", _SLICE, *_AGENTCORE)
    err = capsys.readouterr().err
    assert rc == EXIT_DOMAIN_ERROR
    assert "plan (nothing has run): 2 scenario(s), up to 8 agent turn(s)" in err
    assert "needs --yes" in err
    assert runtime.built == []
    assert _stored(tmp_path / "store") == []


def test_the_budget_must_cover_every_possible_turn(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    runtime = _runtime(monkeypatch, _Runtime([1, 1]))
    _actor(monkeypatch, follow_ups=0)
    # The slice's two scenarios allow 4 turns each: 8 invocations at most.
    monkeypatch.setenv("CALIBRATED_AGENT_EVALUATION_AGENTCORE_MAX_BILLED_CALLS", "7")
    assert _run(tmp_path / "a", _SLICE, *_AGENTCORE, "--yes") != EXIT_OK
    assert "up to 8 billed AgentCore call(s)" in capsys.readouterr().err
    assert runtime.built == []
    # A budget equal to the most invocations runs (these conversations stop after one turn each,
    # so their two invocations and two billed span reads fit in it).
    monkeypatch.setenv("CALIBRATED_AGENT_EVALUATION_AGENTCORE_MAX_BILLED_CALLS", "8")
    assert _run(tmp_path / "b", _SLICE, *_AGENTCORE, "--yes") == EXIT_OK, capsys.readouterr().err
    assert len(_stored(tmp_path / "b")) == 2


def test_a_failed_conversation_stores_none_of_the_run(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    # The first conversation's two turns succeed; the second conversation's first turn fails.
    runtime = _runtime(monkeypatch, _Runtime([2, 2], fail_on=3))
    _actor(monkeypatch, follow_ups=1)
    root = tmp_path / "store"
    rc = _run(root, _SLICE, *_AGENTCORE, "--yes")
    err = capsys.readouterr().err
    assert rc != EXIT_OK
    assert "turn 1 of conversation 'ORD-1002-sim'" in err
    assert _stored(root) == []
    # Both sessions were stopped, the failed one included.
    [clients] = runtime.built
    assert len(clients.calls_to("StopRuntimeSession")) == 2


# --- judging the conversations ------------------------------------------------------------------

_JUDGE = _REPO / "fixtures" / "judge-files" / "agentcore-goal-success.json"
_LIVE_GOAL_SUCCESS = json.loads(
    (
        _REPO / "fixtures" / "raw" / "agentcore" / "evaluate" / "goal-success-evaluator.json"
    ).read_text()
)
_LIVE_RUNGS = _LIVE_GOAL_SUCCESS["evaluatorConfig"]["llmAsAJudge"]["ratingScale"]["numerical"]
# The live GetEvaluator answer for a built-in leaves out the fields the service model marks
# required (its instructions, model and rung definitions), which a Stubber enforces; placeholders
# fill them, as in the committed Correctness replay fixture. Level and scale stay the live ones.
_GOAL_SUCCESS = {
    **_LIVE_GOAL_SUCCESS,
    "evaluatorConfig": {
        "llmAsAJudge": {
            "instructions": "Judge whether the conversation met the user's goal.",
            "modelConfig": {"bedrockEvaluatorModelConfig": {"modelId": "aws-managed"}},
            "ratingScale": {
                "numerical": [{**rung, "definition": f"{rung['label']}."} for rung in _LIVE_RUNGS]
            },
        }
    },
}


class _JudgedRuntime(_Runtime):
    """``_Runtime`` for the run's clients, then a second set that answers the judge's calls.

    The judge's clients are built after every conversation is stored, so each verdict is
    addressed to the session its conversation is sent as, read off the stored trace.
    """

    def __init__(
        self, conversations: list[int], verdicts: dict[str, tuple[float, str]], root: Path
    ) -> None:
        super().__init__(conversations)
        self.verdicts = verdicts
        self.root = root

    def __call__(self, *, region: str | None = None, **options: Any) -> StubbedAwsClients:
        if not self.built:
            return super().__call__(region=region, **options)
        from calibrated_agent_evaluation.agentcore.evaluate_payload import conversation_session_id

        options.pop("profile", None)
        clients = StubbedAwsClients(region=region or REGION, **options)
        clients.stub("bedrock-agentcore-control").add_response(
            "get_evaluator", _GOAL_SUCCESS, {"evaluatorId": "Builtin.GoalSuccessRate"}
        )
        with LocalBackend(root=self.root) as backend:
            stored = {t.input["scenario_id"]: t for t in backend.query()}
        for scenario_id in ("ORD-1001-sim", "ORD-1002-sim"):
            value, label = self.verdicts[scenario_id]
            session = conversation_session_id(stored[scenario_id])
            clients.stub("bedrock-agentcore", view=True).add_response(
                "evaluate",
                {
                    "evaluationResults": [
                        {
                            "evaluatorArn": _GOAL_SUCCESS["evaluatorArn"],
                            "evaluatorId": "Builtin.GoalSuccessRate",
                            "evaluatorName": "Builtin.GoalSuccessRate",
                            "explanation": "judged against the goal",
                            "value": value,
                            "label": label,
                            "context": {"spanContext": {"sessionId": session}},
                        }
                    ]
                },
            )
        self.built.append(clients)
        return clients


def _no_price_lookup(monkeypatch: pytest.MonkeyPatch) -> None:
    def refuse(_clients: Any) -> Any:
        raise RuntimeError("no price lookup offline")

    monkeypatch.setattr(
        "calibrated_agent_evaluation.agentcore.evaluate.lookup_evaluation_rates", refuse
    )


_VERDICTS = {"ORD-1001-sim": (0.0, "No"), "ORD-1002-sim": (1.0, "Yes")}


def test_judging_needs_yes(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    runtime = _runtime(monkeypatch, _JudgedRuntime([1, 1], _VERDICTS, tmp_path / "store"))
    rc = _run(tmp_path / "store", _SLICE, *_AGENTCORE, "--judge", str(_JUDGE))
    err = capsys.readouterr().err
    assert rc == EXIT_DOMAIN_ERROR
    assert "up to 2 billed Evaluate call(s), one per conversation" in err
    assert runtime.built == []


def test_judging_stores_one_advisory_verdict_per_conversation(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    root = tmp_path / "store"
    _no_price_lookup(monkeypatch)
    runtime = _runtime(monkeypatch, _JudgedRuntime([1, 1], _VERDICTS, root))
    _actor(monkeypatch, follow_ups=0)

    rc = _run(root, _SLICE, *_AGENTCORE, "--judge", str(_JUDGE), "--yes")

    out = capsys.readouterr().out
    assert rc == EXIT_OK, out
    with LocalBackend(root=root) as backend:
        verdicts = {t.input["scenario_id"]: backend.get_scores(t.trace_id) for t in backend.query()}
    assert {
        sid: [(lb.outcome, lb.kind, lb.failure_mode) for lb in found]
        for sid, found in verdicts.items()
    } == {
        "ORD-1001-sim": [("FAIL", "llm-judge", "goal_not_met")],
        "ORD-1002-sim": [("PASS", "llm-judge", "goal_not_met")],
    }
    assert "ORD-1001-sim   FAIL" in out
    assert (
        "1 of 2 scenario(s) had no assertions and were judged on their goal alone: ORD-1002-sim"
        in out
    )
    [_run_clients, judge_clients] = runtime.built
    judge_clients.assert_no_pending_responses()


def test_judging_is_refused_in_replay(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setenv("CALIBRATED_AGENT_EVALUATION_CASSETTE_MODE", "replay")
    rc = _run(
        tmp_path / "store",
        _SLICE,
        *_AGENTCORE,
        "--cassette",
        str(tmp_path / "x.cassette.json"),
        "--judge",
        str(_JUDGE),
    )
    assert rc == EXIT_USAGE
    assert "judging is a live, billed Evaluate call" in capsys.readouterr().err


def test_a_workflow_judge_file_is_refused_for_conversations(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    workflow = _REPO / "fixtures" / "judge-files" / "agentcore-correctness.json"
    rc = _run(tmp_path / "store", _SLICE, *_AGENTCORE, "--judge", str(workflow))
    assert rc == EXIT_DOMAIN_ERROR
    assert 'needs a judge file with "level": "session"' in capsys.readouterr().err


# --- offline replay of the committed recording ----------------------------------------------------

_CASSETTE = _REPO / "fixtures" / "raw" / "cassettes" / "orderbot_simulated.cassette.json"
_ORACLE = _REPO / "fixtures" / "expected" / "simulated_runs" / "orderbot-simulated.traces.json"
# Made by the Strands executor on each run, so a replay's differ from the recording's.
_PER_RUN_SPAN_FIELDS = ("span_id", "parent", "started_at", "ended_at")


def _replay(monkeypatch: pytest.MonkeyPatch, root: Path, cassette: Path = _CASSETTE) -> int:
    pytest.importorskip("strands")
    # No usable credentials: a replay that tried AWS or Bedrock would fail.
    monkeypatch.setenv("AWS_ACCESS_KEY_ID", "none")
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "none")
    monkeypatch.setenv("AWS_REGION", "us-west-2")
    monkeypatch.setenv("CALIBRATED_AGENT_EVALUATION_CASSETTE_MODE", "replay")
    return _run(
        root,
        _SLICE,
        "--agent",
        "corpus.agents.orderbot_strands:build",
        "--cassette",
        str(cassette),
    )


def _conversations(root: Path) -> list[dict[str, Any]]:
    from calibrated_agent_evaluation.schema.serde import to_dict

    traces = sorted((to_dict(t) for t in _stored(root)), key=lambda d: d["input"]["scenario_id"])
    for trace in traces:
        for span in trace["spans"]:
            for field in _PER_RUN_SPAN_FIELDS:
                span.pop(field, None)
    return traces


def test_dataset_run_replays_the_committed_conversations_offline(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    rc = _replay(monkeypatch, tmp_path / "store")
    out = capsys.readouterr().out
    assert rc == EXIT_OK, out
    assert "ORD-1001-sim   4 agent turn(s)   max_turns" in out
    assert "ORD-1002-sim   1 agent turn(s)   goal_completed" in out
    oracle = json.loads(_ORACLE.read_text())["traces"]
    for trace in oracle:
        for span in trace["spans"]:
            for field in _PER_RUN_SPAN_FIELDS:
                span.pop(field, None)
    assert _conversations(tmp_path / "store") == oracle


def test_a_replayed_run_is_offline_and_reproduces_the_conversations(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    assert _replay(monkeypatch, tmp_path / "a") == EXIT_OK
    assert _replay(monkeypatch, tmp_path / "b") == EXIT_OK
    capsys.readouterr()
    first, second = _conversations(tmp_path / "a"), _conversations(tmp_path / "b")
    assert first == second
    assert [t["trace_id"] for t in first] == [t["trace_id"] for t in second]


def test_an_edited_recording_is_refused_and_stores_nothing(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    import shutil

    cassette = tmp_path / "edited.cassette.json"
    shutil.copy(_CASSETTE, cassette)
    actor = json.loads(_CASSETTE.with_name("orderbot_simulated.actor.json").read_text())
    # The customer's turn 2 now answers a different agent reply than the one the agent gives.
    for turn in actor["turns"]:
        if turn["scenario_id"] == "ORD-1001-sim" and turn["turn"] == 2:
            turn["agent_reply_sha256"] = "0" * 64
    (tmp_path / "edited.actor.json").write_text(json.dumps(actor))

    rc = _replay(monkeypatch, tmp_path / "store", cassette)

    assert rc != EXIT_OK
    err = capsys.readouterr().err
    assert "ORD-1001-sim" in err
    assert "diverged before turn 2" in err
    assert _stored(tmp_path / "store") == []
