# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Behavior of the AgentCore Code Interpreter sandbox (offline, via the execute test seam)."""

from __future__ import annotations

import json

import pytest

from calibrated_agent_evaluation.agentcore.code_interpreter import CodeInterpreterSandbox
from calibrated_agent_evaluation.agentcore.errors import BilledCallBudgetExceeded, WrongAwsIdentity
from calibrated_agent_evaluation.agentcore.guards import BilledCallBudget
from calibrated_agent_evaluation.execute.cassette import Cassette, Mode
from calibrated_agent_evaluation.execute.errors import SandboxError
from calibrated_agent_evaluation.execute.sandbox import LocalSubprocessSandbox
from calibrated_agent_evaluation.scorers import CodeExecutionScorer
from calibrated_agent_evaluation.scorers.execution import _RESULT_MARKER

# The harness the scorer sends prints this line when its tests pass; the fake service echoes it as
# the session's stdout so a scorer run reads a PASS off the canned response.
_PASS_MARKER = (
    _RESULT_MARKER + json.dumps({"passed": True, "error": None, "timed_out": False}) + "\n"
)


def _ok_response(stdout: str | None = None, exit_code: int = 0) -> dict:
    """A Code Interpreter response in the REAL event-stream shape (confirmed live 2026-10-22).

    ``stdout`` defaults to the harness's passing result marker, so a scorer run off this response
    reads PASS; pass an explicit string to assert raw stdout passthrough.
    """
    text = _PASS_MARKER if stdout is None else stdout
    return {
        "sessionId": "s-test",
        "stream": [
            {
                "result": {
                    "content": [{"type": "text", "text": text}],
                    "structuredContent": {
                        "stdout": text,
                        "stderr": "",
                        "exitCode": exit_code,
                        "executionTime": 0.01,
                    },
                    "isError": False,
                }
            }
        ],
    }


def test_a_successful_response_maps_to_a_clean_result() -> None:
    sandbox = CodeInterpreterSandbox(
        region="us-west-2", execute=lambda *_: _ok_response(stdout="hi")
    )
    result = sandbox.run_code("print('hi')", timeout_s=10)
    assert result.exit_code == 0
    assert result.stdout == "hi"
    assert result.timed_out is False


def test_an_error_response_with_exit_zero_is_forced_to_fail() -> None:
    # The service flagged isError but reported exit 0; the scorer must not read that as a pass.
    sandbox = CodeInterpreterSandbox(
        region="us-west-2",
        execute=lambda *_: {
            "sessionId": "s",
            "stream": [{"result": {"structuredContent": {"exitCode": 0}, "isError": True}}],
        },
    )
    assert sandbox.run_code("boom", timeout_s=10).exit_code != 0


def test_the_verdict_matches_the_local_sandbox_for_the_fixture() -> None:
    # Parity: the same scorer gives the same PASS verdict on both backends for the passing fixture.
    output = {"code": "def add(a, b):\n    return a + b\n"}
    oracle = {"tests": "assert add(2, 3) == 5\n"}
    local = CodeExecutionScorer(LocalSubprocessSandbox()).score(output, oracle, trace_id="p")
    agent = CodeExecutionScorer(
        CodeInterpreterSandbox(region="us-west-2", execute=lambda *_: _ok_response())
    ).score(output, oracle, trace_id="p")
    assert local.outcome == agent.outcome == "PASS"


def test_a_zero_budget_refuses_before_any_call_is_sent() -> None:
    calls: list[str] = []

    def _execute(code: str, language: str, clear: bool) -> dict:
        calls.append(code)
        return _ok_response()

    sandbox = CodeInterpreterSandbox(
        region="us-west-2", budget=BilledCallBudget(0), execute=_execute
    )
    with pytest.raises(BilledCallBudgetExceeded):
        sandbox.run_code("print(1)", timeout_s=10)
    assert calls == []  # nothing was sent


def test_a_wrong_identity_is_refused_before_any_call_is_sent() -> None:
    class _StubClients:
        def caller_identity(self) -> str:
            return "arn:aws:sts::111122223333:assumed-role/SomeOther/session"

    calls: list[str] = []
    sandbox = CodeInterpreterSandbox(
        region="us-west-2",
        clients=_StubClients(),  # type: ignore[arg-type]
        role_arn="arn:aws:iam::111122223333:role/CalibratedAgentEvaluationCodeInterpreter",
        execute=lambda c, *_: calls.append(c) or _ok_response(),
    )
    with pytest.raises(WrongAwsIdentity):
        sandbox.run_code("print(1)", timeout_s=10)
    assert calls == []


def test_a_replay_miss_is_a_sandbox_error_with_no_call(tmp_path) -> None:
    calls: list[str] = []
    cassette = Cassette(tmp_path / "ci.cassette.json", mode=Mode.REPLAY)
    sandbox = CodeInterpreterSandbox(
        region="us-west-2",
        cassette=cassette,
        execute=lambda c, *_: calls.append(c) or _ok_response(),
    )
    with pytest.raises(SandboxError, match="REPLAY"):
        sandbox.run_code("print(1)", timeout_s=10)
    assert calls == []  # REPLAY never falls through to a live call


def test_a_malformed_response_is_refused_and_yields_no_label() -> None:
    sandbox = CodeInterpreterSandbox(region="us-west-2", execute=lambda *_: {"unexpected": "shape"})
    with pytest.raises(SandboxError, match="structuredContent"):
        sandbox.run_code("print(1)", timeout_s=10)
    # And through the scorer, that surfaces as ScorerError (no Label).
    from calibrated_agent_evaluation.scorers.errors import ScorerError

    with pytest.raises(ScorerError):
        CodeExecutionScorer(sandbox).score(
            {"code": "x = 1"}, {"tests": "assert True"}, trace_id="m"
        )


def test_record_then_replay_round_trips(tmp_path) -> None:
    path = tmp_path / "ci.cassette.json"
    calls: list[str] = []

    def _execute(code: str, language: str, clear: bool) -> dict:
        calls.append(code)
        return _ok_response(stdout="recorded")

    record = CodeInterpreterSandbox(
        region="us-west-2", cassette=Cassette(path, mode=Mode.RECORD), execute=_execute
    )
    first = record.run_code("print('x')", timeout_s=10)

    replay = CodeInterpreterSandbox(
        region="us-west-2",
        cassette=Cassette(path, mode=Mode.REPLAY),
        execute=_execute,  # must NOT be called on replay
    )
    second = replay.run_code("print('x')", timeout_s=10)
    assert first.stdout == second.stdout == "recorded"
    assert calls == ["print('x')"]  # the live execute ran once (record), not on replay


def test_normalize_strips_the_managed_shell_exit_note_so_the_real_error_shows() -> None:
    """The Code Interpreter shell appends an IPython 'To exit' note to stderr; the adapter drops
    it so a FAIL's critique shows the real error, not the housekeeping line."""
    from calibrated_agent_evaluation.agentcore.code_interpreter import _clean_stderr, _normalize

    raw = {
        "structuredContent": {
            "stdout": "",
            "stderr": (
                "AssertionError: add returned the wrong value\n"
                "<ipython-input>:1: UserWarning: To exit: use 'exit', 'quit', or Ctrl-D.\n"
                "  warn(\"To exit: use 'exit', 'quit', or Ctrl-D.\", stacklevel=1)\n"
            ),
            "exitCode": 1,
        },
        "isError": True,
    }
    out = _normalize(raw)
    assert "To exit" not in out["stderr"]
    assert out["stderr"].splitlines()[-1] == "AssertionError: add returned the wrong value"
    assert _clean_stderr("real error\nTo exit: use 'exit', 'quit', or Ctrl-D.") == "real error"


def test_a_non_numeric_exit_code_is_refused_as_sandbox_error_not_a_raw_crash() -> None:
    """A malformed field (non-numeric exitCode) is a shape we don't recognise: it must surface as
    SandboxError through run_code, not escape as a bare ValueError past the sandbox boundary."""
    bad_stream = {
        "sessionId": "s-test",
        "stream": [
            {
                "result": {
                    "structuredContent": {
                        "stdout": "",
                        "stderr": "",
                        "exitCode": "not-a-number",
                        "executionTime": 0.01,
                    },
                    "isError": False,
                }
            }
        ],
    }
    sandbox = CodeInterpreterSandbox(region="us-west-2", execute=lambda *_: bad_stream)
    with pytest.raises(SandboxError):
        sandbox.run_code("print(1)", timeout_s=10)


def test_a_response_missing_exit_code_is_refused_not_scored_as_a_pass() -> None:
    """A structuredContent with no exitCode must not default to 0 (a silent PASS); it is a shape we
    cannot read, so it is refused."""
    no_exit = {
        "sessionId": "s-test",
        "stream": [
            {"result": {"structuredContent": {"stdout": "", "stderr": ""}, "isError": False}}
        ],
    }
    sandbox = CodeInterpreterSandbox(region="us-west-2", execute=lambda *_: no_exit)
    with pytest.raises(SandboxError, match="exitCode"):
        sandbox.run_code("print(1)", timeout_s=10)


def test_a_non_python_language_is_refused_before_the_budget_is_charged() -> None:
    """The language is validated up front, so an unsupported one is a typed refusal here rather
    than a bare error from deep in the SDK after the billed call was already charged."""
    calls: list[str] = []
    sandbox = CodeInterpreterSandbox(
        region="us-west-2",
        budget=BilledCallBudget(5),
        execute=lambda c, *_: calls.append(c) or _ok_response(),
    )
    with pytest.raises(SandboxError, match="Python only"):
        sandbox.run_code("console.log(1)", language="javascript", timeout_s=10)
    assert calls == []  # nothing was sent, so nothing was billed
