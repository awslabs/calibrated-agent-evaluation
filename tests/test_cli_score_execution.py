# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""`calibrated-agent-evaluation score --scorer code|sql` grades a trace by *executing* it, under the
same command.

The execution scorers are two more `--scorer` choices beside the compare-to-expected ones: `code`
runs the trace's code against the `--expected` tests in a sandbox; `sql` executes the trace's query
and the `--expected` gold query on a seeded SQLite database. They write the same `kind="code"`
verdicts the run gate reads with `--verdicts scorer`, with no calibration. `--scorer code` runs
locally by default or on the AgentCore Code Interpreter with `--sandbox agentcore`, which is bounded
by a billed-call budget and an optional identity gate.
"""

from __future__ import annotations

import json
from argparse import Namespace
from pathlib import Path

import pytest

from calibrated_agent_evaluation.backends.local import LocalBackend
from calibrated_agent_evaluation.cli import main
from calibrated_agent_evaluation.cli.dispatch import EXIT_DOMAIN_ERROR, EXIT_OK, EXIT_USAGE
from calibrated_agent_evaluation.execute.sandbox import SandboxResult
from calibrated_agent_evaluation.schema.types import Trace
from calibrated_agent_evaluation.scorers.execution import _RESULT_MARKER

_GOOD = "def add(a, b):\n    return a + b\n"
_BAD = "def add(a, b):\n    return a - b\n"
_TESTS = "assert add(2, 3) == 5\n"
_SCHEMA = "CREATE TABLE orders (id INTEGER, status TEXT);"
_SEED = "INSERT INTO orders VALUES (1, 'refunded'), (2, 'shipped');"
_GOLD = "SELECT id FROM orders WHERE status = 'refunded'"
_WRONG = "SELECT id FROM orders WHERE status = 'shipped'"


@pytest.fixture(autouse=True)
def _no_env_root(monkeypatch):
    monkeypatch.delenv("CALIBRATED_AGENT_EVALUATION_LOCAL_ROOT", raising=False)
    monkeypatch.delenv("CALIBRATED_AGENT_EVALUATION_AGENTCORE_MAX_BILLED_CALLS", raising=False)


def _trace(trace_id: str, output: dict) -> Trace:
    return Trace(
        trace_id=trace_id,
        frame="production-representative",
        ownership="probed",
        input={"task": trace_id},
        output=output,
        taxonomy_cell={},
    )


def _seed(root: Path, *traces: Trace) -> None:
    with LocalBackend(root=root) as backend:
        for t in traces:
            backend.put_trace(t)


def _expected_file(path: Path, rows: list[dict]) -> str:
    path.write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8")
    return str(path)


def _code_scores(root: Path, trace_id: str):
    with LocalBackend(root=root) as backend:
        return [lb for lb in backend.get_scores(trace_id) if lb.kind == "code"]


def _pass_marker() -> str:
    """The stdout a sandbox returns when the harness reports all tests passed."""
    body = {"passed": True, "error": None, "tail": "", "timed_out": False}
    return _RESULT_MARKER + json.dumps(body) + "\n"


def test_code_scorer_passes_good_and_fails_bad(tmp_path):
    root = tmp_path / "store"
    _seed(
        root,
        _trace("c-good", {"code": _GOOD, "language": "python"}),
        _trace("c-bad", {"code": _BAD, "language": "python"}),
    )
    spec = _expected_file(
        tmp_path / "e.jsonl",
        [
            {"trace_id": "c-good", "expected": {"tests": _TESTS}},
            {"trace_id": "c-bad", "expected": {"tests": _TESTS}},
        ],
    )
    rc = main(
        [
            "--root",
            str(root),
            "score",
            "--scorer",
            "code",
            "--mode",
            "wrong_code",
            "--expected",
            spec,
        ]
    )
    assert rc == EXIT_OK
    (good,) = _code_scores(root, "c-good")
    (bad,) = _code_scores(root, "c-bad")
    assert good.outcome == "PASS"
    assert bad.outcome == "FAIL"
    assert good.failure_mode == "wrong_code"
    assert good.rater_id is not None
    assert good.rater_id.startswith("scorer:code@")


def test_sql_scorer_matches_the_gold_result_set(tmp_path):
    root = tmp_path / "store"
    _seed(root, _trace("q-good", {"sql": _GOLD}), _trace("q-bad", {"sql": _WRONG}))
    oracle = {"sql": _GOLD, "schema_sql": _SCHEMA, "seed_sql": _SEED}
    spec = _expected_file(
        tmp_path / "e.jsonl",
        [{"trace_id": "q-good", "expected": oracle}, {"trace_id": "q-bad", "expected": oracle}],
    )
    rc = main(
        ["--root", str(root), "score", "--scorer", "sql", "--mode", "wrong_sql", "--expected", spec]
    )
    assert rc == EXIT_OK
    assert _code_scores(root, "q-good")[0].outcome == "PASS"
    assert _code_scores(root, "q-bad")[0].outcome == "FAIL"


def test_code_verdicts_gate_with_run_and_need_no_calibration(tmp_path, capsys):
    root = tmp_path / "store"
    _seed(
        root,
        _trace("c1", {"code": _GOOD, "language": "python"}),
        _trace("c2", {"code": _GOOD, "language": "python"}),
    )
    spec = _expected_file(
        tmp_path / "e.jsonl",
        [
            {"trace_id": "c1", "expected": {"tests": _TESTS}},
            {"trace_id": "c2", "expected": {"tests": _TESTS}},
        ],
    )
    assert (
        main(
            [
                "--root",
                str(root),
                "score",
                "--scorer",
                "code",
                "--mode",
                "wrong_code",
                "--expected",
                spec,
            ]
        )
        == EXIT_OK
    )
    capsys.readouterr()
    assert not (root / "records" / "calibration").exists()
    assert main(["--root", str(root), "run", "--verdicts", "scorer"]) == EXIT_OK
    assert "code-scorer verdicts" in capsys.readouterr().out


def test_score_prints_the_run_gate_next_step_hint(tmp_path, capsys):
    root = tmp_path / "store"
    _seed(root, _trace("c1", {"code": _GOOD, "language": "python"}))
    spec = _expected_file(tmp_path / "e.jsonl", [{"trace_id": "c1", "expected": {"tests": _TESTS}}])
    assert (
        main(["--root", str(root), "score", "--scorer", "code", "--mode", "m", "--expected", spec])
        == EXIT_OK
    )
    assert "calibrated-agent-evaluation run --verdicts scorer" in capsys.readouterr().out


def test_a_refused_score_run_prints_no_gate_hint(tmp_path, capsys):
    # The hint fires only on completion: a run that refuses (no --expected) prints no gate hint.
    root = tmp_path / "store"
    _seed(root, _trace("c1", {"code": _GOOD, "language": "python"}))
    assert main(["--root", str(root), "score", "--scorer", "code", "--mode", "m"]) == EXIT_USAGE
    captured = capsys.readouterr()
    assert "calibrated-agent-evaluation run --verdicts scorer" not in (captured.out + captured.err)


def test_default_sandbox_is_local_and_never_builds_the_agentcore_backend(tmp_path, monkeypatch):
    # The complementary branch of the --sandbox choice: with no --sandbox, code runs on the local
    # sandbox and the agentcore path (which needs the extra + AWS) is never constructed.
    import calibrated_agent_evaluation.cli.stages.score as score_mod

    def _boom(*_a, **_k):
        raise AssertionError("the agentcore sandbox must not be built on the default local path")

    monkeypatch.setattr(score_mod, "_agentcore_sandbox", _boom)
    root = tmp_path / "store"
    _seed(root, _trace("c1", {"code": _GOOD, "language": "python"}))
    spec = _expected_file(tmp_path / "e.jsonl", [{"trace_id": "c1", "expected": {"tests": _TESTS}}])
    assert (
        main(["--root", str(root), "score", "--scorer", "code", "--mode", "m", "--expected", spec])
        == EXIT_OK
    )
    assert _code_scores(root, "c1")[0].outcome == "PASS"


def test_execution_scorers_require_expected(tmp_path, capsys):
    root = tmp_path / "store"
    _seed(root, _trace("c1", {"code": _GOOD, "language": "python"}))
    rc = main(["--root", str(root), "score", "--scorer", "code", "--mode", "m"])
    assert rc == EXIT_USAGE
    assert "--expected" in capsys.readouterr().err


def test_sql_with_agentcore_sandbox_is_refused(tmp_path, capsys):
    root = tmp_path / "store"
    _seed(root, _trace("q1", {"sql": _GOLD}))
    spec = _expected_file(
        tmp_path / "e.jsonl",
        [{"trace_id": "q1", "expected": {"sql": _GOLD, "schema_sql": _SCHEMA, "seed_sql": _SEED}}],
    )
    rc = main(
        [
            "--root",
            str(root),
            "score",
            "--scorer",
            "sql",
            "--mode",
            "m",
            "--expected",
            spec,
            "--sandbox",
            "agentcore",
        ]
    )
    assert rc == EXIT_USAGE
    assert "--sandbox" in capsys.readouterr().err


def test_key_and_threshold_refused_for_execution(tmp_path, capsys):
    root = tmp_path / "store"
    _seed(root, _trace("c1", {"code": _GOOD, "language": "python"}))
    spec = _expected_file(tmp_path / "e.jsonl", [{"trace_id": "c1", "expected": {"tests": _TESTS}}])
    rc = main(
        [
            "--root",
            str(root),
            "score",
            "--scorer",
            "code",
            "--mode",
            "m",
            "--expected",
            spec,
            "--threshold",
            "0.9",
        ]
    )
    assert rc == EXIT_USAGE
    assert "--threshold" in capsys.readouterr().err


def test_billed_budget_flag_beats_env_beats_one_per_trace(monkeypatch):
    from calibrated_agent_evaluation.cli.stages import score

    monkeypatch.delenv("CALIBRATED_AGENT_EVALUATION_AGENTCORE_MAX_BILLED_CALLS", raising=False)
    assert score._billed_budget(Namespace(max_billed_calls=None), 5).limit == 5
    monkeypatch.setenv("CALIBRATED_AGENT_EVALUATION_AGENTCORE_MAX_BILLED_CALLS", "3")
    assert score._billed_budget(Namespace(max_billed_calls=None), 5).limit == 3
    assert score._billed_budget(Namespace(max_billed_calls=1), 5).limit == 1


def test_agentcore_sandbox_threads_profile_region_role_and_budget(monkeypatch):
    import calibrated_agent_evaluation.agentcore as ac
    from calibrated_agent_evaluation.cli.stages import score

    captured: dict = {}

    class _FakeClients:
        def __init__(self, region):
            self.region = region

        @classmethod
        def from_profile(cls, *, region=None, profile=None):
            captured["profile"] = profile
            return cls(region or "us-west-2")

    monkeypatch.setattr(ac, "AwsClients", _FakeClients, raising=False)
    monkeypatch.setattr(
        ac, "CodeInterpreterSandbox", lambda **kw: captured.update(kw), raising=False
    )
    monkeypatch.delenv("CALIBRATED_AGENT_EVALUATION_AGENTCORE_MAX_BILLED_CALLS", raising=False)

    args = Namespace(
        region="us-west-2",
        profile="eval-framework",
        role_arn="arn:aws:iam::1:role/scoped",
        max_billed_calls=None,
    )
    score._agentcore_sandbox(args, n_traces=4)
    assert captured["profile"] == "eval-framework"
    assert captured["region"] == "us-west-2"
    assert captured["role_arn"] == "arn:aws:iam::1:role/scoped"
    assert captured["budget"].limit == 4


def test_agentcore_routes_code_execution_to_the_injected_backend(tmp_path, monkeypatch):
    root = tmp_path / "store"
    _seed(root, _trace("c-bad", {"code": _BAD, "language": "python"}))  # fails locally
    spec = _expected_file(
        tmp_path / "e.jsonl", [{"trace_id": "c-bad", "expected": {"tests": _TESTS}}]
    )

    calls: list[str] = []

    class _FakeSandbox:
        def run_code(self, code, *, language="python", timeout_s, files=None):
            calls.append(code)
            return SandboxResult(
                stdout=_pass_marker(),
                stderr="",
                exit_code=0,
                timed_out=False,
                limits_applied=True,
                duration_s=0.0,
            )

    monkeypatch.setattr(
        "calibrated_agent_evaluation.cli.stages.score._agentcore_sandbox",
        lambda _args, _n: _FakeSandbox(),
    )
    rc = main(
        [
            "--root",
            str(root),
            "score",
            "--scorer",
            "code",
            "--mode",
            "m",
            "--expected",
            spec,
            "--sandbox",
            "agentcore",
        ]
    )
    assert rc == EXIT_OK
    assert calls, "the injected AgentCore sandbox was not used"
    # The fake reports a passing marker, so the locally-failing candidate PASSes — proving the
    # verdict came from the injected fake, not a real local run.
    assert _code_scores(root, "c-bad")[0].outcome == "PASS"


def test_no_partial_verdict_set_when_scoring_fails_partway(tmp_path, monkeypatch):
    """If a billed-call budget (or any infra failure) trips mid-run, nothing is written — the
    handler scores every trace before it writes any (atomic write-back)."""
    from calibrated_agent_evaluation.agentcore.errors import BilledCallBudgetExceeded

    root = tmp_path / "store"
    _seed(
        root,
        _trace("c1", {"code": _GOOD, "language": "python"}),
        _trace("c2", {"code": _GOOD, "language": "python"}),
    )
    spec = _expected_file(
        tmp_path / "e.jsonl",
        [
            {"trace_id": "c1", "expected": {"tests": _TESTS}},
            {"trace_id": "c2", "expected": {"tests": _TESTS}},
        ],
    )

    class _BudgetTrips:
        def __init__(self):
            self.n = 0

        def run_code(self, code, *, language="python", timeout_s, files=None):
            self.n += 1
            if self.n >= 2:  # the second billed call is refused
                raise BilledCallBudgetExceeded("the billed-call budget of 1 is used up")
            return SandboxResult(
                stdout=_pass_marker(),
                stderr="",
                exit_code=0,
                timed_out=False,
                limits_applied=True,
                duration_s=0.0,
            )

    monkeypatch.setattr(
        "calibrated_agent_evaluation.cli.stages.score._code_sandbox", lambda _a, _n: _BudgetTrips()
    )
    rc = main(
        [
            "--root",
            str(root),
            "score",
            "--scorer",
            "code",
            "--mode",
            "m",
            "--expected",
            spec,
            "--sandbox",
            "agentcore",
        ]
    )
    assert rc == EXIT_DOMAIN_ERROR
    # Nothing written for EITHER trace — the first verdict was not persisted before the second
    # trace's scoring failed.
    assert _code_scores(root, "c1") == []
    assert _code_scores(root, "c2") == []


def test_code_scorer_does_not_require_the_scorers_extra(tmp_path, monkeypatch):
    """code/sql are stdlib-only, so the handler must not gate them behind autoevals ([scorers])."""
    import calibrated_agent_evaluation.cli.config as cfg

    required: list[str] = []
    real = cfg.require_extras

    def _spy(*modules: str) -> None:
        required.extend(modules)
        real(*modules)

    monkeypatch.setattr(cfg, "require_extras", _spy)
    root = tmp_path / "store"
    _seed(root, _trace("c1", {"code": _GOOD, "language": "python"}))
    spec = _expected_file(tmp_path / "e.jsonl", [{"trace_id": "c1", "expected": {"tests": _TESTS}}])
    assert (
        main(["--root", str(root), "score", "--scorer", "code", "--mode", "m", "--expected", spec])
        == EXIT_OK
    )
    assert "autoevals" not in required
