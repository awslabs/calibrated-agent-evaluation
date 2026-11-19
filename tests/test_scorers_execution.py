# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Behavior of the code- and SQL-execution scorers."""

from __future__ import annotations

import pytest

from calibrated_agent_evaluation.execute import SandboxError, SandboxResult
from calibrated_agent_evaluation.scorers import CodeExecutionScorer, SqlExecutionScorer
from calibrated_agent_evaluation.scorers.base import ScorerLane
from calibrated_agent_evaluation.scorers.errors import ScorerError

_GOOD = "def add(a, b):\n    return a + b\n"
_BAD = "def add(a, b):\n    return a - b\n"
_TESTS = "assert add(2, 3) == 5\n"
_TESTS_MSG = "assert add(2, 3) == 5, 'add returned the wrong value'\n"

_SCHEMA = "CREATE TABLE t (id INTEGER, name TEXT);"
_SEED = "INSERT INTO t VALUES (1, 'a'), (2, 'b'), (3, 'c');"


def _sql_expected(gold: str) -> dict:
    return {"sql": gold, "schema_sql": _SCHEMA, "seed_sql": _SEED}


# --- code execution ---------------------------------------------------------------------------


def test_a_passing_solution_scores_pass() -> None:
    label = CodeExecutionScorer().score({"code": _GOOD}, {"tests": _TESTS}, trace_id="code-pass")
    assert label.outcome == "PASS"
    assert label.kind == "code"


def test_a_failing_solution_scores_fail_and_the_critique_names_the_failure() -> None:
    label = CodeExecutionScorer().score({"code": _BAD}, {"tests": _TESTS_MSG}, trace_id="code-fail")
    assert label.outcome == "FAIL"
    assert "add returned the wrong value" in label.critique


def test_every_label_is_a_code_label_with_the_trace_id_and_a_critique() -> None:
    label = CodeExecutionScorer().score({"code": _BAD}, {"tests": _TESTS_MSG}, trace_id="t-1")
    assert label.kind == "code"
    assert label.trace_id == "t-1"
    assert label.critique  # non-empty, even on FAIL


def test_the_code_scorer_is_a_ci_gate() -> None:
    assert CodeExecutionScorer().lane == ScorerLane.CI_GATE


def test_the_default_backend_runs_locally_with_no_backend_wiring() -> None:
    # With no sandbox argument and no optional extra, scoring still runs on the local default.
    label = CodeExecutionScorer().score({"code": _GOOD}, {"tests": _TESTS}, trace_id="default")
    assert label.outcome == "PASS"


def test_a_candidate_that_forces_exit_zero_before_the_tests_still_fails() -> None:
    """The verdict comes from the tests, not the exit code: a solution that exits 0 on its own --
    an ``atexit`` hook here -- must not be able to skip a failing assertion and score PASS."""
    sneaky = "import os, atexit\natexit.register(lambda: os._exit(0))\n" + _BAD
    label = CodeExecutionScorer().score({"code": sneaky}, {"tests": _TESTS_MSG}, trace_id="sneak")
    assert label.outcome == "FAIL"


def test_a_main_guard_that_exits_does_not_skip_the_tests() -> None:
    """A conventional ``if __name__ == '__main__': sys.exit(main())`` tail must not pre-empt the
    tests; the candidate runs under a non-main name so the guard never fires."""
    main_guard = (
        "def main():\n"
        "    return 0\n"
        "if __name__ == '__main__':\n"
        "    import sys\n"
        "    sys.exit(main())\n"
    )
    label = CodeExecutionScorer().score(
        {"code": _BAD + main_guard}, {"tests": _TESTS_MSG}, trace_id="guard"
    )
    assert label.outcome == "FAIL"


def test_a_solution_with_a_future_import_is_scored_not_a_syntax_error() -> None:
    """A leading ``from __future__`` import is valid Python and a correct solution using it must
    PASS, not fail on a spurious SyntaxError from being run below a preamble."""
    code = "from __future__ import annotations\n" + _GOOD
    label = CodeExecutionScorer().score({"code": code}, {"tests": _TESTS}, trace_id="future")
    assert label.outcome == "PASS"


def test_a_timeout_scores_fail_citing_the_timeout() -> None:
    label = CodeExecutionScorer(timeout_s=0.5).score(
        {"code": "x = 1\n"}, {"tests": "import time\ntime.sleep(5)\n"}, trace_id="slow"
    )
    assert label.outcome == "FAIL"
    assert "timed out" in label.critique


def test_an_infrastructure_failure_raises_scorer_error_and_returns_no_label() -> None:
    class _BrokenSandbox:
        def run_code(self, code, *, language="python", timeout_s, files=None) -> SandboxResult:
            raise SandboxError("subprocess could not start")

    with pytest.raises(ScorerError):
        CodeExecutionScorer(_BrokenSandbox()).score(
            {"code": _GOOD}, {"tests": _TESTS}, trace_id="broken"
        )


def test_missing_code_is_refused() -> None:
    with pytest.raises(ScorerError, match="code"):
        CodeExecutionScorer().score({}, {"tests": _TESTS}, trace_id="no-code")


def test_tests_must_come_from_expected_not_the_output_under_test() -> None:
    """The oracle tests are required in ``expected``; the output's own fields cannot supply them,
    so a trace cannot grade itself."""
    with pytest.raises(ScorerError, match="tests"):
        CodeExecutionScorer().score({"code": _GOOD, "tests": _TESTS}, None, trace_id="self-grade")


def test_a_non_python_language_is_refused() -> None:
    with pytest.raises(ScorerError, match="Python only"):
        CodeExecutionScorer().score(
            {"code": _GOOD, "language": "javascript"}, {"tests": _TESTS}, trace_id="js"
        )


# --- sql execution ----------------------------------------------------------------------------


def test_matching_result_set_scores_pass() -> None:
    label = SqlExecutionScorer().score(
        {"sql": "SELECT id, name FROM t"},
        _sql_expected("SELECT id, name FROM t"),
        trace_id="sql-pass",
    )
    assert label.outcome == "PASS"


def test_differing_result_set_scores_fail() -> None:
    label = SqlExecutionScorer().score(
        {"sql": "SELECT id, name FROM t WHERE id = 1"},
        _sql_expected("SELECT id, name FROM t"),
        trace_id="sql-fail",
    )
    assert label.outcome == "FAIL"


def test_row_order_is_ignored_without_order_by() -> None:
    label = SqlExecutionScorer().score(
        {"sql": "SELECT id FROM t ORDER BY id DESC"},
        _sql_expected("SELECT id FROM t"),  # no ORDER BY -> multiset compare
        trace_id="sql-unordered",
    )
    assert label.outcome == "PASS"


def test_row_order_is_enforced_when_the_gold_has_order_by() -> None:
    gold = "SELECT id FROM t ORDER BY id DESC"
    wrong_order = SqlExecutionScorer().score(
        {"sql": "SELECT id FROM t ORDER BY id ASC"}, _sql_expected(gold), trace_id="ord-wrong"
    )
    right_order = SqlExecutionScorer().score(
        {"sql": "SELECT id FROM t ORDER BY id DESC"}, _sql_expected(gold), trace_id="ord-right"
    )
    assert wrong_order.outcome == "FAIL"
    assert right_order.outcome == "PASS"


def test_a_candidate_query_error_scores_fail_citing_the_error() -> None:
    label = SqlExecutionScorer().score(
        {"sql": "SELEKT id FROM t"},  # syntax error
        _sql_expected("SELECT id FROM t"),
        trace_id="sql-syntax",
    )
    assert label.outcome == "FAIL"
    assert "errored" in label.critique


def test_a_broken_gold_query_raises_scorer_error() -> None:
    with pytest.raises(ScorerError, match="gold"):
        SqlExecutionScorer().score(
            {"sql": "SELECT id FROM t"},
            _sql_expected("SELEKT id FROM t"),  # broken oracle
            trace_id="sql-bad-gold",
        )


def test_gold_order_by_detection_ignores_string_literals_and_identifiers() -> None:
    """Order-sensitivity keys off a real ORDER BY keyword, not the substring: a column named
    order_by or an 'order by' inside a string literal must not flip the comparison to ordered."""
    from calibrated_agent_evaluation.scorers.execution import _gold_is_ordered

    assert _gold_is_ordered("SELECT id FROM t ORDER BY id")
    assert _gold_is_ordered("select id from t where x=1 order by id desc")
    assert not _gold_is_ordered("SELECT id FROM t")
    assert not _gold_is_ordered("SELECT order_by FROM t")
    assert not _gold_is_ordered("SELECT id FROM t WHERE note = 'order by phone'")


def test_a_candidate_attach_is_denied_and_writes_no_host_file(tmp_path) -> None:
    """The candidate query runs read-only: an ATTACH (which would create a file on the host) is
    refused, the attempt scores FAIL, and no file appears."""
    target = tmp_path / "attack.db"
    label = SqlExecutionScorer().score(
        {"sql": f"ATTACH DATABASE '{target}' AS x"},
        _sql_expected("SELECT id FROM t"),
        trace_id="attach",
    )
    assert label.outcome == "FAIL"
    assert not target.exists()


def test_a_candidate_cannot_read_the_scorers_environment(monkeypatch) -> None:
    monkeypatch.setenv("CALIBRATED_AGENT_EVALUATION_SCORER_CANARY", "leaked")
    monkeypatch.setenv("LANGFUSE_SECRET_KEY", "sk-lf-canary")
    code = (
        "import os\n"
        "SEEN = (os.environ.get('CALIBRATED_AGENT_EVALUATION_SCORER_CANARY'),"
        " os.environ.get('LANGFUSE_SECRET_KEY'))\n"
    )
    label = CodeExecutionScorer(timeout_s=10).score(
        {"code": code}, {"tests": "assert SEEN == (None, None), SEEN\n"}, trace_id="t-env"
    )
    assert label.outcome == "PASS", label.critique


def test_a_candidate_cannot_open_a_socket_to_a_live_listener() -> None:
    import socket
    import threading

    server = socket.create_server(("127.0.0.1", 0))
    server.settimeout(0.2)
    port = server.getsockname()[1]
    accepted: list[object] = []
    stop = threading.Event()

    def _accept() -> None:
        while not stop.is_set():
            try:
                conn, _ = server.accept()
            except TimeoutError:
                continue
            accepted.append(conn)
            conn.close()

    thread = threading.Thread(target=_accept, daemon=True)
    thread.start()
    code = (
        "import socket\n"
        "try:\n"
        f"    socket.create_connection(('127.0.0.1', {port}), timeout=2).close()\n"
        "    CONNECTED = True\n"
        "except OSError:\n"
        "    CONNECTED = False\n"
    )
    try:
        label = CodeExecutionScorer(timeout_s=10).score(
            {"code": code}, {"tests": "assert CONNECTED is False\n"}, trace_id="t-net"
        )
    finally:
        stop.set()
        thread.join(timeout=2)
        server.close()
    assert accepted == []
    assert label.outcome == "PASS", label.critique


def test_a_candidate_that_forges_the_old_result_file_and_exits_still_fails() -> None:
    forged = (
        "import json, os\n"
        "json.dump({'passed': True, 'error': None, 'tail': ''},"
        " open('_calibrated_agent_evaluation_result.json', 'w'))\n"
        "os._exit(0)\n" + _BAD
    )
    label = CodeExecutionScorer(timeout_s=10).score(
        {"code": forged}, {"tests": _TESTS}, trace_id="t-forge"
    )
    assert label.outcome == "FAIL"
    assert "ended before the tests reported" in label.critique


def test_a_candidate_that_rewrites_a_tests_file_cannot_replace_the_oracle() -> None:
    rewriter = "open('_calibrated_agent_evaluation_tests.py', 'w').write('pass\\n')\n" + _BAD
    label = CodeExecutionScorer(timeout_s=10).score(
        {"code": rewriter}, {"tests": _TESTS}, trace_id="t-rewrite"
    )
    assert label.outcome == "FAIL"


def test_a_candidate_that_prints_a_fake_runner_record_still_fails() -> None:
    printer = (
        "import os\n"
        "os.write(1, b'0123456789abcdef0123456789abcdef "
        '{"passed": true, "error": null, "tail": ""}\\n\')\n' + _BAD
    )
    label = CodeExecutionScorer(timeout_s=10).score(
        {"code": printer}, {"tests": _TESTS}, trace_id="t-print"
    )
    assert label.outcome == "FAIL"


def test_the_honest_control_still_passes_and_fails_on_its_merits() -> None:
    scorer = CodeExecutionScorer(timeout_s=10)
    assert scorer.score({"code": _GOOD}, {"tests": _TESTS}, trace_id="t-ok").outcome == "PASS"
    assert scorer.score({"code": _BAD}, {"tests": _TESTS}, trace_id="t-bad").outcome == "FAIL"


def test_two_result_markers_in_the_sandbox_output_fail_rather_than_trust_the_last() -> None:
    marker = "__CALIBRATED_AGENT_EVALUATION_EXEC_RESULT__ "
    stdout = (
        marker
        + '{"passed": false, "error": "x", "tail": "", "timed_out": false}\n'
        + marker
        + '{"passed": true, "error": null, "tail": "", "timed_out": false}\n'
    )
    scripted = _ScriptedSandbox(SandboxResult(stdout, "", 0, False, True, 0.1))
    label = CodeExecutionScorer(scripted).score(
        {"code": _GOOD}, {"tests": _TESTS}, trace_id="t-two"
    )
    assert label.outcome == "FAIL"
    assert "more than one result marker" in label.critique


class _ScriptedSandbox:
    def __init__(self, result: SandboxResult) -> None:
        self._result = result

    def run_code(self, code, *, language="python", timeout_s, files=None) -> SandboxResult:
        return self._result


_COUNT_FOREVER = (
    "WITH RECURSIVE c(x) AS (SELECT 1 UNION ALL SELECT x + 1 FROM c) SELECT count(*) FROM c"
)
_ROWS_FOREVER = "WITH RECURSIVE c(x) AS (SELECT 1 UNION ALL SELECT x + 1 FROM c) SELECT x FROM c"


def test_a_never_ending_recursive_cte_fails_on_the_deadline() -> None:
    import time

    started = time.monotonic()
    label = SqlExecutionScorer(timeout_s=0.5).score(
        {"sql": _COUNT_FOREVER}, _sql_expected("SELECT id FROM t"), trace_id="t-cte"
    )
    assert time.monotonic() - started < 5
    assert label.outcome == "FAIL"
    assert "0.5s time limit" in label.critique


def test_a_candidate_returning_too_many_rows_fails_on_the_row_cap() -> None:
    import time

    started = time.monotonic()
    label = SqlExecutionScorer(max_rows=100).score(
        {"sql": _ROWS_FOREVER}, _sql_expected("SELECT id FROM t"), trace_id="t-rows"
    )
    assert time.monotonic() - started < 5
    assert label.outcome == "FAIL"
    assert "more than 100 rows" in label.critique


def test_a_gold_query_past_the_limits_is_a_broken_oracle_not_a_candidate_fail() -> None:
    with pytest.raises(ScorerError, match="gold"):
        SqlExecutionScorer(timeout_s=0.5).score(
            {"sql": "SELECT id FROM t"}, _sql_expected(_COUNT_FOREVER), trace_id="t-gold"
        )


def test_sql_limits_must_be_positive() -> None:
    with pytest.raises(ValueError, match="timeout_s"):
        SqlExecutionScorer(timeout_s=0)
    with pytest.raises(ValueError, match="max_rows"):
        SqlExecutionScorer(max_rows=0)
