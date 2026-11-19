# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Execution-based scorers: judge an output by running it, not by matching its text.

``CodeExecutionScorer`` runs an output's code against the task's tests in a :class:`SandboxBackend`
(the local default, or the AgentCore Code Interpreter). ``SqlExecutionScorer`` executes the
candidate and gold queries against a seeded in-memory ``sqlite3`` database and compares their
result sets; SQL always runs locally and takes no sandbox (there is no remote-SQL path in v1).

Both implement the existing ``Scorer`` protocol (``lane=CI_GATE``, a ``Label`` with ``kind="code"``,
a binary outcome and a non-empty critique) with no change to it. An *infrastructure* failure
(sandbox cannot start, backend unreachable) is re-wrapped as ``ScorerError`` and no Label is
returned; a program that runs and fails its tests, or a candidate query that errors, is a normal
FAIL verdict.
"""

from __future__ import annotations

import base64
import json
import re
import sqlite3
import time
from collections import Counter
from contextlib import closing
from typing import Any

from calibrated_agent_evaluation.execute.errors import ExecutionError
from calibrated_agent_evaluation.execute.sandbox import (
    NETWORK_BLOCK_SOURCE,
    LocalSubprocessSandbox,
    SandboxBackend,
)
from calibrated_agent_evaluation.schema.types import Label, Outcome
from calibrated_agent_evaluation.scorers.base import ScorerLane
from calibrated_agent_evaluation.scorers.errors import ScorerError

_DEFAULT_TIMEOUT_S = 30.0
_DEFAULT_SQL_TIMEOUT_S = 10.0
_DEFAULT_SQL_MAX_ROWS = 100_000
# SQLite VM instructions between deadline checks: frequent enough to stop within milliseconds.
_SQL_PROGRESS_STEPS = 1_000
# The sandbox wall-clock is only a backstop: the harness enforces the real timeout in-process (so
# it behaves the same on the local and AgentCore backends), and this grace lets the harness win the
# race and emit a clean timed-out result before the sandbox kills the whole process.
_TIMEOUT_GRACE_S = 5.0

# One line the harness writes to the real stdout AFTER the tests finish, carrying the verdict as
# JSON. The verdict rides on this sentinel, never on the process exit code: a candidate can force
# exit 0 (``sys.exit(0)``, ``os._exit(0)``, an ``atexit`` hook) but that only ends its runner early,
# which the harness reports as a FAIL. Exactly one marker must be present; a second one (a candidate
# that found the harness's stdout file and appended to it) makes the result untrustworthy, so it
# FAILs.
_RESULT_MARKER = "__CALIBRATED_AGENT_EVALUATION_EXEC_RESULT__ "

# The runner (run as a CHILD process by the harness below). Before it touches the candidate it
# reads the tests and a fresh per-run nonce from stdin, so a candidate cannot rewrite the tests and
# never sees the nonce on disk. It then imports the candidate under a non-``__main__`` name from its
# own file (so an ``if __name__ == "__main__"`` block never fires and a leading ``from __future__``
# import stays valid), runs the tests in the candidate's namespace, captures the candidate's output
# into a bounded tail, and prints ONE nonce-tagged result record to its real stdout. The harness
# trusts only that record; a missing or extra record, or an early exit, is a FAIL.
#
# This raises the bar; it is not a proof. The tests share the candidate's interpreter, so a
# candidate can in principle reach into the runner (frames, ``gc``, patched builtins or ``__eq__``)
# and subvert them, on either backend. The AgentCore sandbox protects the host and its credentials
# from untrusted model output; it does not make the verdict unforgeable. The runner is a fresh
# interpreter, so it applies the network block itself: a block in the harness would not reach it.
_RUNNER_SRC = (
    NETWORK_BLOCK_SOURCE
    + r"""
import io, json, os, runpy, sys, traceback

sys.path.insert(0, os.getcwd())

_LIMIT = 8000


class _Tail(io.TextIOBase):
    def __init__(self):
        self._buf = ""

    def write(self, s):
        self._buf = (self._buf + s)[-_LIMIT:]
        return len(s)

    def value(self):
        return self._buf


def _main():
    payload = json.loads(sys.stdin.buffer.read().decode("utf-8"))
    nonce, tests = payload["nonce"], payload["tests"]
    del payload
    dumps = json.dumps
    real_out, real_err = sys.stdout, sys.stderr
    cap = _Tail()
    passed, err = False, None
    sys.stdout = sys.stderr = cap
    try:
        ns = runpy.run_path(
            "_calibrated_agent_evaluation_solution.py",
            run_name="_calibrated_agent_evaluation_candidate",
        )
        exec(compile(tests, "_calibrated_agent_evaluation_tests.py", "exec"), ns)
        passed = True
    except BaseException:
        err = traceback.format_exc()
    finally:
        sys.stdout, sys.stderr = real_out, real_err
    record = dumps({"passed": passed, "error": err, "tail": cap.value()})
    real_out.write(nonce + " " + record + "\n")
    real_out.flush()


_main()
"""
)

# The harness, identical on both sandboxes. It runs the candidate + tests as a CHILD process (the
# runner) and enforces the timeout by killing that child, so a wedged candidate never takes THIS
# process down. The harness therefore always exits cleanly and its result marker on stdout is
# delivered reliably (the managed Code Interpreter drops a kernel's stdout when it dies on a hard
# ``os._exit``, which an in-process watchdog would have required). The nonce is generated here at
# run time and handed to the runner over stdin only, never written to a file.
_HARNESS_TEMPLATE = r"""
import base64, json, secrets, subprocess, sys

_MARKER = @@MARKER@@
_TIMEOUT = @@TIMEOUT@@
_TESTS = base64.b64decode(@@TESTS_B64@@).decode("utf-8")
for _name, _b64 in (
    ("_calibrated_agent_evaluation_solution.py", @@CODE_B64@@),
    ("_calibrated_agent_evaluation_runner.py", @@RUNNER_B64@@),
):
    with open(_name, "w", encoding="utf-8") as _f:
        _f.write(base64.b64decode(_b64).decode("utf-8"))

_nonce = secrets.token_hex(16)
_timed_out = False
_stdout = b""
try:
    _stdout = subprocess.run(
        [sys.executable, "-I", "_calibrated_agent_evaluation_runner.py"],
        input=json.dumps({"nonce": _nonce, "tests": _TESTS}).encode("utf-8"),
        timeout=_TIMEOUT,
        capture_output=True,
    ).stdout
except subprocess.TimeoutExpired:
    _timed_out = True
except Exception:
    pass

_result = {"passed": False, "error": None, "tail": "", "timed_out": _timed_out}
if not _timed_out:
    _tag = _nonce + " "
    _records = [
        _line[len(_tag):]
        for _line in _stdout.decode("utf-8", "replace").splitlines()
        if _line.startswith(_tag)
    ]
    if not _records:
        # The runner died before reporting (e.g. the candidate called os._exit): not a pass.
        _result["error"] = "the candidate ended before the tests reported a result"
    elif len(_records) > 1:
        _result["error"] = "the runner reported more than one result; refusing to trust any"
    else:
        try:
            _inner = json.loads(_records[0])
            _result["passed"] = _inner.get("passed") is True
            _result["error"] = _inner.get("error")
            _result["tail"] = _inner.get("tail", "")
        except (ValueError, AttributeError):
            _result["error"] = "the runner's result record was malformed"

sys.stdout.write(_MARKER + json.dumps(_result) + "\n")
sys.stdout.flush()
"""


def _build_code_harness(code: str, tests: str, timeout_s: float) -> str:
    """Render the harness program for one candidate + its oracle tests."""

    def _b64(text: str) -> str:
        return repr(base64.b64encode(text.encode("utf-8")).decode("ascii"))

    return (
        _HARNESS_TEMPLATE.replace("@@MARKER@@", repr(_RESULT_MARKER))
        .replace("@@CODE_B64@@", _b64(code))
        .replace("@@TESTS_B64@@", _b64(tests))
        .replace("@@RUNNER_B64@@", _b64(_RUNNER_SRC))
        .replace("@@TIMEOUT@@", repr(float(timeout_s)))
    )


def _parse_result_marker(stdout: str) -> dict[str, Any] | None:
    """Pull the harness's verdict line out of captured stdout, or ``None`` if it never ran.

    More than one marker line means something other than the harness wrote one, so the result is
    a FAIL rather than whichever line came last.
    """
    markers = [line for line in stdout.splitlines() if line.startswith(_RESULT_MARKER)]
    if len(markers) > 1:
        return {
            "passed": False,
            "error": "the sandbox output carried more than one result marker; trusting none",
        }
    if not markers:
        return None
    try:
        parsed = json.loads(markers[0][len(_RESULT_MARKER) :])
    except json.JSONDecodeError:
        return None
    return parsed if isinstance(parsed, dict) else None


class CodeExecutionScorer:
    """Score an output's code by executing the task's tests against it in a sandbox.

    The candidate code is the trace's output (``output["code"]``, ``output["language"]`` defaulting
    to ``"python"``). The tests are the oracle and come from ``expected`` (``expected["tests"]`` --
    from ``--expected``), not from the output: the thing under test must not supply the assertions
    it is graded by. The scorer wraps both in a harness (see ``_HARNESS_TEMPLATE``) and runs it
    through the sandbox, so the program is identical on the local and AgentCore backends. The
    verdict is read from the harness's result marker, never from the process exit code.

    The tests run in the same interpreter as the candidate, so a determined candidate can in
    principle subvert them (see ``_RUNNER_SRC``). The harness closes the cheap forgeries: the tests
    and a per-run nonce reach the runner over stdin before the candidate runs, and only one
    nonce-tagged record counts. Run untrusted model output on the AgentCore sandbox so that what
    it does cannot reach this machine.
    """

    lane = ScorerLane.CI_GATE

    def __init__(
        self, sandbox: SandboxBackend | None = None, *, timeout_s: float = _DEFAULT_TIMEOUT_S
    ) -> None:
        # Default to the local sandbox so the common path needs no backend wiring and no extra.
        self._sandbox: SandboxBackend = sandbox if sandbox is not None else LocalSubprocessSandbox()
        self._timeout_s = timeout_s

    def score(
        self,
        output: dict[str, Any],
        expected: dict[str, Any] | None,
        *,
        trace_id: str,
    ) -> Label:
        """Run the oracle tests against the output's code and return a PASS/FAIL Label.

        Raises:
            ScorerError: the output lacks a string ``code``, the language is not Python, the
                expected value does not carry the string ``tests`` oracle, or the sandbox could not
                run the program (an infrastructure failure, not a test failure).
        """
        code = output.get("code")
        if not isinstance(code, str):
            raise ScorerError(
                f"CodeExecutionScorer needs the output to carry a string 'code' (got keys "
                f"{sorted(output)})"
            )
        language = output.get("language", "python")
        if language != "python":
            raise ScorerError(f"the code scorer runs Python only in v1, got language={language!r}")
        tests = expected.get("tests") if isinstance(expected, dict) else None
        if not isinstance(tests, str):
            raise ScorerError(
                "CodeExecutionScorer needs the tests in the expected value (pass --expected "
                'carrying {"tests": <assertions>}); the output under test must not supply the '
                "tests it is graded by"
            )
        harness = _build_code_harness(code, tests, self._timeout_s)
        try:
            result = self._sandbox.run_code(
                harness, language="python", timeout_s=self._timeout_s + _TIMEOUT_GRACE_S
            )
        except ExecutionError as exc:
            # Infrastructure failure (sandbox/ backend), not a test result: no verdict.
            raise ScorerError(f"code execution could not be run: {exc}") from exc
        return self._verdict(result, trace_id)

    def _verdict(self, result: Any, trace_id: str) -> Label:
        marker = _parse_result_marker(result.stdout)
        if marker is None:
            # The harness never reported: a wall-clock backstop trip, or the candidate took the
            # process down before the tests completed (e.g. os._exit). Either way it did not pass.
            if result.timed_out:
                return _label(trace_id, "FAIL", self._timeout_critique())
            tail = _last_line(result.stderr or result.stdout) or "no result reported"
            return _label(
                trace_id, "FAIL", f"code execution ended before the tests completed: {tail}"
            )
        if marker.get("timed_out"):
            return _label(trace_id, "FAIL", self._timeout_critique())
        if marker.get("passed"):
            return _label(trace_id, "PASS", "code execution: all tests passed")
        tail = _last_line(marker.get("error")) or "tests failed"
        return _label(trace_id, "FAIL", f"code execution: tests failed: {tail}")

    def _timeout_critique(self) -> str:
        return f"code execution timed out after {self._timeout_s:g}s before the tests finished"


class SqlExecutionScorer:
    """Score a candidate SQL query by executing it against a seeded SQLite database.

    ``output`` carries ``{"sql": <candidate>}``; ``expected`` carries
    ``{"sql": <gold>, "schema_sql": <DDL>, "seed_sql": <inserts>}``. PASS iff the candidate's result
    set equals the gold's. Comparison is order-insensitive unless the gold query has an ``ORDER BY``
    (then row order must match too). SQL runs locally on stdlib ``sqlite3`` regardless of any code
    sandbox selection. The candidate query runs under a read-only authorizer, so it cannot
    ``ATTACH`` a database (which would create a file on the host) or write to the database.

    Each query runs under a wall-clock deadline (``timeout_s``) and may return at most ``max_rows``
    rows. A candidate past either limit is a FAIL; a gold query past either is a broken oracle and
    raises :class:`ScorerError`.
    """

    lane = ScorerLane.CI_GATE

    def __init__(
        self, *, timeout_s: float = _DEFAULT_SQL_TIMEOUT_S, max_rows: int = _DEFAULT_SQL_MAX_ROWS
    ) -> None:
        if not timeout_s > 0:
            raise ValueError(f"timeout_s must be positive, got {timeout_s!r}")
        if max_rows < 1:
            raise ValueError(f"max_rows must be at least 1, got {max_rows!r}")
        self._timeout_s = timeout_s
        self._max_rows = max_rows

    def score(
        self,
        output: dict[str, Any],
        expected: dict[str, Any] | None,
        *,
        trace_id: str,
    ) -> Label:
        """Execute the candidate and gold queries and compare result sets.

        Raises:
            ScorerError: inputs are missing, or the GOLD query errors (a broken oracle, not a
                candidate failure).
        """
        if expected is None:
            raise ScorerError("SqlExecutionScorer requires an expected value with the gold query")
        candidate = output.get("sql")
        gold = expected.get("sql")
        schema_sql = expected.get("schema_sql", "")
        seed_sql = expected.get("seed_sql", "")
        if not isinstance(candidate, str) or not isinstance(gold, str):
            raise ScorerError("SqlExecutionScorer needs a string 'sql' in both output and expected")

        with closing(sqlite3.connect(":memory:")) as conn:
            try:
                if schema_sql:
                    conn.executescript(schema_sql)
                if seed_sql:
                    conn.executescript(seed_sql)
            except sqlite3.Error as exc:
                raise ScorerError(f"SQL fixture schema/seed failed to load: {exc}") from exc
            try:
                gold_rows = self._bounded_fetch(conn, gold)
            except _SqlLimitExceeded as exc:
                raise ScorerError(
                    f"gold SQL query {exc} (broken oracle, or raise the limit)"
                ) from exc
            except sqlite3.Error as exc:
                # A broken oracle is not a candidate failure; refuse rather than score against it.
                raise ScorerError(f"gold SQL query errored (broken oracle): {exc}") from exc
            # From here on the UNTRUSTED candidate runs: restrict it to reads so it cannot ATTACH a
            # file (which would write to the host), or write/alter the database. The fixture and the
            # trusted gold query ran above, before the restriction is installed.
            conn.set_authorizer(_read_only_authorizer)
            try:
                candidate_rows = self._bounded_fetch(conn, candidate)
            except _SqlLimitExceeded as exc:
                return _label(trace_id, "FAIL", f"SQL execution: candidate query {exc}")
            except sqlite3.Error as exc:
                return _label(trace_id, "FAIL", f"SQL execution: candidate query errored: {exc}")

        ordered = _gold_is_ordered(gold)
        if ordered:
            matched = candidate_rows == gold_rows
        else:
            matched = Counter(map(tuple, candidate_rows)) == Counter(map(tuple, gold_rows))
        if matched:
            how = "ordered" if ordered else "as a multiset"
            return _label(trace_id, "PASS", f"SQL execution: result set matches the gold ({how})")
        critique = (
            f"SQL execution: result set differs from gold "
            f"(candidate {len(candidate_rows)} rows, gold {len(gold_rows)} rows)"
        )
        return _label(trace_id, "FAIL", critique)

    def _bounded_fetch(self, conn: sqlite3.Connection, sql: str) -> list[Any]:
        """Run ``sql`` and fetch its rows, aborting past the deadline or the row cap.

        Raises:
            _SqlLimitExceeded: the query ran past ``timeout_s`` or returned over ``max_rows`` rows.
            sqlite3.Error: the query itself errored.
        """
        deadline = time.monotonic() + self._timeout_s
        expired = False

        def _check() -> int:
            nonlocal expired
            expired = time.monotonic() > deadline
            return 1 if expired else 0

        # SQLite evaluates lazily, so the deadline must cover the fetch as well as the execute.
        conn.set_progress_handler(_check, _SQL_PROGRESS_STEPS)
        try:
            rows = conn.execute(sql).fetchmany(self._max_rows + 1)
        except sqlite3.OperationalError as exc:
            if expired:
                raise _SqlLimitExceeded(
                    f"ran past the {self._timeout_s:g}s time limit and was stopped"
                ) from exc
            raise
        finally:
            conn.set_progress_handler(None, 0)
        if len(rows) > self._max_rows:
            raise _SqlLimitExceeded(f"returned more than {self._max_rows} rows, the row cap")
        return rows


class _SqlLimitExceeded(Exception):
    """A query hit the SQL scorer's time limit or row cap."""


# The only sqlite actions a candidate SELECT needs. Everything else -- ATTACH/DETACH, any write or
# DDL, PRAGMA, loading an extension -- is denied, so the candidate cannot touch the host or mutate
# the fixture. A denied action makes sqlite raise, which the scorer records as a candidate FAIL.
_SQL_READ_ACTIONS = frozenset(
    {
        sqlite3.SQLITE_SELECT,
        sqlite3.SQLITE_READ,
        sqlite3.SQLITE_FUNCTION,
        sqlite3.SQLITE_RECURSIVE,
    }
)


def _read_only_authorizer(action: int, *_: Any) -> int:
    """Allow read actions for the candidate query; deny everything else."""
    return sqlite3.SQLITE_OK if action in _SQL_READ_ACTIONS else sqlite3.SQLITE_DENY


def _last_line(text: str | None) -> str:
    """The last non-empty line of ``text`` (a traceback or captured output), or ``""``."""
    lines = (text or "").strip().splitlines()
    return lines[-1] if lines else ""


_STRING_LITERAL = re.compile(r"'(?:''|[^'])*'")
_ORDER_BY = re.compile(r"\border\s+by\b", re.IGNORECASE)


def _gold_is_ordered(gold: str) -> bool:
    """Whether the gold query asks for a specific row order (an ``ORDER BY`` clause).

    Heuristic, not a SQL parser: string literals are stripped first (so ``WHERE note = 'order by
    x'`` does not count) and the match is on the ``order by`` keyword as a whole token (so a column
    named ``order_by`` does not count). Known v1 limitation: an ``ORDER BY`` inside a subquery or a
    window function (``... OVER (ORDER BY ...)``) still reads as ordered even though it does not
    order the outer result set; such a query would be compared order-sensitively. Use a candidate
    whose outer query carries the ``ORDER BY`` to avoid the ambiguity.
    """
    return bool(_ORDER_BY.search(_STRING_LITERAL.sub("", gold)))


def _label(trace_id: str, outcome: Outcome, critique: str) -> Label:
    return Label(trace_id=trace_id, span_id=None, kind="code", outcome=outcome, critique=critique)
