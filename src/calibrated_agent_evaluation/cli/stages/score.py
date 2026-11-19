# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""``score``: grade stored traces with a deterministic code scorer, one mode at a time.

A code scorer compares a trace's output with the expected output and says PASS or FAIL, with no
model and no calibration. Two families, same verb: **compare-to-expected** scorers (exact match,
Levenshtein similarity, numeric closeness, JSON structural similarity, or plain JSON validity --
the ``calibrated_agent_evaluation.scorers`` wrappers over ``autoevals``, the ``[scorers]`` extra);
and **execution** scorers (``code`` runs the trace's code against the ``--expected`` tests in a
sandbox, ``sql`` executes the trace's query and the ``--expected`` gold query against a seeded
SQLite database and compares result sets). Each verdict is written back as a ``kind="code"`` label
stamped with the failure mode it measures (``--mode``) and the scorer that made it
(``rater_id="scorer:<name>@<params-hash>"``), so
``calibrated-agent-evaluation run --verdicts scorer`` gates them and a re-run with other parameters
is a different rater, never merged with this one.

The execution scorers take their oracle from ``--expected`` (not from human labels): for ``code``
each row's ``expected`` is ``{"tests": ...}``; for ``sql`` it is
``{"sql": <gold>, "schema_sql": ..., "seed_sql": ...}``. ``--scorer code`` runs the code in the
local stdlib sandbox by default (best-effort isolation, not a security boundary), or on the
Bedrock AgentCore Code Interpreter with
``--sandbox agentcore`` -- which is bounded: a billed-call budget (``--max-billed-calls``, else
``CALIBRATED_AGENT_EVALUATION_AGENTCORE_MAX_BILLED_CALLS``, else one call per scored trace) refuses
the call past the cap before it is sent, and ``--role-arn`` adds an identity gate. ``sql`` always
runs locally.

The expected output of a trace comes from its human labels by default: the ``expected`` a human
recorded on a FAIL for this mode. ``--expected FILE`` reads it from JSONL instead, one
``{"trace_id": ..., "expected": {...}}`` per line. A trace in scope with no expected value (or two
raters who recorded different ones) is counted and named and is not scored; with none in scope the
command refuses. ``valid-json`` needs no expected value.

Scoring before any human label exists is allowed, with the warning that error analysis comes
first.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any

from calibrated_agent_evaluation._model import replace
from calibrated_agent_evaluation.cli.dispatch import EXIT_OK, StageOutcome
from calibrated_agent_evaluation.cli.errors import CliError, UsageError
from calibrated_agent_evaluation.scenarios.labels import newest_per_rater, resolve_expected
from calibrated_agent_evaluation.schema.types import Label, Trace

SCORERS: tuple[str, ...] = (
    "exact",
    "levenshtein",
    "numeric",
    "json-diff",
    "valid-json",
    "code",
    "sql",
)
_EXECUTION_SCORERS = frozenset({"code", "sql"})
_THRESHOLDED = frozenset({"levenshtein", "numeric", "json-diff"})
_NEEDS_EXPECTED = frozenset(SCORERS) - {"valid-json"}
_SHOW = 5
_TIMEOUT_S = 30.0

NO_EXPECTED = "trace(s) with no expected value for this mode"
CONFLICTING = "trace(s) whose raters recorded different expected values (adjudicate them first)"


def score_handler(args: argparse.Namespace) -> StageOutcome:
    """Score the traces in scope and write one code verdict per trace."""
    from calibrated_agent_evaluation.cli.backends import open_backend
    from calibrated_agent_evaluation.cli.config import require_extras
    from calibrated_agent_evaluation.cli.scope import read_scope, scope_note, skip_lines

    name = str(args.scorer)
    # The compare-to-expected scorers run through autoevals ([scorers] extra); the execution
    # scorers (code/sql) are stdlib-only, so they must not demand it on a bare install.
    if name not in _EXECUTION_SCORERS:
        require_extras("autoevals")
    from calibrated_agent_evaluation.enforce import warn_error_analysis_first

    mode = str(args.mode).strip()
    if not mode:
        raise UsageError("--mode must name the failure mode the scorer measures")
    key = getattr(args, "key", None)
    threshold = getattr(args, "threshold", None)
    expected_file = getattr(args, "expected", None)
    _check_execution_flags(args, name)
    if name in _EXECUTION_SCORERS:
        if not expected_file:
            oracle = "the task's tests" if name == "code" else "the gold query + schema + seed"
            raise UsageError(
                f"--scorer {name} needs --expected FILE carrying {oracle} per trace; an execution "
                "oracle is not recorded on human labels"
            )
        params: dict[str, Any] = {"scorer": name, "key": None}
        # The code scorer needs the sandbox (and its budget), built once the scope is known.
        scorer: Any = None if name == "code" else _sql_execution_scorer()
    else:
        scorer, params = _scorer(name, key, threshold)
    rater_id = f"scorer:{name}@{_params_hash(params)}"
    if expected_file and name not in _NEEDS_EXPECTED:
        raise UsageError(f"--scorer {name} checks the output alone; leave out --expected")

    with open_backend(args) as backend:
        scan = read_scope(backend, args)
        traces = scan.traces
        frame = getattr(args, "frame", None)
        if frame:
            traces = [t for t in traces if t.frame == frame]
        scope = scope_note(args, len(traces))
        if frame and not getattr(args, "trace_id", None):
            scope += f" in the {frame} frame"
        if not traces:
            raise CliError(f"no stored traces to score on {scope}")
        if name == "code":
            # One billed Code Interpreter call per scored trace, so the budget ceiling is the
            # number of traces in scope (the local sandbox ignores the budget).
            scorer = _code_execution_scorer(args, len(traces))
        labels = {t.trace_id: backend.get_scores(t.trace_id) for t in traces}
        n_human = sum(1 for ls in labels.values() for lb in ls if lb.kind == "human")
        if name not in _NEEDS_EXPECTED:
            expected: dict[str, dict[str, Any] | None] = {t.trace_id: None for t in traces}
            missing: dict[str, list[str]] = {}
            unknown: list[str] = []
        elif expected_file:
            expected, missing, unknown = _from_file(Path(expected_file), traces)
        else:
            expected, missing, unknown = _from_labels(traces, labels, mode)
        if not expected:
            where = (
                f"in {expected_file}" if expected_file else f"on the human FAIL labels for {mode!r}"
            )
            raise CliError(
                f"no trace on {scope} has an expected value {where}, so there is "
                "nothing to compare its output with. Record the correct output when labeling "
                "(`calibrated-agent-evaluation label --serve`), or pass --expected FILE"
                + _missing_text(missing)
            )
        verdicts = []
        for trace in traces:
            if trace.trace_id not in expected:
                continue
            label = _score_trace(scorer, trace, expected[trace.trace_id])
            verdicts.append(replace(label, failure_mode=mode, rater_id=rater_id))
        for label in verdicts:
            backend.put_score(label)

    outcomes = Counter(lb.outcome for lb in verdicts)
    n_missing = sum(len(ids) for ids in missing.values())
    warning = warn_error_analysis_first(rater_id, n_human)
    summary = [
        f"Scored {len(verdicts)} trace(s) for {mode} with the {name} scorer: "
        f"{outcomes['FAIL']} FAIL, {outcomes['PASS']} PASS.",
    ]
    if n_missing:
        summary.append(
            f"{n_missing} trace(s) in scope had no expected value to compare with and were not "
            "scored; --explain names them."
        )
    if unknown:
        summary.append(
            f"{len(unknown)} trace id(s) in {expected_file} are not in scope and were ignored."
        )
    if warning is not None:
        summary.append(f"Warning: {warning.message} {warning.nudge or ''}".rstrip())
    summary.append(
        "Next: gate these verdicts with `calibrated-agent-evaluation run --verdicts scorer` (they "
        "are code verdicts, "
        "not a judge's, so no calibration is involved)."
    )
    detail = [
        f"wrote {len(verdicts)} code verdict(s) as rater {rater_id!r} (parameters "
        f"{json.dumps(params, sort_keys=True)}) for mode {mode!r} on {scope}",
        f"  expected values from {expected_file or 'the human FAIL labels for the mode'}",
        *_missing_lines(missing),
        *([f"  ignored {len(unknown)} id(s) not in scope: {_shown(unknown)}"] if unknown else []),
        *skip_lines(scan),
    ]
    from calibrated_agent_evaluation.cli import plain

    summary += plain.skipped(scan)
    return StageOutcome(
        stage="score", exit_code=EXIT_OK, summary="\n".join(summary), detail="\n".join(detail)
    )


def _scorer(name: str, key: str | None, threshold: float | None) -> tuple[Any, dict[str, Any]]:
    from calibrated_agent_evaluation.scorers import (
        ExactMatchScorer,
        JSONDiffScorer,
        LevenshteinScorer,
        NumericDiffScorer,
        ValidJSONScorer,
    )

    if name not in SCORERS:
        raise UsageError(f"--scorer must be one of {', '.join(SCORERS)}, got {name!r}")
    if threshold is not None and name not in _THRESHOLDED:
        raise UsageError(
            f"--scorer {name} is PASS/FAIL already; --threshold applies to "
            f"{', '.join(sorted(_THRESHOLDED))}"
        )
    if threshold is not None and not 0.0 <= threshold <= 1.0:
        raise UsageError(f"--threshold must be in [0, 1], got {threshold}")
    params: dict[str, Any] = {"scorer": name, "key": key}
    if name == "exact":
        return ExactMatchScorer(key=key), params
    if name == "valid-json":
        return ValidJSONScorer(key=key), params
    cls, default = {
        "levenshtein": (LevenshteinScorer, 0.9),
        "numeric": (NumericDiffScorer, 0.99),
        "json-diff": (JSONDiffScorer, 1.0),
    }[name]
    params["threshold"] = default if threshold is None else threshold
    return cls(threshold=params["threshold"], key=key), params


def _params_hash(params: dict[str, Any]) -> str:
    return hashlib.sha256(json.dumps(params, sort_keys=True).encode()).hexdigest()[:8]


def _from_labels(
    traces: list[Trace], labels: dict[str, list[Label]], mode: str
) -> tuple[dict[str, dict[str, Any] | None], dict[str, list[str]], list[str]]:
    """Each trace's expected value from its human FAIL labels for ``mode``, newest per rater."""
    expected: dict[str, dict[str, Any] | None] = {}
    missing: dict[str, list[str]] = {}
    for trace in traces:
        newest = newest_per_rater(
            lb
            for lb in labels[trace.trace_id]
            if lb.kind == "human" and lb.failure_mode == mode and lb.expected is not None
        )
        resolved = resolve_expected(newest.values())
        if resolved.conflicting:
            missing.setdefault(CONFLICTING, []).append(trace.trace_id)
        elif resolved.value is None:
            missing.setdefault(NO_EXPECTED, []).append(trace.trace_id)
        else:
            expected[trace.trace_id] = resolved.value
    return expected, missing, []


def _from_file(
    path: Path, traces: list[Trace]
) -> tuple[dict[str, dict[str, Any] | None], dict[str, list[str]], list[str]]:
    """Each trace's expected value from a JSONL file of ``{"trace_id", "expected"}``."""
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise CliError(f"cannot read --expected {path}: {exc.strerror or exc}") from exc
    except UnicodeDecodeError as exc:
        raise CliError(f"--expected {path} is not UTF-8 text: {exc}") from exc
    rows: dict[str, dict[str, Any]] = {}
    for n, line in enumerate(text.splitlines(), start=1):
        if not line.strip():
            continue
        try:
            doc = json.loads(line)
        except json.JSONDecodeError as exc:
            raise CliError(f"--expected {path} line {n} is not JSON: {exc}") from exc
        if (
            not isinstance(doc, dict)
            or not isinstance(doc.get("trace_id"), str)
            or not isinstance(doc.get("expected"), dict)
            or set(doc) - {"trace_id", "expected"}
        ):
            raise CliError(
                f"--expected {path} line {n} must be an object with a string trace_id and an "
                "object expected, and nothing else"
            )
        if doc["trace_id"] in rows:
            raise CliError(f"--expected {path} names trace {doc['trace_id']!r} twice")
        rows[doc["trace_id"]] = doc["expected"]
    in_scope = {t.trace_id for t in traces}
    expected: dict[str, dict[str, Any] | None] = {
        t.trace_id: rows[t.trace_id] for t in traces if t.trace_id in rows
    }
    missing = {NO_EXPECTED: [t.trace_id for t in traces if t.trace_id not in rows]}
    unknown = sorted(set(rows) - in_scope)
    return expected, {k: v for k, v in missing.items() if v}, unknown


def _shown(ids: list[str]) -> str:
    more = f" and {len(ids) - _SHOW} more" if len(ids) > _SHOW else ""
    return ", ".join(repr(i) for i in ids[:_SHOW]) + more


def _missing_lines(missing: dict[str, list[str]]) -> list[str]:
    return [f"  not scored: {len(ids)} {why}: {_shown(ids)}" for why, ids in missing.items()]


def _missing_text(missing: dict[str, list[str]]) -> str:
    lines = _missing_lines(missing)
    return ("\n" + "\n".join(lines)) if lines else ""


# ---- execution scorers (code / sql) ----------------------------------------------------------


def _check_execution_flags(args: argparse.Namespace, name: str) -> None:
    """Refuse execution-only flags where they don't apply, before any work."""
    sandbox = str(getattr(args, "sandbox", "local") or "local")
    if name != "code" and sandbox != "local":
        raise UsageError(
            f"--sandbox applies only to --scorer code (it runs code); --scorer {name} runs "
            "locally, so drop --sandbox"
            + ("" if name != "sql" else " (SQL executes on in-process SQLite in v1)")
        )
    if name in _EXECUTION_SCORERS:
        if getattr(args, "key", None) is not None:
            raise UsageError(f"--scorer {name} compares by executing, not by field; drop --key")
        if getattr(args, "threshold", None) is not None:
            raise UsageError(f"--scorer {name} is PASS/FAIL already; drop --threshold")


def _score_trace(scorer: Any, trace: Trace, oracle: dict[str, Any] | None) -> Label:
    """Score one trace: the candidate is the trace's output, the oracle is ``expected``.

    Both execution scorers take the candidate from the output and the oracle from ``expected`` (for
    ``code`` the oracle carries the tests, for ``sql`` the gold query + fixture). The output under
    test never supplies its own oracle.
    """
    return scorer.score(output=trace.output, expected=oracle, trace_id=trace.trace_id)


def _sql_execution_scorer() -> Any:
    from calibrated_agent_evaluation.scorers.execution import SqlExecutionScorer

    return SqlExecutionScorer()


def _code_execution_scorer(args: argparse.Namespace, n_traces: int) -> Any:
    from calibrated_agent_evaluation.scorers.execution import CodeExecutionScorer

    return CodeExecutionScorer(_code_sandbox(args, n_traces), timeout_s=_TIMEOUT_S)


def _code_sandbox(args: argparse.Namespace, n_traces: int) -> Any:
    """The sandbox for ``--scorer code``: the local stdlib default, or AgentCore when selected."""
    sandbox = str(getattr(args, "sandbox", "local") or "local")
    if sandbox == "agentcore":
        return _agentcore_sandbox(args, n_traces)
    from calibrated_agent_evaluation.execute.sandbox import LocalSubprocessSandbox

    return LocalSubprocessSandbox()


def _agentcore_sandbox(args: argparse.Namespace, n_traces: int) -> Any:
    """Build the AgentCore Code Interpreter sandbox, bounded by budget and optional identity gate.

    The billed-call budget and (if ``--role-arn`` is given) ``require_role`` both refuse before a
    request is sent. Lazy-imported: the local path never loads boto3, and without the
    ``[agentcore]`` extra this raises ``AgentCoreExtraMissing``.
    """
    from calibrated_agent_evaluation.agentcore import AwsClients, CodeInterpreterSandbox

    clients = AwsClients.from_profile(
        region=getattr(args, "region", None), profile=getattr(args, "profile", None)
    )
    region = clients.region
    if not region:
        raise CliError(
            "--sandbox agentcore needs an AWS region; pass --region or set AWS_REGION (or "
            "configure one on the profile)"
        )
    return CodeInterpreterSandbox(
        region=region,
        clients=clients,
        budget=_billed_budget(args, n_traces),
        role_arn=getattr(args, "role_arn", None),
    )


def _billed_budget(args: argparse.Namespace, n_traces: int) -> Any:
    """The billed-call budget: ``--max-billed-calls`` > env var > one call per scored trace.

    One scored trace is one billed call, so the default ceiling is the number of traces in scope:
    it never blocks a normal run and caps a retry loop or a bug that would bill more.
    """
    import os

    from calibrated_agent_evaluation.agentcore import BilledCallBudget

    explicit = getattr(args, "max_billed_calls", None)
    if explicit is not None:
        return BilledCallBudget(explicit)
    if os.environ.get("CALIBRATED_AGENT_EVALUATION_AGENTCORE_MAX_BILLED_CALLS", "").strip():
        return BilledCallBudget.from_env(os.environ)
    return BilledCallBudget(n_traces)
