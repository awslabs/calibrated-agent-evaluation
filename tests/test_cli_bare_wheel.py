# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Every subcommand, run with real arguments on a bare install of the built wheel, ends cleanly.

``test_extras_closure.py`` hides distributions inside the dev environment; this installs the wheel
with no extra into a fresh virtual environment, seeds a root with ``quickstart --certify``, and runs
each subcommand against it. None may print a traceback, and one that needs an extra must say which
and how to install it. ``--help`` alone covered only the parser, so a stage that imported numpy or
boto3 before checking for it crashed on first real use. Opt-in with the other wheel tests:
``just test-extras``.
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest

from test_extras_wheel import _clean_env, _fresh_install, wheel  # noqa: F401  (fixture)

pytestmark = pytest.mark.wheel

_REPO = Path(__file__).resolve().parent.parent
_OTLP = _REPO / "fixtures" / "raw" / "otel" / "ORD-1001.events.otlp.json"
_PARQUET = _REPO / "fixtures" / "raw" / "hf" / "swebench-slice.parquet"


def _commands(root: Path, work: Path) -> list[tuple[tuple[str, ...], str | None]]:
    """(argv after ``--root ROOT``, the extra a refusal must name or ``None`` when it runs)."""
    kit = root / "quickstart"
    judge, cassette = str(kit / "judge-v1.json"), str(kit / "judge.cassette.json")
    region = ("--region", "us-east-1")
    return [
        (("doctor",), None),
        (("ingest", str(_OTLP)), None),
        (("ingest", str(_PARQUET)), "datasets"),
        (("ingest", "--format", "hf", "org/dataset"), "hf"),
        (("define", "--taxonomy", str(work / "taxonomy.json")), "sampling"),
        (("generate",), "sampling"),
        (("label", "--serve", "--rater", "r1"), "ui"),
        (("cluster",), "stats"),
        (("calibrate", "--judge", judge, "--cassette", cassette), "stats"),
        (
            (
                "bias",
                *("--judge", judge, "--cassette", cassette),
                *("--probes", str(kit / "bias-probes-1")),
                *("--thresholds", str(kit / "bias-thresholds.json")),
            ),
            "stats",
        ),
        (("judge", "--judge", judge, "--cassette", cassette), "stats"),
        (("judge", "--check", judge), "stats"),
        (("score", "--scorer", "exact", "--mode", "m"), "scorers"),
        (("score", "--scorer", "sql", "--mode", "m", "--expected", str(work / "sql.jsonl")), None),
        (("run", "--verdicts", "human"), "stats"),
        (("improve", "--register", "m", "--min-reduction", "0.1"), "stats"),
        (("plan", "--gate", "--expected-pass", "0.95"), "stats"),
        (("results", "--out", str(work / "results.csv")), "stats"),
        (("dump", str(work / "dump")), None),
        (("dump", "--judges-only", "--judge", judge, str(work / "kit")), "stats"),
        (("dataset", "build", "--out", str(work / "slice.json")), None),
        (("audit", "abtest", "--ab-test-id", "ab-1", *region), "agentcore"),
        (("audit", "recommendation", "--recommendation-id", "rec-1", *region), "agentcore"),
        (("online", "create-evaluator", "--judge", judge, *region), "agentcore"),
        (("online", "read", "--judge", judge, *region), "stats"),
    ]


def _cli(python: Path, cwd: Path, *argv: str) -> subprocess.CompletedProcess[str]:
    env = _clean_env()
    env.pop("CALIBRATED_AGENT_EVALUATION_EXPLAIN", None)
    return subprocess.run(
        [str(python), "-I", "-m", "calibrated_agent_evaluation", *argv],
        capture_output=True,
        text=True,
        encoding="utf-8",
        cwd=cwd,
        env=env,
        check=False,
        timeout=300,
    )


def test_every_subcommand_ends_without_a_traceback_on_a_bare_install(
    wheel: Path,  # noqa: F811
    tmp_path: Path,
) -> None:
    python = _fresh_install(wheel, tmp_path / "venv", "")
    work = tmp_path / "work"
    work.mkdir()
    root = work / "root"
    (work / "taxonomy.json").write_text(
        json.dumps({"channel": ["chat", "email"], "tier": ["a", "b"]}), encoding="utf-8"
    )
    sql = {"sql": "SELECT 1", "schema_sql": "CREATE TABLE t (a INTEGER);", "seed_sql": ""}
    (work / "sql.jsonl").write_text(
        json.dumps({"trace_id": "Q-10000", "expected": sql}) + "\n", encoding="utf-8"
    )
    seeded = _cli(python, work, "--root", str(root), "quickstart", "--certify")
    assert seeded.returncode == 0, seeded.stderr[-3000:]

    failures: list[str] = []
    for argv, extra in _commands(root, work):
        result = _cli(python, work, "--root", str(root), *argv)
        output = result.stdout + result.stderr
        shown = " ".join(argv[:3])
        if "Traceback" in output:
            failures.append(f"{shown}: traceback\n{output[-1500:]}")
        elif extra is not None and (result.returncode == 0 or f"[{extra}]" not in output):
            failures.append(f"{shown}: expected the [{extra}] install message\n{output[-800:]}")
        elif extra is not None and "install" not in output:
            failures.append(f"{shown}: no install command\n{output[-800:]}")
    assert not failures, "\n\n".join(failures)
