# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Walk the bundled judge kit through its trust gates, printing each command and its output.

Seeds ``calibrated-agent-evaluation quickstart --certify`` into a fresh store, then runs the named
steps in order. Every judge call replays the kit's recorded cassette, so nothing reaches a model.
Run from the repository root with the ``stats`` extra::

    uv run --extra stats python docs/site/examples/guides/certify_chain.py calibrate-v1 bias-v1

A step name ending in ``+explain`` adds ``--explain``; one starting with ``_`` runs without
printing (a prerequisite). ``seed`` prints the quickstart output itself; otherwise the seed is
silent.
"""

from __future__ import annotations

import subprocess
import sys
import tempfile
from pathlib import Path

from calibrated_agent_evaluation.backends.local import LocalBackend

CALIBRATED_AGENT_EVALUATION = str(Path(sys.executable).parent / "calibrated-agent-evaluation")
KIT = "store/quickstart"
CASSETTE = ["--cassette", f"{KIT}/judge.cassette.json"]


def _bias(version: str) -> list[str]:
    return [
        "bias",
        "--judge",
        f"{KIT}/judge-{version}.json",
        "--probes",
        f"{KIT}/bias-probes-{version[1]}",
        "--thresholds",
        f"{KIT}/bias-thresholds.json",
        *CASSETTE,
    ]


def _calibrate(version: str) -> list[str]:
    cell = f"label_round={version[1]}"
    judge = f"{KIT}/judge-{version}.json"
    return ["calibrate", "--judge", judge, *CASSETTE, "--rate-safe", "--cell", cell]


def _one_production_trace(work: Path) -> str:
    with LocalBackend(work / "store") as store:
        traces = store.query(frame="production-representative")
    return min(t.trace_id for t in traces)


STEPS = {
    "calibrate-v1": lambda _w: _calibrate("v1"),
    "calibrate-v2": lambda _w: _calibrate("v2"),
    "bias-v1": lambda _w: _bias("v1"),
    "bias-v2": lambda _w: _bias("v2"),
    "judge-v2": lambda _w: [
        "judge",
        "--judge",
        f"{KIT}/judge-v2.json",
        "--frame",
        "production-representative",
        *CASSETTE,
    ],
    "judge-v2-one": lambda w: [
        "judge",
        "--judge",
        f"{KIT}/judge-v2.json",
        "--trace",
        _one_production_trace(w),
        *CASSETTE,
    ],
    "run-v2": lambda _w: ["run", "--judge", f"{KIT}/judge-v2.json", "--ci"],
}


def run(args: list[str], cwd: Path, *, quiet: bool = False) -> None:
    done = subprocess.run(  # noqa: S603 - fixed argv
        [CALIBRATED_AGENT_EVALUATION, "--root", "store", *args],
        cwd=cwd,
        capture_output=True,
        text=True,
        check=False,
        timeout=120,
    )
    if quiet:
        if done.returncode != 0:
            raise SystemExit(
                f"calibrated-agent-evaluation {' '.join(args)} failed:\n{done.stdout}{done.stderr}"
            )
        return
    print(f"$ calibrated-agent-evaluation --root store {' '.join(args)}")
    print((done.stdout + done.stderr).rstrip())
    print(f"(exit {done.returncode})\n")


def main(names: list[str]) -> None:
    with tempfile.TemporaryDirectory() as tmp:
        work = Path(tmp)
        run(["quickstart", "--certify"], work, quiet=names[:1] != ["seed"])
        for name in names:
            if name == "seed":
                continue
            quiet = name.startswith("_")
            step, _, flag = name.removeprefix("_").partition("+")
            args = STEPS[step](work)
            run([*args, "--explain"] if flag == "explain" else args, work, quiet=quiet)


if __name__ == "__main__":
    main(sys.argv[1:])
