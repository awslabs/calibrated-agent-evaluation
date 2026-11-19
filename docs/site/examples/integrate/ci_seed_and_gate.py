# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Seed a CI job's store from a committed dump, then run the gate, as a CI job would.

Builds an evaluation set (the bundled sample corpus) in one store and writes it out with
``calibrated-agent-evaluation dump``, the directory you would commit. A second, empty store stands
in for the CI job: it restores the set with ``calibrated-agent-evaluation load`` and runs
``calibrated-agent-evaluation run --ci``. Each command runs as its own process. Run from the
repository root with the ``stats`` extra.

    uv run --extra stats python docs/site/examples/integrate/ci_seed_and_gate.py
"""

from __future__ import annotations

import subprocess
import sys
import tempfile
from pathlib import Path

CALIBRATED_AGENT_EVALUATION = str(Path(sys.executable).parent / "calibrated-agent-evaluation")


def run(args: list[str], cwd: Path, *, quiet: bool = False) -> None:
    done = subprocess.run(  # noqa: S603 - fixed argv
        [CALIBRATED_AGENT_EVALUATION, *args],
        cwd=cwd,
        capture_output=True,
        text=True,
        check=False,
        timeout=120,
    )
    print(f"$ calibrated-agent-evaluation {' '.join(args)}")
    if not quiet:
        print((done.stdout + done.stderr).rstrip())
    print(f"(exit {done.returncode})\n")


with tempfile.TemporaryDirectory() as tmp:
    work = Path(tmp)
    print("# once, on your machine: build the evaluation set and dump it for the repo\n")
    run(["--root", "eval-store", "quickstart"], work, quiet=True)
    run(["--root", "eval-store", "dump", "eval-set"], work)

    print("# in the CI job: restore the set into a fresh store and gate\n")
    run(["--root", ".calibrated-agent-evaluation", "load", "eval-set"], work)
    run(["--root", ".calibrated-agent-evaluation", "run", "--verdicts", "human", "--ci"], work)
