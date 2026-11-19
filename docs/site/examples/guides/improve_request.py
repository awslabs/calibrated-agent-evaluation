# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Register a fix's success bar, then judge the fix from a request file, as
``calibrated-agent-evaluation improve`` does.

Seeds the sample corpus, pre-registers the target with
``calibrated-agent-evaluation improve --register``, writes the request file below, and evaluates it
with ``calibrated-agent-evaluation improve --request``. The pairs are a held-out set of 20 traces: 8
failed before the fix, 3 after. Run from the repository root with the ``stats`` extra::

    uv run --extra stats python docs/site/examples/guides/improve_request.py [--explain]
    uv run --extra stats python docs/site/examples/guides/improve_request.py --print-request
"""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from pathlib import Path

CALIBRATED_AGENT_EVALUATION = str(Path(sys.executable).parent / "calibrated-agent-evaluation")

UNCHANGED_RATE = {
    "point": 0.06,
    "lo": 0.01,
    "hi": 0.11,
    "method": "wilson",
    "level": 0.95,
    "sided": "two-sided",
    "n": 100,
}
REQUEST = {
    "target_metric": "trusted_broken_policy_tool",
    "pairs": [
        {
            "trace_id": f"ORD-conf-{i}",
            "before_fail": i < 8,
            "after_fail": i < 3,
            "labels_are_judge_produced": False,
        }
        for i in range(20)
    ],
    "development_ids": ["ORD-1001", "ORD-1002"],
    "slices": [
        {
            "name": "misread_amount",
            "kind": "other",
            "regression_p": 0.7,
            "rate_before": UNCHANGED_RATE,
            "rate_after": UNCHANGED_RATE,
        }
    ],
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
        return
    print(f"$ calibrated-agent-evaluation --root store {' '.join(args)}")
    print((done.stdout + done.stderr).rstrip())
    print(f"(exit {done.returncode})\n")


def main(flags: list[str]) -> None:
    if flags == ["--print-request"]:
        print(json.dumps(REQUEST, indent=2))
        return
    with tempfile.TemporaryDirectory() as tmp:
        work = Path(tmp)
        run(["quickstart"], work, quiet=True)
        register = [
            "improve",
            "--register",
            "trusted_broken_policy_tool",
            "--min-reduction",
            "0.02",
        ]
        run(register, work)
        (work / "request.json").write_text(json.dumps(REQUEST, indent=2), encoding="utf-8")
        run(["improve", "--request", "request.json", *flags], work)


if __name__ == "__main__":
    main(sys.argv[1:])
