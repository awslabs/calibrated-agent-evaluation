# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Evaluate a crew fix from the run before it and the run after it, as
``calibrated-agent-evaluation improve`` does.

Builds two local stores of 40 ClaimsCrew traces (copies of the four in ``fixtures/normalized``),
labelled by a reviewer for one workflow failure mode, ``wrong_claim_decision``. Before the fix,
intake caused 14 wrong decisions. After the fix to intake, intake causes 2 and the policy checker
12, on the same claims. The crew file declares the agents and handoffs; the command registers the
metric as a crew metric and evaluates the fix. ``--single`` instead cuts the 14 failures to 2 and
evaluates the fix as a plain metric, with a second mode as its guard. Run from the repository root
with the ``stats`` extra::

    uv run --extra stats python docs/site/examples/guides/improve_crew.py [--explain]
    uv run --extra stats python docs/site/examples/guides/improve_crew.py --print-crew
    uv run --extra stats python docs/site/examples/guides/improve_crew.py --single
"""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from pathlib import Path

from calibrated_agent_evaluation.backends.local import LocalBackend
from calibrated_agent_evaluation.schema import Label, Trace
from calibrated_agent_evaluation.schema.serde import trace_from_dict

CALIBRATED_AGENT_EVALUATION = str(Path(sys.executable).parent / "calibrated-agent-evaluation")
CORPUS = Path("fixtures") / "normalized"
BASES = ("CLM-4471", "CLM-4473", "CLM-4475", "CLM-4477")
MODE = "wrong_claim_decision"

CREW = {
    "agents": ["supervisor", "intake", "policy-checker", "fraud-scorer", "payout", "escalation"],
    "handoffs": [
        "intake->policy-checker",
        "policy-checker->fraud-scorer",
        "fraud-scorer->payout",
        "fraud-scorer->escalation",
    ],
}


def copy(i: int) -> Trace:
    """Corpus trace ``BASES[i % 4]`` under a new trace id, its span ids renamed to match."""
    base = BASES[i % len(BASES)]
    new = f"{base}-r{i:02d}"
    doc = json.loads((CORPUS / f"{base}.json").read_text(encoding="utf-8"))
    doc["trace_id"] = new
    for span in doc["spans"]:
        span["trace_id"] = new
        span["span_id"] = span["span_id"].replace(base, new, 1)
        if span["parent"]:
            span["parent"] = span["parent"].replace(base, new, 1)
    return trace_from_dict(doc)


def verdict(trace: Trace, culprit: str | None) -> Label:
    """The reviewer's trace-level verdict: PASS, or a FAIL blaming the ``culprit`` agent's span."""
    if culprit is None:
        return Label(
            trace_id=trace.trace_id, span_id=None, kind="human", outcome="PASS", critique="correct"
        )
    return Label(
        trace_id=trace.trace_id,
        span_id=None,
        kind="human",
        outcome="FAIL",
        critique="the claim got the wrong decision",
        failure_mode=MODE,
        culprit_span=f"{trace.trace_id}/{culprit}",
        manifestation_span=f"{trace.trace_id}/payout",
        rater_id="reviewer",
    )


def store(path: Path, culprits: dict[int, str]) -> None:
    with LocalBackend(root=path) as backend:
        for i in range(40):
            trace = copy(i)
            backend.put_trace(trace)
            backend.put_score(verdict(trace, culprits.get(i)))


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


GUARD = "missed_escalation"


def single_agent(work: Path) -> None:
    """The same 14 failures cut to 2 by a fix, read as one plain metric with no crew check.

    Traces 30 to 35 fail another mode in both runs, so the guard panel has a slice to watch.
    """
    for name, failing in (("before", range(14)), ("after", range(2))):
        with LocalBackend(root=work / name) as backend:
            for i in range(40):
                trace = copy(i)
                backend.put_trace(trace)
                label = verdict(trace, "intake" if i in failing else None)
                if 30 <= i < 36:
                    label = label.model_copy(
                        update={"outcome": "FAIL", "failure_mode": GUARD, "rater_id": "reviewer"}
                    )
                backend.put_score(label)
    (work / "dev.txt").write_text("CLM-4471\nCLM-4473\n", encoding="utf-8")
    run(["improve", "--register", MODE, "--min-reduction", "0.10"], work, quiet=True)
    run(
        [
            "improve",
            "--evaluate",
            MODE,
            "--verdicts",
            "human",
            "--before-root",
            "before",
            "--after-root",
            "after",
            "--development",
            "dev.txt",
        ],
        work,
    )


def main(flags: list[str]) -> None:
    if flags == ["--print-crew"]:
        print(json.dumps(CREW, indent=2))
        return
    if flags == ["--single"]:
        with tempfile.TemporaryDirectory() as tmp:
            single_agent(Path(tmp))
        return
    with tempfile.TemporaryDirectory() as tmp:
        work = Path(tmp)
        store(work / "before", dict.fromkeys(range(14), "intake"))
        store(
            work / "after",
            {**dict.fromkeys(range(2), "intake"), **dict.fromkeys(range(2, 14), "policy-checker")},
        )
        (work / "crew.json").write_text(json.dumps(CREW, indent=2), encoding="utf-8")
        (work / "dev.txt").write_text("CLM-4471\nCLM-4473\n", encoding="utf-8")
        run(["improve", "--register", MODE, "--min-reduction", "0.05", "--crew"], work)
        evaluate = [
            "improve",
            "--evaluate",
            MODE,
            "--crew-file",
            "crew.json",
            "--touched-agent",
            "intake",
            "--verdicts",
            "human",
            "--before-root",
            "before",
            "--after-root",
            "after",
            "--development",
            "dev.txt",
        ]
        run([*evaluate, *flags], work)


if __name__ == "__main__":
    main(sys.argv[1:])
