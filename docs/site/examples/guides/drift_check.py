# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Check whether production traffic still looks like the traces a judge was calibrated on.

Seeds the bundled judge kit, then compares two dimensions (the length of the customer's message,
and the days since delivery the policy tool reported) between the calibration traces and the
production sample with ``calibrated_agent_evaluation.run.input_drift``. Run from the repository root
with the ``stats`` extra::

    uv run --extra stats python docs/site/examples/guides/drift_check.py
"""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from pathlib import Path

from calibrated_agent_evaluation.backends.local import LocalBackend
from calibrated_agent_evaluation.run import input_drift
from calibrated_agent_evaluation.schema import Trace

CALIBRATED_AGENT_EVALUATION = str(Path(sys.executable).parent / "calibrated-agent-evaluation")


def message_length(trace: Trace) -> float:
    return float(len(json.dumps(trace.input)))


def days_since_delivery(trace: Trace) -> float:
    """What the policy tool reported; the hunted calibration traces lean toward late returns."""
    for span in trace.spans:
        for call in span.tool_calls:
            if call.name == "check_refund_policy" and isinstance(call.result, dict):
                return float(call.result["days_since_delivery"])
    raise ValueError(f"{trace.trace_id}: no check_refund_policy result")


DETAIL = "--detail" in sys.argv[1:]


def show(title: str, dimension: str, baseline: list[float], fresh: list[float]) -> None:
    report = input_drift(baseline, fresh, dimension=dimension, seed=0)
    if not DETAIL:
        verdict = (
            "drifted: the two samples differ by more than chance and by enough to matter"
            if report.any_exceeded
            else "no drift: the two samples look alike"
        )
        print(f"{dimension} ({report.n_baseline} vs {report.n_fresh} traces): {verdict}")
        return
    print(f"{title}, {dimension} ({report.n_baseline} vs {report.n_fresh} traces)")
    for m in report.metrics:
        if m.name == "jsd":
            print(f"  {m.name:<12} {m.value:.3f}  p={m.p_value:.3f}  reported only")
            continue
        bar = "none" if m.threshold is None else f"{m.threshold:.2f}"
        flag = "DRIFT" if m.exceeded else "ok"
        print(f"  {m.name:<12} {m.value:.3f}  p={m.p_value:.3f}  bar={bar:<5} {flag}")
    print()


with tempfile.TemporaryDirectory() as tmp:
    store = Path(tmp) / "store"
    subprocess.run(  # noqa: S603 - fixed argv
        [CALIBRATED_AGENT_EVALUATION, "--root", str(store), "quickstart", "--certify"],
        capture_output=True,
        check=True,
        timeout=120,
    )
    with LocalBackend(store) as backend:
        calibration = backend.query(frame="discovery")
        production = backend.query(frame="production-representative")

for name, feature in (
    ("message length", message_length),
    ("days since delivery", days_since_delivery),
):
    show(
        "calibration traces vs production sample",
        name,
        [feature(t) for t in calibration],
        [feature(t) for t in production],
    )
