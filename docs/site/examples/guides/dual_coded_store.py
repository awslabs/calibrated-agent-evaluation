# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Seed a local store with two raters' labels on the same 40 traces, for ``label --agreement``.

alice and bob both label every trace under codebook version 1. They agree on 12 FAILs and 26
PASSes and split on 2 traces, which is close agreement on too few shared traces for the gate to be
sure. Run from the repository root; the store goes to ``$CALIBRATED_AGENT_EVALUATION_LOCAL_ROOT``
(default ``.calibrated-agent-evaluation``)::

    uv run python docs/site/examples/guides/dual_coded_store.py
    uv run calibrated-agent-evaluation label --agreement --raters alice,bob --codebook-version 1
"""

from __future__ import annotations

import os
from pathlib import Path

from calibrated_agent_evaluation.backends.local import LocalBackend
from calibrated_agent_evaluation.schema import Label, Trace

N_TRACES = 40
BOTH_FAIL = 12
SPLIT = 2


def _trace(trace_id: str) -> Trace:
    return Trace(
        trace_id=trace_id,
        frame="production-representative",
        ownership="probed",
        input={"message": f"Please refund order {trace_id}"},
        output={"text": "Your refund has been issued."},
        taxonomy_cell={"feature": "refund_handling"},
    )


def _label(trace_id: str, rater: str, outcome: str) -> Label:
    return Label(
        trace_id=trace_id,
        span_id=None,
        kind="human",
        outcome=outcome,  # type: ignore[arg-type]
        critique=f"{rater}: {'refund outside the window' if outcome == 'FAIL' else 'correct'}",
        rater_id=rater,
        codebook_version="1",
    )


def main() -> None:
    root = Path(
        os.environ.get("CALIBRATED_AGENT_EVALUATION_LOCAL_ROOT", ".calibrated-agent-evaluation")
    )
    with LocalBackend(root=root) as backend:
        for i in range(N_TRACES):
            trace_id = f"ORD-{2000 + i}"
            backend.put_trace(_trace(trace_id))
            alice = "FAIL" if i < BOTH_FAIL + SPLIT else "PASS"
            bob = "FAIL" if i < BOTH_FAIL else "PASS"
            backend.put_score(_label(trace_id, "alice", alice))
            backend.put_score(_label(trace_id, "bob", bob))
    print(f"seeded {N_TRACES} traces labeled by alice and bob under codebook version 1 in {root}")


if __name__ == "__main__":
    main()
