# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Render the kit's corrected rate next to its answer key from a real run, so it can't drift.

Run after the judge kit's ``calibrated-agent-evaluation run --ci`` against the same store. It reads
the latest run record for the judged failure rate (raw and corrected), then grades the same 200
production traces with the sample set's policy rule. The rule is what made the synthetic gold, so it
gives an answer key a real deployment never has.

    python docs/site/examples/tutorial/value_panel.py demo
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(REPO_ROOT))

from corpus.generate.quickstart_set import (  # noqa: E402 - needs the repo root on the path
    FAILURE_MODE,
    gold_for_run,
    production_requests,
)

from calibrated_agent_evaluation._quickstart import load_certify_set  # noqa: E402


def answer_key() -> tuple[int, int]:
    """How many production traces the policy rule marks as this failure, out of how many."""
    traces, _ = load_certify_set()
    requests = {r.order_id: r for r in production_requests()}
    production = [t for t in traces if t.frame == "production-representative"]
    failures = 0
    for trace in production:
        calls = [call for span in trace.spans for call in span.tool_calls]
        gold = gold_for_run(requests[trace.trace_id], calls)
        if gold is not None and gold["failure_mode"] == FAILURE_MODE:
            failures += 1
    return failures, len(production)


def pct(x: float) -> str:
    return f"{100 * x:.1f}%"


def main(root: Path) -> None:
    runs = json.loads((root / "records" / "run" / "run_record.json").read_text(encoding="utf-8"))
    (trend,) = (t for t in runs[-1]["trend"] if t["mode"]["failure_type"] == FAILURE_MODE)
    raw, corrected = trend["raw_rate"], trend["corrected_rate"]
    flagged = round(raw["point"] * raw["n"])
    key_fail, key_n = answer_key()
    print(
        "In the bundled judge kit's run on the OrderBot sample, the revised judge flagged "
        f"{pct(raw['point'])} of {raw['n']} sample traces ({flagged} of them). Calibrated Agent "
        "Evaluation reports "
        f"{pct(corrected['point'])}, likely between {pct(corrected['lo'])} and "
        f"{pct(corrected['hi'])}, and the rule that made the sample's labels says "
        f"{pct(key_fail / key_n)} ({key_fail} of {key_n})."
    )


if __name__ == "__main__":
    main(Path(sys.argv[1]))
