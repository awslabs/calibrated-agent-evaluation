# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""``calibrated-agent-evaluation results --out FILE``: every stored estimate as one flat CSV table.

Each stage writes its records under ``<root>/records/<kind>/`` as an archive: one row per record
with JSON cells. This command reads every kind and writes
:func:`calibrated_agent_evaluation.dumpload.results_table` over all of them, one row per interval or
bound, refusals included (an empty ``point`` with the ``refusal_reason``), each row carrying its
method, level, sidedness, ``n`` and provenance. A kind the table has no extractor for is refused
rather than left out.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from calibrated_agent_evaluation.cli.dispatch import EXIT_OK, StageOutcome
from calibrated_agent_evaluation.cli.errors import CliError, UsageError

_NO_ESTIMATE = frozenset({"agentcore-online", "agentcore-session", "agentcore-read"})
"""Record kinds that report no estimate and are not passed to the results table."""


def results_handler(args: argparse.Namespace) -> StageOutcome:
    """Write the results table over every record under the local root."""
    from calibrated_agent_evaluation.cli.backends import local_root
    from calibrated_agent_evaluation.cli.config import require_extras

    require_extras("scipy")
    from calibrated_agent_evaluation.cli.records import RECORD_KINDS, load_records_of, records_dir
    from calibrated_agent_evaluation.dumpload import results_table, write_results_csv

    if str(args.out) == "-":
        raise UsageError(
            "--out - would write a file named '-'; the results table is written to a file, so "
            "name the CSV file to write"
        )
    out = Path(args.out)
    if out.exists() and out.is_dir():
        raise CliError(f"--out {out} is a directory; name the CSV file to write")
    root = local_root(args)
    # The AgentCore online records name a config and the sessions a run sent; they report no
    # estimate, and leaving them out here keeps the results table free of the plug-in.
    by_kind = {
        kind: load_records_of(root, kind) for kind in RECORD_KINDS if kind not in _NO_ESTIMATE
    }
    records = [r for recs in by_kind.values() for r in recs]
    if not records:
        held = {kind: len(load_records_of(root, kind)) for kind in sorted(_NO_ESTIMATE)}
        if any(held.values()):
            kinds = ", ".join(f"{n} {kind}" for kind, n in held.items() if n)
            raise CliError(
                f"the stored records under {root / 'records'} ({kinds}) report no estimate, so "
                "there is nothing to tabulate; run a stage that records its result (calibrate, "
                "bias, run, improve) first"
            )
        raise CliError(
            f"no stored records under {root / 'records'} to tabulate; run a stage that records "
            "its result (calibrate, bias, run, improve) first"
        )
    rows = results_table(records)
    written = write_results_csv(rows, out)
    counts = ", ".join(f"{len(recs)} {kind}" for kind, recs in by_kind.items() if recs)
    refused = sum(1 for r in rows if r.point is None)
    summary = [f"Wrote {written} estimate(s) from {len(records)} record(s) ({counts}) to {out}."]
    if refused:
        summary.append(
            f"{refused} of them were refused and have an empty point with the reason in "
            "refusal_reason."
        )
    detail = [
        f"read {', '.join(str(records_dir(root, k)) for k, recs in by_kind.items() if recs)}",
        "  one row per interval or bound; columns: estimand, slice, point, lo, hi, level, sided, "
        "method, n, n_resamples, seed, refusal_reason, source and the provenance stamp",
    ]
    return StageOutcome(
        stage="results", exit_code=EXIT_OK, summary="\n".join(summary), detail="\n".join(detail)
    )
