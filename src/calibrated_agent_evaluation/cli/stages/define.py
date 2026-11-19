# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""``define``: turn a taxonomy into the discovery frame the later stages sample from.

Reuses ``calibrated_agent_evaluation.sampling.discovery_frame``: a covering array over the taxonomy
grid, so every ``--strength``-way combination of dimension values appears in at least one row. The
grid is a ``--taxonomy`` file, or the stored traces' ``taxonomy_cell`` values (``--trace`` narrows
them). ``--out`` writes the frame for ``calibrated-agent-evaluation generate --frame``; without it
the frame is reported and not kept, and the summary says so.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from calibrated_agent_evaluation.cli.dispatch import EXIT_OK, StageOutcome
from calibrated_agent_evaluation.cli.errors import CliError


def define_handler(args: argparse.Namespace) -> StageOutcome:
    """Build the discovery frame and report (and optionally write) it."""
    from calibrated_agent_evaluation.cli.config import require_extras

    require_extras("allpairspy")
    from calibrated_agent_evaluation.cli.backends import open_backend
    from calibrated_agent_evaluation.cli.scope import read_scope, skip_lines
    from calibrated_agent_evaluation.cli.taxonomy import (
        frame_to_doc,
        grid_from_file,
        grid_from_traces,
    )
    from calibrated_agent_evaluation.sampling import discovery_frame

    taxonomy = getattr(args, "taxonomy", None)
    padded: list[str] = []
    skipped: list[str] = []
    if taxonomy:
        grid = grid_from_file(taxonomy)
        source = f"taxonomy file {taxonomy}"
    else:
        with open_backend(args) as backend:
            scan = read_scope(backend, args)
            grid, padded = grid_from_traces(scan.traces)
            skipped = skip_lines(scan)
        source = "taxonomy of the stored traces"
    strength = int(getattr(args, "strength", 2))
    frame = discovery_frame(grid, strength=strength)

    dims = ", ".join(f"{d} ({len(v)})" for d, v in grid.dimensions.items())
    lines = [
        f"discovery frame from the {source}: {len(frame.rows)} row(s) covering every "
        f"{frame.coverage_strength}-way combination of {len(grid.dimensions)} dimension(s): {dims}"
    ]
    lines += skipped
    if padded:
        lines.append(
            f"  note: {', '.join(padded)} had one stored value; a placeholder second value was "
            "added so the frame can cross it, and rows on it are cells no trace was seen in"
        )
    out = getattr(args, "out", None)
    if out:
        path = Path(out)
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(
                json.dumps(frame_to_doc(frame), indent=2, sort_keys=True), encoding="utf-8"
            )
        except OSError as exc:
            raise CliError(f"cannot write the frame to {path}: {exc.strerror or exc}") from exc
        lines.append(
            f"wrote the frame to {path} (read it back with `calibrated-agent-evaluation generate "
            f"--frame {path}`)"
        )
    else:
        lines.append("frame not kept (pass --out FILE to write it)")
    return StageOutcome(stage="define", exit_code=EXIT_OK, summary="\n".join(lines))
