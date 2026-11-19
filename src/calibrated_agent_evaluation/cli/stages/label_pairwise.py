# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""``label --serve --pairwise --atoms FILE``: serve the A/B preference page.

The glue between the CLI and :func:`calibrated_agent_evaluation.review.pairwise.build_pairwise_app`:
it reads the atom set, resolves the traces in scope the way the labeling page does (``--frame``,
``--cell``, ``--trace``, ``--limit``), and hands the app a store that appends each choice to
``<root>/records/preference/``, the records ``calibrated-agent-evaluation prefer --rank`` reads. The
bind is checked by ``label_handler`` before this runs: loopback only.
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

from calibrated_agent_evaluation.cli.dispatch import EXIT_OK, StageOutcome
from calibrated_agent_evaluation.cli.errors import CliError, UsageError


def preference_store(root: Path) -> Callable[[Sequence[Any]], None]:
    """Append preference records under ``root``, failing in the subjective store's error family.

    The A/B page shows a store failure to the rater and keeps the comparison current; it catches
    that family only, so a CLI-layer refusal (an id clash, a lock it cannot take) is re-raised as
    one.
    """
    from calibrated_agent_evaluation.cli.records import append_records
    from calibrated_agent_evaluation.subjective import SubjectiveStoreError

    def store(records: Sequence[Any]) -> None:
        try:
            append_records(root, "preference", records)
        except CliError as exc:
            raise SubjectiveStoreError(str(exc)) from exc

    return store


def serve_pairwise(
    args: argparse.Namespace, *, host: str, port: int, rater_id: str
) -> StageOutcome:
    """Serve the A/B page on ``host``:``port`` until interrupted."""
    from calibrated_agent_evaluation.backends.base import scan_traces
    from calibrated_agent_evaluation.cli.backends import local_root, open_backend
    from calibrated_agent_evaluation.cli.records import load_records_of
    from calibrated_agent_evaluation.cli.scope import parse_cells
    from calibrated_agent_evaluation.cli.stages.label import _bind_message, _require_bindable
    from calibrated_agent_evaluation.review.addresses import surface_url
    from calibrated_agent_evaluation.review.guards import allowed_hosts_for
    from calibrated_agent_evaluation.review.pairwise import build_pairwise_app
    from calibrated_agent_evaluation.subjective import load_atom_set

    atoms = getattr(args, "atoms", None)
    if not atoms:
        raise UsageError(
            "--pairwise needs --atoms FILE: the atom set whose subjective dimensions you compare on"
        )
    atom_set = load_atom_set(str(atoms))
    root = local_root(args)

    store = preference_store(root)

    with open_backend(args) as backend:
        trace_id = getattr(args, "trace_id", None)
        if trace_id:
            raise UsageError(
                "--pairwise compares traces with each other, so --trace (one trace) leaves nothing "
                "to compare; narrow with --frame, --cell or --limit"
            )
        cells = parse_cells(getattr(args, "cell", None))
        scan = scan_traces(
            backend, frame=getattr(args, "frame", None), cell=dict(cells) if cells else None
        )
        traces = list(scan.traces)
        limit = getattr(args, "limit", None)
        if limit is not None:
            if limit < 2:
                raise UsageError(f"--limit {limit} leaves fewer than two traces to compare")
            traces = traces[:limit]
        app = build_pairwise_app(
            backend,
            atom_set=atom_set,
            rater_id=rater_id,
            traces=traces,
            store=store,
            done=load_records_of(root, "preference"),
            allowed_hosts=allowed_hosts_for(host),
        )
        print(
            f"{len(app.state.pairwise.pending)} comparison(s) for rater {rater_id!r} over "
            f"{len(traces)} trace(s) on "
            f"{', '.join(d.name for d in atom_set.subjective_dimensions)}; choices go to "
            f"{root / 'records' / 'preference'}",
            file=sys.stderr,
        )
        _require_bindable(host, port)

        import uvicorn

        try:
            uvicorn.run(app, host=host, port=port)
        except OSError as exc:
            raise CliError(_bind_message(host, port, exc)) from exc
    return StageOutcome(
        stage="label", exit_code=EXIT_OK, summary=f"A/B page served at {surface_url(host, port)}"
    )
