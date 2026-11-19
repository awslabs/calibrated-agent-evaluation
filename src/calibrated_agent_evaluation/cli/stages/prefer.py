# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""``prefer``: import pairwise preferences, and rank traces on a subjective dimension.

Two modes:

  * ``calibrated-agent-evaluation prefer --atoms FILE --pairs FILE`` imports preferences
    (``calibrated_agent_evaluation.subjective``'s JSON Lines format, see
    :mod:`calibrated_agent_evaluation.subjective.pairfile`). Every line is checked against the atom
    set with ``record_preference``: an objective or undeclared dimension, or a trace compared with
    itself, refuses the whole file and nothing is stored. Each trace named must be in the store. The
    preferences go to ``<root>/records/preference/``; importing the same file again adds nothing.
  * ``calibrated-agent-evaluation prefer --rank DIMENSION`` fits Bradley-Terry ratings with 95% Wald
    intervals over the stored preferences for that dimension. Ratings are tracked, never gated: no
    command gates on them. Preferences from different criteria or different judge kinds (a human's
    and an LLM judge's) are ranked separately, never pooled. Sequential Elo is shown only under
    ``--explain``: it depends on the order the preferences were stored in and has no interval.
"""

from __future__ import annotations

import argparse
import hashlib
from collections import defaultdict
from pathlib import Path
from typing import Any

from calibrated_agent_evaluation.cli.dispatch import EXIT_DOMAIN_ERROR, EXIT_OK, StageOutcome
from calibrated_agent_evaluation.cli.errors import CliError, UsageError

TRACKED = "tracked, never gated"


def prefer_handler(args: argparse.Namespace) -> StageOutcome:
    """Import preferences (``--pairs``) or rank one dimension (``--rank``)."""
    pairs = getattr(args, "pairs", None)
    rank = getattr(args, "rank", None)
    if bool(pairs) == bool(rank):
        raise UsageError(
            "calibrated-agent-evaluation prefer needs exactly one of --pairs FILE (import "
            "preferences, with --atoms) "
            "or --rank DIMENSION (rank the stored preferences on one subjective dimension)"
        )
    if pairs:
        return _import(args, Path(str(pairs)))
    return _rank(args, str(rank))


def _atoms(args: argparse.Namespace, *, required: bool) -> Any:
    from calibrated_agent_evaluation.subjective import load_atom_set

    path = getattr(args, "atoms", None)
    if not path:
        if required:
            raise UsageError(
                "--pairs needs --atoms FILE: the atom set that declares which dimensions are "
                "subjective, so an objective one is never ranked"
            )
        return None
    return load_atom_set(str(path))


def _import(args: argparse.Namespace, path: Path) -> StageOutcome:
    from calibrated_agent_evaluation.cli.backends import local_root, open_backend
    from calibrated_agent_evaluation.cli.records import (
        append_records,
        load_records_of,
        records_dir,
        records_lock,
    )
    from calibrated_agent_evaluation.subjective import preference_record, read_preference_file

    atom_set = _atoms(args, required=True)
    pairs = read_preference_file(path, atom_set)
    named = sorted({t for _, p in pairs for t in (p.trace_id_a, p.trace_id_b)})
    with open_backend(args) as backend:
        missing = [t for t in named if not _stored(backend, t)]
    if missing:
        shown = ", ".join(repr(t) for t in missing[:5])
        more = f" and {len(missing) - 5} more" if len(missing) > 5 else ""
        raise CliError(
            f"{len(missing)} trace id(s) in {path} are not in the store ({shown}{more}); "
            "nothing was imported. Ingest those traces first, or fix the ids"
        )
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    root = local_root(args)
    with records_lock(root, "preference"):
        stored = {r.preference_id for r in load_records_of(root, "preference")}
        fresh = []
        for number, pair in pairs:
            pid = "pref-" + hashlib.sha256(f"{digest}:{number}".encode()).hexdigest()[:16]
            if pid in stored:
                continue
            fresh.append(
                preference_record(
                    pair,
                    atom_set,
                    preference_id=pid,
                    source="file",
                    source_ref=f"{path}:{number}",
                )
            )
        added = append_records(root, "preference", fresh, locked=True)
    again = len(pairs) - added
    dims = sorted({p.dimension for _, p in pairs})
    skipped = f"; {again} were already imported from this file" if again else ""
    return StageOutcome(
        stage="prefer",
        exit_code=EXIT_OK,
        summary=(
            f"Imported {added} preference(s) on {', '.join(repr(d) for d in dims)} for "
            f"{atom_set.criterion!r}{skipped}.\n"
            f"Next: rank them with `calibrated-agent-evaluation prefer --rank {dims[0]}`."
        ),
        detail=(
            f"read {len(pairs)} preference(s) from {path} (sha256 {digest[:12]}), each checked "
            f"against the atom set (subjective: "
            f"{', '.join(d.name for d in atom_set.subjective_dimensions)}); wrote {added} to "
            f"{records_dir(root, 'preference')}"
        ),
    )


def _stored(backend: Any, trace_id: str) -> bool:
    from calibrated_agent_evaluation.schema.errors import TraceNotFoundError

    try:
        backend.get_trace(trace_id)
    except TraceNotFoundError:
        return False
    return True


def _rank(args: argparse.Namespace, dimension: str) -> StageOutcome:
    from calibrated_agent_evaluation.cli.backends import local_root
    from calibrated_agent_evaluation.cli.records import load_records_of
    from calibrated_agent_evaluation.subjective import SubjectiveError, is_objective, is_subjective

    atom_set = _atoms(args, required=False)
    if atom_set is not None and is_objective(dimension, atom_set):
        raise SubjectiveError(
            f"{dimension!r} is an objective atom of {atom_set.criterion!r}: it is judged PASS or "
            "FAIL and gated with `calibrated-agent-evaluation run`, not ranked"
        )
    if atom_set is not None and not is_subjective(dimension, atom_set):
        raise SubjectiveError(
            f"{dimension!r} is not declared in the atom set for {atom_set.criterion!r}; declare it "
            "as a subjective dimension first"
        )
    root = local_root(args)
    records = [r for r in load_records_of(root, "preference") if r.dimension == dimension]
    if atom_set is not None:
        records = [r for r in records if r.criterion == atom_set.criterion]
    if not records:
        raise CliError(
            f"no stored preference on {dimension!r}"
            + (f" for {atom_set.criterion!r}" if atom_set is not None else "")
            + "; import some with `calibrated-agent-evaluation prefer --atoms FILE --pairs FILE` "
            "or record them with "
            "`calibrated-agent-evaluation label --serve --pairwise --atoms FILE`"
        )
    groups: dict[tuple[str, str], list[Any]] = defaultdict(list)
    for record in records:
        groups[(record.criterion, record.judge_kind)].append(record)
    summary: list[str] = []
    detail: list[str] = []
    refused = False
    for (criterion, kind), members in sorted(groups.items()):
        text, explain, ok = _group(dimension, criterion, kind, members)
        summary += text
        detail += explain
        refused = refused or not ok
    if len(groups) > 1:
        summary.append(
            "Each criterion and judge kind is ranked on its own preferences; they are never pooled."
        )
    summary.append(
        f"These ratings are {TRACKED}. Gate on the objective atoms with "
        "`calibrated-agent-evaluation run`; "
        "collect more comparisons to narrow the intervals."
    )
    return StageOutcome(
        stage="prefer",
        exit_code=EXIT_DOMAIN_ERROR if refused else EXIT_OK,
        summary="\n".join(summary),
        detail="\n".join(detail),
    )


def _group(
    dimension: str, criterion: str, kind: str, members: list[Any]
) -> tuple[list[str], list[str], bool]:
    """The plain lines, the ``--explain`` lines and whether the fit succeeded, for one group."""
    from calibrated_agent_evaluation.subjective import (
        SubjectiveError,
        bradley_terry,
        compute_ratings,
    )

    pairs = [r.pair() for r in members]
    head = (
        f"{dimension!r} for {criterion!r}, {kind} preferences "
        f"({len(pairs)} comparison(s)), {TRACKED}:"
    )
    try:
        fit = bradley_terry(dimension, pairs)
    except SubjectiveError as exc:
        return [head, f"  not ranked: {exc}"], [], False
    lines = [head]
    width = max(len(r.trace_id) for r in fit.ratings)
    for place, rating in enumerate(fit.ratings, start=1):
        iv = rating.rating
        lines.append(
            f"  {place}. {rating.trace_id:<{width}}  {iv.point:.0f} "
            f"(95% interval {iv.lo:.0f} to {iv.hi:.0f}), {rating.n_comparisons} comparison(s)"
        )
    elo = compute_ratings(dimension, pairs)
    explain = [
        f"{criterion!r} / {kind}: Bradley-Terry maximum likelihood "
        f"({fit.ratings[0].rating.method}), 95% Wald intervals from the Fisher information "
        "under a sum-to-zero constraint, on the "
        "Elo scale 1500 + 400 * log-strength / ln 10 (a 400-point gap is 10:1 odds); "
        "order-invariant",
        f"  sequential Elo, display only (K={elo.k_factor:g}, start {elo.initial_rating:g}, "
        "in stored order; it changes with the order and has no interval):",
        *(
            f"    {e.trace_id}: {e.rating:.0f} ({e.n_comparisons} comparison(s))"
            for e in elo.ratings
        ),
    ]
    return lines, explain, True
