# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""``cluster --ratify MAP``: stamp human-ratified failure modes onto the stored FAIL labels.

The map (:mod:`calibrated_agent_evaluation.cluster.ratify_map`) is checked whole against the store
before anything is written. Each tagged label replaces the stored one (the tagged copy is written,
then the original retracted), and the ratified vocabulary is stored as a ``codebook`` record under
``<root>/records/codebook/``, which
``calibrated-agent-evaluation label --serve --codebook-version V`` then offers as its failure-mode
list.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from calibrated_agent_evaluation.cli.dispatch import EXIT_OK, StageOutcome
from calibrated_agent_evaluation.cli.errors import CliError
from calibrated_agent_evaluation.schema.errors import CalibratedAgentEvaluationDataError


def _parent_version(stored: list[Any], version: str) -> str | None:
    """The version this ratification follows: the stored parent of ``version``, else the newest."""
    same = [r for r in stored if r.version == version]
    if same:
        return same[-1].parent_version
    return stored[-1].version if stored else None


def ratify_handler(args: argparse.Namespace) -> StageOutcome:
    """Plan the ratification over the whole store, then write the labels and the codebook record."""
    from calibrated_agent_evaluation._provenance import capture_provenance
    from calibrated_agent_evaluation.cli.backends import local_root, open_backend
    from calibrated_agent_evaluation.cli.proposals import RecordsProposalStore
    from calibrated_agent_evaluation.cli.records import append_records
    from calibrated_agent_evaluation.cli.scope import read_scope
    from calibrated_agent_evaluation.cli.stages.codebook import modes_for, stored_codebooks
    from calibrated_agent_evaluation.cluster.ratify_map import (
        load_ratification_map,
        plan_ratification,
    )
    from calibrated_agent_evaluation.review.record import CodebookRecord, with_content_id

    map_path = Path(args.ratify)
    ratification = load_ratification_map(map_path)
    root = local_root(args)
    stored = stored_codebooks(root)
    version = ratification.codebook_version
    proposals = (
        RecordsProposalStore(root).pending(version)
        if ratification.adopt or ratification.merge
        else ()
    )
    with open_backend(args) as backend:
        traces = read_scope(backend, args).traces
        wanted = {a.trace_id for a in ratification.assign}
        if ratification.merge:
            wanted |= {t.trace_id for t in traces}  # a merged proposal's labels may be anywhere
        labels = [
            lb for t in traces if t.trace_id in wanted for lb in backend.get_scores(t.trace_id)
        ]
        plan = plan_ratification(
            ratification,
            labels,
            parent_version=_parent_version(stored, version),
            proposals=proposals,
            ratified=modes_for(stored, version),
        )
        written = 0
        for old, new in plan.replacements:
            try:
                backend.put_score(new)
                backend.retract_score(old)
            except CalibratedAgentEvaluationDataError as exc:
                raise CliError(
                    f"tagging stopped after {written} of {len(plan.replacements)} label(s): "
                    f"{exc}. Run the same --ratify again: labels already tagged are skipped"
                ) from exc
            written += 1
    codebook = plan.codebook
    record = with_content_id(
        CodebookRecord(
            codebook_id="",
            version=codebook.version,
            modes=plan.modes,
            ratified_by=codebook.ratified_by,
            ratified_at=codebook.ratified_at.isoformat(timespec="seconds"),
            parent_version=codebook.parent_version,
            n_tagged=written,
            source=map_path.name,
            merged=dict(plan.merged),
            provenance=capture_provenance(),
        ),
        "codebook_id",
        "codebook",
    )
    append_records(root, "codebook", [record])
    already = f"; {plan.unchanged} already carried their mode" if plan.unchanged else ""
    settled = ""
    if plan.adopted or plan.merged:
        parts = [f"adopted {', '.join(plan.adopted)}"] if plan.adopted else []
        parts += [f"merged {source} into {target}" for source, target in plan.merged.items()]
        settled = f"Proposed modes settled: {'; '.join(parts)}.\n"
    summary = (
        f"ratified {len(plan.modes)} failure mode(s) under codebook version {codebook.version!r} "
        f"(by {codebook.ratified_by}) and tagged {written} FAIL label(s){already}.\n"
        f"{settled}"
        "Next: `calibrated-agent-evaluation cluster` counts the named modes; "
        "`calibrated-agent-evaluation label --serve --rater <id> "
        f"--codebook-version {codebook.version}` offers them in the review form."
    )
    detail = "\n".join(
        [
            f"codebook {record.codebook_id} (parent version {codebook.parent_version!r}) in "
            f"{root / 'records' / 'codebook'}",
            *(
                f"  {m.name}: {m.description}" if m.description else f"  {m.name}"
                for m in plan.modes
            ),
            *(
                f"  tagged {new.trace_id}"
                + (f"/{new.span_id}" if new.span_id else "")
                + f" ({new.rater_id or 'unattributed'}) -> {new.failure_mode}"
                for _, new in plan.replacements
            ),
        ]
    )
    return StageOutcome(stage="cluster", exit_code=EXIT_OK, summary=summary, detail=detail)
