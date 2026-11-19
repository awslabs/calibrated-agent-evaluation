# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""``cluster``: check failure-mode saturation, blind spots and draft modes.

It answers whether the failure-mode taxonomy is saturated, where its blind spots are, and (with an
embedder) which draft modes the critiques fall into.

Reuses ``calibrated_agent_evaluation.cluster`` over the human FAIL labels in scope (``--trace``
narrows them):

  * **saturation**: ``saturation_check`` over how many traces show each human-ratified failure
    mode (a trace counts once per mode, whatever its labels, raters and span defects), the
    "should I keep labeling?" answer, with the Good-Turing coverage lower bound and the Chao1
    richness interval. Not saturated is information, not a failure: the stage exits 0.
  * **coverage**: ``coverage_report`` of the FAIL traces' taxonomy cells over the grid (a
    ``--taxonomy`` file, else the values the stored traces carry), naming the blind spots.
  * **draft**: with ``--embedder bedrock`` the FAIL critiques are embedded, grouped by
    ``hierarchical_cluster`` at ``--threshold``, assembled by ``build_draft``, and written with
    ``--out``. A draft is unratified: a human names and ratifies the modes before any label is
    tagged with one. Without an embedder the draft is skipped and the output says so.
  * **ratify**: ``--ratify MAP`` stamps the modes a human ratified onto the stored FAIL labels and
    stores the codebook (``cli.stages.ratify``); nothing else runs.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

from calibrated_agent_evaluation.cli.dispatch import EXIT_OK, StageOutcome
from calibrated_agent_evaluation.cli.errors import CliError
from calibrated_agent_evaluation.schema.types import Label

_BLIND_SPOTS_SHOWN = 5


def cluster_handler(args: argparse.Namespace) -> StageOutcome:
    """Report saturation and coverage over the human FAIL labels; draft modes with an embedder."""
    from calibrated_agent_evaluation.cli.config import require_extras

    require_extras("scipy")
    if getattr(args, "ratify", None):
        from calibrated_agent_evaluation.cli.stages.ratify import ratify_handler

        return ratify_handler(args)
    from calibrated_agent_evaluation.cli.backends import open_backend
    from calibrated_agent_evaluation.cli.scope import read_scope, scope_note, skip_lines

    with open_backend(args) as backend:
        scan = read_scope(backend, args)
        traces = scan.traces
        fails: list[Label] = [
            lb
            for t in traces
            for lb in backend.get_scores(t.trace_id)
            if lb.kind == "human" and lb.outcome == "FAIL"
        ]
    if not fails:
        raise CliError(
            f"no human FAIL labels on {scope_note(args, len(traces))} to cluster: label traces "
            "first (`calibrated-agent-evaluation label --serve --rater <your-id>`)"
        )

    from calibrated_agent_evaluation.cli import plain

    named = _traces_per_mode(fails)
    saturation = _saturation(named)
    coverage, coverage_lines = _coverage(args, traces, fails)
    labeled = plain.labeled_note([lb.rater_id or "" for lb in fails], "FAIL label(s)")
    lines = [f"cluster over {labeled} on {scope_note(args, len(traces))}"]
    lines += skip_lines(scan)
    lines += _saturation_lines(fails, named, saturation)
    lines += coverage_lines
    lines += _draft_lines(args, fails)
    summary = plain.cluster(
        named,
        saturation,
        n_blind=None if coverage is None else len(coverage.blind_spots),
        n_cells=None if coverage is None else coverage.total_cells,
    )
    unnamed = sum(1 for lb in fails if not lb.failure_mode)
    if unnamed and named:
        summary.append(
            f"{unnamed} FAIL label(s) have no failure mode yet and aren't counted: name them in "
            "`calibrated-agent-evaluation cluster --ratify MAP` (or in the review UI with a "
            "codebook)."
        )
    summary += plain.skipped(scan)
    return StageOutcome(
        stage="cluster", exit_code=EXIT_OK, summary="\n".join(summary), detail="\n".join(lines)
    )


def _traces_per_mode(fails: list[Label]) -> Counter[str]:
    """How many distinct traces show each named failure mode.

    The saturation estimators read a mode's count as how many independent times it was observed.
    A trace-level FAIL and a span defect on the same trace, or two raters coding one trace, are one
    observation of the mode, not two: counting labels would turn singletons into doubletons and
    report the taxonomy as more saturated than it is.
    """
    return Counter(mode for _, mode in {(lb.trace_id, lb.failure_mode) for lb in fails} if mode)


def _saturation(named: Counter[str]) -> Any:
    from calibrated_agent_evaluation.cluster import saturation_check

    return saturation_check(list(named.values())) if named else None


def _saturation_lines(fails: list[Label], named: Counter[str], sat: Any) -> list[str]:
    unnamed = sum(1 for lb in fails if not lb.failure_mode)
    lines = [
        "ratified failure modes (failing traces per mode): "
        + (", ".join(f"{m} ({n})" for m, n in named.most_common()) or "none yet")
    ]
    if unnamed:
        lines.append(f"  {unnamed} FAIL label(s) have no failure mode yet and are not counted")
    if not named:
        return [*lines, "saturation: not checked (no labeled failure mode to count)"]
    cov = sat.coverage
    verdict = "saturated" if sat.saturated else f"NOT saturated ({sat.reason})"
    lines.append(
        f"saturation: {verdict}; Good-Turing coverage {cov.point:.3f}, one-sided 95% lower bound "
        f"{cov.lb95:.3f} (n={cov.n} observations, one per failing trace and mode)"
    )
    if sat.chao1 is not None:
        c = sat.chao1
        point = "refused" if c.point is None else f"{c.point:.1f}"
        lines.append(
            f"  Chao1 richness estimate {point} mode(s), {c.level:.0%} CI [{c.lo:.1f}, {c.hi:.1f}]"
            f" ({sat.total_modes} observed, {sat.singletons} seen once)"
        )
    return lines


def _coverage(
    args: argparse.Namespace, traces: list[Any], fails: list[Label]
) -> tuple[Any, list[str]]:
    """The coverage report (``None`` when there is no taxonomy to check) and its detail lines."""
    from calibrated_agent_evaluation.cli.taxonomy import grid_from_file
    from calibrated_agent_evaluation.cluster import coverage_report
    from calibrated_agent_evaluation.sampling import TaxonomyGrid

    taxonomy = getattr(args, "taxonomy", None)
    if taxonomy:
        grid = grid_from_file(taxonomy)
    else:
        dims: dict[str, set[str]] = {}
        for t in traces:
            for dim, val in t.taxonomy_cell.items():
                dims.setdefault(dim, set()).add(val)
        if not dims:
            return None, ["coverage: not checked (no taxonomy: tag traces or pass --taxonomy FILE)"]
        grid = TaxonomyGrid(dimensions={d: sorted(v) for d, v in dims.items()})
    cells = {t.trace_id: t.taxonomy_cell for t in traces}
    failing_ids = sorted({lb.trace_id for lb in fails})
    report = coverage_report(grid, [dict(cells[tid]) for tid in failing_ids if tid in cells])
    lines = [
        f"coverage: failures seen in {report.covered_cells} of {report.total_cells} taxonomy "
        f"cell(s); {len(report.blind_spots)} blind spot(s)"
    ]
    for cell in report.blind_spots[:_BLIND_SPOTS_SHOWN]:
        lines.append("  blind spot: " + ", ".join(f"{k}={v}" for k, v in sorted(cell.items())))
    if len(report.blind_spots) > _BLIND_SPOTS_SHOWN:
        lines.append(f"  ... and {len(report.blind_spots) - _BLIND_SPOTS_SHOWN} more")
    if report.unclassified:
        lines.append(f"  {len(report.unclassified)} failing trace(s) fall outside the grid")
    return report, lines


def _draft_lines(args: argparse.Namespace, fails: list[Label]) -> list[str]:
    from calibrated_agent_evaluation.cluster import (
        build_draft,
        hierarchical_cluster,
        saturation_check,
    )

    embedder = _embedder(args)
    if embedder is None:
        return ["draft modes: skipped (needs an embedder: --embedder bedrock, the [bedrock] extra)"]
    critiques = [lb.critique for lb in fails]
    if len(critiques) < 2:
        return ["draft modes: skipped (clustering needs at least 2 FAIL critiques)"]
    threshold = float(getattr(args, "threshold", 1.0))
    embeddings = embedder.embed(critiques)
    assignments = hierarchical_cluster(embeddings, threshold=threshold)
    draft = build_draft(critiques, embeddings, assignments)
    sat = saturation_check([len(c.member_indices) for c in draft.clusters])
    lines = [
        f"draft modes (unratified): {len(draft.clusters)} cluster(s) at threshold {threshold}; "
        f"coverage lower bound {sat.coverage.lb95:.3f}"
        + ("" if sat.saturated else f", not saturated ({sat.reason})")
    ]
    for c in draft.clusters:
        lines.append(f"  {c.name}: {len(c.member_indices)} critique(s)")
    out = getattr(args, "out", None)
    if out:
        _write_draft(Path(out), draft, fails)
        lines.append(f"wrote the draft to {out}; name and ratify its modes before tagging labels")
    else:
        lines.append("draft not kept (pass --out FILE to write it)")
    return lines


def _write_draft(path: Path, draft: Any, fails: list[Label]) -> None:
    doc = {
        "ratified": False,
        "clusters": [
            {
                "name": c.name,
                "members": [
                    {
                        "trace_id": fails[i].trace_id,
                        "span_id": fails[i].span_id,
                        "critique": fails[i].critique,
                    }
                    for i in c.member_indices
                ],
            }
            for c in draft.clusters
        ],
    }
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(doc, indent=2, sort_keys=True), encoding="utf-8")
    except OSError as exc:
        raise CliError(f"cannot write the draft to {path}: {exc.strerror or exc}") from exc


def _embedder(args: argparse.Namespace) -> Any:
    """The embedder ``--embedder`` names, or ``None``. Bedrock is opt-in: it is a paid call."""
    choice = getattr(args, "embedder", None)
    if choice is None:
        return None
    from calibrated_agent_evaluation.cli.config import probe_extra

    probe = probe_extra("boto3")
    if not probe.available:
        raise CliError(f"--embedder bedrock needs the bedrock extra: {probe.install_hint}")
    from calibrated_agent_evaluation.execute.bedrock_embedder import BedrockEmbedder

    model = getattr(args, "embed_model", None)
    return BedrockEmbedder(model_id=model) if model else BedrockEmbedder()
