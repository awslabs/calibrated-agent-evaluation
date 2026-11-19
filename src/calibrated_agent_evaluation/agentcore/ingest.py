# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Orchestrate an AgentCore ingest: traces through the shared pull path, scores as advisory labels.

Vendor-free: this speaks the ``TraceSource`` / ``Trace`` / ``Label`` surface and the destination
backend only, never boto3. The trace half reuses the shared pull orchestration (``ingest_langfuse``
is source-generic); the score half maps evaluator results to advisory labels and persists them
with a de-dup so a re-read never double-counts.
"""

from __future__ import annotations

import dataclasses
import logging
from collections.abc import Iterator
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

from calibrated_agent_evaluation.agentcore.errors import (
    AgentCoreNoPassDirection,
    AgentCoreScoreIncomplete,
)
from calibrated_agent_evaluation.agentcore.scores import result_to_label
from calibrated_agent_evaluation.backends.base import TraceSource
from calibrated_agent_evaluation.ingest import ingest_langfuse
from calibrated_agent_evaluation.ingest.core import IngestReport
from calibrated_agent_evaluation.schema.serde import label_identity

if TYPE_CHECKING:
    from calibrated_agent_evaluation.backends.base import TraceStoreBackend
    from calibrated_agent_evaluation.schema.types import Frame

logger = logging.getLogger(__name__)


@runtime_checkable
class AgentCoreReadSource(TraceSource, Protocol):
    """A :class:`TraceSource` that also reads evaluator scores and records mixed-session re-keys.

    What :func:`ingest_agentcore` needs; ``AgentCoreTraceSource`` satisfies it, and so does a test
    double, since it is structural.
    """

    rekeyed: list[tuple[str, str]]

    def read_scores(self) -> Iterator[dict[str, Any]]:
        """Yield the source's evaluator-result records (empty when none/unrequested)."""
        ...


def ingest_agentcore(
    source: AgentCoreReadSource,
    *,
    backend: TraceStoreBackend,
    dry_run: bool = False,
    require_owned: bool = False,
    with_scores: bool = False,
    frame: Frame | None = None,
) -> IngestReport:
    """Pull AgentCore traces into ``backend``; optionally read evaluator scores as advisory labels.

    The traces go through the same gate and report as any pulled source. A mixed-session trace the
    source re-keyed is noted on its result. When ``with_scores`` is set, evaluator results are
    mapped to advisory ``llm-judge`` labels and persisted, skipping any whose identity is already
    stored (so re-reading the same results adds no duplicate). ``frame`` stamps every pulled trace
    with that sampling frame, as :func:`~calibrated_agent_evaluation.ingest.ingest_langfuse` does.
    """
    report = ingest_langfuse(
        source,
        backend=backend,
        dry_run=dry_run,
        require_owned=require_owned,
        source_kind="agentcore",
        frame=frame,
    )
    rekeyed = dict(source.rekeyed)  # original trace id -> runtime session id it was stored under
    report = _note_rekeys(report, rekeyed)
    if with_scores and not dry_run:
        ingested_ids = {
            r.trace_id for r in report.results if r.status == "ingested" and r.trace_id is not None
        }
        _ingest_scores(source, backend=backend, ingested_ids=ingested_ids, rekeyed=rekeyed)
    return report


def _note_rekeys(report: IngestReport, rekeyed: dict[str, str]) -> IngestReport:
    """Record on each re-keyed trace's result that it was collapsed from a mixed-session id."""
    if not rekeyed:
        return report
    original_of = {new: old for old, new in rekeyed.items()}
    results = [
        dataclasses.replace(r, reason=f"re-keyed from mixed-session id {original_of[r.trace_id]!r}")
        if r.trace_id in original_of and r.status == "ingested"
        else r
        for r in report.results
    ]
    return dataclasses.replace(report, results=results)


def _ingest_scores(
    source: AgentCoreReadSource,
    *,
    backend: TraceStoreBackend,
    ingested_ids: set[str],
    rekeyed: dict[str, str],
) -> int:
    """Persist advisory labels for the evaluator results that belong to this pull's traces.

    A result is stored only when it addresses a trace that actually ingested in THIS pull: a
    re-keyed trace's id is mapped to the runtime session id it was stored under, and a result for
    any other trace (another runtime's evaluation that shared the window, or a trace that was
    refused) is dropped rather than written against an id that is not in the store. A result that
    cannot be mapped to a label is collected and reported, never raised, so one bad record does not
    abort the pass after partial writes; an already-stored identity is skipped so a re-read adds no
    duplicate.
    """
    stored = dropped = failed = 0
    seen: dict[str, set[str]] = {}
    for result in source.read_scores():
        context = result.setdefault("context", {})
        span_context = context.setdefault("spanContext", {})
        raw_id = span_context.get("traceId") or span_context.get("sessionId")
        final_id = rekeyed.get(raw_id, raw_id) if isinstance(raw_id, str) else None
        if final_id not in ingested_ids:
            dropped += 1
            continue
        # Attach the label to the id the trace was actually stored under (post-re-key).
        span_context["traceId"] = final_id
        try:
            label = result_to_label(result)
        except (AgentCoreScoreIncomplete, AgentCoreNoPassDirection):
            failed += 1
            continue
        known = seen.get(label.trace_id)
        if known is None:
            known = {label_identity(existing) for existing in backend.get_scores(label.trace_id)}
            seen[label.trace_id] = known
        identity = label_identity(label)
        if identity in known:
            continue
        backend.put_score(label)
        known.add(identity)
        stored += 1
    logger.info("stored %d advisory AgentCore evaluator label(s)", stored)
    if dropped:
        logger.info(
            "dropped %d evaluator result(s) not addressed to a trace ingested in this pull", dropped
        )
    if failed:
        logger.warning("skipped %d evaluator result(s) that could not be mapped to a label", failed)
    return stored
