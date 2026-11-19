# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""run_judge: the judge-execution runtime.

Runs a promoted judge over a batch of traces/spans: gate the judge once up front (refuse if not
certified), then for each target check the verdict cache, on a miss call the judge
model through the ``execute/`` seam with bounded retry and force a structured verdict,
map it to a validated ``Label``, write it back through the trace store,
and finally emit a ``JudgeRunRecord`` pinning the frozen judge version. No vendor type
appears here; the model call is behind ``JudgeModelBackend``.
"""

from __future__ import annotations

import hashlib
from collections.abc import Sequence
from dataclasses import dataclass

from calibrated_agent_evaluation.backends.base import TraceStoreBackend
from calibrated_agent_evaluation.calibrate.errors import (
    UnpairedObservationError,
    UnratifiedGoldError,
)
from calibrated_agent_evaluation.calibrate.pairing import refuse_duplicate_targets
from calibrated_agent_evaluation.calibrate.record import (
    CalibrationRecord,
    CalibrationSet,
    LabelPair,
)
from calibrated_agent_evaluation.enforce import enforce_score_gate
from calibrated_agent_evaluation.execute import JudgeModelBackend
from calibrated_agent_evaluation.execute.cassette import CassetteMiss
from calibrated_agent_evaluation.execute.errors import ExecutionError, JudgeConfigurationError
from calibrated_agent_evaluation.judges.cache import VerdictCache
from calibrated_agent_evaluation.judges.errors import JudgeRuntimeError
from calibrated_agent_evaluation.judges.executable import ExecutableJudge
from calibrated_agent_evaluation.judges.levels import (
    assert_not_workflow_for_agent_rate,
    assert_span_gold_human,
)
from calibrated_agent_evaluation.judges.run_record import JudgeRunRecord
from calibrated_agent_evaluation.judges.types import JudgeDefinition
from calibrated_agent_evaluation.judges.verdict import reply_to_label
from calibrated_agent_evaluation.judges.verdict_schema import build_verdict_schema
from calibrated_agent_evaluation.schema.types import Label


@dataclass(frozen=True)
class JudgeTarget:
    """One trace (or span) to judge.

    ``content`` is the already-rendered text the judge examines; the caller renders the trace/span
    into the judge's prompt (the runtime stays decoupled from how a trace becomes a prompt).
    ``span_id`` is ``None`` for a whole-trace (workflow) judgment. ``sample`` numbers a repeat of
    the same target (0 is the first judgment), so a repeat is a fresh model draw.
    """

    trace_id: str
    span_id: str | None
    content: str
    sample: int = 0


@dataclass(frozen=True)
class JudgeRunResult:
    """The output of a judge run: the verdict labels + reproducibility record."""

    labels: tuple[Label, ...]
    record: JudgeRunRecord


def _derive_run_id(judge: ExecutableJudge, targets: list[JudgeTarget]) -> str:
    """A deterministic run id from the frozen judge version + the scored targets (no clock/random).

    Deterministic so tests are reproducible and re-running the identical batch yields the same id;
    the ledger of *distinct* runs is the caller's concern (pass an explicit ``run_id`` to force).
    """
    material = "|".join(
        [
            judge.judge_id,
            judge.model,
            judge.prompt_version,
            judge.fingerprint,
            *(t.trace_id for t in targets),
        ]
    )
    return "run-" + hashlib.sha256(material.encode("utf-8")).hexdigest()[:12]


def run_judge(
    judge: ExecutableJudge,
    targets: list[JudgeTarget],
    *,
    calib_record: CalibrationRecord,
    model_backend: JudgeModelBackend,
    store: TraceStoreBackend | None = None,
    cache: VerdictCache | None = None,
    max_retries: int = 2,
    per_agent_rate: bool = False,
    run_id: str | None = None,
) -> JudgeRunResult:
    """Run ``judge`` over ``targets`` and return the verdict labels + run record.

    Refuses (``UncalibratedScoreError``) before any model call if ``calib_record`` does not certify
    ``judge`` at its current version. If ``per_agent_rate`` is requested, refuses a
    workflow-level judge (``JudgeLevelError``). For each target: a cache hit is returned
    without a model call; a miss forces a structured verdict via ``model_backend`` with up
    to ``max_retries`` retries on a transient model failure, maps it to a validated
    ``Label``, caches it, and (if ``store`` is given) writes it back via ``put_score``.
    A cache hit is NOT written back (it is already stored). A backend ``BackendError`` on write-back
    propagates un-swallowed.

    Args:
        judge: The judge to run.
        targets: The traces or spans to score.
        calib_record: The calibration record that certifies ``judge``.
        model_backend: The model the judge's prompts are sent to.
        store: Where to write the verdicts back; nothing is written when omitted.
        cache: Verdicts already computed, reused without a model call.
        max_retries: Retries on a transient model failure.
        per_agent_rate: The verdicts will feed a per-agent rate.
        run_id: The run id; derived from the inputs when omitted.

    Returns:
        The verdict labels and the run record.

    Raises:
        UncalibratedScoreError: ``calib_record`` doesn't certify ``judge`` at its current
            version.
        JudgeLevelError: A per-agent rate was requested from a workflow-level judge.
        JudgeRuntimeError: The model failed after retries, or returned an invalid verdict.
        BackendError: The store is unreachable on write-back.
    """
    # Read the certification (never recompute). Refuses before any
    # model call, so an uncertified judge cannot spend a token or write a Label.
    enforce_score_gate(calib_record, judge)

    # A workflow judge cannot attribute per agent. Reuse the level guard via the
    # level-carrying JudgeDefinition it reads (single source of the refusal rule).
    if per_agent_rate:
        assert_not_workflow_for_agent_rate(
            JudgeDefinition(
                judge_id=judge.judge_id,
                level=judge.level,
                model=judge.model,
                prompt_version=judge.prompt_version,
            )
        )

    schema = build_verdict_schema(judge)
    labels: list[Label] = []
    n_cached = 0
    n_computed = 0

    for target in targets:
        cached = (
            cache.get(judge, target.trace_id, target.span_id, target.sample)
            if cache is not None
            else None
        )
        if cached is not None:
            labels.append(cached)
            n_cached += 1
        else:
            label = _judge_one(judge, target, schema, model_backend, max_retries)
            if cache is not None:
                cache.put(judge, label, target.sample)
            labels.append(label)
            n_computed += 1
        # Write-back per COMPUTED verdict only. A cache hit was already written durably on the run
        # that computed it, so re-writing it added a second row to an append-only score store: a
        # warm-cache re-run doubled every label it touched and moved every rate computed over them.
        # Read-side dedup (`serde.dedup_labels`) makes that harmless, but not writing is the honest
        # form -- "this verdict is already stored" is exactly what a cache hit means.
        #
        # A BackendError (e.g. the Langfuse adapter's synchronous reachability guard firing)
        # propagates un-swallowed: the runtime does NOT catch or wrap it, so an unreachable backend
        # is never reported as a durable write.
        if store is not None and cached is None:
            store.put_score(labels[-1])

    record = JudgeRunRecord(
        run_id=run_id or _derive_run_id(judge, targets),
        judge_id=judge.judge_id,
        model=judge.model,
        prompt_version=judge.prompt_version,
        calib_id=calib_record.calib_id,
        level=judge.level,
        trace_ids=tuple(t.trace_id for t in targets),
        span_ids=tuple(t.span_id for t in targets),
        n_cached=n_cached,
        n_computed=n_computed,
        judge_fingerprint=judge.fingerprint,
    )
    return JudgeRunResult(labels=tuple(labels), record=record)


def locked_test_set(
    judge: ExecutableJudge,
    gold: Sequence[tuple[JudgeTarget, Label]],
    *,
    set_id: str,
    model_backend: JudgeModelBackend,
    max_retries: int = 2,
) -> CalibrationSet:
    """Run ``judge`` over its locked test and pair each verdict with its human gold.

    The one judge run that needs no calibration record, because it is the run calibration is made
    from: :func:`run_judge` refuses an uncertified judge, so it cannot produce the verdicts
    :func:`calibrated_agent_evaluation.calibrate.certify` consumes. The verdicts here are not
    scores. Nothing is written and no verdict is returned on its own; they come back only inside the
    ``locked_test`` :class:`CalibrationSet`, whose one use is ``certify``.

    ``gold`` pairs each :class:`JudgeTarget` with the human label it is graded against, keyed on
    the same ``(trace_id, span_id)``. Refused before any model call: an empty set, a gold label
    that is not ``kind="human"``, a target that is not its gold's, a target graded more than once
    (:class:`~calibrated_agent_evaluation.calibrate.errors.DuplicateGoldError`), and for a span
    judge any gold without a ``span_id`` (calibrate at the level you attribute at).
    """
    if not gold:
        raise UnpairedObservationError(
            f"judge {judge.judge_id!r}: no gold to calibrate against; a locked test needs at least "
            "one human-labeled target"
        )
    labels = [g for _, g in gold]
    if judge.level == "span":
        assert_span_gold_human(labels)
    for target, label in gold:
        if label.kind != "human":
            raise UnratifiedGoldError(
                f"gold label for {label.trace_id} has kind={label.kind!r}, not 'human'; a judge is "
                "certified only against human gold"
            )
        if (target.trace_id, target.span_id) != (label.trace_id, label.span_id):
            raise UnpairedObservationError(
                f"target {(target.trace_id, target.span_id)} is graded against gold "
                f"{(label.trace_id, label.span_id)}; a target and its gold must name the same "
                "trace and span"
            )
    refuse_duplicate_targets((label.trace_id, label.span_id) for label in labels)
    schema = build_verdict_schema(judge)
    pairs = tuple(
        LabelPair(gold=label, verdict=_judge_one(judge, target, schema, model_backend, max_retries))
        for target, label in gold
    )
    return CalibrationSet(pairs=pairs, split="locked_test", set_id=set_id)


def _judge_one(
    judge: ExecutableJudge,
    target: JudgeTarget,
    schema: dict[str, object],
    model_backend: JudgeModelBackend,
    max_retries: int,
) -> Label:
    """Force one structured verdict, with bounded retry, and map it to a validated Label.

    Retries a *transient* model failure (``ExecutionError``) up to ``max_retries`` times, then wraps
    the last cause in :class:`JudgeRuntimeError`; never a raw vendor exception, never a
    fabricated verdict. A ``CassetteMiss`` is deterministic (a missing recording, not transient), so
    it is surfaced immediately as ``JudgeRuntimeError`` rather than retried. ``reply_to_label`` may
    raise ``JudgeRuntimeError`` for a malformed payload; that is a validation failure,
    not a transient one, so it propagates without retry.
    """
    last_exc: Exception | None = None
    for _attempt in range(max_retries + 1):
        try:
            # A first judgment calls without ``sample``, so a backend written before repeats
            # existed keeps working.
            reply = (
                model_backend.generate_structured(
                    system=judge.prompt_template,
                    user=target.content,
                    schema=schema,
                    sample=target.sample,
                )
                if target.sample
                else model_backend.generate_structured(
                    system=judge.prompt_template, user=target.content, schema=schema
                )
            )
        except CassetteMiss as exc:
            raise JudgeRuntimeError(
                f"judge {judge.judge_id!r} on trace {target.trace_id!r}: no recorded verdict "
                f"(cassette miss); {exc}"
            ) from exc
        except JudgeConfigurationError:
            # Permanent by construction: retrying cannot change the outcome, and spending the budget
            # would relabel a configuration fault as flakiness.
            raise
        except ExecutionError as exc:
            last_exc = exc
            continue  # transient; retry up to the bound
        else:
            return reply_to_label(reply, judge, trace_id=target.trace_id, span_id=target.span_id)
    raise JudgeRuntimeError(
        f"judge {judge.judge_id!r} on trace {target.trace_id!r}: model failed after "
        f"{max_retries + 1} attempts; {last_exc}"
    ) from last_exc
