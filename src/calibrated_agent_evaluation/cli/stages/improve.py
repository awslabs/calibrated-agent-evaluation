# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""``improve``: pre-register a fix's target, then evaluate the fix against it.

Two steps, in the order the guard demands:

  * ``calibrated-agent-evaluation improve --register METRIC --min-reduction X`` freezes the target
    metric and the minimum absolute fail-rate reduction the fix must show into the append-only
    pre-registration ledger at ``<root>/records/improve/prereg_ledger.json``
    (``calibrated_agent_evaluation.improve.register_prereg``). Do this before the fix exists; a
    metric is registered once.
  * ``calibrated-agent-evaluation improve --evaluate METRIC --before-root A --after-root B
    --development FILE`` evaluates the fix from the stores: each trace's verdict for the metric
    before the fix (store ``A``) and after it (store ``B``), paired by trace id, and every other
    failure mode with verdicts on both sides as a guard slice, tested with McNemar for a regression
    and Holm corrected at 0.05 (:mod:`calibrated_agent_evaluation.cli.pairing`).
    ``--verdicts judge|scorer|human`` picks the verdict source: the LLM-judge verdicts (the
    default), the code-scorer verdicts or the human labels, never two of them pooled. A source with
    no verdict for the target, or a trace or verdict on one side only, is a typed refusal naming it.
    With ``human``, a trace whose only label on a side is a FAIL for another mode is excluded from
    that mode's pairs, the target's and each guard slice's, and the output and the record's inputs
    give the counts and the ids.
  * ``calibrated-agent-evaluation improve --request FILE`` is the fallback for a fix measured
    elsewhere: it evaluates the paired before/after observations in the request file (below), whose
    guard slices arrive with their p-values already computed.
  * Both evaluate with ``calibrated_agent_evaluation.improve.evaluate_improvement``, reading the
    threshold from the ledger, never from the request. A discordant pair scored by a judge needs a
    human verdict on the after run's output for the target: a trace-level human FAIL for the target,
    or a PASS for it or for no mode, read from the after store (``--evaluate``) or the ``--root``
    store (``--request``). Each rater's newest such label counts, raters who disagree are refused,
    and the human verdict replaces the judge's in the pair before McNemar runs. The verdict is
    persisted as an ``ImproveRecord`` in ``<root>/records/improve/``; a rejected fix exits 1.
  * The ledger is append-only on disk, must still hold every pre-registration an earlier
    evaluation recorded, and with ``--evaluate`` its entry must predate the earliest capture or
    span start time in the after store. When the after store records no time, the output and the
    record say the order was not checked. Registering and evaluating also refuse when a stored run
    record or judge verdict was made against a ledger head
    (``calibrated_agent_evaluation.improve.ledger_head``) the ledger no longer holds: the ledger was
    replaced after that run used it.

The request file is one JSON object::

    {
      "target_metric": "trusted_broken_policy_tool",
      "pairs": [{"trace_id": "c-1", "before_fail": true, "after_fail": false,
                 "labels_are_judge_produced": false}, ...],
      "confirmation_ids": ["c-1", ...],
      "development_ids": ["d-1", ...],
      "slices": [{"name": "misread_amount", "kind": "other", "regression_p": 0.7,
                  "rate_before": <Interval>, "rate_after": <Interval>}, ...]
    }

``confirmation_ids`` defaults to the pairs' trace ids. ``slices`` is the guard panel and must be
non-empty: the library refuses a declared-but-empty panel rather than report "no regression". An
``Interval`` is the object the run records store (``point``, ``lo``, ``hi``, ``method``, ``level``,
``sided``, ``n``). A crew improvement is not taken from a request file; evaluate it from the
stores with ``--crew-file``.

A crew fix (a metric registered with ``--crew``) is evaluated with ``--evaluate METRIC --crew-file
CREW.json --touched-agent NAME``. METRIC is one trace-level failure mode, the crew's workflow
outcome, paired like any target. The crew file declares the agents and handoffs
(:func:`calibrated_agent_evaluation.improve.crew_definition_from_file`); each agent other than the
touched one and each handoff becomes a guard slice, counted per trace from the same source's FAIL
verdicts, attributed to agents by the rule ``calibrated-agent-evaluation run`` uses for per-agent
rates. The relocation check reads the workflow outcome rate against the touched agent's. A FAIL with
a disputed culprit counts for no agent and is named. A slice the source cannot measure stays in the
panel at p = 1 and is listed as not measured.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import secrets
from collections import Counter
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from calibrated_agent_evaluation.cli.dispatch import EXIT_DOMAIN_ERROR, EXIT_OK, StageOutcome
from calibrated_agent_evaluation.cli.errors import CliError, UsageError
from calibrated_agent_evaluation.cli.plain import points

_REQUEST_FIELDS = {"target_metric", "pairs", "confirmation_ids", "development_ids", "slices"}
_SLICE_KINDS = {"target", "agent", "handoff", "other"}
_CREW_FIELDS = {"crew", "crew_before", "crew_after", "touched_agent"}


def improve_handler(args: argparse.Namespace) -> StageOutcome:
    """Register a target (``--register``) or evaluate a fix (``--request``)."""
    from calibrated_agent_evaluation.cli.config import require_extras

    require_extras("scipy", "statsmodels")
    register = getattr(args, "register", None)
    evaluate = getattr(args, "evaluate", None)
    request = getattr(args, "request", None)
    if sum(bool(x) for x in (register, evaluate, request)) != 1:
        raise UsageError(
            "calibrated-agent-evaluation improve needs exactly one of --register METRIC (before "
            "the fix), --evaluate "
            "METRIC (the fix, from the before and after stores) or --request FILE (the fix, from "
            "a request file)"
        )
    if not evaluate and (getattr(args, "crew_file", None) or getattr(args, "touched_agent", None)):
        raise UsageError(
            "--crew-file and --touched-agent go with --evaluate METRIC: a crew fix is evaluated "
            "from the before and after stores"
        )
    if register:
        return _register(args, register)
    if evaluate:
        return evaluate_store(args, evaluate)
    return _evaluate(args, Path(str(request)))


def _register(args: argparse.Namespace, metric: str) -> StageOutcome:
    from calibrated_agent_evaluation.cli.backends import local_root
    from calibrated_agent_evaluation.cli.records import (
        load_ledger,
        records_dir,
        records_lock,
        save_ledger,
    )
    from calibrated_agent_evaluation.improve import (
        PreRegEntry,
        register_prereg,
        verify_anchors,
        verify_heads,
        verify_ledger,
    )

    reduction = getattr(args, "min_reduction", None)
    if reduction is None:
        raise UsageError(
            "--register needs --min-reduction X: the absolute fail-rate drop the fix must show"
        )
    root = local_root(args)
    # Read, extend and write the ledger under one lock, so two registrations cannot both chain
    # onto the same tail and lose one.
    with records_lock(root, "improve"):
        ledger = load_ledger(root)
        verify_ledger(ledger)
        verify_anchors(ledger, _anchors(root))
        verify_heads(ledger, _stamped_heads(root))
        entry = PreRegEntry(
            prereg_id=f"prereg-{len(ledger.entries) + 1}-{metric}",
            target_metric=metric,
            success_threshold=float(reduction),
            is_crew=bool(getattr(args, "crew", False)),
            code_ref=getattr(args, "code_ref", None),
        )
        ledger = register_prereg(ledger, entry)
        save_ledger(root, ledger, locked=True)
    frozen = ledger.entries[-1]
    return StageOutcome(
        stage="improve",
        exit_code=EXIT_OK,
        summary=(
            f"Pre-registered {metric!r}: the fix must cut its failure rate by at least "
            f"{points(frozen.success_threshold)}. The threshold is now frozen.\n"
            "Next: after the fix, compare the run before it with the run after it: "
            f"`calibrated-agent-evaluation improve --evaluate {metric} --before-root A "
            "--after-root B --development "
            "FILE" + (" --crew-file CREW.json --touched-agent NAME`." if frozen.is_crew else "`.")
        ),
        detail=(
            f"pre-registered {metric!r}: the fix must cut its fail rate by at least "
            f"{frozen.success_threshold:.2f}, an absolute difference (the whole paired-effect "
            "interval at or below "
            f"-{frozen.success_threshold:.2f}); frozen at {frozen.frozen_at}, hash "
            f"{frozen.content_hash[:12]}, entry {len(ledger.entries)} of the ledger in "
            f"{records_dir(root, 'improve')}"
        ),
    )


def _evaluate(args: argparse.Namespace, path: Path) -> StageOutcome:
    from calibrated_agent_evaluation.cli.backends import local_root, open_backend
    from calibrated_agent_evaluation.cli.records import append_records, load_ledger
    from calibrated_agent_evaluation.improve import (
        ImproveRequest,
        discordant_pairs,
        evaluate_improvement,
        record_of,
        slice_id,
        verify_heads,
    )

    doc = _read_request(path)
    pairs = _pairs(doc, path)
    confirmation = tuple(doc.get("confirmation_ids") or [p.trace_id for p in pairs])
    development = tuple(_strings(doc, "development_ids", path))
    if not development:
        raise CliError(
            f"request {path} needs development_ids: the traces the fix was developed on, so the "
            "guard can prove the confirmation pairs are held out"
        )
    root = local_root(args)
    metric = str(doc["target_metric"])
    judged = [p for p in discordant_pairs(pairs) if p.labels_are_judge_produced]
    labels: dict[str, list[Any]] = {}
    if judged:
        with open_backend(args) as backend:
            labels = {p.trace_id: backend.get_scores(p.trace_id) for p in judged}
    adjudications = _adjudications(pairs, labels, metric)
    ledger = load_ledger(root)
    _require_crew_match(ledger, metric, crew=False, how="--request")
    verify_heads(ledger, _stamped_heads(root))
    request = ImproveRequest(
        ledger=ledger,
        target_metric=metric,
        pairs=pairs,
        confirmation_ids=confirmation,
        development_ids=development,
        slices=_slices(doc, path),
        adjudications=adjudications,
        anchors=_anchors(root),
    )
    verdict = evaluate_improvement(request)
    improve_id = getattr(args, "improve_id", None) or _improve_id()
    record = record_of(
        verdict,
        improve_id=improve_id,
        confirmation_slice_id=slice_id("confirm", confirmation),
        development_set_id=slice_id("develop", development),
        inputs={
            "request": str(path),
            "request_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "root": str(root),
            "n_pairs": str(len(pairs)),
            **_adjudication_inputs(pairs, adjudications),
            **_order_inputs(verdict, ledger, None),
        },
    )
    append_records(root, "improve", [record])
    return StageOutcome(
        stage="improve",
        exit_code=EXIT_OK if verdict.verdict == "accepted" else EXIT_DOMAIN_ERROR,
        summary="\n".join(
            [
                *_plain().improve(verdict),
                _plain().improve_order_unchecked(_REQUEST_HAS_NO_TIMES),
            ]
        ),
        detail="\n".join(
            [
                *_verdict_lines(verdict),
                *_adjudication_lines(pairs, adjudications),
                _order_line(verdict, None, _REQUEST_HAS_NO_TIMES),
                f"recorded {improve_id} in {root / 'records' / 'improve'}",
            ]
        ),
    )


def evaluate_store(args: argparse.Namespace, metric: str) -> StageOutcome:
    """Evaluate a fix from the verdicts in two stores, paired per trace."""
    from calibrated_agent_evaluation.cli.backends import local_root
    from calibrated_agent_evaluation.cli.pairing import guard_slice, pair_stores, source_name
    from calibrated_agent_evaluation.cli.records import append_records, load_ledger
    from calibrated_agent_evaluation.improve import (
        ImproveRequest,
        PairedObs,
        evaluate_improvement,
        record_of,
        slice_id,
        verify_heads,
    )

    before_root, after_root, dev_path = _evaluate_inputs(args)
    crew_in = _crew_input(args)
    development = _id_file(dev_path)
    source = str(getattr(args, "verdicts", None) or "judge")
    root = local_root(args)
    ledger = load_ledger(root)
    _require_crew_match(ledger, metric, crew=crew_in is not None, how="--evaluate")
    before, _, before_spans = _store_labels(before_root)
    after, after_started, after_spans = _store_labels(after_root)
    paired = pair_stores(before, after, metric, source)

    # A code scorer's verdict is no more human gold than a judge's, so its discordant pairs are
    # adjudicated the same way; only human labels pair without adjudication.
    judged = source != "human"
    pairs = tuple(
        PairedObs(trace_id=t, before_fail=b, after_fail=a, labels_are_judge_produced=judged)
        for t, (b, a) in sorted(paired.target.items())
    )
    slices = tuple(guard_slice(mode, p) for mode, p in paired.guards.items() if p)
    if not slices and crew_in is None:
        raise _no_guard_slice(metric, source, paired)
    # The after store is where the fix's output was reviewed; a human label in the before store
    # is a verdict on the old agent's output, not on whether the fix worked.
    adjudications = _adjudications(pairs, after, metric) if judged else {}
    confirmation = tuple(p.trace_id for p in pairs)
    verify_heads(ledger, _stamped_heads(root, before_root, after_root))
    crew_view = (
        None
        if crew_in is None
        else _crew_view(
            crew_in,
            pairs,
            adjudications,
            (before, before_spans),
            (after, after_spans),
            source=source,
            metric=metric,
        )
    )
    if crew_view is not None:
        slices = (*crew_view.slices, *slices)
    request = ImproveRequest(
        ledger=ledger,
        target_metric=metric,
        pairs=pairs,
        confirmation_ids=confirmation,
        development_ids=development,
        slices=slices,
        adjudications=adjudications,
        anchors=_anchors(root),
        after_run_started_at=after_started,
        **(crew_view.request_fields() if crew_view is not None else {}),
    )
    verdict = evaluate_improvement(request)
    improve_id = getattr(args, "improve_id", None) or _improve_id()
    record = record_of(
        verdict,
        improve_id=improve_id,
        confirmation_slice_id=slice_id("confirm", confirmation),
        development_set_id=slice_id("develop", development),
        inputs={
            "before_root": str(before_root),
            "after_root": str(after_root),
            "verdict_source": source,
            "development": str(dev_path),
            "development_sha256": hashlib.sha256(dev_path.read_bytes()).hexdigest(),
            "root": str(root),
            "n_pairs": str(len(pairs)),
            **_adjudication_inputs(pairs, adjudications),
            **_order_inputs(verdict, ledger, after_started),
            **_exclusion_inputs(paired),
            **(crew_view.record_inputs() if crew_view is not None else {}),
        },
    )
    append_records(root, "improve", [record])
    slice_lines = [_slice_line(s) for s in slices]
    no_time = f"no trace in {after_root} records a time"
    return StageOutcome(
        stage="improve",
        exit_code=EXIT_OK if verdict.verdict == "accepted" else EXIT_DOMAIN_ERROR,
        summary="\n".join(
            [
                *_plain().improve(verdict),
                *(crew_view.plain_lines() if crew_view else []),
                *([] if after_started else [_plain().improve_order_unchecked(no_time)]),
            ]
        ),
        detail="\n".join(
            [
                *_verdict_lines(verdict),
                f"  paired {len(pairs)} trace(s) by id between {before_root} (before) and "
                f"{after_root} (after), {source_name(source)}",
                *_exclusion_lines(metric, paired),
                *_adjudication_lines(pairs, adjudications),
                _order_line(verdict, after_started, no_time),
                *(crew_view.detail_lines(verdict) if crew_view is not None else []),
                "  guard slices:",
                *slice_lines,
                f"recorded {improve_id} in {root / 'records' / 'improve'}",
            ]
        ),
    )


def _evaluate_inputs(args: argparse.Namespace) -> tuple[Path, Path, Path]:
    """The before and after stores and the development-id file ``--evaluate`` reads."""
    if str(getattr(args, "backend", "local")) != "local":
        raise CliError(
            "--before-root and --after-root are local stores; bring a Langfuse project's runs "
            "into two roots with `calibrated-agent-evaluation ingest --from-langfuse` and "
            "evaluate those"
        )
    before_root, after_root = _side_root(args, "before"), _side_root(args, "after")
    if before_root.resolve() == after_root.resolve():
        raise UsageError("--before-root and --after-root name the same store; a fix needs two runs")
    dev_path = getattr(args, "development", None)
    if not dev_path:
        raise UsageError(
            "--evaluate needs --development FILE: the trace ids the fix was developed on, one per "
            "line, so the guard can prove the paired traces are held out"
        )
    return before_root, after_root, Path(str(dev_path))


def _no_guard_slice(metric: str, source: str, paired: Any) -> CliError:
    """The refusal when no other failure mode is paired, so a regression could not be seen."""
    from calibrated_agent_evaluation.cli.pairing import excluded_note, source_name

    how = {
        "judge": "score at least one other mode in both stores "
        "(`calibrated-agent-evaluation judge` with that mode's judge file)",
        "scorer": "store code-scorer verdicts for at least one other mode in both stores",
        "human": "label at least one other mode in both stores",
    }[source]
    return CliError(
        f"no guard slice for {metric!r}: no other failure mode has {source_name(source)} "
        f"paired on both sides{excluded_note(paired)}, so a collateral regression could not "
        f"be seen. {how[0].upper()}{how[1:]}"
    )


def _adjudications(pairs: Any, labels: Mapping[str, Sequence[Any]], metric: str) -> dict[str, Any]:
    """The human verdict on each judge-scored discordant pair's after-run output for ``metric``.

    Among a trace's trace-level human labels that speak to ``metric``, each rater's newest counts.
    Raters who disagree are refused, naming the traces: picking one would decide the pair by
    which label happened to be written last. A trace with human labels but none about ``metric``
    passes its newest one through, so the guard's refusal names what is wrong with it.
    """
    from calibrated_agent_evaluation.improve import (
        AdjudicationRequiredError,
        adjudication_problem,
        discordant_pairs,
    )

    adjudications: dict[str, Any] = {}
    split: list[str] = []
    for pair in discordant_pairs(pairs):
        if not pair.labels_are_judge_produced:
            continue
        human = [lb for lb in labels.get(pair.trace_id, ()) if lb.kind == "human"]
        usable = [lb for lb in human if adjudication_problem(pair.trace_id, lb, metric) is None]
        if not usable:
            if human:
                adjudications[pair.trace_id] = human[-1]
            continue
        newest = {lb.rater_id: lb for lb in usable}
        if len({lb.outcome for lb in newest.values()}) > 1:
            said = ", ".join(f"{r or 'an unnamed rater'} {lb.outcome}" for r, lb in newest.items())
            split.append(f"{pair.trace_id!r} ({said})")
            continue
        adjudications[pair.trace_id] = usable[-1]
    if split:
        raise AdjudicationRequiredError(
            f"the human raters disagree about {metric!r} on {len(split)} judge-scored flip(s): "
            f"{'; '.join(split[:5])}. Settle each (one rater relabels, or a third adjudicates and "
            "the others' labels are replaced) before evaluating the fix."
        )
    return adjudications


def _adjudication_inputs(pairs: Any, adjudications: Mapping[str, Any]) -> dict[str, str]:
    """The record's count of adjudicated flips and of those the human overturned."""
    return {
        "n_adjudicated": str(len(adjudications)),
        "n_adjudication_overturned": str(len(_overturned(pairs, adjudications))),
    }


def _adjudication_lines(pairs: Any, adjudications: Mapping[str, Any]) -> list[str]:
    if not adjudications:
        return []
    overturned = _overturned(pairs, adjudications)
    return [
        f"  {len(adjudications)} judge-scored flip(s) adjudicated by a human, whose verdict on the "
        f"after run replaced the judge's; {len(overturned)} overturned"
        + (f" ({_shown(overturned)})" if overturned else "")
    ]


def _overturned(pairs: Any, adjudications: Mapping[str, Any]) -> tuple[str, ...]:
    return tuple(
        p.trace_id
        for p in pairs
        if p.trace_id in adjudications
        and (adjudications[p.trace_id].outcome == "FAIL") != p.after_fail
    )


def _anchors(root: Path) -> tuple[Any, ...]:
    """The pre-registrations earlier evaluations recorded, which the ledger must still hold."""
    from calibrated_agent_evaluation.cli.records import load_records_of

    return tuple(r.prereg for r in load_records_of(root, "improve"))


def _stamped_heads(root: Path, *stores: Path) -> list[tuple[str, str]]:
    """Each stored run or judge record that names the ledger head it was made against.

    Read from the run records under ``root`` and from every label in the local stores at ``root``
    and ``stores`` that carries a head. Records made before heads were stamped carry none and are
    not checked. One record is named per distinct head.
    """
    from calibrated_agent_evaluation.backends.local import LocalBackend
    from calibrated_agent_evaluation.cli.records import RECORD_KINDS, load_records_of
    from calibrated_agent_evaluation.improve import LEDGER_HEAD_FIELD

    seen: dict[str, str] = {}
    for kind in ("run", "judge"):
        if kind not in RECORD_KINDS:
            continue
        for record in load_records_of(root, kind):
            # Judge run records carry the head as a field; run records keep it in their inputs.
            head = getattr(record, LEDGER_HEAD_FIELD, None) or getattr(record, "inputs", {}).get(
                LEDGER_HEAD_FIELD
            )
            if head:
                rid = getattr(record, "run_id", None) or "unnamed"
                seen.setdefault(str(head), f"{kind} record {rid!r}")
    for store in dict.fromkeys(p.resolve() for p in (root, *stores)):
        if not (store / "store.sqlite3").exists():
            continue
        with LocalBackend(root=store) as backend:
            for label in backend.all_scores():
                head = getattr(label, LEDGER_HEAD_FIELD, None)
                if head:
                    seen.setdefault(
                        str(head), f"judge verdict on trace {label.trace_id!r} in {store}"
                    )
    return [(what, head) for head, what in seen.items()]


_REQUEST_HAS_NO_TIMES = "a request file carries no run times"


def _order_inputs(verdict: Any, ledger: Any, after_started: datetime | None) -> dict[str, str]:
    """The record's account of when the target was frozen against when the after run began."""
    from calibrated_agent_evaluation.improve import ledger_head

    return {
        "prereg_frozen_at": verdict.prereg.frozen_at,
        "after_run_started_at": after_started.isoformat() if after_started else "not recorded",
        "ledger_head": ledger_head(ledger),
        "ledger_entries": str(len(ledger.entries)),
    }


def _order_line(verdict: Any, after_started: datetime | None, why_unknown: str) -> str:
    frozen = verdict.prereg.frozen_at
    if after_started is None:
        return f"  pre-registration frozen at {frozen}; not checked against the after run: " + (
            why_unknown
        )
    return f"  pre-registration frozen at {frozen}, before the after run began at {after_started}"


def _exclusion_inputs(paired: Any) -> dict[str, str]:
    """The record's account of which traces were left out of the pairs and the guard slices."""
    return {
        "n_excluded": str(len(paired.excluded)),
        "excluded_ids": ",".join(paired.excluded) or "none",
        "n_guard_excluded": str(sum(len(ids) for ids in paired.guard_excluded.values())),
        "guard_excluded_ids": ";".join(
            f"{mode}={','.join(ids)}" for mode, ids in paired.guard_excluded.items()
        )
        or "none",
    }


def _exclusion_lines(metric: str, paired: Any) -> list[str]:
    """The ``--explain`` lines naming every trace left out, and why."""
    why = "a human FAIL for another failure mode is not evidence about it"
    lines = (
        [
            f"  excluded {len(paired.excluded)} trace(s) from the pairs for {metric!r}: {why} "
            f"({', '.join(repr(t) for t in paired.excluded)})"
        ]
        if paired.excluded
        else []
    )
    lines += [
        f"  excluded {len(ids)} trace(s) from the guard slice {mode!r}: {why} ({_shown(ids)})"
        + ("; no pair left, so no slice" if not paired.guards[mode] else "")
        for mode, ids in paired.guard_excluded.items()
    ]
    return lines


def _side_root(args: argparse.Namespace, side: str) -> Path:
    value = getattr(args, f"{side}_root", None)
    if not value:
        raise UsageError(f"--evaluate needs --{side}-root PATH, the store of the {side} run")
    path = Path(str(value))
    if not path.is_dir():
        raise CliError(f"--{side}-root {path} is not a directory")
    return path


def _store_labels(
    root: Path,
) -> tuple[dict[str, list[Any]], datetime | None, dict[tuple[str, str], str]]:
    """Each trace's labels in the store at ``root``, its earliest time and its span agents.

    The span agents are keyed ``(trace_id, span_id)``. The time is the earliest timezone-aware
    ``captured_at`` or span ``started_at`` among the traces, ``None`` when none records one.
    """
    from calibrated_agent_evaluation.attribution.labels import span_agents
    from calibrated_agent_evaluation.backends.local import LocalBackend
    from calibrated_agent_evaluation.schema.serde import latest_verdicts

    with LocalBackend(root=root) as backend:
        traces = backend.query()
        labels = {t.trace_id: latest_verdicts(backend.get_scores(t.trace_id)) for t in traces}
    stamps = [
        when
        for t in traces
        for raw in (t.captured_at, *(sp.started_at for sp in t.spans))
        if (when := _aware(raw)) is not None
    ]
    spans = span_agents(traces)
    return labels, min(stamps, default=None), spans


def _aware(raw: str | None) -> datetime | None:
    if not raw:
        return None
    try:
        when = datetime.fromisoformat(raw)
    except ValueError:
        return None
    return when if when.utcoffset() is not None else None


def _id_file(path: Path) -> tuple[str, ...]:
    try:
        text = path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError) as exc:
        raise CliError(f"cannot read --development {path}: {exc}") from exc
    ids = tuple(line.strip() for line in text.splitlines() if line.strip())
    if not ids:
        raise CliError(f"--development {path} lists no trace ids")
    return ids


def _shown(ids: tuple[str, ...], limit: int = 5) -> str:
    """The first ``limit`` ids and how many more; the record keeps every one."""
    head = ", ".join(repr(t) for t in ids[:limit])
    return head + (f" and {len(ids) - limit} more" if len(ids) > limit else "")


def _plain() -> Any:
    from calibrated_agent_evaluation.cli import plain

    return plain


def _rate(iv: Any) -> str:
    return f"{iv.point:.3f} [{iv.lo:.3f}, {iv.hi:.3f}] (n={iv.n})"


def _verdict_lines(verdict: Any) -> list[str]:
    paired = verdict.paired
    effect = paired.effect
    need = verdict.prereg.success_threshold
    refused = f"REFUSED ({effect.refusal_reason})"
    shown = refused if effect.point is None else f"{effect.point:+.3f}"
    p_text = "no p-value" if paired.p_value is None else f"p={paired.p_value:.4g}"
    head = f"improve {verdict.prereg.target_metric!r}: {verdict.verdict.upper()}"
    lines = [
        head,
        f"  paired fail-rate change (after - before) {shown}, {effect.level:.0%} CI "
        f"[{effect.lo:+.3f}, {effect.hi:+.3f}] ({effect.method}, n={effect.n}); McNemar "
        f"{paired.method}, {paired.n_discordant} discordant pair(s), {p_text}",
        f"  pre-registered minimum reduction {need:.2f}: the interval's upper bound must be at "
        f"or below {-need:+.3f}",
        f"  guard panel: {verdict.panel.m} slice(s), Holm; regressed: "
        + (", ".join(verdict.panel.regressed) or "none"),
    ]
    if paired.undefined_reason:
        lines.append(f"  {paired.undefined_reason}")
    lines += [f"  reason: {r}" for r in verdict.reasons]
    return lines


def _read_request(path: Path) -> dict[str, Any]:
    try:
        doc = json.loads(path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise CliError(f"cannot read request {path}: {exc.strerror or exc}") from exc
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise CliError(f"request {path} is not valid JSON: {exc}") from exc
    if not isinstance(doc, dict):
        raise CliError(f"request {path} must hold one JSON object")
    crew = sorted(_CREW_FIELDS & set(doc))
    if crew:
        raise CliError(
            f"request {path} describes a crew improvement ({', '.join(crew)}); the CLI does not "
            "take one yet, so evaluate it through "
            "calibrated_agent_evaluation.improve.evaluate_improvement"
        )
    unknown = sorted(set(doc) - _REQUEST_FIELDS)
    if unknown:
        raise CliError(f"request {path} has unknown field(s): {', '.join(unknown)}")
    if not isinstance(doc.get("target_metric"), str) or not doc["target_metric"]:
        raise CliError(f"request {path} needs target_metric, the pre-registered metric")
    return doc


def _strings(doc: dict[str, Any], key: str, path: Path) -> list[str]:
    values = doc.get(key) or []
    if not isinstance(values, list) or not all(isinstance(v, str) for v in values):
        raise CliError(f"request {path}: {key} must be a list of trace ids")
    return values


def _pairs(doc: dict[str, Any], path: Path) -> tuple[Any, ...]:
    from calibrated_agent_evaluation.improve import PairedObs

    raw = doc.get("pairs")
    if not isinstance(raw, list) or not raw:
        raise CliError(f"request {path} needs a non-empty pairs list")
    try:
        pairs = tuple(
            PairedObs(
                trace_id=str(p["trace_id"]),
                before_fail=_bool(p["before_fail"]),
                after_fail=_bool(p["after_fail"]),
                labels_are_judge_produced=_bool(p.get("labels_are_judge_produced", False)),
            )
            for p in raw
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise CliError(f"request {path}: a pair is malformed ({exc!r})") from exc
    counts = Counter(p.trace_id for p in pairs)
    repeated = tuple(sorted(t for t, n in counts.items() if n > 1))
    if repeated:
        raise CliError(
            f"request {path} pairs {len(repeated)} trace id(s) more than once "
            f"({_shown(repeated)}); McNemar would count each copy as another input. List each "
            "trace once."
        )
    return pairs


def _bool(value: Any) -> bool:
    if not isinstance(value, bool):
        raise TypeError(f"expected true/false, got {value!r}")
    return value


def _slices(doc: dict[str, Any], path: Path) -> tuple[Any, ...]:
    from calibrated_agent_evaluation.calibrate.record import interval_from_dict
    from calibrated_agent_evaluation.improve import GuardSlice

    out = []
    for raw in doc.get("slices") or []:
        if not isinstance(raw, dict) or raw.get("kind") not in _SLICE_KINDS:
            raise CliError(
                f"request {path}: each guard slice needs a kind, one of {sorted(_SLICE_KINDS)}"
            )
        try:
            out.append(
                GuardSlice(
                    name=str(raw["name"]),
                    kind=raw["kind"],
                    regression_p=float(raw["regression_p"]),
                    rate_before=interval_from_dict(raw["rate_before"]),
                    rate_after=interval_from_dict(raw["rate_after"]),
                )
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise CliError(f"request {path}: a guard slice is malformed ({exc!r})") from exc
    return tuple(out)


def _improve_id() -> str:
    stamp = datetime.now(tz=UTC).strftime("%Y%m%dT%H%M%SZ")
    return f"improve-{stamp}-{secrets.token_hex(3)}"


def _slice_line(s: Any) -> str:
    from calibrated_agent_evaluation.improve.crew import METHOD_NOT_MEASURED

    if s.rate_before.method == METHOD_NOT_MEASURED:
        return f"    {s.name} ({s.kind}): {s.rate_before.refusal_reason}; held at p=1"
    return (
        f"    {s.name}: fail rate {_rate(s.rate_before)} before, {_rate(s.rate_after)} after; "
        f"one-sided McNemar regression p={s.regression_p:.4g}"
    )


class _CrewInput:
    """The crew file, read and checked before either store is opened."""

    def __init__(self, crew: Any, path: Path, sha256: str, touched: str) -> None:
        self.crew = crew
        self.path = path
        self.sha256 = sha256
        self.touched = touched


def _crew_input(args: argparse.Namespace) -> _CrewInput | None:
    """The ``--crew-file`` and ``--touched-agent`` pair, or ``None`` for a single-agent fix."""
    from calibrated_agent_evaluation.attribution import CrewDefinitionError
    from calibrated_agent_evaluation.improve import ImproveError, crew_definition_from_file

    raw_path = getattr(args, "crew_file", None)
    touched = getattr(args, "touched_agent", None)
    if not raw_path and not touched:
        return None
    if not raw_path:
        raise UsageError("--touched-agent needs --crew-file CREW.json, the crew it belongs to")
    path = Path(str(raw_path))
    if not touched:
        raise UsageError(
            f"--crew-file {path} needs --touched-agent NAME: the agent the fix changed, whose "
            "failures the relocation check follows"
        )
    try:
        data = path.read_bytes()
        doc = json.loads(data.decode("utf-8"))
    except OSError as exc:
        raise CliError(f"cannot read --crew-file {path}: {exc.strerror or exc}") from exc
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise CliError(f"--crew-file {path} is not valid JSON: {exc}") from exc
    if not isinstance(doc, dict):
        raise CliError(f"--crew-file {path} must hold one JSON object")
    try:
        crew = crew_definition_from_file(doc)
    except (ImproveError, CrewDefinitionError) as exc:
        raise CliError(f"--crew-file {path}: {exc}") from exc
    if touched not in crew.agents:
        raise CliError(
            f"--touched-agent {touched!r} is not an agent in {path} ({', '.join(crew.agents)})"
        )
    return _CrewInput(crew, path, hashlib.sha256(data).hexdigest(), str(touched))


def _require_crew_match(ledger: Any, metric: str, *, crew: bool, how: str) -> None:
    """Refuse a crew file for a single-agent metric, and a crew metric without one."""
    entries = [e for e in ledger.entries if e.target_metric == metric]
    if not entries:
        return  # evaluate_improvement refuses an unregistered metric with its own message
    registered_crew = entries[-1].is_crew
    if crew and not registered_crew:
        raise CliError(
            f"{metric!r} was not registered as a crew metric, so its threshold is not about the "
            "workflow outcome; drop --crew-file, or register a new metric with --register METRIC "
            "--crew"
        )
    if registered_crew and not crew:
        raise CliError(
            f"{metric!r} was registered as a crew metric (--crew); evaluate it with --evaluate "
            f"{metric} --crew-file CREW.json --touched-agent NAME"
            + (", not --request" if how == "--request" else "")
        )


_SideView = tuple[Mapping[str, Sequence[Any]], dict[tuple[str, str], str]]


class _CrewView:
    """The crew counts of one evaluation, and how the output and the record show them."""

    def __init__(
        self,
        crew_in: _CrewInput,
        comparison: Any,
        outcomes: tuple[Any, Any],
        slices: tuple[Any, ...],
        source: str,
    ) -> None:
        self.crew_in = crew_in
        self.comparison = comparison
        self.outcomes = outcomes
        self.slices = slices
        self.source = source

    def request_fields(self) -> dict[str, Any]:
        return {
            "crew": self.crew_in.crew,
            "crew_before": self.outcomes[0],
            "crew_after": self.outcomes[1],
            "touched_agent": self.crew_in.touched,
        }

    def not_measured(self) -> list[str]:
        c = self.comparison
        return [*c.not_measured_agents, *c.not_measured_handoffs]

    def record_inputs(self) -> dict[str, str]:
        c = self.comparison

        def counts(failed: Mapping[str, tuple[str, ...]]) -> str:
            return ",".join(f"{k}={len(v)}" for k, v in failed.items()) or "none"

        return {
            "crew_file": str(self.crew_in.path),
            "crew_file_sha256": self.crew_in.sha256,
            "touched_agent": self.crew_in.touched,
            "agent_failures_before": counts(c.before.agent_failed),
            "agent_failures_after": counts(c.after.agent_failed),
            "handoff_failures_before": counts(c.before.handoff_failed),
            "handoff_failures_after": counts(c.after.handoff_failed),
            "not_measured": ",".join(self.not_measured()) or "none",
            "disputed_before": ",".join(c.before.disputed) or "none",
            "disputed_after": ",".join(c.after.disputed) or "none",
        }

    def plain_lines(self) -> list[str]:
        from calibrated_agent_evaluation.cli.pairing import source_name

        c = self.comparison
        lines = []
        disputed = sorted(set(c.before.disputed) | set(c.after.disputed))
        if disputed:
            lines.append(
                f"{len(disputed)} trace(s) have a failure whose culprit is disputed, so it counts "
                f"for no agent (it still counts for the crew): {_shown(tuple(disputed))}."
            )
        missing = self.not_measured()
        if missing:
            lines.append(
                f"Not measured by the {source_name(self.source)}, so not guarded: "
                f"{', '.join(missing)}."
            )
        return lines

    def detail_lines(self, verdict: Any) -> list[str]:
        from calibrated_agent_evaluation.cli.pairing import source_name

        c = self.comparison
        before, after = self.outcomes
        touched = self.crew_in.touched
        lines = [
            f"  crew {self.crew_in.path} (sha256 {self.crew_in.sha256[:12]}), touched agent "
            f"{touched!r}: workflow failures {before.workflow_failures} of "
            f"{before.workflow_total} before, {after.workflow_failures} of "
            f"{after.workflow_total} after"
        ]
        relocation = verdict.panel.relocation
        if relocation is not None:
            lines.append(f"  relocation check: {relocation.verdict}")
        for agent, failed in c.before.agent_failed.items():
            if agent in c.not_measured_agents:
                continue
            lines.append(
                f"    agent {agent}: {len(failed)} failed trace(s) before, "
                f"{len(c.after.agent_failed[agent])} after"
                + (" (touched)" if agent == touched else "")
            )
        for seam, failed in c.before.handoff_failed.items():
            if seam in c.not_measured_handoffs:
                continue
            lines.append(
                f"    handoff {seam}: {len(failed)} failed trace(s) before, "
                f"{len(c.after.handoff_failed[seam])} after"
            )
        if touched in c.not_measured_agents:
            lines.append(
                f"  the {source_name(self.source)} do not measure the touched agent {touched!r}, "
                "so the relocation check cannot see its failures fall"
            )
        for side, ids in (("before", c.before.disputed), ("after", c.after.disputed)):
            if ids:
                lines.append(
                    f"  {len(ids)} trace(s) {side} with a disputed culprit, counted for no agent "
                    f"and still toward the workflow outcome: {_shown(ids)}"
                )
        return lines


def _crew_view(
    crew_in: _CrewInput,
    pairs: tuple[Any, ...],
    adjudications: Mapping[str, Any],
    before: _SideView,
    after: _SideView,
    *,
    source: str,
    metric: str,
) -> _CrewView:
    """Count the crew's agent and handoff failures on each side and build their guard slices."""
    from calibrated_agent_evaluation.attribution.labels import attributed_agent
    from calibrated_agent_evaluation.cli.pairing import SOURCE_KINDS, guard_slice, source_name
    from calibrated_agent_evaluation.improve import (
        adjudicate_pairs,
        build_crew_slices,
        compare_crew_sides,
        count_crew_side,
        crew_outcomes,
        discordant_pairs,
        not_measured_rate,
        require_adjudication,
    )

    crew = crew_in.crew
    kinds = SOURCE_KINDS[source]
    paired_ids = [p.trace_id for p in pairs]

    def is_verdict(label: Any) -> bool:
        # The run gate's reading: a span defect is attribution evidence, not a verdict, and a
        # human's verdict is the trace-level one.
        return (
            label.kind in kinds
            and not label.span_defect
            and (source != "human" or label.span_id is None)
        )

    def side(view: _SideView, name: str) -> Any:
        labels, spans = view
        ran: dict[str, set[str]] = {}
        for (trace_id, _span), agent in spans.items():
            ran.setdefault(trace_id, set()).add(agent)
        return count_crew_side(
            crew,
            labels,
            paired_ids,
            is_verdict=is_verdict,
            attribute=lambda lb: attributed_agent(lb, spans),
            ran=ran if source == "human" else None,
            side=name,
        )

    comparison = compare_crew_sides(
        crew, side(before, "before"), side(after, "after"), paired_ids, source=source_name(source)
    )
    require_adjudication(
        discordant_pairs(pairs),
        adjudications,
        labels_are_judge_produced=any(p.labels_are_judge_produced for p in pairs),
        target_metric=metric,
    )
    workflow = adjudicate_pairs(pairs, adjudications, target_metric=metric)
    outcomes = crew_outcomes(comparison, workflow)

    unmeasured = not_measured_rate(source_name(source))
    p_values: dict[str, float] = {}
    rates: dict[str, tuple[Any, Any]] = {}
    tables = {**comparison.agent_pairs, **comparison.handoff_pairs}
    for name in (*(a for a in crew.agents if a != crew_in.touched), *crew.handoffs):
        table = tables.get(name)
        if table is None:
            p_values[name], rates[name] = 1.0, (unmeasured, unmeasured)
        else:
            s = guard_slice(name, table)
            p_values[name], rates[name] = s.regression_p, (s.rate_before, s.rate_after)
    seams = set(crew.handoffs)
    slices = build_crew_slices(
        crew,
        touched_agent=crew_in.touched,
        agent_regression_p={k: v for k, v in p_values.items() if k not in seams},
        agent_rates={k: v for k, v in rates.items() if k not in seams},
        handoff_regression_p={k: v for k, v in p_values.items() if k in seams},
        handoff_rates={k: v for k, v in rates.items() if k in seams},
    )
    return _CrewView(crew_in, comparison, outcomes, slices, source)
