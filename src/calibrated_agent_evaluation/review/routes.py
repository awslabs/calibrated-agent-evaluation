# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""The review surface's HTTP routes, registered on a FastHTML app against one ReviewContext.

Browsing routes move the view and render traces; verdict routes write labels and keep the store,
the session and the client in agreement. Every handler is ``async`` deliberately: Starlette runs
sync handlers in a worker threadpool, but async handlers run on the event-loop thread, the same
thread ``uvicorn.run(app)`` and the backend connection run on. Keeping all backend access on one
thread lets the single-user local app work against any backend with no cross-thread issue (SQLite
thread affinity, say). Blocking backend I/O on the loop is fine here: there is one labeler.
"""

from __future__ import annotations

import logging
from datetime import UTC, datetime
from typing import Any

from fasthtml.common import Div
from starlette.requests import Request
from starlette.responses import JSONResponse

from calibrated_agent_evaluation._provenance import capture_provenance
from calibrated_agent_evaluation.review.context import ReviewContext, parse_expected, with_status
from calibrated_agent_evaluation.review.errors import ReviewError
from calibrated_agent_evaluation.review.labeling import build_label
from calibrated_agent_evaluation.review.record import ProposedModeRecord, mode_key, with_content_id
from calibrated_agent_evaluation.review.renderers.trace_view import (
    completion_view,
    history_view,
    stored_labels_view,
)
from calibrated_agent_evaluation.schema import CalibratedAgentEvaluationDataError, Label

__all__ = ["register_browse_routes", "register_verdict_routes"]

logger = logging.getLogger("calibrated_agent_evaluation.review.app")


def register_browse_routes(app: Any, ctx: ReviewContext) -> None:
    """Register the routes that show traces and move the view, and write no label.

    Args:
        app: The FastHTML app.
        ctx: The session state the routes read.
    """

    @app.get("/")  # nosemgrep: useless-inner-function  # FastHTML route handler
    async def _index() -> Any:
        return ctx.page(ctx.current())

    @app.get("/trace/{trace_id}")  # nosemgrep: useless-inner-function  # FastHTML route handler
    async def _trace_route(request: Request, trace_id: str) -> Any:  # view a specific trace
        try:
            if trace_id in ctx.session.pending:
                return ctx.fragment_or_page(request, ctx.trace_fragment(trace_id))
            # Not pending: a trace labeled earlier, shown read-only with what the store holds.
            trace = ctx.trace(trace_id)
            rendered, fallback = ctx.registry.render_with_status(trace)
            fragment = Div(
                Div(fallback, cls="hd-notice", role="status") if fallback is not None else "",
                rendered,
                stored_labels_view(ctx.backend.get_scores(trace_id)),
                cls="hd-review-inner",
            )
            return ctx.fragment_or_page(request, fragment)
        except CalibratedAgentEvaluationDataError as exc:
            notice = Div(f"No such trace {trace_id!r}: {exc}", cls="hd-notice", role="alert")
            return ctx.fragment_or_page(request, notice)

    @app.get("/history")  # nosemgrep: useless-inner-function  # FastHTML route handler
    async def _history(request: Request) -> Any:  # read-only list of this session's verdicts
        return ctx.fragment_or_page(request, history_view(ctx.session.done))

    @app.get("/next")  # nosemgrep: useless-inner-function  # FastHTML route handler
    async def _next(after: str = "") -> Any:  # pure prefetch of the next trace; no write
        # Keyed on the trace_id the client is currently showing (`after`), NOT the session head:
        # an optimistic client advances its view before the label write's `record()` moves the
        # server head, so a head-relative "next" would race and re-serve the just-shown trace.
        # Resolving "the item after `after` in the work-list" is stable across that lag.
        pending = ctx.session.pending
        if after and after in pending:
            idx = pending.index(after)
            if idx + 1 < len(pending):
                return ctx.trace_fragment(pending[idx + 1], after=after)
            # `after` is the last item: labeling it leaves the view cursor clamped onto the item
            # before it, so prefetch that one. "All labeled" only when `after` is the only item.
            if idx > 0:
                return ctx.trace_fragment(pending[idx - 1], after=after)
            return completion_view(ctx.session.summary())
        # Fallback (no/unknown `after`, e.g. the very first prefetch): the item after the head.
        if len(pending) >= 2:
            return ctx.trace_fragment(pending[1], after=pending[0])
        return completion_view(ctx.session.summary())

    @app.get("/view")  # nosemgrep: useless-inner-function  # FastHTML route handler
    async def _view() -> Any:  # the current view, unmoved: the client's resync after a failure
        return ctx.current()

    # POST only: moving the view is a write, so the same-origin guard covers it.
    @app.post("/nav")  # nosemgrep: useless-inner-function  # FastHTML route handler
    async def _nav(to: str = "next") -> Any:  # move the view without recording a verdict
        if ctx.session.pending:
            step = 1 if to == "next" else -1
            ctx.view_index = max(0, min(ctx.view_index + step, len(ctx.session.pending) - 1))
        return ctx.current()

    @app.get("/summary")  # nosemgrep: useless-inner-function  # FastHTML route handler
    async def _summary() -> Any:  # session throughput summary
        return JSONResponse(ctx.session.summary())


def register_verdict_routes(app: Any, ctx: ReviewContext) -> None:
    """Register the routes that write: label, defer and undo.

    Every response carries an ``HD-Label-Status`` header so the optimistic client
    (``static/hotkeys.js``) knows whether it must reconcile the view it already advanced:
    ``recorded`` keeps the advance; anything else means the write did not land as shown.

    Args:
        app: The FastHTML app.
        ctx: The session state the routes read and update.
    """

    @app.post("/label")  # nosemgrep: useless-inner-function  # FastHTML route handler
    async def _label(
        request: Request,
        trace_id: str,
        outcome: str = "",
        critique: str = "",
        failure_mode: str = "",
        culprit_span: str = "",
        manifestation_span: str = "",
        handoff_implicated: str = "",
        defect_span: str = "",
        expected: str = "",
        ratified: str = "",
        new_mode_name: str = "",
        new_mode_note: str = "",
    ) -> Any:
        submitted = {
            "trace_id": trace_id,
            "critique": critique,
            "failure_mode": failure_mode,
            "culprit_span": culprit_span,
            "manifestation_span": manifestation_span,
            "handoff_implicated": handoff_implicated,
            "defect_span": defect_span,
            "expected": expected,
            "new_mode_name": new_mode_name,
            "new_mode_note": new_mode_note,
        }
        # Guard a double-submit BEFORE writing: a trace no longer pending must NOT be written again
        # (no duplicate label in the store) and must not 500.
        if trace_id not in ctx.session.pending:
            logger.warning(
                "label for %r ignored; not in the pending work-list (double-submit)", trace_id
            )
            return with_status(
                ctx.current(notice="Not advanced: already labeled or advanced past."), "noop"
            )
        fields = {
            "trace_id": trace_id,
            "outcome": outcome,
            "critique": critique,
            "failure_mode": failure_mode,
            "culprit_span": culprit_span,
            "manifestation_span": manifestation_span,
            "handoff_implicated": handoff_implicated,
            "defect_span": defect_span,
            "expected": expected,
            "ratified": ratified,
        }
        proposal: ProposedModeRecord | None = None
        notice: str | None = None
        try:
            if new_mode_name.strip():
                mode, proposal, notice = _new_mode(ctx, fields, new_mode_name, new_mode_note)
                fields["failure_mode"] = mode
            extra = (proposal.name,) if proposal is not None else ()
            label, defect = _build_labels(ctx, extra_modes=extra, **fields)
        except ReviewError as exc:  # empty critique or mode on a FAIL / bad outcome, expected, span
            return ctx.back_to(trace_id, f"Not recorded: {exc}", submitted, "refused")
        except (
            CalibratedAgentEvaluationDataError
        ) as exc:  # the trace could not be read back to check span ids
            logger.exception("label submit could not read trace %s", trace_id)
            return ctx.back_to(trace_id, f"Not recorded: {exc}", submitted, "error")

        if proposal is not None:
            stored = _store_proposal(ctx, proposal)
            if isinstance(stored, str):
                return ctx.back_to(trace_id, f"Not recorded: {stored}", submitted, "error")
            if stored.name != proposal.name:  # another session proposed it first
                fields["failure_mode"] = stored.name
                label, defect = _build_labels(ctx, extra_modes=(stored.name,), **fields)
                notice = (
                    f"{stored.name!r} was proposed in another session a moment ago; the label "
                    "uses it."
                )
            else:
                notice = (
                    f"Proposed failure mode {stored.name!r} recorded for the codebook owner; it "
                    "is in the picker now."
                )

        failed = _write_labels(ctx, label, defect, submitted)
        if failed is not None:
            return failed

        # The next trace is the one after this in the work-list: what `/next?after=` served the
        # optimistic client, and so what its view now shows. It slides into this trace's index.
        position = ctx.session.pending.index(trace_id)
        entry = ctx.session.record(label)  # safe: guarded above that trace_id is pending
        if defect is not None:
            entry = ctx.session.attach_span_label(defect)  # so undo retracts it with the verdict
        ctx.view_index = position
        log = ctx.log
        log_failure = ctx.log_write(
            lambda: log.label(entry, rater_id=ctx.rater_id) if log else None
        )
        if log_failure is not None:
            # The verdict is stored; only its timing record is missing. Say so on screen.
            return with_status(ctx.current(notice=f"Verdict recorded; {log_failure}"), "error")
        if notice is not None:
            return with_status(ctx.current(notice=notice), "recorded")
        if request.headers.get("HD-Optimistic") == "1":
            # The optimistic client already shows the next trace and discards this body, so do not
            # render one. Its timer still starts now, which is when the client swapped it in.
            nxt = ctx.shown_id()
            if nxt is not None:
                ctx.session.mark_shown(nxt)
            return with_status("", "recorded")
        return with_status(ctx.current(), "recorded")

    @app.post("/defer")  # nosemgrep: useless-inner-function  # FastHTML route handler
    async def _defer(trace_id: str = "") -> Any:  # defer the trace on screen
        # Default to the trace shown, not the queue head: after navigation they differ, and
        # deferring the head moved a trace the labeler was not looking at. The view index stays,
        # so the trace that slides into place is shown next.
        ctx.session.defer(trace_id or ctx.shown_id())
        return ctx.current()

    @app.post("/undo")  # nosemgrep: useless-inner-function  # FastHTML route handler
    async def _undo() -> Any:  # retract the last label, from the store first
        return _undo_last(ctx)


def _new_mode(
    ctx: ReviewContext, fields: dict[str, str], name: str, note: str
) -> tuple[str, ProposedModeRecord | None, str | None]:
    """The mode a "new mode" submission labels with, and the proposal to store for it.

    A name that matches a listed or proposed mode, ignoring case and spacing, is that mode: the
    label uses it and nothing is proposed, so two labelers never create the same mode twice.

    Raises:
        ReviewError: The session cannot take proposals, the verdict is not a FAIL, a listed mode
            was picked as well, or a genuinely new mode has no note.
    """
    if not ctx.can_propose or ctx.codebook_version is None:
        raise ReviewError(
            "this session has no codebook version and failure-mode list to propose a mode to; "
            "label with --codebook-version V (or --codebook FILE) to propose modes"
        )
    if fields["outcome"] != "FAIL":
        raise ReviewError("only a FAIL names a failure mode, so only a FAIL can propose one")
    name = " ".join(name.split())
    picked = fields["failure_mode"]
    if picked and mode_key(picked) != mode_key(name):
        raise ReviewError(
            f"pick the listed mode {picked!r} or propose {name!r}, not both; clear one of them"
        )
    for mode in (*ctx.failure_modes, *ctx.proposed_modes()):
        if mode_key(mode.name) == mode_key(name):
            return (
                mode.name,
                None,
                f"{name!r} is already in the picker as {mode.name!r}; the label uses it.",
            )
    if not note.strip():
        raise ReviewError(
            f"a new failure mode needs a note saying what {name!r} covers, so the codebook owner "
            "can decide whether to adopt it"
        )
    proposal = with_content_id(
        ProposedModeRecord(
            proposal_id="",
            codebook_version=ctx.codebook_version,
            name=name,
            note=note.strip(),
            proposed_by=ctx.rater_id,
            trace_id=fields["trace_id"],
            proposed_at=datetime.now(tz=UTC).isoformat(timespec="microseconds"),
            provenance=capture_provenance(),
        ),
        "proposal_id",
        "proposal",
    )
    return name, proposal, None


def _store_proposal(ctx: ReviewContext, proposal: ProposedModeRecord) -> ProposedModeRecord | str:
    """The stored proposal (this one, or a same-named one already pending), or why it failed."""
    if ctx.proposals is None:  # _new_mode refuses a session without a store first
        return "this session has no store for proposed failure modes"
    try:
        return ctx.proposals.propose(proposal)
    except ReviewError as exc:
        logger.exception("storing the proposed mode %r failed", proposal.name)
        return f"the proposed failure mode was not stored, so the label was not either: {exc}"


def _build_labels(
    ctx: ReviewContext,
    *,
    extra_modes: tuple[str, ...] = (),
    trace_id: str,
    outcome: str,
    critique: str,
    failure_mode: str,
    culprit_span: str,
    manifestation_span: str,
    handoff_implicated: str,
    defect_span: str,
    expected: str,
    ratified: str,
) -> tuple[Label, Label | None]:
    """The verdict label and, for a FAIL naming a defect span, its span-defect label."""
    trace = ctx.trace(trace_id)
    # The submitted span ids must name spans of this trace: a typo would otherwise become a
    # phantom per-agent failure mode in the CI gate.
    span_ids = frozenset(s.span_id for s in trace.spans)
    known = (*ctx.failure_modes, *ctx.proposed_modes())
    mode_names = frozenset(m.name for m in known) | frozenset(extra_modes)
    # Gold stays human. Whether the form was prefilled is the server's own record of what it
    # suggested, not a client field. Ratification is an explicit accept (`ratified=1`, sent only
    # by the Accept action) or an override visible here: another verdict or an edited critique. A
    # verdict key on an unedited suggestion is neither, and build_label refuses it.
    shown = ctx.suggest(trace)
    overridden = shown is not None and (
        outcome != shown.outcome or critique.strip() != shown.critique.strip()
    )
    # `expected` is parsed as JSON here so a malformed value is a typed refusal, never a silent
    # drop. It never substitutes for the critique; the backend validate re-checks.
    label = build_label(
        trace_id=trace_id,
        outcome=outcome,
        critique=critique,
        failure_mode=failure_mode or None,
        culprit_span=culprit_span or None,
        manifestation_span=manifestation_span or None,
        handoff_implicated=handoff_implicated or None,
        expected=parse_expected(expected),
        from_suggestion=shown is not None,
        ratified=ratified == "1" or overridden,
        rater_id=ctx.rater_id,
        span_ids=span_ids,
        codebook_version=ctx.codebook_version,
        failure_modes=mode_names,
    )
    if outcome != "FAIL" or not defect_span:
        return label, None
    defect = build_label(
        trace_id=trace_id,
        span_id=defect_span,
        outcome="FAIL",
        critique=critique,
        failure_mode=failure_mode or None,
        culprit_span=defect_span,
        span_defect=True,
        rater_id=ctx.rater_id,
        span_ids=span_ids,
        codebook_version=ctx.codebook_version,
        failure_modes=mode_names,
    )
    return label, defect


def _write_labels(
    ctx: ReviewContext, label: Label, defect: Label | None, submitted: dict[str, str]
) -> Any | None:
    """Store the verdict and its defect label together, or neither; a response if that failed."""
    trace_id = label.trace_id
    try:
        ctx.backend.put_score(label)
    except CalibratedAgentEvaluationDataError as exc:  # surface, leave the trace unlabeled
        logger.exception("label submit failed for %s", trace_id)
        return ctx.back_to(
            trace_id, f"Submit failed, trace left unlabeled: {exc}", submitted, "error"
        )
    if defect is None:
        return None
    try:
        ctx.backend.put_score(defect)
    except CalibratedAgentEvaluationDataError as exc:
        logger.exception("span-defect submit failed for %s/%s", trace_id, defect.span_id)
        try:
            ctx.backend.retract_score(label)
        except CalibratedAgentEvaluationDataError as rollback_exc:
            logger.exception("rolling back the verdict for %s failed", trace_id)
            # The verdict is stored without its defect label: record it so the session tells the
            # truth, and say exactly what is missing.
            ctx.session.record(label)
            return with_status(
                ctx.current(
                    notice=(
                        f"Verdict recorded but its span-defect label was not ({exc}), "
                        f"and rolling the verdict back failed ({rollback_exc}). "
                        "Undo it and resubmit."
                    )
                ),
                "error",
            )
        return ctx.back_to(
            trace_id,
            f"Not recorded: the span-defect write failed, so the verdict was rolled back: {exc}",
            submitted,
            "error",
        )
    return None


def _undo_last(ctx: ReviewContext) -> Any:
    """Retract the last verdict and its span labels from the store, then move the session.

    The store is the record every gate reads, so it changes first and as one unit: the span labels
    go before the verdict, and a failure part way puts back what was already retracted. Only then
    does the session move and the timing log get its line, so a log failure can never leave a span
    label behind without its verdict.
    """
    entry = ctx.session.last()
    if entry is None:  # nothing labeled yet: a benign no-op
        return with_status(ctx.current(), "noop")
    retracted: list[Label] = []
    for label in (*entry.span_labels, entry.label):
        try:
            ctx.backend.retract_score(label)
        except CalibratedAgentEvaluationDataError as exc:
            logger.exception("undo failed to retract a label for %s", entry.trace_id)
            restore_failure = _restore(ctx, retracted)
            notice = f"Undo failed: the verdict is still stored: {exc}"
            if restore_failure is not None:
                notice += f"; putting back its span labels also failed: {restore_failure}"
            return with_status(ctx.current(notice=notice), "error")
        retracted.append(label)
    ctx.session.undo()
    ctx.view_index = 0
    log = ctx.log
    log_failure = ctx.log_write(lambda: log.undo(entry, rater_id=ctx.rater_id) if log else None)
    if log_failure is not None:
        return with_status(ctx.current(notice=f"Verdict retracted; {log_failure}"), "error")
    return with_status(ctx.current(), "recorded")


def _restore(ctx: ReviewContext, labels: list[Label]) -> CalibratedAgentEvaluationDataError | None:
    """Put back labels an undo had already retracted; the first failure, or ``None``."""
    for label in labels:
        try:
            ctx.backend.put_score(label)
        except CalibratedAgentEvaluationDataError as exc:
            logger.exception("undo could not put back %s/%s", label.trace_id, label.span_id)
            return exc
    return None
