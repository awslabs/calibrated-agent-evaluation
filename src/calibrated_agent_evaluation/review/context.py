# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""The state one review session's routes share, and the fragments they render from it.

:func:`calibrated_agent_evaluation.review.app.build_app` builds one :class:`ReviewContext` and hands
it to the route modules (:mod:`calibrated_agent_evaluation.review.routes`), so every handler reads
the same backend, work-list, caches and view cursor without closing over ``build_app``'s locals.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Callable
from importlib.resources import files  # nosemgrep: python37-compatibility-importlib2
from typing import Any, cast

from fasthtml.common import Div, HttpHeader, NotStr, Script, Titled
from starlette.requests import Request

from calibrated_agent_evaluation.backends.base import TraceStoreBackend
from calibrated_agent_evaluation.review.errors import ReviewError
from calibrated_agent_evaluation.review.prefill import Prefill, Suggestion
from calibrated_agent_evaluation.review.proposals import ProposalStore
from calibrated_agent_evaluation.review.record import CodebookMode, mode_key
from calibrated_agent_evaluation.review.renderers import RendererRegistry
from calibrated_agent_evaluation.review.renderers.trace_view import (
    completion_view,
    hotkey_legend,
    labeling_form,
    progress_bar,
)
from calibrated_agent_evaluation.review.session import ReviewSession
from calibrated_agent_evaluation.review.session_log import SessionLog
from calibrated_agent_evaluation.schema import Trace

__all__ = ["ReviewContext", "parse_expected", "with_status"]

logger = logging.getLogger("calibrated_agent_evaluation.review.app")

_HOTKEYS_JS = (files("calibrated_agent_evaluation.review") / "static" / "hotkeys.js").read_text(
    encoding="utf-8"
)


def parse_expected(raw: str) -> dict[str, Any] | None:
    """Parse the labeler's optional objective "correct output", a JSON object string.

    Args:
        raw: The form field as submitted.

    Returns:
        The parsed object, or ``None`` when the field is empty (a subjective label or a PASS).

    Raises:
        ReviewError: The value is malformed JSON or not an object; it is never silently dropped.
    """
    raw = (raw or "").strip()
    if not raw:
        return None
    try:
        value = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ReviewError(
            f"'expected' must be a JSON object (the correct output); could not parse it: {exc}"
        ) from exc
    if not isinstance(value, dict):
        raise ReviewError('\'expected\' must be a JSON object (e.g. {"action": "REFUND_DENIED"}).')
    return value


class ReviewContext:
    """One labeler's session: the store, the work-list, the caches and the view cursor.

    Single-user and single-threaded: the routes are ``async`` so they all run on the event-loop
    thread, the one the backend connection was opened on.
    """

    def __init__(
        self,
        *,
        backend: TraceStoreBackend,
        session: ReviewSession,
        registry: RendererRegistry,
        prefill: Prefill,
        rater_id: str,
        log: SessionLog | None,
        traces: list[Trace],
        codebook_version: str | None = None,
        failure_modes: tuple[CodebookMode, ...] = (),
        proposals: ProposalStore | None = None,
    ) -> None:
        self.backend = backend
        self.codebook_version = codebook_version
        self.failure_modes = failure_modes
        self.proposals = proposals
        self.session = session
        self.registry = registry
        self.prefill = prefill
        self.rater_id = rater_id
        self.log = log
        # Traces are read once per session. The startup probe already holds every trace it
        # resolved (on Langfuse each one cost a `trace.get`), and the hot path used to re-read the
        # same trace for the `/label` span check, the response fragment and the `/next` prefetch.
        self._traces: dict[str, Trace] = {t.trace_id: t for t in traces}
        # Suggestions are cached too: a real Prefill is an LLM call, and the ratification check on
        # submit must compare against the suggestion that was shown, not a fresh, different one.
        self._suggestions: dict[str, Suggestion | None] = {}
        self.view_index = 0  # transient view cursor for nav

    @property
    def can_propose(self) -> bool:
        """Whether the form offers a new mode: a listed codebook, a version and a store."""
        return (
            bool(self.failure_modes)
            and self.codebook_version is not None
            and (self.proposals is not None)
        )

    def proposed_modes(self) -> tuple[CodebookMode, ...]:
        """The pending proposals under this version that no listed mode already names.

        Read from the store on every call, so another session's proposal shows up here.
        """
        if not self.failure_modes or self.proposals is None or self.codebook_version is None:
            return ()
        listed = {mode_key(m.name) for m in self.failure_modes}
        return tuple(
            CodebookMode(name=p.name, description=f"proposed by {p.proposed_by}: {p.note}")
            for p in self.proposals.pending(self.codebook_version)
            if mode_key(p.name) not in listed
        )

    def _form(self, trace: Trace, submitted: dict[str, str] | None = None) -> Any:
        warning: Any = ""
        try:
            proposed = self.proposed_modes()
        except ReviewError as exc:
            logger.exception("reading the proposed failure modes failed")
            proposed = ()
            warning = Div(
                f"Proposed failure modes could not be read, so the picker shows the codebook's "
                f"only: {exc}",
                cls="hd-notice",
                role="alert",
            )
        form = labeling_form(
            trace,
            self.suggest(trace),
            submitted=submitted,
            failure_modes=self.failure_modes,
            proposed_modes=proposed,
            can_propose=self.can_propose,
        )
        return Div(warning, form) if warning else form

    def trace(self, trace_id: str) -> Trace:
        """The trace, read from the store once and cached for the session."""
        if trace_id not in self._traces:
            self._traces[trace_id] = self.backend.get_trace(trace_id)
        return self._traces[trace_id]

    def suggest(self, trace: Trace) -> Suggestion | None:
        """The prefill suggestion shown for ``trace``, computed once per session."""
        if trace.trace_id not in self._suggestions:
            self._suggestions[trace.trace_id] = self.prefill.suggest(trace)
        return self._suggestions[trace.trace_id]

    def shown_id(self) -> str | None:
        """The pending trace the view cursor points at, or ``None`` when the queue is empty."""
        if not self.session.pending:
            return None
        self.view_index = max(0, min(self.view_index, len(self.session.pending) - 1))
        return self.session.pending[self.view_index]

    def log_write(self, write: Callable[[], None]) -> str | None:
        """Run a session-log write; a failure is reported as a notice, never swallowed."""
        try:
            write()
        except ReviewError as exc:
            logger.exception("session log write failed")
            return f"the session timing log was not written: {exc}"
        return None

    def progress(self, trace_id: str, *, after: str | None = None) -> Any:
        """The progress bar for ``trace_id``, or nothing for a trace outside the queue."""
        session = self.session
        if trace_id not in session.pending:
            return ""
        # Position counts verdicts already standing plus this trace's place in the queue, so it
        # is the same whether the trace is rendered now or prefetched before `after` is labeled.
        ahead = session.pending.index(trace_id)
        if after is not None and after in session.pending[:ahead]:
            ahead -= 1  # `after` is about to be labeled; it will no longer be ahead in the queue
        position = len(session.done) + (1 if after is not None else 0) + ahead + 1
        total = len(session.done) + len(session.pending)
        return progress_bar(position, total, float(session.summary()["elapsed_seconds"]))

    def trace_fragment(self, trace_id: str, *, after: str | None = None) -> Any:
        """A pending trace with its labeling form, as the prefetch and direct views render it."""
        trace = self.trace(trace_id)
        rendered, fallback = self.registry.render_with_status(trace)
        return Div(
            self.progress(trace_id, after=after),
            Div(fallback, cls="hd-notice", role="status") if fallback is not None else "",
            rendered,
            self._form(trace),
            cls="hd-review-inner",
        )

    def current(self, notice: str | None = None, submitted: dict[str, str] | None = None) -> Any:
        """The trace under the view cursor, or the completion view once the queue is empty."""
        cur = self.shown_id()
        if cur is None:
            return completion_view(self.session.summary())
        trace = self.trace(cur)
        self.session.mark_shown(cur)  # time the trace actually rendered, not the queue head
        parts: list[Any] = [self.progress(cur)]
        if notice is not None:
            parts.append(Div(notice, cls="hd-notice", role="alert"))
        rendered, fallback = self.registry.render_with_status(trace)  # guards a raising renderer
        if fallback is not None:
            parts.append(Div(fallback, cls="hd-notice", role="status"))
        parts.append(rendered)
        # Re-seed the form only on the trace the submission was for.
        keep = submitted if submitted is not None and submitted.get("trace_id") == cur else None
        parts.append(self._form(trace, keep))
        return Div(*parts, cls="hd-review-inner")

    def back_to(self, trace_id: str, notice: str, submitted: dict[str, str], status: str) -> Any:
        """Return the labeler to ``trace_id`` with what they typed, after a write that failed."""
        if trace_id in self.session.pending:
            self.view_index = self.session.pending.index(trace_id)
        return with_status(self.current(notice=notice, submitted=submitted), status)

    @staticmethod
    def page(fragment: Any) -> Any:
        """A full page around a fragment, for a route visited directly rather than swapped in."""
        return Titled(
            "Calibrated Agent Evaluation Review",
            hotkey_legend(),  # outside the swapped root, so an open legend stays open
            Div(fragment, id="review-root"),
            Script(cast(str, NotStr(_HOTKEYS_JS))),  # NotStr is a str at runtime; JS unescaped
        )

    def fragment_or_page(self, request: Request, fragment: Any) -> Any:
        """The bare fragment for an htmx swap, the full page for a direct visit."""
        return fragment if request.headers.get("HX-Request") else self.page(fragment)


def with_status(fragment: Any, status: str) -> Any:
    """Attach the ``HD-Label-Status`` header the optimistic client reads to decide to reconcile."""
    return fragment, HttpHeader("HD-Label-Status", status)
