# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""In-memory review session state.

Holds the labeling work-list and the flow operations the hotkeys drive: defer (re-queue to
the end), undo (retract the last label and return to it), per-trace timing +
a throughput summary, and the completion state when the queue empties.

State is per-process, single-user (the localhost-only v1 scope). Timing uses an injected
``clock`` so tests are deterministic (no wall-clock in unit tests); the real app
passes ``time.monotonic``.
"""

from __future__ import annotations

import time
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass, field

from calibrated_agent_evaluation._model import replace
from calibrated_agent_evaluation.backends.base import SkippedTrace, TraceStoreBackend, scan_traces
from calibrated_agent_evaluation.review.errors import ReviewError
from calibrated_agent_evaluation.schema import Frame, Label, Trace


@dataclass(frozen=True)
class LabeledEntry:
    """One completed submission: kept for undo and the throughput summary.

    ``span_labels`` holds the span-level defect labels written with the verdict, so undo can
    retract everything the submission put in the store, not only the trace-level label.
    """

    trace_id: str
    label: Label
    seconds_on_trace: float
    span_labels: tuple[Label, ...] = ()


@dataclass
class ReviewSession:
    """The work-list for one labeling session.

    ``pending`` is the ordered queue of ``trace_id``s awaiting a verdict; ``done`` is the stack
    of completed entries (undo pops the last). Construct with the ids returned by
    ``LabelBackend.next_item`` / ``TraceStoreBackend.query`` in a deterministic order.
    """

    pending: list[str]
    done: list[LabeledEntry] = field(default_factory=list)
    clock: Callable[[], float] = time.monotonic
    _shown_id: str | None = field(default=None, repr=False)
    _shown_at: float | None = field(default=None, repr=False)
    # Time spent on traces that left the screen without a verdict that still stands: deferred,
    # navigated past, or labeled then undone. The session total counts it; dropping it understated
    # the session.
    _unlabeled_seconds: float = field(default=0.0, repr=False)

    @classmethod
    def from_ids(
        cls, ids: Iterable[str], *, clock: Callable[[], float] = time.monotonic
    ) -> ReviewSession:
        """Build a session from an ordered iterable of ``trace_id``s (dedup, order-preserving)."""
        seen: dict[str, None] = {}
        for tid in ids:
            if tid:
                seen.setdefault(tid, None)
        return cls(pending=list(seen), clock=clock)

    @property
    def is_complete(self) -> bool:
        """True when no trace remains to label (show the completion state)."""
        return not self.pending

    def current(self) -> str | None:
        """The trace at the head of the work-list, or ``None`` when the queue is empty."""
        return self.pending[0] if self.pending else None

    def mark_shown(self, trace_id: str | None = None) -> str | None:
        """Start timing ``trace_id`` (default: the queue head) as the trace on screen.

        Called whenever a trace is rendered, with the id actually rendered: after navigation that
        is not the queue head, and timing the head instead charged the interval to the wrong
        trace. Time on the trace previously shown is banked. Re-rendering the trace already being
        timed (a refused submit) keeps its clock running.
        """
        target = trace_id if trace_id is not None else self.current()
        if target is not None and target == self._shown_id and self._shown_at is not None:
            return target
        self._stop_timer()
        if target is not None:
            self._shown_id = target
            self._shown_at = self.clock()
        return target

    def _elapsed(self) -> float:
        if self._shown_at is None:
            return 0.0
        return max(0.0, self.clock() - self._shown_at)

    def _stop_timer(self) -> float:
        """Stop the running timer, banking its time as unlabeled; return the seconds it ran."""
        spent = self._elapsed()
        self._unlabeled_seconds += spent
        self._shown_id = None
        self._shown_at = None
        return spent

    def record(self, label: Label) -> LabeledEntry:
        """Record ``label`` against its trace, remove it from the queue, stop its timer.

        Raises ``ReviewError`` if the label's trace is not in the pending work-list; a label for
        a trace the session isn't tracking is a caller bug, not something to swallow.
        """
        tid = label.trace_id
        if tid not in self.pending:
            raise ReviewError(f"{tid!r} is not in the pending work-list; cannot record its label.")
        # The interval belongs to this label only if this trace is the one being timed; otherwise
        # the running time is banked as unlabeled and the label carries none.
        seconds = self._elapsed() if self._shown_id == tid else 0.0
        if self._shown_id == tid:
            self._shown_id = None
            self._shown_at = None
        else:
            self._stop_timer()
        entry = LabeledEntry(trace_id=tid, label=label, seconds_on_trace=seconds)
        self.pending.remove(tid)
        self.done.append(entry)
        return entry

    def last(self) -> LabeledEntry | None:
        """The most recently recorded entry (what ``undo`` would retract), or ``None``."""
        return self.done[-1] if self.done else None

    def attach_span_label(self, label: Label) -> LabeledEntry:
        """Record a span-level label written alongside the most recent verdict.

        Raises ``ReviewError`` if nothing has been recorded or the label is for another trace.
        """
        entry = self.last()
        if entry is None or entry.trace_id != label.trace_id:
            raise ReviewError(
                f"no recorded verdict for {label.trace_id!r} to attach a span-level label to."
            )
        amended = replace(entry, span_labels=(*entry.span_labels, label))
        self.done[-1] = amended
        return amended

    def defer(self, trace_id: str | None = None) -> str | None:
        """Move ``trace_id`` (default: the current trace) to the end of the work-list.

        Returns the new current trace. A no-op returning ``None`` when the queue is empty.
        """
        target = trace_id if trace_id is not None else self.current()
        if target is None or target not in self.pending:
            return self.current()
        self.pending.remove(target)
        self.pending.append(target)
        self._stop_timer()
        return self.current()

    def undo(self) -> LabeledEntry | None:
        """Retract the most recently recorded label and re-queue its trace at the front.

        Returns the retracted entry, or ``None`` when nothing has been labeled yet (empty-stack
        no-op; undoing an empty history is a benign no-op, not an error).
        """
        if not self.done:
            return None
        entry = self.done.pop()
        self.pending.insert(0, entry.trace_id)
        self._stop_timer()
        self._unlabeled_seconds += entry.seconds_on_trace  # the time was spent; the verdict is gone
        return entry

    def summary(self) -> dict[str, float | int]:
        """Session throughput.

        ``labeled_seconds`` is the time on traces whose verdict stands; ``elapsed_seconds`` is all
        on-trace time, including deferred, navigated-past, undone, and the trace on screen now.
        """
        labeled = sum(e.seconds_on_trace for e in self.done)
        return {
            "labeled_count": len(self.done),
            "labeled_seconds": labeled,
            "elapsed_seconds": labeled + self._unlabeled_seconds + self._elapsed(),
        }


@dataclass(frozen=True)
class WorkScope:
    """Which traces a review session covers (``label --trace/--frame/--cell/--limit``).

    ``frame`` and ``cell`` narrow the backend query; ``trace_id`` names one trace; ``limit`` caps
    the work-list after resume filtering, so ``limit=50`` is 50 traces still to label. By default
    a trace this rater already labeled at trace level is left out (resume); ``relabel`` keeps it.
    """

    frame: Frame | None = None
    cell: Mapping[str, str] | None = None
    trace_id: str | None = None
    limit: int | None = None
    relabel: bool = False


@dataclass(frozen=True)
class WorkList:
    """The traces a session will show, plus the ids skipped because this rater already did them.

    ``not_calibrated_agent_evaluation`` holds the traces of a shared store that Calibrated Agent
    Evaluation did not write, which the session cannot show; the caller reports them rather than
    losing them unseen.
    """

    traces: list[Trace]
    already_labeled: list[str]
    not_calibrated_agent_evaluation: tuple[SkippedTrace, ...] = ()


def _matches(trace: Trace, scope: WorkScope) -> bool:
    if scope.frame is not None and trace.frame != scope.frame:
        return False
    return all(trace.taxonomy_cell.get(k) == v for k, v in (scope.cell or {}).items())


def _labeled_by(backend: TraceStoreBackend, trace_id: str, rater_id: str) -> bool:
    # Only this rater's own trace-level human verdict counts. Another rater's label on the same
    # trace is exactly what a dual-coding session is for, so it must not hide the trace.
    return any(
        lbl.kind == "human" and lbl.span_id is None and lbl.rater_id == rater_id
        for lbl in backend.get_scores(trace_id)
    )


def build_work_list(backend: TraceStoreBackend, scope: WorkScope, *, rater_id: str) -> WorkList:
    """Resolve ``scope`` against the backend into the traces to review, in stable order.

    This is the startup reachability probe too: an unreachable backend raises its typed
    ``BackendError`` here, before any session opens. Raises ``BackendError`` for a
    ``trace_id`` the backend does not hold, and ``ReviewError`` for a bad ``limit`` or a
    narrowed scope (frame, cell or trace) that matches no trace (it would open onto "all done").
    """
    if scope.limit is not None and scope.limit < 1:
        raise ReviewError(f"limit must be a positive number of traces; got {scope.limit}.")
    skipped: tuple[SkippedTrace, ...] = ()
    if scope.trace_id is not None:
        candidates = [backend.get_trace(scope.trace_id)]
    else:
        scan = scan_traces(
            backend, frame=scope.frame, cell=dict(scope.cell) if scope.cell else None
        )
        candidates, skipped = scan.traces, scan.skipped
    candidates = [t for t in candidates if _matches(t, scope)]
    narrowed = scope.frame is not None or bool(scope.cell) or scope.trace_id is not None
    if narrowed and not candidates:
        # A typo in --frame/--cell would otherwise open straight onto the completion screen.
        raise ReviewError(f"no trace in the backend matches the requested scope {scope}.")

    traces: list[Trace] = []
    already: list[str] = []
    for trace in candidates:
        if scope.limit is not None and len(traces) >= scope.limit:
            break  # stop before reading labels for traces the session will never show
        if not scope.relabel and _labeled_by(backend, trace.trace_id, rater_id):
            already.append(trace.trace_id)
            continue
        traces.append(trace)
    return WorkList(traces=traces, already_labeled=already, not_calibrated_agent_evaluation=skipped)
