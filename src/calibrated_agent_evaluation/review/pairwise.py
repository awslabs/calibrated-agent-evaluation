# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""The review surface's A/B page: pick the better of two traces on one subjective dimension.

``build_pairwise_app`` serves one rater a queue of comparisons: every unordered pair of the traces
in scope, once per subjective dimension of the atom set, in a seeded shuffled order with the sides
assigned by the same shuffle (so neither trace sits on the left by habit). ``a`` or ``1`` picks
the left trace, ``b`` or ``2`` the right one, ``s`` skips a pair the rater cannot call. A choice
is checked with :func:`~calibrated_agent_evaluation.subjective.preference_record` and handed to the
injected ``store`` callable; the CLI passes one that appends to ``<root>/records/preference/``. A
comparison this rater already recorded is left out, so a restarted session resumes.

The app carries the same guards as the labeling page
(:mod:`calibrated_agent_evaluation.review.guards`): it answers only to its allowed Host names and
refuses any write that does not come from the page itself, and the CLI serves it on a loopback
address only.
"""

from __future__ import annotations

import itertools
import json
import logging
import random
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from importlib.resources import files  # nosemgrep: python37-compatibility-importlib2
from typing import Any, cast

from fasthtml.common import H2, Button, Div, FastHTML, Meta, NotStr, P, Script, Span, Style, Titled
from starlette.middleware import Middleware

from calibrated_agent_evaluation.backends.base import TraceStoreBackend
from calibrated_agent_evaluation.review.errors import ReviewError
from calibrated_agent_evaluation.review.guards import (
    DEFAULT_ALLOWED_HOSTS,
    SameOriginWrites,
    TrustedHosts,
    sessionless,
)
from calibrated_agent_evaluation.review.renderers import RendererRegistry, make_default_registry
from calibrated_agent_evaluation.schema import CalibratedAgentEvaluationDataError, Trace
from calibrated_agent_evaluation.subjective import (
    AtomSet,
    PreferencePair,
    PreferenceRecord,
    preference_record,
)

__all__ = ["Comparison", "build_pairwise_app", "comparisons"]

logger = logging.getLogger("calibrated_agent_evaluation.review.pairwise")

_STATIC = files("calibrated_agent_evaluation.review") / "static"
_STYLES_CSS = (_STATIC / "styles.css").read_text(encoding="utf-8")
_HTMX_JS = (_STATIC / "vendor" / "htmx.min.js").read_text(encoding="utf-8")
_PAIRWISE_JS = (_STATIC / "pairwise.js").read_text(encoding="utf-8")
_PAIRWISE_CSS = """
.hd-ab-pair { display: grid; grid-template-columns: 1fr 1fr; gap: 1rem; align-items: start; }
.hd-ab-side { min-width: 0; }
.hd-ab-label { margin: 0 0 .5rem; }
@media (max-width: 900px) { .hd-ab-pair { grid-template-columns: 1fr; } }
"""

# Fixed so the same scope always yields the same queue and sides; recorded in each source_ref.
DEFAULT_SEED = 0


@dataclass(frozen=True)
class Comparison:
    """One queued comparison: ``trace_id_a`` is shown on the left, ``trace_id_b`` on the right."""

    item_id: str
    trace_id_a: str
    trace_id_b: str
    dimension: str


def comparisons(
    trace_ids: Sequence[str],
    atom_set: AtomSet,
    *,
    done: Sequence[PreferenceRecord] = (),
    rater_id: str,
    seed: int = DEFAULT_SEED,
) -> list[Comparison]:
    """Every unordered pair of ``trace_ids`` on every subjective dimension, minus the ones done.

    A pair counts as done when ``done`` holds a record by ``rater_id`` for the same two traces (in
    either order), the same dimension and the same criterion.

    Raises:
        ReviewError: Fewer than two traces, or an atom set with no subjective dimension.
    """
    ids = sorted(dict.fromkeys(trace_ids))
    if len(ids) < 2:
        raise ReviewError(
            f"the A/B page compares two traces at a time, and the scope holds {len(ids)}; widen "
            "--frame/--cell/--limit"
        )
    dims = [d.name for d in atom_set.subjective_dimensions]
    if not dims:
        raise ReviewError(
            f"the atom set for {atom_set.criterion!r} declares no subjective dimension, so there "
            "is nothing to compare on; label its objective atoms PASS or FAIL instead"
        )
    seen = {
        (frozenset((r.trace_id_a, r.trace_id_b)), r.dimension)
        for r in done
        if r.rater_id == rater_id and r.criterion == atom_set.criterion
    }
    rng = random.Random(seed)  # noqa: S311  # ordering and side assignment, not security
    queue: list[Comparison] = []
    for dimension in dims:
        for left, right in itertools.combinations(ids, 2):
            if (frozenset((left, right)), dimension) in seen:
                continue
            a, b = (left, right) if rng.random() < 0.5 else (right, left)
            queue.append(Comparison(f"{dimension}:{a}:{b}", a, b, dimension))
    rng.shuffle(queue)
    return queue


class _Session:
    """One rater's queue and how far through it they are. Single-user, single-threaded."""

    def __init__(
        self,
        *,
        backend: TraceStoreBackend,
        atom_set: AtomSet,
        rater_id: str,
        queue: list[Comparison],
        traces: dict[str, Trace],
        registry: RendererRegistry,
        store: Callable[[Sequence[PreferenceRecord]], object],
        seed: int,
    ) -> None:
        self.backend = backend
        self.atom_set = atom_set
        self.rater_id = rater_id
        self.pending = queue
        self.traces = traces
        self.registry = registry
        self.store = store
        self.seed = seed
        self.recorded = 0
        self.skipped = 0

    def trace(self, trace_id: str) -> Trace:
        if trace_id not in self.traces:
            self.traces[trace_id] = self.backend.get_trace(trace_id)
        return self.traces[trace_id]

    def describe(self, dimension: str) -> str:
        return next(
            (d.description for d in self.atom_set.subjective_dimensions if d.name == dimension), ""
        )

    def choose(self, item_id: str, winner: str) -> str | None:
        """Record or skip the head comparison; returns a notice when nothing was recorded."""
        if not self.pending or self.pending[0].item_id != item_id:
            return (
                "That comparison is no longer the current one; the page now shows the current one."
            )
        item = self.pending[0]
        if winner == "skip":
            self.pending.pop(0)
            self.skipped += 1
            return None
        if winner not in ("a", "b"):
            return f"Pick a or b (or s to skip); got {winner!r}."
        pair = PreferencePair(
            trace_id_a=item.trace_id_a,
            trace_id_b=item.trace_id_b,
            dimension=item.dimension,
            winner=winner,  # type: ignore[arg-type]  # checked just above
            judge_kind="human",
        )
        try:
            record = preference_record(
                pair,
                self.atom_set,
                preference_id=f"pref-{item.dimension}-{item.trace_id_a}-{item.trace_id_b}-"
                f"{self.rater_id}",
                source="review",
                source_ref=f"review A/B page, seed {self.seed}",
                rater_id=self.rater_id,
            )
            self.store([record])
        except CalibratedAgentEvaluationDataError as exc:
            logger.exception("preference write failed")
            return f"Not recorded: {exc}"
        self.pending.pop(0)
        self.recorded += 1
        return None


def _side(session: _Session, label: str, key: str, trace_id: str) -> Any:
    trace = session.trace(trace_id)
    rendered, fallback = session.registry.render_with_status(trace)
    return Div(
        H2(f"{label} ({key})", cls="hd-ab-label"),
        Div(fallback, cls="hd-notice", role="status") if fallback is not None else "",
        rendered,
        cls="hd-ab-side",
    )


def _choice(item: Comparison, winner: str, text: str, cls: str) -> Any:
    return Button(
        text,
        type="button",
        cls=cls,
        hx_post="/choose",
        hx_vals=json.dumps({"item_id": item.item_id, "winner": winner}),
        hx_target="#review-root",
        hx_swap="innerHTML",
        data_choice=winner,
    )


def _current(session: _Session, notice: str | None = None) -> Any:
    parts: list[Any] = []
    if notice is not None:
        parts.append(Div(notice, cls="hd-notice", role="alert"))
    if not session.pending:
        parts.append(
            Div(
                H2("All comparisons done"),
                P(
                    f"Recorded {session.recorded}, skipped {session.skipped}. Rank them with "
                    "`calibrated-agent-evaluation prefer --rank DIMENSION`; the ratings are "
                    "tracked, never gated."
                ),
                cls="hd-complete",
            )
        )
        return Div(*parts, cls="hd-review-inner")
    item = session.pending[0]
    done = session.recorded + session.skipped
    total = done + len(session.pending)
    description = session.describe(item.dimension)
    parts += [
        Div(
            Span(f"{done + 1} of {total}", cls="hd-muted"),
            H2(f"Which is better on {item.dimension!r}?"),
            P(description, cls="hd-muted") if description else "",
            P(f"Criterion: {session.atom_set.criterion}", cls="hd-muted"),
            cls="hd-ab-head",
        ),
        Div(
            _side(session, "A", "a", item.trace_id_a),
            _side(session, "B", "b", item.trace_id_b),
            cls="hd-ab-pair",
        ),
        Div(
            _choice(item, "a", "A is better (a)", "hd-pass"),
            _choice(item, "b", "B is better (b)", "hd-pass"),
            _choice(item, "skip", "Cannot tell, skip (s)", "hd-skip"),
            cls="hd-verdict",
        ),
    ]
    return Div(*parts, cls="hd-review-inner", data_item=item.item_id)


def build_pairwise_app(
    backend: TraceStoreBackend,
    *,
    atom_set: AtomSet,
    rater_id: str,
    traces: Sequence[Trace],
    store: Callable[[Sequence[PreferenceRecord]], object],
    done: Sequence[PreferenceRecord] = (),
    registry: RendererRegistry | None = None,
    allowed_hosts: Sequence[str] = DEFAULT_ALLOWED_HOSTS,
    seed: int = DEFAULT_SEED,
) -> Any:
    """Build the A/B comparison app over ``traces``.

    ``store`` persists each accepted choice (one :class:`PreferenceRecord` per call); a failure it
    raises in the ``CalibratedAgentEvaluationDataError`` family is shown to the rater and the
    comparison stays current. ``done`` are the preferences already stored, so this rater's finished
    comparisons are left out. The returned app exposes ``.state.pairwise`` (the queue) for tests.

    Raises:
        ReviewError: An empty ``rater_id``, fewer than two traces, or no subjective dimension.
    """
    rater_id = (rater_id or "").strip()
    if not rater_id:
        raise ReviewError("the A/B page needs a non-empty rater_id naming who chooses")
    queue = comparisons(
        [t.trace_id for t in traces], atom_set, done=done, rater_id=rater_id, seed=seed
    )
    session = _Session(
        backend=backend,
        atom_set=atom_set,
        rater_id=rater_id,
        queue=queue,
        traces={t.trace_id: t for t in traces},
        registry=registry or make_default_registry(),
        store=store,
        seed=seed,
    )
    app: Any = FastHTML(
        default_hdrs=False,
        **sessionless(),
        middleware=[
            Middleware(TrustedHosts, allowed_hosts=list(allowed_hosts)),
            Middleware(SameOriginWrites),
        ],
        hdrs=(
            Meta(charset="utf-8"),
            Meta(name="viewport", content="width=device-width, initial-scale=1"),
            Style(cast(str, NotStr(_STYLES_CSS + _PAIRWISE_CSS))),
            Script(cast(str, NotStr(_HTMX_JS))),
        ),
    )

    def page(fragment: Any) -> Any:
        return Titled(
            "Calibrated Agent Evaluation A/B",
            P("Keys: a or 1 picks A, b or 2 picks B, s skips.", cls="hd-muted"),
            Div(fragment, id="review-root"),
            Script(cast(str, NotStr(_PAIRWISE_JS))),
        )

    # Async handlers run on the event-loop thread, the one the backend connection was opened on.
    @app.get("/")  # nosemgrep: useless-inner-function  # FastHTML route handler
    async def _index() -> Any:
        return page(_current(session))

    @app.post("/choose")  # nosemgrep: useless-inner-function  # FastHTML route handler
    async def _choose(item_id: str = "", winner: str = "") -> Any:
        return _current(session, session.choose(item_id, winner))

    app.state.pairwise = session
    return app
