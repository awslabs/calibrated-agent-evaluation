# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""FastHTML application factory for the review surface.

Wires the backends, the session, the renderer registry, and the labeling logic into a
local, single-user web app. Reads traces and writes labels **only** through the backend
interfaces; no vendor SDK type appears here. ``fasthtml``/``starlette``
are the view framework, not data-plane vendors.

Startup does a synchronous reachability probe via ``TraceStoreBackend.query`` (covered by
the ``BackendError`` contract): an unreachable backend raises before any session opens.
The verdict flow is optimistic on the client (``static/hotkeys.js``) and reconciled on
the server; a failed label write re-queues the trace as unlabeled rather than presenting it as
done.
"""

from __future__ import annotations

from collections.abc import Sequence
from importlib.resources import files  # nosemgrep: python37-compatibility-importlib2
from pathlib import Path
from typing import Any, cast

from fasthtml.common import FastHTML, Meta, NotStr, Script, Style
from starlette.middleware import Middleware

from calibrated_agent_evaluation.backends.base import TraceStoreBackend, scan_traces
from calibrated_agent_evaluation.review.context import ReviewContext
from calibrated_agent_evaluation.review.errors import ReviewError
from calibrated_agent_evaluation.review.guards import (
    DEFAULT_ALLOWED_HOSTS,
    SameOriginWrites,
    TrustedHosts,
    sessionless,
)
from calibrated_agent_evaluation.review.prefill import NoPrefill, Prefill
from calibrated_agent_evaluation.review.proposals import ProposalStore
from calibrated_agent_evaluation.review.record import CodebookMode
from calibrated_agent_evaluation.review.renderers import RendererRegistry, make_default_registry
from calibrated_agent_evaluation.review.routes import (
    register_browse_routes,
    register_verdict_routes,
)
from calibrated_agent_evaluation.review.session import (
    ReviewSession,
    WorkList,
    WorkScope,
    build_work_list,
)
from calibrated_agent_evaluation.review.session_log import SessionLog

_STATIC = files("calibrated_agent_evaluation.review") / "static"
# One bundled, dependency-free default theme; inlined into the page (no CDN,
# no framework, no build step).
_STYLES_CSS = (_STATIC / "styles.css").read_text(encoding="utf-8")
# htmx served from a vendored local copy (0BSD), not a CDN, so
# the surface works offline. FastHTML's CDN headers are disabled (default_hdrs=False, below).
_HTMX_JS = (_STATIC / "vendor" / "htmx.min.js").read_text(encoding="utf-8")


def build_app(
    backend: TraceStoreBackend,
    *,
    rater_id: str,
    prefill: Prefill | None = None,
    registry: RendererRegistry | None = None,
    session: ReviewSession | None = None,
    scope: WorkScope | None = None,
    session_log: str | Path | None = None,
    allowed_hosts: Sequence[str] = DEFAULT_ALLOWED_HOSTS,
    codebook_version: str | None = None,
    failure_modes: Sequence[CodebookMode] = (),
    proposals: ProposalStore | None = None,
) -> Any:
    """Build the review FastHTML app over the given trace-store backend.

    Reads traces and reads/writes labels through the ``TraceStoreBackend`` surface
    (``query`` / ``get_trace`` / ``put_score`` / ``get_scores``); the interface every v1 backend
    implements (``LocalBackend``, ``LangfuseBackend``). (The ``LabelBackend`` work-list protocol has
    no v1 implementer; it is the future managed-labeling seam.)

    ``prefill`` defaults to a no-op (offline, no LLM); ``registry`` defaults to the tool-call-tree
    renderer; ``session`` defaults to the work-list ``scope`` resolves to (``build_work_list``):
    by default every trace this rater has not yet labeled, in stable order, so a restarted session
    resumes where it stopped.
    ``rater_id`` names the human labeling in this session and is stamped on every label the app
    writes. It is required, not defaulted: a guessed identity (the OS login, say) silently merges
    two raters who share a machine, and an agreement statistic keyed on it would then measure
    nothing. Raises ``ReviewError`` for an empty ``rater_id``. ``session_log``, when given, is a
    JSONL file that gets one line per verdict and per undo, with the time on the trace
    (``review.session_log``); ``ReviewError`` at startup if it cannot be written.
    ``allowed_hosts`` lists the Host header names the app answers to (``guards.allowed_hosts_for``
    builds it for a bind address); any other Host is refused, which blocks DNS rebinding. Every
    state-changing request must also come from the review page itself (``guards.SameOriginWrites``).
    ``codebook_version`` is stamped on every label written (``None``: the unversioned bucket).
    ``failure_modes``, when non-empty, is the closed failure-mode list the form offers; every FAIL
    must then name one of them. ``proposals``, with a codebook version and a list, lets the
    labeler propose a new mode (name and note) from the form: the FAIL saves with it and the
    proposal is stored there for the codebook owner. The picker re-reads the pending proposals on
    every render, so a mode proposed in any session on the store is offered in the next one.
    Raises ``BackendError`` if the backend is unreachable at startup; no session is opened.
    The returned app exposes ``.state.review_session``, ``.state.codebook_version`` and
    ``.state.failure_modes`` for inspection/tests.
    """
    if not isinstance(rater_id, str) or not rater_id.strip():
        raise ReviewError("the review surface needs a non-empty rater_id naming the labeler")
    rater_id = rater_id.strip()
    log = SessionLog(session_log) if session_log is not None else None

    # Synchronous reachability probe; propagate the typed BackendError, open no session.
    if session is None:
        work = build_work_list(backend, scope or WorkScope(), rater_id=rater_id)
        session = ReviewSession.from_ids([t.trace_id for t in work.traces])
    else:  # an injected session: probe without resolving a scope
        scan = scan_traces(backend)
        work = WorkList(
            traces=scan.traces, already_labeled=[], not_calibrated_agent_evaluation=scan.skipped
        )

    ctx = ReviewContext(
        backend=backend,
        session=session,
        registry=registry or make_default_registry(),
        prefill=prefill or NoPrefill(),
        rater_id=rater_id,
        log=log,
        traces=list(work.traces),
        codebook_version=codebook_version,
        failure_modes=tuple(failure_modes),
        proposals=proposals,
    )

    # FastHTML adds .route/.get/.post/.state dynamically; type as Any so those resolve for pyright.
    # default_hdrs=False disables FastHTML's CDN-loaded htmx/surreal/fasthtml-js/canonical headers;
    # we supply our own head; charset/viewport, the bundled theme, and
    # the vendored local htmx; so nothing is fetched from a CDN and the surface works offline.
    middleware = [
        Middleware(TrustedHosts, allowed_hosts=list(allowed_hosts)),
        Middleware(SameOriginWrites),
    ]
    app: Any = FastHTML(
        default_hdrs=False,
        **sessionless(),
        middleware=middleware,
        hdrs=(
            Meta(charset="utf-8"),
            Meta(name="viewport", content="width=device-width, initial-scale=1"),
            Style(cast(str, NotStr(_STYLES_CSS))),  # bundled default theme
            Script(cast(str, NotStr(_HTMX_JS))),  # vendored htmx, local (not CDN)
        ),
    )
    register_browse_routes(app, ctx)
    register_verdict_routes(app, ctx)

    app.state.review_session = session
    app.state.already_labeled = list(work.already_labeled)
    app.state.not_calibrated_agent_evaluation = list(work.not_calibrated_agent_evaluation)
    app.state.codebook_version = codebook_version
    app.state.failure_modes = tuple(failure_modes)
    return app
