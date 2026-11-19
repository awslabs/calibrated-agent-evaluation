# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Plugin-renderer registry.

The extension seam that lets a domain register a trace renderer that activates on trace
metadata; "the framework is the floor, not the ceiling". The registry is
an *instance* the app owns (no global mutable state), and is deliberately
FastHTML-free: a ``Renderer`` is any callable ``Trace -> component``, so this module and its
tests need no ``[ui]`` extra. The concrete renderers (``default.py``, ``trace_view.py``) are the
ones that build FastHTML components.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from calibrated_agent_evaluation.schema import Trace

logger = logging.getLogger(__name__)

RendererPredicate = Callable[[Trace], bool]
Renderer = Callable[
    [Trace], Any
]  # returns a view component (FastHTML FT); kept Any to stay UI-free


@dataclass
class RendererRegistry:
    """Resolve a trace's renderer: the first registered predicate that matches, else the default.

    Construct with the ``default`` renderer (the app wires the tool-call-tree renderer). Register
    domain renderers with :meth:`register`. :meth:`render` is the safe entry point; it guards a
    raising renderer back to the default and logs, so one bad plugin never ends the session.
    """

    default: Renderer
    _rules: list[tuple[RendererPredicate, Renderer]] = field(default_factory=list, repr=False)

    def register(self, predicate: RendererPredicate, renderer: Renderer) -> None:
        """Register ``renderer`` for any trace where ``predicate(trace)`` is true.

        Registration order is match order; the first matching predicate wins.
        """
        self._rules.append((predicate, renderer))

    def resolve(self, trace: Trace) -> Renderer:
        """Return the first renderer whose predicate matches ``trace``, else the default.

        Note: a predicate that itself raises is treated as non-matching (guarded + logged) so a
        broken predicate cannot mask the default.
        """
        for predicate, renderer in self._rules:
            try:
                matched = predicate(trace)
            except Exception:
                logger.exception(
                    "renderer predicate raised for trace %s; treating as no-match", trace.trace_id
                )
                continue
            if matched:
                return renderer
        return self.default

    def render(self, trace: Trace) -> Any:
        """Render ``trace`` with the resolved renderer; fall back to the default on error.

        A registered renderer that raises is caught and logged (``logging.exception``) and the
        default renderer is used instead; the labeling session continues rather than crashing.
        Use :meth:`render_with_status` to also learn that a fallback happened.
        """
        output, _fallback = self.render_with_status(trace)
        return output

    def render_with_status(self, trace: Trace) -> tuple[Any, str | None]:
        """Like :meth:`render`, plus a message naming the failure when the default stood in.

        The message is for the labeler's screen. A fallback that only reached the log looked, to
        the person labeling, exactly like a trace the plugin was never meant to render.
        """
        renderer = self.resolve(trace)
        try:
            return renderer(trace), None
        except Exception as exc:
            logger.exception(
                "plugin renderer failed for trace %s; falling back to the default renderer",
                trace.trace_id,
            )
            name = getattr(renderer, "__name__", type(renderer).__name__)
            message = (
                f"The {name!r} renderer for this trace failed ({type(exc).__name__}: {exc}); "
                "showing the default view instead."
            )
            return self.default(trace), message
