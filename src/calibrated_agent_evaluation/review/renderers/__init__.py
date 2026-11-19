# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Plugin-renderer seam for the review surface.

``RendererRegistry`` (in ``base``) is the FastHTML-free resolver; ``tool_call_tree_renderer``
(in ``default``) is the built-in fallback view. ``make_default_registry`` wires them together:
the app builds one, and a domain registers its own renderers on it via ``register(predicate, fn)``.
The concrete renderers need ``fasthtml`` (the ``[ui]`` extra), so they load on first use. Importing
this package for ``RendererRegistry`` alone, as ``calibrated_agent_evaluation.review`` does, needs
no extra.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from calibrated_agent_evaluation.review.renderers.base import (
    Renderer,
    RendererPredicate,
    RendererRegistry,
)

if TYPE_CHECKING:
    from calibrated_agent_evaluation.review.renderers.default import tool_call_tree_renderer


def make_default_registry() -> RendererRegistry:
    """A fresh registry whose default is the built-in tool-call-tree renderer."""
    from calibrated_agent_evaluation.review.renderers.default import tool_call_tree_renderer

    return RendererRegistry(default=tool_call_tree_renderer)


def __getattr__(name: str) -> object:
    if name == "tool_call_tree_renderer":
        from calibrated_agent_evaluation.review.renderers.default import tool_call_tree_renderer

        return tool_call_tree_renderer
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "Renderer",
    "RendererPredicate",
    "RendererRegistry",
    "make_default_registry",
    "tool_call_tree_renderer",
]
