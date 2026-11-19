# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""The smaller review-surface issues, one test per item."""

from __future__ import annotations

import logging
from collections.abc import Callable
from typing import Any

from fasthtml.common import to_xml

from calibrated_agent_evaluation._model import replace
from calibrated_agent_evaluation.review import RendererRegistry
from calibrated_agent_evaluation.review.renderers import make_default_registry
from calibrated_agent_evaluation.review.renderers.default import FOLD_LINES, tool_call_tree_renderer
from calibrated_agent_evaluation.schema import Trace

from .conftest import FakeBackend


def test_the_completion_page_has_one_review_root(
    make_client: Callable[..., Any], make_backend: Callable[..., Any]
) -> None:
    client, _app = make_client(make_backend(traces=[]))
    assert client.get("/").text.count('id="review-root"') == 1


def test_a_failing_plugin_renderer_is_announced_on_screen(
    make_client: Callable[..., Any], backend: FakeBackend, caplog: Any
) -> None:
    def broken(trace: Trace) -> Any:
        raise RuntimeError("template missing")

    registry = make_default_registry()
    registry.register(lambda t: True, broken)
    client, _app = make_client(backend, registry=registry)
    with caplog.at_level(logging.ERROR):
        page = client.get("/").text
    assert "renderer for this trace failed" in page
    assert "template missing" in page
    assert 'data-trace-id="ORD-1001"' in page  # the default view still rendered


def test_render_with_status_reports_no_fallback_when_the_plugin_works(
    corpus_traces: list[Trace],
) -> None:
    registry = RendererRegistry(default=lambda t: "DEFAULT")
    registry.register(lambda t: True, lambda t: "PLUGIN")
    assert registry.render_with_status(corpus_traces[0]) == ("PLUGIN", None)


def test_a_large_json_block_is_folded_but_not_truncated(corpus_traces: list[Trace]) -> None:
    big = {f"field_{i}": i for i in range(FOLD_LINES * 2)}
    trace = replace(corpus_traces[0], output=big)
    page = to_xml(tool_call_tree_renderer(trace))
    assert "<details" in page
    assert f"field_{FOLD_LINES * 2 - 1}" in page  # every line is still there, behind the fold


def test_a_small_json_block_is_not_folded(corpus_traces: list[Trace]) -> None:
    trace = replace(
        corpus_traces[0], ownership="probed", spans=[], input={"q": "hi"}, output={"a": "ok"}
    )
    page = to_xml(tool_call_tree_renderer(trace))
    assert "<details" not in page
    assert "&quot;a&quot;: &quot;ok&quot;" in page or '"a": "ok"' in page


def test_history_lists_this_sessions_verdicts_and_opens_them_read_only(
    make_client: Callable[..., Any], backend: FakeBackend
) -> None:
    client, _app = make_client(backend)
    client.post("/label", data={"trace_id": "ORD-1001", "outcome": "FAIL", "critique": "late"})
    client.post("/label", data={"trace_id": "ORD-1002", "outcome": "PASS", "critique": ""})

    history = client.get("/history", headers={"HX-Request": "true"}).text
    assert history.index("ORD-1002") < history.index("ORD-1001")  # newest first
    assert 'hx-get="/trace/ORD-1001"' in history

    labeled = client.get("/trace/ORD-1001", headers={"HX-Request": "true"}).text
    assert "Stored labels (read-only)" in labeled
    assert "late" in labeled
    assert 'id="label-form"' not in labeled  # nothing to submit from a history view
    assert "<script" not in labeled  # a fragment when swapped in


def test_a_route_visited_directly_gets_the_whole_page(
    make_client: Callable[..., Any], backend: FakeBackend
) -> None:
    client, _app = make_client(backend)
    client.post("/label", data={"trace_id": "ORD-1001", "outcome": "PASS", "critique": ""})
    page = client.get("/trace/ORD-1001").text
    assert 'id="review-root"' in page
    assert "createController" in page  # hotkeys.js came with it
