# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Plugin-renderer resolution + guarded fallback."""

from __future__ import annotations

import logging

from calibrated_agent_evaluation.review import RendererRegistry
from calibrated_agent_evaluation.schema import Trace


def _by_id(traces: list[Trace], trace_id: str) -> Trace:
    return next(t for t in traces if t.trace_id == trace_id)


def _is_refund(trace: Trace) -> bool:
    return trace.taxonomy_cell.get("feature") == "refund_handling"


def test_matching_predicate_uses_registered_renderer(corpus_traces: list[Trace]) -> None:
    registry = RendererRegistry(default=lambda t: "DEFAULT")
    registry.register(_is_refund, lambda t: "REFUND-VIEW")
    assert registry.render(_by_id(corpus_traces, "ORD-1001")) == "REFUND-VIEW"


def test_nonmatching_trace_falls_to_default(corpus_traces: list[Trace]) -> None:
    registry = RendererRegistry(default=lambda t: "DEFAULT")
    registry.register(_is_refund, lambda t: "REFUND-VIEW")
    # CLM-4471 is claim_adjudication, not refund_handling → default renderer.
    assert registry.render(_by_id(corpus_traces, "CLM-4471")) == "DEFAULT"


def test_raising_renderer_falls_back_to_default_and_logs(
    corpus_traces: list[Trace], caplog
) -> None:
    def boom(trace: Trace) -> str:
        raise RuntimeError("renderer blew up")

    registry = RendererRegistry(default=lambda t: "DEFAULT")
    registry.register(lambda t: True, boom)
    with caplog.at_level(logging.ERROR):
        result = registry.render(_by_id(corpus_traces, "ORD-1001"))
    assert result == "DEFAULT"  # session survives a broken plugin
    assert any("falling back" in r.getMessage() for r in caplog.records)
