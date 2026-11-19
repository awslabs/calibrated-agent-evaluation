# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""End-to-end: labeling the corpus through the route yields the gold outcomes."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from calibrated_agent_evaluation.schema import Label


def test_corpus_labels_match_gold_outcomes(
    make_client: Callable[..., Any], backend: Any, gold_labels: list[Label]
) -> None:
    client, app = make_client(backend)
    gold = {lab.trace_id: lab.outcome for lab in gold_labels if lab.span_id is None}
    # The review session serves the seeded worklist (the corpus), not every gold trace — the
    # ClaimsCrew contrast traces exist in gold but aren't in this review backend's worklist.
    served = [t.trace_id for t in backend.query()]

    for trace_id in served:
        client.post(
            "/label",
            data={
                "trace_id": trace_id,
                "outcome": gold[trace_id],
                "critique": "human review note",
            },
        )

    for trace_id in served:
        trace_level = [lab for lab in backend.get_scores(trace_id) if lab.span_id is None]
        assert trace_level, f"no label written for {trace_id}"
        assert trace_level[0].outcome == gold[trace_id]
        assert trace_level[0].kind == "human"

    assert app.state.review_session.is_complete
