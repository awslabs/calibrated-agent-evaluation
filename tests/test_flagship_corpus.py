# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the corpus completion: the authored oracles + the guard.

The cluster-stage taxonomy oracle and the attribution transform oracle are re-derived from the
human gold and compared against the committed ``fixtures/expected/{cluster,attribution}/*`` files
(so the flagship checks real oracle values, not "it ran"); the corpus-build ratification guard
refuses a gold label that is not human-ratified (no model-written label in a gold set).
"""

from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

import pytest

from calibrated_agent_evaluation.schema.serde import label_from_dict

_REPO = Path(__file__).resolve().parent.parent
_FIXTURES = _REPO / "fixtures"
# fixtures/ is not an installed package; put the repo root on sys.path so the deferred
# `fixtures.validate_corpus` import resolves inside the tests.
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))


def _gold() -> list[dict]:
    return json.loads((_FIXTURES / "labels" / "gold.json").read_text(encoding="utf-8"))


@pytest.mark.parametrize(("agent", "prefix"), [("orderbot", "ORD"), ("claimscrew", "CLM")])
def test_cluster_oracle_recovers_the_gold_taxonomy(agent: str, prefix: str) -> None:
    """The cluster oracle equals the human-ratified failure-mode taxonomy on the gold."""
    oracle = json.loads(
        (_FIXTURES / "expected" / "cluster" / f"{agent}.json").read_text(encoding="utf-8")
    )["expected"]
    fails = [
        r
        for r in _gold()
        if r["trace_id"].startswith(prefix) and r["outcome"] == "FAIL" and r.get("failure_mode")
    ]
    modes = Counter(r["failure_mode"] for r in fails)
    assert oracle["n_modes"] == len(modes)
    assert oracle["modes"] == dict(modes)
    assert oracle["n_fail_labels"] == len(fails)


def test_attribution_oracle_matches_walkback_transform() -> None:
    """The attribution oracle equals the real validate_walkback output over the crew."""
    from _flagship_helpers import corpus
    from calibrated_agent_evaluation.flagship.reference import (
        claimscrew_definition,
        walkback_from_gold,
    )

    oracle = json.loads(
        (_FIXTURES / "expected" / "attribution" / "claimscrew.json").read_text(encoding="utf-8")
    )["expected"]
    crew = claimscrew_definition()
    reference = corpus()
    gold = reference.gold_labels()
    for trace_id, exp in oracle.items():
        att = walkback_from_gold(reference.trace(trace_id), crew, gold[(trace_id, None)])
        assert att.status == exp["status"], trace_id
        assert list(att.culprit_spans) == exp["culprit_spans"], trace_id
        assert att.manifestation_span == exp["manifestation_span"], trace_id
        assert att.handoff_implicated == exp["handoff_implicated"], trace_id


def test_unratified_gold_label_is_refused() -> None:
    """A non-human gold label with no `ratified_by` is refused by the corpus guard."""
    from fixtures.validate_corpus import validate_label

    errors: list[str] = []
    # An llm-judge gold label with no human ratification is a violation.
    validate_label(
        {
            "trace_id": "T-1",
            "span_id": None,
            "kind": "llm-judge",
            "outcome": "FAIL",
            "critique": "auto",
            "span_defect": False,
        },
        0,
        {"T-1"},
        errors,
    )
    assert any("ratified_by" in e for e in errors)

    # The same label WITH a recorded human ratification is accepted (the ratification path works).
    ok_errors: list[str] = []
    validate_label(
        {
            "trace_id": "T-1",
            "span_id": None,
            "kind": "llm-judge",
            "outcome": "FAIL",
            "critique": "auto, ratified by a human",
            "span_defect": False,
            "ratified_by": "reviewer@example",
        },
        0,
        {"T-1"},
        ok_errors,
    )
    assert not any("ratified_by" in e for e in ok_errors)


def test_gold_labels_are_all_human_ratified() -> None:
    """Every label in the shipped gold corpus is human (ratified by definition)."""
    for row in _gold():
        label = label_from_dict(row)
        assert label.kind == "human", f"{label.trace_id} gold label is {label.kind!r}, not human"
