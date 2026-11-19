# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Contract test binding the fixture oracles to the real linter.

The oracle files under ``fixtures/expected/enforce/`` are authored **independently** of the linter
(hand-written with the judge-definition fixtures, like the ``normalized/`` oracle), so
matching them is a real contract test, not a tautology. For each committed judge fixture this runs
the shipped ``lint_judge_definition`` under the oracle's ``EnforcementContext`` and asserts:

  * the typed error raised (by class name) equals ``expected_error`` (``null`` ⇒ no refusal); and
  * the decisions the layer **records** — the returned list on a clean lint, the error-attached
    ``err.decision`` on a hard refusal — match ``expected_decisions`` on (rule, tier, outcome).

Equality is on the (rule, tier, outcome) triple the oracle carries (message/inputs/nudge are
free-text and covered field-for-field by the round-trip test), mirroring the normalizer golden
test's structural-match discipline.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from calibrated_agent_evaluation.enforce import (
    Axis,
    ChecklistItem,
    EnforcementContext,
    EnforcementError,
    JudgeDefinition,
    lint_judge_definition,
)
from calibrated_agent_evaluation.enforce import errors as enforce_errors

_FIXTURES = Path(__file__).resolve().parent.parent / "fixtures"
_JUDGES = _FIXTURES / "judges"
_ORACLES = _FIXTURES / "expected" / "enforce"

# The committed fixture pair; the oracle file shares the judge file's name.
_CASES = ["orderbot_binary.json", "bad_multi_likert.json"]


def _judgedef(name: str) -> JudgeDefinition:
    doc = json.loads((_JUDGES / name).read_text(encoding="utf-8"))
    return JudgeDefinition(
        judge_id=doc["judge_id"],
        axes=tuple(Axis(name=a["name"], scale=a["scale"]) for a in doc["axes"]),
        output_fields=tuple(doc["output_fields"]),
        scorer=doc.get("scorer"),
        checklist=tuple(
            ChecklistItem(name=c["name"], critical=c["critical"]) for c in doc.get("checklist", [])
        ),
        weighted_collapse=doc.get("weighted_collapse", False),
    )


def _triples(decisions: Any) -> list[tuple[str, str, str]]:
    return [(d.rule, d.tier, d.outcome) for d in decisions]


def _oracle_triples(expected: list[dict[str, str]]) -> list[tuple[str, str, str]]:
    return [(d["rule"], d["tier"], d["outcome"]) for d in expected]


# --- the fixture oracles match what the shipped linter emits ----------------


@pytest.mark.parametrize("name", _CASES)
def test_fixture_oracle_matches_linter(name: str) -> None:
    oracle = json.loads((_ORACLES / name).read_text(encoding="utf-8"))
    defn = _judgedef(name)
    ctx = EnforcementContext(**oracle["lint_context"])
    expected_triples = _oracle_triples(oracle["expected_decisions"])

    if oracle["expected_error"] is None:
        # Clean lint: no refusal; the returned decisions ARE the recorded ones.
        decisions = lint_judge_definition(defn, ctx)
        assert _triples(decisions) == expected_triples
        return

    # Hard refusal: the named typed error fires and carries its recorded decision.
    expected_cls = getattr(enforce_errors, oracle["expected_error"])
    with pytest.raises(expected_cls) as exc:
        lint_judge_definition(defn, ctx)
    assert isinstance(exc.value, EnforcementError)  # every refusal is in the typed family
    recorded = [exc.value.decision] if exc.value.decision is not None else []
    assert _triples(recorded) == expected_triples


def test_every_judge_fixture_has_a_matching_oracle() -> None:
    # Guard: a new judge fixture must ship its oracle (and vice-versa) — no orphaned corpus file.
    judge_files = {p.name for p in _JUDGES.glob("*.json")}
    oracle_files = {p.name for p in _ORACLES.glob("*.json")}
    assert judge_files == oracle_files
