# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""No public record field carries a bare float estimate.

A structural check over ``src/calibrated_agent_evaluation`` (outside ``stats/``, whose primitives
return scalars that their callers wrap): any dataclass field named like an estimate (``*_rate``,
``*_kappa``, ``*coverage``, ``value``, ``rating``) annotated as a bare ``float`` fails, unless it is
listed below with the field that carries its confidence treatment. Before the
statistical-correctness fixes this flagged ``AgreementReport.outcome_kappa`` / ``culprit_kappa``,
``AgreementDisplay.value``, ``DriftMetric.value``, ``NumericPoint.value`` and
``EloRating.rating``, alongside the ``SaturationResult`` floats fixed earlier.
"""

from __future__ import annotations

import ast
import re
from pathlib import Path

SRC = Path(__file__).resolve().parent.parent / "src" / "calibrated_agent_evaluation"
ESTIMATE_NAME = re.compile(r"(_rate|_kappa|coverage|^value|^rating|^kappa)$")

# (module, class, field) -> the sibling field that carries its confidence treatment, or a reason.
ALLOWED: dict[tuple[str, str, str], str] = {
    ("run/report.py", "DriftMetric", "value"): "p_value",
    ("run/report.py", "NumericPoint", "value"): "lo",
    # Sequential Elo is a display transform; bradley_terry is the estimator of record.
    ("subjective/types.py", "EloRating", "rating"): "display-only",
    # A rung of an evaluator's published rating scale: a fixed point the vendor defines, not an
    # estimate, so it has no confidence treatment to carry.
    ("agentcore/judge.py", "ScaleValue", "value"): "display-only",
}


def _is_bare_float(annotation: ast.expr) -> bool:
    text = ast.unparse(annotation).replace(" ", "")
    return text in {"float", "float|None", "None|float", "Optional[float]"}


def _bare_estimate_fields() -> list[tuple[str, str, str, set[str]]]:
    found: list[tuple[str, str, str, set[str]]] = []
    for path in sorted(SRC.rglob("*.py")):
        rel = path.relative_to(SRC).as_posix()
        if rel.startswith("stats/"):
            continue
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if not isinstance(node, ast.ClassDef) or node.name.startswith("_"):
                continue
            fields = {
                item.target.id
                for item in node.body
                if isinstance(item, ast.AnnAssign) and isinstance(item.target, ast.Name)
            }
            for item in node.body:
                if (
                    isinstance(item, ast.AnnAssign)
                    and isinstance(item.target, ast.Name)
                    and ESTIMATE_NAME.search(item.target.id)
                    and _is_bare_float(item.annotation)
                ):
                    found.append((rel, node.name, item.target.id, fields))
    return found


def test_no_bare_float_estimate_fields() -> None:
    offenders = []
    for rel, cls, name, fields in _bare_estimate_fields():
        companion = ALLOWED.get((rel, cls, name))
        if companion is None:
            offenders.append(f"{rel}::{cls}.{name}")
        elif companion != "display-only":
            assert companion in fields, f"{rel}::{cls}.{name} lost its {companion!r} companion"
    assert offenders == [], (
        "bare float estimates without a confidence treatment (carry an Interval/Bound, or list "
        f"the companion field in ALLOWED with a reason): {offenders}"
    )


def test_the_rule_catches_a_bare_rate() -> None:
    """The check is live: a synthetic dataclass with a bare ``*_rate`` field is matched."""
    tree = ast.parse("class R:\n    pass_rate: float\n")
    cls = tree.body[0]
    assert isinstance(cls, ast.ClassDef)
    item = cls.body[0]
    assert isinstance(item, ast.AnnAssign)
    assert isinstance(item.target, ast.Name)
    assert ESTIMATE_NAME.search(item.target.id)
    assert _is_bare_float(item.annotation)
