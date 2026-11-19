# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the oracle comparison.

compare_oracle returns matched=True when a stage reproduces the committed oracle's pinned fields and
matched=False naming the divergence otherwise; a missing oracle for a stage that claims one is
a proof-integrity failure that raises the typed FlagshipError (never a silent skip).
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from calibrated_agent_evaluation.flagship.errors import FlagshipError
from calibrated_agent_evaluation.flagship.oracle import compare_oracle, load_oracle


def _write_oracle(tmp_path: Path, doc: dict) -> Path:
    path = tmp_path / "oracle.json"
    path.write_text(json.dumps(doc))
    return path


def test_matches_when_pinned_fields_reproduced(tmp_path: Path) -> None:
    """Only the fields the oracle pins are compared; extra produced fields are allowed."""
    oracle = _write_oracle(tmp_path, {"expected": {"passed": False, "m": 2}})
    check = compare_oracle("run", {"passed": False, "m": 2, "extra_field": "ignored"}, oracle)
    assert check.matched is True
    assert check.divergence is None
    assert check.oracle_path == str(oracle)


def test_diverges_when_pinned_field_differs(tmp_path: Path) -> None:
    """A produced value differing from a pinned oracle value -> matched=False naming it."""
    oracle = _write_oracle(tmp_path, {"expected": {"passed": False}})
    check = compare_oracle("run", {"passed": True}, oracle)
    assert check.matched is False
    assert check.divergence is not None
    assert "passed" in check.divergence


def test_missing_oracle_raises_not_silent_skip(tmp_path: Path) -> None:
    """A stage claiming an absent oracle is a proof-integrity failure, not a pass."""
    missing = tmp_path / "does_not_exist.json"
    with pytest.raises(FlagshipError, match="missing"):
        compare_oracle("cluster", {"modes": 3}, missing)
    with pytest.raises(FlagshipError, match="missing"):
        load_oracle(missing)


def test_missing_pinned_key_is_a_divergence(tmp_path: Path) -> None:
    """A produced dict lacking a key the oracle pins is a divergence, not a match."""
    oracle = _write_oracle(tmp_path, {"expected": {"observed_tools": ["lookup_order"]}})
    check = compare_oracle("probe", {}, oracle)
    assert check.matched is False
    assert "observed_tools" in (check.divergence or "")


@pytest.mark.parametrize("doc", [{}, {"expected": {}}, {"note": "pins nothing"}, {"expected": []}])
def test_an_oracle_that_pins_no_field_is_refused(tmp_path: Path, doc: dict) -> None:
    oracle = _write_oracle(tmp_path, doc)
    with pytest.raises(FlagshipError, match="pins no fields"):
        compare_oracle("run", {"anything": True}, oracle)


def test_an_oracle_that_is_not_an_object_is_refused(tmp_path: Path) -> None:
    path = tmp_path / "oracle.json"
    path.write_text("[1, 2]")
    with pytest.raises(FlagshipError, match="JSON object"):
        compare_oracle("run", {"anything": True}, path)
