# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Risk-zone tests, including the structural label-block."""

from __future__ import annotations

import inspect
import warnings

import pytest

import calibrated_agent_evaluation.synth as synth
from calibrated_agent_evaluation.sampling.types import FrameResult, FrameRow
from calibrated_agent_evaluation.synth import RiskZoneWarning, generate, scan_risk_zones
from test_synth_helpers import FakeRenderer


def _frame(cells: list[dict[str, str]]) -> FrameResult:
    rows = [FrameRow(cell=c, frame="discovery") for c in cells]  # type: ignore[arg-type]
    return FrameResult(rows=rows, frame="discovery")  # type: ignore[arg-type]


class TestRiskZoneWarning:
    """A risk-zone taxonomy value warns; a routine one does not."""

    def test_medical_value_warns_naming_domain(self) -> None:
        frame = _frame([{"domain": "medical", "persona": "patient"}])
        with pytest.warns(RiskZoneWarning, match="medical"):
            matches = scan_risk_zones(frame)
        assert "medical" in matches

    def test_routine_taxonomy_no_warning(self) -> None:
        frame = _frame([{"feature": "returns", "persona": "consumer"}])
        with warnings.catch_warnings():
            warnings.simplefilter("error")  # any RiskZoneWarning would raise
            matches = scan_risk_zones(frame)
        assert matches == {}

    def test_warning_fires_through_generate(self) -> None:
        frame = _frame([{"domain": "legal"}])
        with pytest.warns(RiskZoneWarning, match="legal"):
            generate(frame, FakeRenderer())


class TestConfigurableLexicon:
    """Caller-supplied lexicon replaces the default; empty lexicon disables scanning."""

    def test_custom_lexicon_flags_domain_default_misses(self) -> None:
        frame = _frame([{"domain": "aerospace"}])
        # 'aerospace' is not in the default lexicon → no warning by default.
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            assert scan_risk_zones(frame) == {}
        # But a custom lexicon flags it.
        with pytest.warns(RiskZoneWarning, match="aerospace"):
            matches = scan_risk_zones(frame, lexicon=frozenset({"aerospace"}))
        assert "aerospace" in matches

    def test_empty_lexicon_disables_warnings(self) -> None:
        frame = _frame([{"domain": "medical"}])  # would warn under the default
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            matches = scan_risk_zones(frame, lexicon=frozenset())
        assert matches == {}


class TestNoLabelPath:
    """Label half: the synth public surface exposes no Label-writing callable."""

    def test_synth_surface_exposes_no_label_returning_callable(self) -> None:
        # Structural block: the engine emits inputs + real-agent Traces only, never a Label.
        for name in synth.__all__:
            obj = getattr(synth, name)
            if not callable(obj) or inspect.isclass(obj):
                continue
            ret = inspect.signature(obj).return_annotation
            assert "Label" not in str(ret), f"{name} must not return a Label (gold is human-only)"

    def test_label_not_in_public_surface(self) -> None:
        assert "Label" not in synth.__all__
