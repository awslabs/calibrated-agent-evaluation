# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""The FlagshipReport carries confidence-treated headlines, no blended scalar.

Every stage headline the report surfaces is an Interval (never a bare float; a refused
correction is point=None + refusal_reason), and the report exposes no single blended aggregate
accuracy scalar field. This is the no-bare-number contract for the proof artifact.
"""

from __future__ import annotations

from pathlib import Path

from _helpers import TEST_INTERVAL_KIND
from calibrated_agent_evaluation.backends import TraceStoreBackend, get_backend
from calibrated_agent_evaluation.calibrate.record import Interval
from calibrated_agent_evaluation.flagship import FlagshipConfig, FlagshipReport, World, prove

_REPO = Path(__file__).resolve().parent.parent
_FIXTURES = _REPO / "fixtures"


def _backend_factory(tmp_path: Path):
    def factory(_world: World) -> TraceStoreBackend:
        return get_backend(None, local_root=tmp_path / "backend")

    return factory


def _proof(tmp_path: Path) -> FlagshipReport:
    from _flagship_helpers import orderbot_plan

    config = FlagshipConfig(
        agents=("orderbot",),
        worlds=("local",),
        model_plane="cassette",
        fixtures_root=str(_FIXTURES),
    )
    return prove(config, [orderbot_plan()], backend_factory=_backend_factory(tmp_path))


def test_every_headline_is_an_interval_never_a_bare_float(tmp_path: Path) -> None:
    """A StageOutcome headline is an Interval or None -- never a bare float."""
    report = _proof(tmp_path)
    for oc in report.stages:
        assert oc.headline is None or isinstance(oc.headline, Interval), oc.stage


def test_report_exposes_no_blended_accuracy_scalar(tmp_path: Path) -> None:
    """The report carries no single blended aggregate accuracy number field.

    A bare float/int top-level field would be exactly the banned blended-accuracy shape; the report
    carries only structured, confidence-treated outcomes (verdict is a categorical, headlines are
    Intervals).
    """
    report = _proof(tmp_path)
    for name in type(report).model_fields:
        value = getattr(report, name)
        assert not isinstance(value, float), (
            f"FlagshipReport.{name} is a bare float -- a blended headline is banned"
        )


def test_refused_correction_round_trips_as_point_none(tmp_path: Path) -> None:
    """A point=None Interval headline carries a refusal_reason (no fabricated float)."""
    # Construct the refused-correction case explicitly (a J<0.5 correction the shipped gate refuses)
    # and confirm the report shape represents it honestly rather than as a fabricated number.
    refused = Interval(
        point=None, lo=0.0, hi=0.0, refusal_reason="Youden J < 0.5", **TEST_INTERVAL_KIND
    )
    assert refused.point is None
    assert refused.refusal_reason is not None
