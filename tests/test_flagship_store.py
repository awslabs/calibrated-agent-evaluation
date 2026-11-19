# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the FlagshipReport round-trip store.

Property/round-trip: a FlagshipReport (with nested stage outcomes + oracle checks + intervals +
gate refs + world-parity) dumps and loads field-for-field equal in both CSV and JSON;
a schema-version mismatch refuses with the typed FlagshipStoreError (not a raw JSONDecodeError);
the FlagshipReport empty-inputs guard raises.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from _helpers import TEST_INTERVAL_KIND, TEST_STAMP
from calibrated_agent_evaluation.calibrate.record import Interval
from calibrated_agent_evaluation.flagship.errors import FlagshipError, FlagshipStoreError
from calibrated_agent_evaluation.flagship.report import (
    FlagshipReport,
    GateDecisionRef,
    OracleCheck,
    StageOutcome,
    WorldParity,
)
from calibrated_agent_evaluation.flagship.store import dump_flagship_reports, load_flagship_reports


def _report() -> FlagshipReport:
    """A fully-populated report exercising every nested type the codec must reconstruct."""
    return FlagshipReport(
        provenance=TEST_STAMP,
        flagship_id="flagship-001",
        verdict="proven",
        agents=("orderbot", "claimscrew"),
        worlds=("local", "langfuse"),
        model_plane="cassette",
        fixture_version="1",
        stages=(
            StageOutcome(
                agent="orderbot",
                world="local",
                stage="run",
                ok=True,
                oracle=OracleCheck(
                    stage="run",
                    oracle_path="fixtures/expected/run/gate_decision.json",
                    matched=True,
                    divergence=None,
                ),
                headline=Interval(
                    point=0.06, lo=0.02, hi=0.11, refusal_reason=None, **TEST_INTERVAL_KIND
                ),
                artifact_ids=("ORD-1001",),
                note="run gate reproduced",
            ),
            StageOutcome(
                agent="claimscrew",
                world="local",
                stage="improve",
                ok=True,
                oracle=OracleCheck(
                    stage="improve",
                    oracle_path="fixtures/expected/improve/relocation_claimscrew.json",
                    matched=True,
                ),
                headline=Interval(
                    point=None,
                    lo=0.0,
                    hi=0.0,
                    refusal_reason="Youden J < 0.5",
                    **TEST_INTERVAL_KIND,
                ),
            ),
        ),
        gates=(
            GateDecisionRef(
                judge_id="orderbot-refund-resolved",
                certified=True,
                bias_verdict="CLEAR",
                score_gate_admitted=True,
            ),
        ),
        parity=(
            WorldParity(
                agent="orderbot", compared=("local", "langfuse"), equal=True, divergence=None
            ),
        ),
        reasons=(),
        inputs={"agents": "orderbot,claimscrew", "worlds": "local,langfuse"},
    )


@pytest.mark.parametrize("fmt", ["csv", "json"])
def test_flagship_report_round_trips_field_for_field(fmt: str, tmp_path: Path) -> None:
    """A FlagshipReport dumps and loads field-for-field equal in both CSV and JSON."""
    original = _report()
    dump_flagship_reports([original], tmp_path, fmt=fmt)
    loaded = load_flagship_reports(tmp_path, fmt=fmt)
    assert loaded == [original]


def test_schema_version_mismatch_refuses_loudly(tmp_path: Path) -> None:
    """A stored schema_version != the running one raises the typed FlagshipStoreError.

    Not a raw JSONDecodeError, and not a silent coercion (the no-coercion precedent).
    """
    dump_flagship_reports([_report()], tmp_path, fmt="json")
    file = tmp_path / "flagship_report.json"
    docs = json.loads(file.read_text())
    docs[0]["schema_version"] = "999"
    file.write_text(json.dumps(docs))
    with pytest.raises(FlagshipStoreError, match="schema_version"):
        load_flagship_reports(tmp_path, fmt="json")


def test_empty_inputs_report_is_refused_at_construction() -> None:
    """A FlagshipReport with no recorded inputs is not reproducible -> refuse it."""
    with pytest.raises(FlagshipError, match="requires non-empty inputs"):
        FlagshipReport(
            provenance=TEST_STAMP,
            flagship_id="x",
            verdict="failed",
            agents=(),
            worlds=(),
            model_plane="cassette",
            fixture_version="1",
            stages=(),
            gates=(),
            parity=(),
            reasons=(),
            inputs={},
        )


def test_malformed_dump_refuses_loudly(tmp_path: Path) -> None:
    """A syntactically-valid but structurally-malformed dump refuses, not a partial load."""
    (tmp_path / "flagship_report.json").write_text(json.dumps([{"flagship_id": "x"}]))
    with pytest.raises(FlagshipStoreError):
        load_flagship_reports(tmp_path, fmt="json")
