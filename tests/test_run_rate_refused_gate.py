# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""``calibrated-agent-evaluation run --ci`` gates a judge whose corrected rate is refused.

The per-mode gate needs only the certified judge's verdicts, so a record that cannot back a
corrected rate (gold below the coverage floor here) still gets its gate and its exit code. The rate
is reported as not available, with the reason the calibration engine gave, and the reason is kept
in the run record.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from calibrated_agent_evaluation._model import replace
from calibrated_agent_evaluation.cli import main
from calibrated_agent_evaluation.cli.dispatch import EXIT_DOMAIN_ERROR, EXIT_OK
from calibrated_agent_evaluation.cli.records import load_records_of
from cli_store_helpers import (
    certified_calibration,
    clear_battery,
    seed_verdicts,
    store_battery,
    store_calibration,
    write_judge_file,
)


@pytest.fixture(autouse=True)
def _no_env_root(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("CALIBRATED_AGENT_EVALUATION_LOCAL_ROOT", raising=False)


def _rate_unsafe():
    # 80 gold per class at an observed 0.95, whose coverage floor is 80, and a planned floor of 130:
    # certified, but below the floor the corrected rate needs.
    record = replace(certified_calibration(), planned_per_class=80, planned_coverage_floor=130)
    assert record.below_coverage_floor
    return record


def _run(root: Path, judge: Path) -> int:
    return main(["--root", str(root), "run", "--ci", "--n-resamples", "199", "--judge", str(judge)])


def test_a_rate_unsafe_judge_whose_gate_passes_exits_ok(tmp_path, capsys):
    root = tmp_path / "store"
    seed_verdicts(root, 200, 0)
    judge = write_judge_file(tmp_path / "judge.json")
    store_calibration(root, _rate_unsafe())
    store_battery(root, clear_battery())
    assert _run(root, judge) == EXIT_OK
    out = capsys.readouterr().out
    assert "PASSED" in out
    assert "REFUSED (gold below the coverage floor:" in out
    assert "calibrate on the new gold" in out
    (record,) = load_records_of(root, "run")
    assert record.gate_passed is True
    (point,) = record.trend
    assert point.corrected_rate.point is None
    assert point.corrected_rate.refusal_reason.startswith("gold below the coverage floor:")


def test_a_rate_unsafe_judge_whose_gate_fails_exits_nonzero(tmp_path, capsys):
    root = tmp_path / "store"
    seed_verdicts(root, 150, 50)
    judge = write_judge_file(tmp_path / "judge.json")
    store_calibration(root, _rate_unsafe())
    store_battery(root, clear_battery())
    assert _run(root, judge) == EXIT_DOMAIN_ERROR
    captured = capsys.readouterr()
    out = captured.out + captured.err
    assert "FAILED" in out
    assert "REFUSED (gold below the coverage floor:" in out
    (record,) = load_records_of(root, "run")
    assert record.gate_passed is False
    assert record.trend[0].corrected_rate.refusal_reason is not None


def test_a_rate_safe_judge_still_gets_its_corrected_rate(tmp_path, capsys):
    root = tmp_path / "store"
    seed_verdicts(root, 180, 20)
    judge = write_judge_file(tmp_path / "judge.json")
    store_calibration(root, certified_calibration())
    store_battery(root, clear_battery())
    assert _run(root, judge) == EXIT_OK
    out = capsys.readouterr().out
    assert "REFUSED" not in out
    (record,) = load_records_of(root, "run")
    (point,) = record.trend
    assert point.corrected_rate.point is not None
    assert point.corrected_rate.refusal_reason is None
