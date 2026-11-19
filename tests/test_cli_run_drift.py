# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""`calibrated-agent-evaluation run --drift-baseline STORE --drift-feature F`: input drift against a
baseline store.

Real local stores on both sides, the real `calibrated-agent-evaluation run` and the shipped
`input_drift`. Drift is passive unless `--fail-on-drift` is set, which gives a flag exit code 3
while a failing gate under `--ci` keeps exit 1.
"""

from __future__ import annotations

import pytest

from calibrated_agent_evaluation.cli import main
from calibrated_agent_evaluation.cli.dispatch import (
    EXIT_DOMAIN_ERROR,
    EXIT_DRIFT,
    EXIT_OK,
    EXIT_USAGE,
)
from calibrated_agent_evaluation.cli.records import load_records_of
from calibrated_agent_evaluation.run.drift_report import METHOD_JSD_COUNTS
from cli_store_helpers import (
    certified_calibration,
    clear_battery,
    owned_trace,
    seed_traces,
    store_battery,
    store_calibration,
    write_judge_file,
)

N = 60


@pytest.fixture(autouse=True)
def _no_ambient_root(monkeypatch):
    monkeypatch.delenv("CALIBRATED_AGENT_EVALUATION_LOCAL_ROOT", raising=False)


def _traces(prefix: str, *, chars: int, steps: int = 1, cells: tuple[str, ...] = ("refund",)):
    return [
        owned_trace(
            f"{prefix}{i:03d}",
            chars=chars + i % 20,
            steps=steps + i % 3,
            tool_calls=i % 4,
            cell={"feature": cells[i % len(cells)]},
        )
        for i in range(N)
    ]


@pytest.fixture
def baseline(tmp_path):
    root = tmp_path / "baseline"
    seed_traces(root, _traces("B", chars=80))
    return root


def _run(root, *argv: str) -> int:
    return main(["--root", str(root), "run", "--n-resamples", "199", *argv])


def _drift_args(baseline, feature: str = "input-chars") -> tuple[str, ...]:
    return ("--drift-baseline", str(baseline), "--drift-feature", feature)


def test_traffic_like_the_baseline_reports_no_drift_and_records_the_report(
    tmp_path, baseline, capsys
):
    root = tmp_path / "now"
    seed_traces(root, _traces("T", chars=80), n_fail=2)
    assert _run(root, "--ci", "--fail-on-drift", *_drift_args(baseline)) == EXIT_OK
    out = capsys.readouterr().out
    assert "Input drift on input-chars: none found (60 current traces against 60" in out

    (record,) = load_records_of(root, "run")
    drift = record.input_drift
    assert drift is not None
    assert drift.dimension == "input-chars"
    assert not drift.any_exceeded
    assert (drift.n_baseline, drift.n_fresh) == (N, N)
    assert record.inputs["drift_feature"] == "input-chars"


def test_drifted_traffic_is_reported_but_never_changes_the_gate_without_the_flag(
    tmp_path, baseline, capsys
):
    root = tmp_path / "now"
    seed_traces(root, _traces("T", chars=300), n_fail=2)
    assert _run(root, "--ci", *_drift_args(baseline)) == EXIT_OK
    out = capsys.readouterr().out
    assert "CI gate passed" in out
    assert "differ from the baseline by more than chance" in out
    (record,) = load_records_of(root, "run")
    assert record.gate_passed is True
    assert record.input_drift is not None
    assert record.input_drift.any_exceeded


def test_fail_on_drift_exits_three_when_the_gate_passes_and_drift_is_flagged(
    tmp_path, baseline, capsys
):
    root = tmp_path / "now"
    seed_traces(root, _traces("T", chars=300), n_fail=2)
    assert _run(root, "--ci", "--fail-on-drift", *_drift_args(baseline)) == EXIT_DRIFT
    err = capsys.readouterr().err
    assert "CI gate passed" in err
    assert "Exit code 3" in err


def test_a_failing_gate_under_ci_keeps_exit_one_even_with_drift_flagged(tmp_path, baseline, capsys):
    root = tmp_path / "now"
    seed_traces(root, _traces("T", chars=300), n_fail=20)
    assert _run(root, "--ci", "--fail-on-drift", *_drift_args(baseline)) == EXIT_DOMAIN_ERROR
    err = capsys.readouterr().err
    assert "CI gate failed" in err
    assert "Exit code 3" not in err


def test_the_taxonomy_cell_mix_is_always_compared_and_never_flags(tmp_path, baseline, capsys):
    root = tmp_path / "now"
    seed_traces(root, _traces("T", chars=80, cells=("refund", "exchange")), n_fail=2)
    assert _run(root, "--explain", *_drift_args(baseline)) == EXIT_OK
    out = capsys.readouterr().out
    assert "taxonomy-cell JSD" in out
    assert "over 2 cell(s)" in out
    (record,) = load_records_of(root, "run")
    jsd = next(m for m in record.input_drift.metrics if m.name == "jsd")
    assert jsd.method == METHOD_JSD_COUNTS
    assert jsd.value > 0.1
    assert jsd.exceeded is False


@pytest.mark.parametrize("feature", ["turns", "tool-calls"])
def test_step_and_tool_call_features_are_compared_on_owned_traces(
    tmp_path, baseline, feature, capsys
):
    root = tmp_path / "now"
    seed_traces(root, _traces("T", chars=80, steps=5), n_fail=2)
    assert _run(root, *_drift_args(baseline, feature)) == EXIT_OK
    (record,) = load_records_of(root, "run")
    assert record.input_drift is not None
    assert record.input_drift.dimension == feature
    flagged = record.input_drift.any_exceeded
    # The step counts moved from 1-3 to 5-7; tool calls did not move.
    assert flagged is (feature == "turns")


def test_too_few_traces_is_a_named_note_not_a_no_drift(tmp_path, baseline, capsys):
    root = tmp_path / "now"
    seed_traces(root, _traces("T", chars=80)[:20])
    assert _run(root, "--fail-on-drift", *_drift_args(baseline)) == EXIT_OK
    out = capsys.readouterr().out
    assert "Input drift on input-chars: not measured, too few traces to compare" in out
    assert "20 now" in out
    (record,) = load_records_of(root, "run")
    assert record.input_drift is None


def test_probed_traces_have_no_steps_so_the_step_feature_is_not_measured(
    tmp_path, baseline, capsys
):
    from cli_store_helpers import seed_verdicts

    root = tmp_path / "now"
    seed_verdicts(root, 58, 2)
    assert _run(root, "--explain", *_drift_args(baseline, "turns")) == EXIT_OK
    out = capsys.readouterr().out
    assert "60 current trace(s) have no turns" in out
    assert "not measured" in out


def test_a_drifted_admitted_judge_gets_a_staleness_warning_that_fails_with_the_flag(
    tmp_path, baseline, capsys
):
    root = tmp_path / "now"
    seed_traces(root, _traces("T", chars=300), n_fail=2)
    store_calibration(root, certified_calibration())
    store_battery(root, clear_battery())
    judge = write_judge_file(tmp_path / "judge.json")
    argv = ("--judge", str(judge), "--ci", "--fail-on-drift", *_drift_args(baseline))
    assert _run(root, *argv) == EXIT_DRIFT
    err = capsys.readouterr().err
    assert "Judge 'refund-judge' may need recalibrating: its inputs have drifted" in err
    (record,) = load_records_of(root, "run")
    (stale,) = [d for d in record.decisions if d.inputs.get("judge_id") == "refund-judge"]
    assert stale.tier == "warning"
    assert stale.inputs["drift_exceeded"] == "True"


def test_an_admitted_judge_on_undrifted_traffic_gets_no_staleness_warning(
    tmp_path, baseline, capsys
):
    root = tmp_path / "now"
    seed_traces(root, _traces("T", chars=80), n_fail=2)
    store_calibration(root, certified_calibration())
    store_battery(root, clear_battery())
    judge = write_judge_file(tmp_path / "judge.json")
    assert _run(root, "--judge", str(judge), "--fail-on-drift", *_drift_args(baseline)) == EXIT_OK
    assert "may need recalibrating" not in capsys.readouterr().out
    (record,) = load_records_of(root, "run")
    assert record.decisions == ()


def test_the_feature_is_required_with_a_baseline(tmp_path, baseline, capsys):
    root = tmp_path / "now"
    seed_traces(root, _traces("T", chars=80), n_fail=2)
    assert _run(root, "--drift-baseline", str(baseline)) == EXIT_USAGE
    assert "--drift-baseline needs --drift-feature" in capsys.readouterr().err
    assert load_records_of(root, "run") == []


def test_an_unknown_feature_is_a_usage_error(tmp_path, baseline):
    with pytest.raises(SystemExit) as exc:
        _run(tmp_path / "now", *_drift_args(baseline, "latency"))
    assert exc.value.code == 2


@pytest.mark.parametrize(
    ("argv", "message"),
    [
        (("--drift-feature", "turns"), "--drift-feature needs --drift-baseline"),
        (("--fail-on-drift",), "--fail-on-drift needs --drift-baseline"),
    ],
)
def test_drift_flags_without_a_baseline_are_refused(tmp_path, argv, message, capsys):
    root = tmp_path / "now"
    seed_traces(root, _traces("T", chars=80), n_fail=2)
    assert _run(root, *argv) == EXIT_USAGE
    assert message in capsys.readouterr().err


def test_a_baseline_that_is_not_a_directory_is_refused(tmp_path, capsys):
    root = tmp_path / "now"
    seed_traces(root, _traces("T", chars=80), n_fail=2)
    assert _run(root, *_drift_args(tmp_path / "missing")) == EXIT_DOMAIN_ERROR
    assert "is not a directory" in capsys.readouterr().err


def test_a_baseline_with_no_random_sample_is_refused(tmp_path, capsys):
    hunted = tmp_path / "hunted"
    seed_traces(hunted, [owned_trace(f"H{i}", frame="discovery") for i in range(N)], rater_id=None)
    root = tmp_path / "now"
    seed_traces(root, _traces("T", chars=80), n_fail=2)
    assert _run(root, *_drift_args(hunted)) == EXIT_DOMAIN_ERROR
    err = capsys.readouterr().err
    assert "none of them production-representative" in err
    assert load_records_of(root, "run") == []
