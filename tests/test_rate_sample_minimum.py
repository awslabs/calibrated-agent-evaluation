# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""A corrected rate needs a rate sample of at least ``MIN_RATE_SAMPLE`` traces.

Below that size the interval was measured to under-cover at low prevalence, so ``corrected_rate``
refuses with the number of traces still needed, the trend keeps the mode with that reason, and
``calibrated-agent-evaluation run`` prints it.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import pytest

from _helpers import make_label, rate_ids, rate_ids_by_mode
from calibrated_agent_evaluation.calibrate import (
    MIN_RATE_SAMPLE,
    InMemoryLockedTestLedger,
    RateSampleTooSmallError,
    plan_calibration,
)
from calibrated_agent_evaluation.calibrate.engine import GATE_THRESHOLD, certify, corrected_rate
from calibrated_agent_evaluation.calibrate.record import (
    CalibrationRecord,
    CalibrationSet,
    LabelPair,
)
from calibrated_agent_evaluation.cli import main
from calibrated_agent_evaluation.cli.dispatch import EXIT_OK
from calibrated_agent_evaluation.cli.records import load_records_of
from calibrated_agent_evaluation.run import ModeCounts, ModeKey, trend_report
from cli_store_helpers import (
    certified_calibration,
    clear_battery,
    seed_verdicts,
    store_battery,
    store_calibration,
    write_judge_file,
)

_REPO = Path(__file__).resolve().parent.parent
_DERIVED = _REPO / "fixtures" / "expected" / "coverage" / "rate_sample_minimum.json"


@dataclass(frozen=True)
class _Judge:
    judge_id: str = "refund-judge"
    model: str = "bedrock/claude-sonnet-4"
    prompt_version: str = "v1"


def _rate_safe_record() -> CalibrationRecord:
    """Certified at 0.95 on 100 per class, above that accuracy's coverage floor of 80."""
    pairs = []
    for cls, flip in (("FAIL", "PASS"), ("PASS", "FAIL")):
        for i in range(100):
            said = flip if i < 5 else cls
            pairs.append(
                LabelPair(
                    gold=make_label(trace_id=f"{cls}-{i}", kind="human", outcome=cls),
                    verdict=make_label(trace_id=f"{cls}-{i}", kind="llm-judge", outcome=said),
                )
            )
    locked = CalibrationSet(pairs=tuple(pairs), split="locked_test", set_id="locked")
    record = certify(
        _Judge(),
        locked,
        plan=plan_calibration(0.95, GATE_THRESHOLD),
        ledger=InMemoryLockedTestLedger(),
    )
    assert record.result == "CERTIFIED"
    assert not record.below_coverage_floor
    return record


def _sample(n: int) -> list[int]:
    fails = n // 5
    return [1] * fails + [0] * (n - fails)


def test_the_minimum_matches_its_derivation_from_the_committed_cells():
    from calibrated_agent_evaluation.stats.studies.rate_sample_minimum import cells_from, derive

    doc = json.loads(_DERIVED.read_text(encoding="utf-8"))
    assert doc["minimum"] == MIN_RATE_SAMPLE
    assert derive(cells_from(doc))["minimum"] == MIN_RATE_SAMPLE


def test_one_trace_short_of_the_minimum_is_refused_with_the_count_still_needed():
    with pytest.raises(RateSampleTooSmallError) as exc:
        corrected_rate(
            _rate_safe_record(),
            _sample(MIN_RATE_SAMPLE - 1),
            rate_set_id="prod",
            rate_trace_ids=rate_ids(_sample(MIN_RATE_SAMPLE - 1)),
        )
    assert (exc.value.n, exc.value.minimum) == (MIN_RATE_SAMPLE - 1, MIN_RATE_SAMPLE)
    message = str(exc.value)
    assert f"has {MIN_RATE_SAMPLE - 1} trace(s)" in message
    assert f"needs at least {MIN_RATE_SAMPLE}" in message
    assert "score 1 more" in message


def test_a_sample_exactly_at_the_minimum_gets_a_corrected_rate():
    out = corrected_rate(
        _rate_safe_record(),
        _sample(MIN_RATE_SAMPLE),
        rate_set_id="prod",
        rate_trace_ids=rate_ids(_sample(MIN_RATE_SAMPLE)),
    )
    interval = out.corrected_rate
    assert interval is not None
    assert interval.point is not None
    assert interval.n == MIN_RATE_SAMPLE
    assert out.provenance.thresholds["min_rate_sample"] == MIN_RATE_SAMPLE


def test_clustered_verdicts_count_distinct_traces_not_verdicts():
    record = _rate_safe_record()
    short = [f"t{i // 2}" for i in range(2 * (MIN_RATE_SAMPLE - 1))]
    with pytest.raises(RateSampleTooSmallError) as exc:
        corrected_rate(
            record,
            _sample(len(short)),
            rate_set_id="prod",
            rate_cluster_ids=short,
            rate_trace_ids=rate_ids(_sample(len(short))),
        )
    assert exc.value.n == MIN_RATE_SAMPLE - 1
    enough = [f"t{i // 2}" for i in range(2 * MIN_RATE_SAMPLE)]
    out = corrected_rate(
        record,
        _sample(len(enough)),
        rate_set_id="prod",
        rate_cluster_ids=enough,
        rate_trace_ids=rate_ids(_sample(len(enough))),
    )
    assert out.corrected_rate is not None


def test_the_trend_keeps_a_mode_whose_rate_sample_is_too_small_with_the_reason():
    mode = ModeKey(failure_type="over_refund")
    (point,) = trend_report(
        {mode: ModeCounts(mode=mode, n_total=40, n_judged_fail=8)},
        {mode: _rate_safe_record()},
        {mode: _sample(40)},
        n_resamples=199,
        rate_trace_ids=rate_ids_by_mode({mode: _sample(40)}),
    )
    interval = point.corrected_rate
    assert interval.point is None
    assert (interval.lo, interval.hi, interval.n) == (0.0, 1.0, 40)
    assert interval.refusal_reason is not None
    assert interval.refusal_reason.startswith("rate sample too small:")
    assert "score 110 more" in interval.refusal_reason
    assert point.raw_rate.point == pytest.approx(8 / 40)


def test_calibrated_agent_evaluation_run_prints_the_refusal_and_records_it(
    tmp_path, monkeypatch, capsys
):
    monkeypatch.delenv("CALIBRATED_AGENT_EVALUATION_LOCAL_ROOT", raising=False)
    root = tmp_path / "store"
    seed_verdicts(root, 40, 10)
    judge = write_judge_file(tmp_path / "judge.json")
    store_calibration(root, certified_calibration())
    store_battery(root, clear_battery())
    code = main(["--root", str(root), "run", "--n-resamples", "199", "--judge", str(judge)])
    assert code == EXIT_OK
    out = capsys.readouterr().out
    assert "REFUSED (rate sample too small:" in out
    assert "score 100 more" in out
    (record,) = load_records_of(root, "run")
    assert record.trend[0].corrected_rate.refusal_reason is not None
    assert record.trend[0].corrected_rate.n == 50
