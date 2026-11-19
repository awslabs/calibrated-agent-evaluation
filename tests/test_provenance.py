# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Every persisted record carries what produced it, so its numbers can be recomputed.

A record that stores an interval but not the seed, resample count,
thresholds and library versions behind it cannot be reproduced from itself.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime, timedelta, timezone
from importlib.metadata import version
from pathlib import Path

import pytest

from _bias_helpers import run_complete_battery
from _helpers import make_label, rate_ids, rate_ids_by_mode
from calibrated_agent_evaluation._model import replace
from calibrated_agent_evaluation._provenance import (
    STATS_DISTRIBUTIONS,
    ProvenanceError,
    ProvenanceStamp,
    capture_provenance,
)
from calibrated_agent_evaluation.bias import (
    BiasBatteryRecord,
    BiasProbe,
    BiasThresholds,
    ProbeInput,
    ProbeVerdict,
    dump_battery,
    load_battery,
)
from calibrated_agent_evaluation.calibrate import (
    MIN_RATE_SAMPLE,
    InMemoryLockedTestLedger,
    dump_records,
    load_records,
)
from calibrated_agent_evaluation.calibrate.engine import (
    GATE_THRESHOLD,
    YOUDEN_FLOOR,
    certify,
    corrected_rate,
)
from calibrated_agent_evaluation.calibrate.record import (
    CalibrationRecord,
    CalibrationSet,
    LabelPair,
)
from calibrated_agent_evaluation.flagship import FlagshipReport
from calibrated_agent_evaluation.improve import ImproveRecord
from calibrated_agent_evaluation.run import ModeCounts, ModeKey, trend_report
from calibrated_agent_evaluation.run.report import RunRecord


@dataclass(frozen=True)
class _Judge:
    judge_id: str = "j"
    model: str = "bedrock/claude"
    prompt_version: str = "v1"


def _certified(set_id: str = "hunted") -> CalibrationRecord:
    pairs: list[LabelPair] = []
    for i in range(80):
        pairs.append(
            LabelPair(
                gold=make_label(trace_id=f"p{i}", kind="human", outcome="FAIL"),
                verdict=make_label(
                    trace_id=f"p{i}", kind="llm-judge", outcome="FAIL" if i < 72 else "PASS"
                ),
            )
        )
        pairs.append(
            LabelPair(
                gold=make_label(trace_id=f"n{i}", kind="human", outcome="PASS"),
                verdict=make_label(
                    trace_id=f"n{i}", kind="llm-judge", outcome="PASS" if i < 72 else "FAIL"
                ),
            )
        )
    locked = CalibrationSet(pairs=tuple(pairs), split="locked_test", set_id=set_id)
    return certify(_Judge(), locked, ledger=InMemoryLockedTestLedger())


EVAL = [1] * 148 + [0] * 852


def test_capture_records_the_environment_that_decides_the_digits() -> None:
    stamp = capture_provenance(seed=3, n_resamples=999, thresholds={"gate": 0.8})
    assert set(stamp.dep_versions) == set(STATS_DISTRIBUTIONS)
    assert all(stamp.dep_versions[d] for d in ("numpy", "scipy", "statsmodels"))
    assert stamp.seed == 3
    assert stamp.n_resamples == 999
    assert stamp.thresholds == {"gate": 0.8}
    assert datetime.fromisoformat(stamp.run_at).utcoffset() == timedelta(0)
    assert stamp.python_version.count(".") == 2
    # A git checkout knows its commit; an unpacked source tree or an installed wheel reports None.
    if not (Path(__file__).resolve().parents[1] / ".git").exists():
        assert stamp.git_sha is None
        return
    assert stamp.git_sha is not None
    sha, _, dirty = stamp.git_sha.partition("-")
    assert len(sha) == 40
    assert dirty in ("", "dirty")


def test_capture_converts_a_non_utc_clock_and_refuses_a_naive_one() -> None:
    plus_two = datetime(2026, 10, 13, 12, 0, tzinfo=timezone(timedelta(hours=2)))
    assert capture_provenance(now=plus_two).run_at == "2026-10-13T10:00:00+00:00"
    with pytest.raises(ProvenanceError, match="timezone"):
        capture_provenance(now=datetime(2026, 10, 13, 12, 0))  # noqa: DTZ001


_FIXED_NOW = datetime(2026, 10, 27, 9, 30, tzinfo=UTC)
# Any distribution every install has will do; a core dependency keeps the test off the extras.
_EXTRA = "pydantic"


def test_extra_distributions_are_recorded_in_dep_versions() -> None:
    stamp = capture_provenance(now=_FIXED_NOW, extra_distributions=(_EXTRA,))
    plain = capture_provenance(now=_FIXED_NOW)

    assert stamp.dep_versions[_EXTRA] == version(_EXTRA)
    assert set(stamp.dep_versions) == {*STATS_DISTRIBUTIONS, _EXTRA}
    assert set(plain.dep_versions) == set(STATS_DISTRIBUTIONS)
    assert {d: stamp.dep_versions[d] for d in STATS_DISTRIBUTIONS} == plain.dep_versions


def test_a_missing_extra_distribution_is_recorded_as_none() -> None:
    stamp = capture_provenance(
        now=_FIXED_NOW, extra_distributions=("calibrated-agent-evaluation-no-such-dist",)
    )

    assert stamp.dep_versions["calibrated-agent-evaluation-no-such-dist"] is None
    assert all(stamp.dep_versions[d] for d in STATS_DISTRIBUTIONS)


@pytest.mark.parametrize(
    ("field", "value", "match"),
    [
        ("run_at", "2026-10-13T12:00:00", "timezone"),
        ("run_at", "yesterday", "ISO-8601"),
        ("seed", True, "seed"),
        ("n_resamples", 0, "positive"),
        ("thresholds", {"gate": "high"}, "number"),
    ],
)
def test_a_malformed_stamp_is_refused(field: str, value: object, match: str) -> None:
    good = capture_provenance()
    with pytest.raises(ProvenanceError, match=match):
        replace(good, **{field: value})


def test_a_stamp_round_trips_and_a_truncated_one_is_refused() -> None:
    stamp = capture_provenance(seed=1, n_resamples=10, thresholds={"gate": 0.8})
    assert ProvenanceStamp.from_dict(stamp.to_dict()) == stamp
    doc = stamp.to_dict()
    del doc["seed"]
    with pytest.raises(ProvenanceError, match="seed"):
        ProvenanceStamp.from_dict(doc)


@pytest.mark.parametrize(
    "record_type", [CalibrationRecord, RunRecord, ImproveRecord, BiasBatteryRecord, FlagshipReport]
)
def test_provenance_is_a_required_field_on_every_persisted_record(record_type: type) -> None:
    fields = record_type.model_fields
    assert "provenance" in fields
    assert fields["provenance"].is_required()


def test_certify_stamps_the_gate_thresholds_and_no_seed() -> None:
    rec = _certified()
    assert rec.provenance.thresholds == {
        "gate": GATE_THRESHOLD,
        "youden_floor": YOUDEN_FLOOR,
        "min_rate_sample": float(MIN_RATE_SAMPLE),
    }
    assert rec.provenance.seed is None
    assert rec.provenance.n_resamples is None


def test_corrected_rate_stamps_its_seed_and_resample_count() -> None:
    out = corrected_rate(
        _certified(),
        EVAL,
        rate_set_id="prod",
        seed=7,
        n_resamples=499,
        rate_trace_ids=rate_ids(EVAL),
    )
    assert out.provenance.seed == 7
    assert out.provenance.n_resamples == 499
    assert out.provenance.thresholds == {
        "gate": GATE_THRESHOLD,
        "youden_floor": YOUDEN_FLOOR,
        "min_rate_sample": float(MIN_RATE_SAMPLE),
    }


def test_corrected_rate_defaults_are_the_shipped_bootstrap() -> None:
    out = corrected_rate(_certified(), EVAL, rate_set_id="prod", rate_trace_ids=rate_ids(EVAL))
    assert out.provenance.seed == 0
    assert out.provenance.n_resamples == 9999


def test_n_resamples_actually_reaches_the_bootstrap() -> None:
    a = corrected_rate(
        _certified("h1"),
        EVAL,
        rate_set_id="prod",
        seed=1,
        n_resamples=99,
        rate_trace_ids=rate_ids(EVAL),
    )
    b = corrected_rate(
        _certified("h2"),
        EVAL,
        rate_set_id="prod",
        seed=1,
        n_resamples=4999,
        rate_trace_ids=rate_ids(EVAL),
    )
    assert a.corrected_rate != b.corrected_rate


def test_a_stored_record_reproduces_its_own_interval(tmp_path: Path) -> None:
    """The Done-when: re-run the computation from the stored stamp and get the same digits."""
    original = corrected_rate(
        _certified("h1"),
        EVAL,
        rate_set_id="prod",
        seed=11,
        n_resamples=777,
        rate_trace_ids=rate_ids(EVAL),
    )
    dump_records([original], tmp_path, fmt="json")
    (loaded,) = load_records(tmp_path, fmt="json")
    assert loaded.provenance == original.provenance

    recomputed = corrected_rate(
        _certified("h1"),
        EVAL,
        rate_set_id="prod",
        seed=loaded.provenance.seed if loaded.provenance.seed is not None else -1,
        n_resamples=loaded.provenance.n_resamples or -1,
        rate_trace_ids=rate_ids(EVAL),
    )
    assert recomputed.corrected_rate == loaded.corrected_rate


def test_a_calibration_record_from_an_older_schema_is_refused(tmp_path: Path) -> None:
    import json

    from calibrated_agent_evaluation.calibrate.errors import CalibrationVersionError

    dump_records([_certified()], tmp_path, fmt="json")
    path = tmp_path / "calibration_record.json"
    docs = json.loads(path.read_text())
    docs[0]["schema_version"] = "1"
    del docs[0]["provenance"]
    path.write_text(json.dumps(docs))
    with pytest.raises(CalibrationVersionError):
        load_records(tmp_path, fmt="json")


def test_trend_report_threads_seed_and_resamples_into_every_mode() -> None:
    mode = ModeKey(failure_type="refund")
    counts = {mode: ModeCounts(mode=mode, n_total=len(EVAL), n_judged_fail=sum(EVAL))}
    report = trend_report(
        counts,
        {mode: _certified()},
        {mode: EVAL},
        seed=5,
        n_resamples=199,
        rate_trace_ids=rate_ids_by_mode({mode: EVAL}),
    )
    (point,) = report
    direct = corrected_rate(
        _certified("h3"),
        EVAL,
        rate_set_id="prod",
        seed=5,
        n_resamples=199,
        rate_trace_ids=rate_ids(EVAL),
    )
    assert point.corrected_rate == direct.corrected_rate


class _FairRunner:
    def run(self, judge_model: str, prompt_version: str, probe_input: ProbeInput) -> ProbeVerdict:
        return ProbeVerdict(label="PASS")


def test_a_bias_battery_stamps_the_thresholds_it_gated_on(tmp_path: Path) -> None:
    judge = _Judge()
    probe = BiasProbe(
        bias="length",
        is_must_pass=True,
        trials=tuple(
            (
                ProbeInput(bias="length", probe_id=f"{i}a", payload={}),
                ProbeInput(bias="length", probe_id=f"{i}b", payload={}),
            )
            for i in range(20)
        ),
    )
    thresholds = BiasThresholds(by_model={judge.model: {"length": 0.3}})
    record = run_complete_battery(judge, _FairRunner(), thresholds, [probe])
    assert record.provenance.thresholds == {
        "length": 0.3,
        **{r.bias: r.threshold for r in record.results if r.bias != "length"},
    }
    assert record.provenance.seed is None

    dump_battery([record], tmp_path)
    (loaded,) = load_battery(tmp_path)
    assert loaded.provenance == record.provenance


def test_stamps_are_utc_now() -> None:
    before = datetime.now(tz=UTC) - timedelta(seconds=5)
    stamp = capture_provenance()
    assert datetime.fromisoformat(stamp.run_at) >= before.replace(microsecond=0)
