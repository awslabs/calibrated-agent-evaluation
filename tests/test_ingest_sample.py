# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""A sampled ingest draws reproducibly, tags the draw, and records how it was drawn."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from _helpers import TEST_STAMP, make_span, make_trace
from calibrated_agent_evaluation.backends.local import LocalBackend
from calibrated_agent_evaluation.cli.dispatch import EXIT_USAGE, main
from calibrated_agent_evaluation.cli.records import load_records_of
from calibrated_agent_evaluation.dumpload.results import results_table
from calibrated_agent_evaluation.ingest import IngestError, ingest_langfuse, run_ingest
from calibrated_agent_evaluation.ingest.sample import (
    SampleRecord,
    draw_sample,
    dump_samples,
    load_samples,
)
from calibrated_agent_evaluation.schema.types import Trace
from test_ingest_langfuse import _SourceBackend

_RAW = Path(__file__).resolve().parent.parent / "fixtures" / "raw"


def sample_record() -> SampleRecord:
    """A sample record with a fixed stamp; shared with the dump/load round-trip test."""
    return SampleRecord(
        sample_id="sample-fixture",
        source="langfuse:demo",
        population_size=100,
        sample_size=2,
        drawn=("T-1", "T-2"),
        trace_ids=("T-1", "T-2"),
        provenance=TEST_STAMP.model_copy(update={"seed": 4}),
    )


def _population(n: int, *, cell: dict[str, str] | None = None) -> list[Trace]:
    return [
        make_trace(
            trace_id=f"POP-{i:03d}",
            spans=[make_span(trace_id=f"POP-{i:03d}", span_id=f"POP-{i:03d}/root")],
            taxonomy_cell=dict(cell or {"feature": "refund"}),
        )
        for i in range(n)
    ]


class TestDraw:
    def test_the_same_seed_and_population_draw_the_same_sample(self) -> None:
        population = [f"T-{i}" for i in range(50)]
        first = draw_sample(population, size=8, seed=7)
        again = draw_sample(list(reversed(population)), size=8, seed=7)
        assert first == again
        assert len(set(first)) == 8

    def test_a_different_seed_draws_a_different_sample(self) -> None:
        population = [f"T-{i}" for i in range(200)]
        assert draw_sample(population, size=20, seed=1) != draw_sample(population, size=20, seed=2)

    @pytest.mark.parametrize(
        ("size", "population", "message"),
        [
            (0, ["T-1"], "at least one"),
            (3, ["T-1", "T-2"], "population holds 2"),
            (1, [], "population is empty"),
        ],
    )
    def test_an_impossible_draw_is_refused(
        self, size: int, population: list[str], message: str
    ) -> None:
        with pytest.raises(IngestError, match=message):
            draw_sample(population, size=size, seed=1)


class TestSampledPull:
    def test_a_sampled_pull_ingests_the_draw_and_records_it(self, tmp_path: Path) -> None:
        dest = LocalBackend(root=str(tmp_path / "store"))
        report = ingest_langfuse(
            _SourceBackend(_population(40)),
            backend=dest,
            sample_size=6,
            sample_seed=11,
            sample_id="sample-x",
            source_label="langfuse:demo",
        )

        assert report.exit_code == 0
        assert report.ingested == 6
        record = report.sample
        assert record is not None
        assert record.population_size == 40
        assert record.sample_size == 6
        assert record.provenance.seed == 11
        assert record.source == "langfuse:demo"
        assert sorted(record.trace_ids) == sorted(record.drawn)
        assert {t.trace_id for t in dest.query()} == set(record.drawn)

    def test_the_sampled_traces_are_tagged_production_representative(self, tmp_path: Path) -> None:
        dest = LocalBackend(root=str(tmp_path / "store"))
        population = _population(12)
        assert all(trace.frame == "discovery" for trace in population)

        ingest_langfuse(_SourceBackend(population), backend=dest, sample_size=4, sample_seed=3)

        stored = dest.query()
        assert len(stored) == 4
        assert {trace.frame for trace in stored} == {"production-representative"}

    def test_an_unsampled_pull_leaves_the_frame_and_writes_no_record(self, tmp_path: Path) -> None:
        dest = LocalBackend(root=str(tmp_path / "store"))
        report = ingest_langfuse(_SourceBackend(_population(5)), backend=dest)
        assert report.sample is None
        assert {trace.frame for trace in dest.query()} == {"discovery"}

    def test_a_cell_restricts_the_population_and_counts_what_it_left_out(
        self, tmp_path: Path
    ) -> None:
        refunds = _population(10, cell={"feature": "refund"})
        returns = [
            make_trace(
                trace_id=f"RET-{i}",
                spans=[make_span(trace_id=f"RET-{i}", span_id=f"RET-{i}/root")],
                taxonomy_cell={"feature": "returns"},
            )
            for i in range(5)
        ]
        dest = LocalBackend(root=str(tmp_path / "store"))
        report = ingest_langfuse(
            _SourceBackend([*refunds, *returns]),
            backend=dest,
            sample_size=3,
            sample_seed=5,
            cell={"feature": "refund"},
        )

        record = report.sample
        assert record is not None
        assert record.population_size == 10
        assert record.cell == {"feature": "refund"}
        assert record.excluded_unknown_cell == 5
        assert all(trace_id.startswith("POP-") for trace_id in record.drawn)

    def test_a_sample_without_a_seed_is_refused(self, tmp_path: Path) -> None:
        dest = LocalBackend(root=str(tmp_path / "store"))
        with pytest.raises(IngestError, match="needs a seed"):
            ingest_langfuse(_SourceBackend(_population(5)), backend=dest, sample_size=2)
        assert dest.query() == []

    def test_a_sample_larger_than_the_population_is_refused(self, tmp_path: Path) -> None:
        dest = LocalBackend(root=str(tmp_path / "store"))
        with pytest.raises(IngestError, match="population holds 3"):
            ingest_langfuse(
                _SourceBackend(_population(3)), backend=dest, sample_size=9, sample_seed=1
            )
        assert dest.query() == []


class TestSampledFiles:
    def _copies(self, directory: Path, n: int) -> None:
        source = (_RAW / "strands" / "CLM-4471.otlp.json").read_text(encoding="utf-8")
        for i in range(n):
            (directory / f"trace-{i:02d}.otlp.json").write_text(source, encoding="utf-8")

    def test_a_sampled_file_ingest_draws_and_tags(self, tmp_path: Path) -> None:
        sources = tmp_path / "exports"
        sources.mkdir()
        self._copies(sources, 8)
        dest = LocalBackend(root=str(tmp_path / "store"))

        report = run_ingest(
            sorted(sources.glob("*.otlp.json")),
            backend=dest,
            sample_size=3,
            sample_seed=42,
            sample_id="sample-files",
        )

        assert report.ingested == 3
        record = report.sample
        assert record is not None
        assert record.population_size == 8
        assert record.sample_size == 3
        assert len(record.drawn) == 3
        assert all(name.endswith(".otlp.json") for name in record.drawn)
        assert {trace.frame for trace in dest.query()} == {"production-representative"}

    def test_the_same_seed_ingests_the_same_files(self, tmp_path: Path) -> None:
        sources = tmp_path / "exports"
        sources.mkdir()
        self._copies(sources, 8)
        files = sorted(sources.glob("*.otlp.json"))

        first = run_ingest(
            files, backend=LocalBackend(":memory:"), sample_size=3, sample_seed=9
        ).sample
        again = run_ingest(
            files, backend=LocalBackend(":memory:"), sample_size=3, sample_seed=9
        ).sample
        assert first is not None
        assert again is not None
        assert first.drawn == again.drawn


class TestSampleRecordStore:
    def test_a_sample_record_round_trips_through_its_store(self, tmp_path: Path) -> None:
        record = sample_record()
        dump_samples([record], tmp_path, fmt="json")
        assert load_samples(tmp_path, fmt="json") == [record]

    def test_the_results_table_reports_the_sampling_fraction(self) -> None:

        record = SampleRecord(
            sample_id="sample-1",
            source="langfuse:demo",
            population_size=200,
            sample_size=50,
            drawn=tuple(f"T-{i}" for i in range(50)),
            trace_ids=tuple(f"T-{i}" for i in range(50)),
            cell={"feature": "refund"},
            provenance=TEST_STAMP.model_copy(update={"seed": 4}),
        )
        [row] = results_table([record])
        assert row.estimand == "sampling_fraction"
        assert row.point == pytest.approx(0.25)
        assert row.seed == 4
        assert row.source == "sample:sample-1"


class _PullSource(_SourceBackend):
    """A source that pulls like Langfuse: Calibrated Agent Evaluation traces normalized, a foreign
    one as OTLP."""

    def __init__(self, traces: list[Trace], foreign: list[dict[str, Any]]) -> None:
        super().__init__(traces)
        self._foreign = foreign

    def pull(self) -> Any:
        from calibrated_agent_evaluation.backends.base import SourceTrace

        for trace in self._traces:
            yield SourceTrace(source_id=trace.trace_id, trace=trace)
        for i, otlp in enumerate(self._foreign):
            yield SourceTrace(source_id=f"foreign-{i}", otlp=otlp)


class TestForeignTracesInAStratum:
    def test_a_foreign_trace_cannot_join_a_cell_stratum(self, tmp_path: Path) -> None:
        foreign = json.loads((_RAW / "strands" / "CLM-4473.otlp.json").read_text(encoding="utf-8"))
        source = _PullSource(_population(6), [foreign])
        dest = LocalBackend(root=str(tmp_path / "store"))

        report = ingest_langfuse(
            source,
            backend=dest,
            sample_size=2,
            sample_seed=2,
            cell={"feature": "refund"},
        )

        record = report.sample
        assert record is not None
        assert record.population_size == 6
        assert record.excluded_unknown_cell == 1
        assert all(not name.startswith("foreign-") for name in record.drawn)


class TestSampleCli:
    def test_the_cli_persists_the_sample_record_and_says_what_it_drew(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        sources = tmp_path / "exports"
        sources.mkdir()
        TestSampledFiles()._copies(sources, 6)
        root = tmp_path / "root"

        code = main(
            [
                "--root",
                str(root),
                "ingest",
                "--sample",
                "2",
                "--seed",
                "13",
                *[str(p) for p in sorted(sources.glob("*.otlp.json"))],
            ]
        )
        out = capsys.readouterr().out

        assert code == 0
        assert "sampled 2 of 6 trace(s)" in out
        assert "seed 13" in out
        assert "production-representative" in out
        [record] = load_records_of(root, "sample")
        assert record.sample_size == 2
        assert record.provenance.seed == 13

    @pytest.mark.parametrize(
        ("argv", "message"),
        [
            (["--sample", "2"], "--sample needs --seed"),
            (["--seed", "3"], "--seed only applies to --sample"),
            (["--sample", "0", "--seed", "1"], "at least one trace"),
            (["--cell", "feature=refund"], "--cell narrows a --sample draw"),
        ],
    )
    def test_an_incomplete_sample_request_is_refused(
        self,
        tmp_path: Path,
        capsys: pytest.CaptureFixture[str],
        argv: list[str],
        message: str,
    ) -> None:
        source = tmp_path / "trace.otlp.json"
        source.write_text(
            (_RAW / "strands" / "CLM-4471.otlp.json").read_text(encoding="utf-8"), encoding="utf-8"
        )
        code = main(["--root", str(tmp_path / "root"), "ingest", *argv, str(source)])
        captured = capsys.readouterr()
        assert code == EXIT_USAGE
        assert message in captured.out + captured.err

    def test_a_cell_sample_over_files_is_refused_with_the_reason(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        source = tmp_path / "trace.otlp.json"
        source.write_text(
            (_RAW / "strands" / "CLM-4471.otlp.json").read_text(encoding="utf-8"), encoding="utf-8"
        )
        code = main(
            [
                "--root",
                str(tmp_path / "root"),
                "ingest",
                "--sample",
                "1",
                "--seed",
                "1",
                "--cell",
                "feature=refund",
                str(source),
            ]
        )
        captured = capsys.readouterr()
        assert code == EXIT_USAGE
        assert "only known once it is normalized" in captured.out + captured.err
