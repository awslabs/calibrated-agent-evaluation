# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the onboarding meta-commands: ``calibrated-agent-evaluation quickstart`` and
``calibrated-agent-evaluation doctor``.

These are the first-run on-ramp: quickstart must produce a real result from the bundled corpus with
zero setup (no extras, no network, no external files), and doctor must report the install honestly
without importing any vendor SDK. Both must exit 0 (they are demonstrations/diagnostics, not gates).
"""

from __future__ import annotations

import pytest

from calibrated_agent_evaluation._quickstart import load_sample_corpus
from calibrated_agent_evaluation.cli import main


class TestBundledCorpus:
    """The seed slice that ships inside the wheel (so an installed user has data to run on)."""

    def test_loads_six_traces_and_gold_labels(self):
        """The bundled corpus parses into schema objects via importlib.resources (no disk path)."""
        traces, labels = load_sample_corpus()
        assert len(traces) == 6
        assert len(labels) >= 6
        # Every gold label references a trace in the bundled set (no dangling refs).
        trace_ids = {t.trace_id for t in traces}
        assert {lb.trace_id for lb in labels} <= trace_ids

    def test_corpus_has_both_failures_and_a_passing_control(self):
        """The sample must show real signal: at least one FAIL and one PASS."""
        _traces, labels = load_sample_corpus()
        outcomes = {lb.outcome for lb in labels}
        assert "FAIL" in outcomes
        assert "PASS" in outcomes


class TestQuickstart:
    def test_quickstart_exits_zero_and_reports_a_real_result(self, capsys, tmp_path):
        """`calibrated-agent-evaluation quickstart` seeds the corpus and reports the failure count +
        modes, exit 0."""
        code = main(["quickstart", "--root", str(tmp_path / "store")])
        assert code == 0
        out = capsys.readouterr().out
        # Reports the seeded counts and a concrete first-pass result (5 of 6 traces fail).
        assert "Seeded 6 agent traces" in out
        assert "of 6 traces have a labeled failure" in out
        # Surfaces at least one real named failure mode from the gold labels.
        assert "trusted_broken_policy_tool" in out

    def test_each_failure_mode_counts_distinct_traces_not_labels(self, capsys, tmp_path):
        """A trace with two FAIL labels for one mode counts once, matching `cluster`."""
        from collections import defaultdict

        _traces, labels = load_sample_corpus()
        traces_per_mode: dict[str, set[str]] = defaultdict(set)
        for lb in labels:
            if lb.outcome == "FAIL" and lb.failure_mode:
                traces_per_mode[lb.failure_mode].add(lb.trace_id)
        main(["quickstart", "--root", str(tmp_path / "store")])
        out = capsys.readouterr().out
        for mode, trace_ids in traces_per_mode.items():
            assert f"    - {mode}: {len(trace_ids)}\n" in out
        assert "    - trusted_broken_policy_tool: 2\n" in out

    def test_the_printed_next_steps_reach_the_seeded_store(self, capsys, monkeypatch):
        """Following a printed step must find the sample, not an empty default store."""
        import shlex

        pytest.importorskip("scipy")
        monkeypatch.delenv("CALIBRATED_AGENT_EVALUATION_LOCAL_ROOT", raising=False)
        assert main(["quickstart"]) == 0
        out = capsys.readouterr().out
        lines = (line.strip().removeprefix("- ") for line in out.splitlines())
        steps = [
            shlex.split(line)
            for line in lines
            if line.startswith("calibrated-agent-evaluation --root ")
        ]
        runnable = {argv[3]: argv[1:] for argv in steps if argv[3] in ("cluster", "run")}
        assert set(runnable) == {"cluster", "run"}
        assert any(argv[3] == "label" for argv in steps)

        assert main(runnable["cluster"]) == 0
        assert "Found 4 failure mode(s)" in capsys.readouterr().out
        main(runnable["run"])
        gated = capsys.readouterr()
        assert "CI gate" in gated.out + gated.err
        assert "0 stored trace" not in gated.out + gated.err

    def test_quickstart_actually_persists_into_the_local_store(self, capsys, tmp_path):
        """The seed is real: after quickstart, the local backend at --root holds the traces."""
        root = tmp_path / "store"
        main(["quickstart", "--root", str(root)])
        from calibrated_agent_evaluation.backends.local import LocalBackend

        backend = LocalBackend(root=str(root))
        assert len(backend.query()) == 6

    def test_quickstart_points_at_next_steps(self, capsys, tmp_path):
        """The on-ramp names the next actions (doctor, the [ui]/[stats] extras)."""
        main(["quickstart", "--root", str(tmp_path / "store")])
        out = capsys.readouterr().out
        assert "calibrated-agent-evaluation doctor" in out
        from calibrated_agent_evaluation._install import install_hint

        assert install_hint("ui") in out
        assert install_hint("stats") in out


class TestDoctor:
    def test_doctor_exits_zero_and_reports_python_and_extras(self, capsys):
        """`calibrated-agent-evaluation doctor` reports the Python version and every optional extra,
        exit 0."""
        code = main(["doctor"])
        assert code == 0
        out = capsys.readouterr().out
        assert "Python:" in out
        # Every documented extra is named in the report.
        for extra in ("stats", "ui", "langfuse", "inspect", "strands", "bedrock"):
            assert f"[{extra}]" in out

    def test_doctor_reports_local_backend_health(self, capsys):
        """The offline default path is confirmed reachable."""
        main(["doctor"])
        out = capsys.readouterr().out
        assert "Local backend:" in out
        assert "offline" in out.lower()

    def test_doctor_shows_install_hint_for_a_missing_extra(self, capsys, monkeypatch):
        """A not-installed extra names what is missing and the exact `pip install` fix line."""
        import calibrated_agent_evaluation.cli.stages.doctor as doctor

        real = doctor._installed
        monkeypatch.setattr(doctor, "_installed", lambda dist: dist != "boto3" and real(dist))
        main(["doctor"])
        out = capsys.readouterr().out
        assert "  [ ] [bedrock]" in out
        assert "missing: boto3" in out
        from calibrated_agent_evaluation._install import install_hint

        assert f"enable: {install_hint('bedrock')}" in out

    def test_doctor_lists_every_extra_the_package_declares(self, capsys):
        """The list was hand-kept and omitted otel, synth and probe-ws."""
        from importlib.metadata import metadata

        declared = metadata("calibrated-agent-evaluation").get_all("Provides-Extra") or []
        assert declared
        main(["doctor"])
        out = capsys.readouterr().out
        for extra in declared:
            assert f"[{extra}]" in out, extra
        assert f"/{len(declared)} optional extras installed" in out

    def test_one_missing_distribution_marks_only_its_extras(self, capsys, monkeypatch):
        """statsmodels is in [stats] only, so [sampling], which shares scipy with it, stays
        installed: the two rows do not probe the same module."""
        import calibrated_agent_evaluation.cli.stages.doctor as doctor

        real = doctor._installed
        monkeypatch.setattr(doctor, "_installed", lambda d: d != "statsmodels" and real(d))
        main(["doctor"])
        out = capsys.readouterr().out
        assert "  [ ] [stats]" in out
        assert "  [x] [sampling]" in out

    def test_every_declared_extra_has_a_description(self):
        from calibrated_agent_evaluation.cli.stages.doctor import _UNLOCKS, extras_and_requirements

        assert set(extras_and_requirements()) == set(_UNLOCKS)


def test_meta_commands_do_not_bloat_the_pipeline_stage_list():
    """quickstart/doctor/ingest/capture/dataset/audit/online are meta-commands, NOT pipeline
    stages — STAGES stays the ten-stage pipeline+IO set (ingest/capture bring traces in, dataset
    moves test scenarios out, audit records a vendor's A/B claims and online manages AgentCore
    judges; none of them is a stage)."""
    from calibrated_agent_evaluation.cli.parser import META_COMMANDS, STAGES

    metas = ("quickstart", "doctor", "ingest", "capture", "dataset", "audit", "online")
    for meta in metas:
        assert meta not in STAGES
    assert set(META_COMMANDS) == set(metas)


@pytest.mark.parametrize("cmd", ["quickstart", "doctor"])
def test_meta_commands_are_registered_and_dispatch(cmd, capsys, tmp_path):
    """Both meta-commands parse and dispatch through the real `main` entry point."""
    argv = [cmd, "--root", str(tmp_path)] if cmd == "quickstart" else [cmd]
    assert main(argv) == 0
