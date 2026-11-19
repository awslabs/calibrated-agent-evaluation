# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""`calibrated-agent-evaluation run --publish local|langfuse`: the recorded run goes to a dashboard.

The publish runs after the run record is appended, so a failed publish exits 1 with the gate
verdict still printed and the run still recorded. These drive the real CLI, the real local sink,
and for Langfuse a real connection to a port nothing listens on; the live Langfuse proof is
`tests/integration/test_cli_run_publish_live.py`.
"""

from __future__ import annotations

import pytest

from calibrated_agent_evaluation.backends.dashboard import LocalDashboardSink
from calibrated_agent_evaluation.cli import main
from calibrated_agent_evaluation.cli.dispatch import EXIT_DOMAIN_ERROR, EXIT_OK
from calibrated_agent_evaluation.cli.records import load_records_of
from cli_store_helpers import seed_verdicts


@pytest.fixture
def root(tmp_path, monkeypatch):
    monkeypatch.delenv("CALIBRATED_AGENT_EVALUATION_LOCAL_ROOT", raising=False)
    root = tmp_path / "store"
    seed_verdicts(root, 48, 2)
    return root


def _run(root, *argv: str) -> int:
    return main(["--root", str(root), "run", "--n-resamples", "199", *argv])


def test_publish_local_writes_the_series_under_the_run_id(root, capsys):
    assert _run(root, "--ci", "--publish", "local", "--explain") == EXIT_OK
    out = capsys.readouterr().out
    (record,) = load_records_of(root, "run")
    where = root / "dashboard" / record.run_id
    assert f"published 0 rate and drift point(s) to {where} (none to chart" in out
    assert (where / "dashboard_series.csv").is_file()
    # No --judge, so no corrected rate: the series is empty, and still written.
    assert LocalDashboardSink(where).load() == []


def test_a_successful_publish_says_where_in_the_plain_output(root, capsys):
    assert _run(root, "--ci", "--publish", "local") == EXIT_OK
    out = capsys.readouterr().out
    (record,) = load_records_of(root, "run")
    assert f"Published the run to {root / 'dashboard' / record.run_id}." in out


def test_a_failed_local_publish_exits_one_with_the_verdict_and_the_run_recorded(root, capsys):
    (root / "dashboard").write_text("not a directory", encoding="utf-8")
    assert _run(root, "--ci", "--publish", "local") == EXIT_DOMAIN_ERROR
    err = capsys.readouterr().err
    (record,) = load_records_of(root, "run")
    assert "CI gate passed" in err
    assert f"Run recorded as {record.run_id}; publish failed:" in err
    assert record.gate_passed is True


def test_an_unreachable_langfuse_fails_the_publish_after_the_run_is_recorded(
    root, monkeypatch, capsys
):
    pytest.importorskip("langfuse")
    monkeypatch.setenv("LANGFUSE_HOST", "http://127.0.0.1:9")
    monkeypatch.setenv("LANGFUSE_PUBLIC_KEY", "pk-lf-nobody")
    monkeypatch.setenv("LANGFUSE_SECRET_KEY", "sk-lf-nobody")
    assert _run(root, "--publish", "langfuse") == EXIT_DOMAIN_ERROR
    err = capsys.readouterr().err
    (record,) = load_records_of(root, "run")
    assert "CI gate passed" in err
    assert f"Run recorded as {record.run_id}; publish failed:" in err
    assert "unreachable" in err
    # The local copy is written first and stays authoritative.
    assert (root / "dashboard" / record.run_id / "dashboard_series.json").is_file()


def test_publish_langfuse_without_keys_is_refused_before_the_run(root, monkeypatch, capsys):
    monkeypatch.delenv("LANGFUSE_PUBLIC_KEY", raising=False)
    monkeypatch.delenv("LANGFUSE_SECRET_KEY", raising=False)
    assert _run(root, "--publish", "langfuse") == EXIT_DOMAIN_ERROR
    assert "LANGFUSE_PUBLIC_KEY" in capsys.readouterr().err
    assert load_records_of(root, "run") == []


def test_a_failing_gate_with_a_good_publish_keeps_the_gate_exit(root, capsys):
    assert _run(root, "--ci", "--threshold", "0.95", "--publish", "local") == EXIT_DOMAIN_ERROR
    err = capsys.readouterr().err
    assert "CI gate failed" in err
    assert "publish failed" not in err


def test_published_drift_points_match_the_recorded_drift(tmp_path, monkeypatch, capsys):
    from cli_store_helpers import owned_trace, seed_traces

    monkeypatch.delenv("CALIBRATED_AGENT_EVALUATION_LOCAL_ROOT", raising=False)
    baseline, now = tmp_path / "baseline", tmp_path / "now"
    seed_traces(baseline, [owned_trace(f"B{i}", chars=80 + i % 20) for i in range(60)])
    seed_traces(now, [owned_trace(f"T{i}", chars=90 + i % 20) for i in range(60)], n_fail=2)
    argv = ("--drift-baseline", str(baseline), "--drift-feature", "input-chars")
    assert _run(now, "--publish", "local", *argv) == EXIT_OK
    (record,) = load_records_of(now, "run")
    series = LocalDashboardSink(now / "dashboard" / record.run_id).load()
    assert record.input_drift is not None
    published = {p.name: (p.value, p.p_value) for p in series}
    assert published == {
        f"drift.input-chars.{m.name}": (m.value, m.p_value) for m in record.input_drift.metrics
    }


def test_an_unknown_publish_target_is_a_usage_error(root):
    with pytest.raises(SystemExit) as exc:
        _run(root, "--publish", "grafana")
    assert exc.value.code == 2
