# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""`calibrated-agent-evaluation run --ci` gate-exit tests.

The CI gate reads the backend's stored verdicts, gates each mode on its pass-rate 95% lower bound,
and maps the decision to a process exit code: 0 when it passes, non-zero when a mode fails (so a
regression blocks CI).
"""

from __future__ import annotations

from calibrated_agent_evaluation.cli import main
from calibrated_agent_evaluation.cli.dispatch import EXIT_DOMAIN_ERROR, EXIT_OK
from calibrated_agent_evaluation.run import CiGateRequest, ModeKey, run_ci_gate
from calibrated_agent_evaluation.schema.types import Label


def test_ci_gate_fails_nonzero_when_a_mode_regresses(seeded_local_root):
    """Negative: the seeded gold corpus is all-FAIL, so the gate fails → non-zero exit."""
    rc = main(["run", "--ci", "--verdicts", "human"])
    assert rc == EXIT_DOMAIN_ERROR


def test_ci_gate_passes_exit_zero_over_passing_verdicts(tmp_path, monkeypatch):
    """Positive: when every gated mode clears its lower bound, `run --ci` exits 0."""
    from cli_store_helpers import seed_verdicts

    monkeypatch.delenv("CALIBRATED_AGENT_EVALUATION_LOCAL_ROOT", raising=False)
    root = tmp_path / "store"
    seed_verdicts(root, 48, 2)
    # Sanity: this input really does pass the shipped gate (guards against a vacuous test).
    mode = ModeKey(failure_type="over_refund")
    labels = [
        Label(
            trace_id=f"t{i}",
            span_id=None,
            kind="llm-judge",
            outcome="PASS" if i < 48 else "FAIL",
            critique="c",
            failure_mode="over_refund",
        )
        for i in range(50)
    ]
    gold: dict[tuple[str, str | None], ModeKey] = {(lb.trace_id, None): mode for lb in labels}
    assert run_ci_gate(CiGateRequest(labels=labels, gold=gold, threshold=0.80)).passed

    assert main(["--root", str(root), "run", "--ci", "--n-resamples", "199"]) == EXIT_OK


def test_without_ci_flag_reports_but_does_not_fail_process(seeded_local_root):
    """Without --ci a failing gate is reported but the process still exits 0 (cadence
    run)."""
    rc = main(["run", "--verdicts", "human"])
    assert rc == EXIT_OK


# --- a re-executed node is the same agent, statistically -------------------------------


def test_the_occurrence_suffix_does_not_split_one_agents_failure_mode() -> None:
    """A repair loop re-runs a node, so its culprit span may be `.../sql_generator#2`. Keying that
    as its own mode would split one agent's rate across buckets that are not different agents, and
    inflate the multiplicity correction with modes that do not exist."""
    from calibrated_agent_evaluation.cli.stages.run import _mode_for

    first = Label(
        trace_id="t",
        span_id=None,
        kind="human",
        outcome="FAIL",
        failure_mode="bad_sql",
        critique="c",
        culprit_span="t/sql_generator",
    )
    retry = Label(
        trace_id="t",
        span_id=None,
        kind="human",
        outcome="FAIL",
        failure_mode="bad_sql",
        critique="c",
        culprit_span="t/sql_generator#2",
    )
    assert _mode_for(first) == _mode_for(retry)
    assert _mode_for(retry).agent == "sql_generator"


def test_a_label_with_no_culprit_span_still_keys_on_the_failure_type_alone() -> None:
    from calibrated_agent_evaluation.cli.stages.run import _mode_for

    single = Label(
        trace_id="t",
        span_id=None,
        kind="human",
        outcome="FAIL",
        failure_mode="bad_sql",
        critique="c",
    )
    assert _mode_for(single) == ModeKey(failure_type="bad_sql", agent=None)
