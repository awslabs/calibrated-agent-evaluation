# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""`calibrated-agent-evaluation quickstart --certify`: the bundled certify set, its rebuild, and the
documented commands.

The set is real OrderBot runs frozen under fixtures/raw/quickstart/ with synthetic rule-derived
gold, in two hunted rounds. The kit ships two versions of one judge and replays one recorded run of
the judge model for each: v1 is certified and probed on round 1, and v2 (the pre-registered prompt
revision) on round 2 only. The documented command
sequence runs here offline, replay only, with AWS credentials pointed at a profile that does not
exist, and its outcomes are pinned: they are what the docs quote, so a change that moves them fails
here.
"""

from __future__ import annotations

import filecmp
import gzip
import json
import re
import sys
from pathlib import Path

import pytest

from calibrated_agent_evaluation._quickstart import (
    CASSETTE_FILE,
    JUDGE_FILES,
    JUDGE_ROUNDS,
    KIT_DIR,
    ROUND_KEY,
    documented_commands,
    load_certify_set,
    probes_dir,
)
from calibrated_agent_evaluation.backends.local import LocalBackend
from calibrated_agent_evaluation.cli import main
from calibrated_agent_evaluation.cli.dispatch import EXIT_DOMAIN_ERROR, EXIT_OK
from calibrated_agent_evaluation.cli.judgefile import load_judge_file

pytest.importorskip("scipy")

_REPO = Path(__file__).resolve().parent.parent
_CERTIFY = _REPO / "src" / "calibrated_agent_evaluation" / "_quickstart" / "data" / "certify"
# The rebuild test imports the generator from the repo-root corpus/ tree, as
# tests/test_strands_executor.py does.
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))


@pytest.fixture(autouse=True)
def _offline(monkeypatch):
    """No live model call can happen: replay only, and AWS points at a profile that is not there."""
    for var in ("AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY", "AWS_SESSION_TOKEN"):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setenv("AWS_PROFILE", "calibrated-agent-evaluation-test-no-such-profile")
    monkeypatch.setenv("CALIBRATED_AGENT_EVALUATION_CASSETTE_MODE", "replay")


def _run(capsys, root: Path, *argv: str) -> tuple[int, str]:
    code = main(["--root", str(root), *argv])
    captured = capsys.readouterr()
    return code, captured.out + captured.err


class TestTheBundledSet:
    def test_each_round_follows_the_rate_safe_plan(self):
        from calibrated_agent_evaluation.calibrate import plan_calibration
        from calibrated_agent_evaluation.calibrate.engine import GATE_THRESHOLD

        per_class = plan_calibration(0.90, GATE_THRESHOLD, rate_safe=True).per_class
        traces, labels = load_certify_set()
        rounds = {t.trace_id: t.taxonomy_cell.get(ROUND_KEY) for t in traces}
        for rnd in ("1", "2"):
            outcomes = [lb.outcome for lb in labels if rounds[lb.trace_id] == rnd]
            assert outcomes.count("FAIL") == outcomes.count("PASS") == per_class, rnd
        gold_ids = {lb.trace_id for lb in labels}
        hunted = {t.trace_id for t in traces if t.frame == "discovery"}
        production = {t.trace_id for t in traces if t.frame == "production-representative"}
        assert hunted == gold_ids
        assert len(production) == 200
        assert not production & gold_ids
        assert {rounds[t] for t in production} == {None}

    def test_every_gold_label_says_it_is_synthetic(self):
        _, labels = load_certify_set()
        assert {lb.rater_id for lb in labels} == {"synthetic-rule-oracle"}
        assert all(lb.critique.startswith("Synthetic gold (rule):") for lb in labels)

    def test_bias_probes_use_runs_outside_every_sample(self):
        traces, _ = load_certify_set()
        stored = {t.trace_id for t in traces}
        seen: set[str] = set()
        for rnd in JUDGE_ROUNDS.values():
            for path in (_CERTIFY / probes_dir(rnd)).glob("*.json"):
                probe = json.loads(path.read_text(encoding="utf-8"))
                for trial in probe["trials"]:
                    run = trial["trial_id"].split("-", 1)[1]
                    assert run not in stored
                    assert run not in seen  # no run probes two biases, or both rounds
                    seen.add(run)

    def test_round_two_probes_carry_the_preregistered_trial_count(self):
        for path in (_CERTIFY / probes_dir("2")).glob("*.json"):
            probe = json.loads(path.read_text(encoding="utf-8"))
            assert len(probe["trials"]) == 30, path.name
            assert all(t["trial_id"].split("-", 2)[1] == "R" for t in probe["trials"])

    def test_rebuild_from_the_frozen_runs_is_byte_identical(self, tmp_path):
        from corpus.generate.quickstart_set import build

        build(out=tmp_path)
        for name in ("traces.jsonl.gz", "gold.jsonl.gz", "manifest.json"):
            assert (tmp_path / name).read_bytes() == (_CERTIFY / name).read_bytes(), name
        for rnd in JUDGE_ROUNDS.values():
            cmp = filecmp.dircmp(tmp_path / probes_dir(rnd), _CERTIFY / probes_dir(rnd))
            assert not cmp.left_only
            assert not cmp.right_only
            assert not cmp.diff_files

    def test_the_two_versions_differ_only_in_prompt(self):
        v1, v2 = (load_judge_file(_CERTIFY / JUDGE_FILES[v]).judge for v in ("v1", "v2"))
        assert (v1.judge_id, v1.model) == (v2.judge_id, v2.model)
        assert (v1.prompt_version, v2.prompt_version) == ("v1", "v2")
        assert v1.prompt_template != v2.prompt_template

    def test_the_cassette_was_recorded_for_the_shipped_judges(self):
        doc = json.loads(gzip.decompress((_CERTIFY / (CASSETTE_FILE + ".gz")).read_bytes()))
        for name in JUDGE_FILES.values():
            assert doc["recorded_model"] == load_judge_file(_CERTIFY / name).judge.model


class TestTheDocumentedCommands:
    """The sequence `calibrated-agent-evaluation quickstart --certify` prints, with the recorded
    judges' outcomes."""

    def test_v1_is_refused_for_bias_and_v2_clears_both_gates(self, capsys, tmp_path):
        root = tmp_path / "store"
        code, out = _run(capsys, root, "quickstart", "--certify")
        assert code == EXIT_OK
        assert "Seeded 848 OrderBot refund traces" in out
        for rnd in ("1", "2"):
            assert (
                f"round {rnd} (label_round={rnd}): 324 hunted traces (discovery frame) carry "
                "gold, 162 FAIL for trusted_broken_policy_tool and 162 PASS"
            ) in out
        assert "The gold is SYNTHETIC" in out
        printed = [line.strip() for line in out.splitlines()]
        steps = documented_commands(root)
        for argv in steps:
            assert any(line.endswith(" ".join(argv)) for line in printed), argv
        calibrate_v1, bias_v1, calibrate_v2, bias_v2, judge_v2, run_v2 = steps

        code, out = _run(capsys, root, *calibrate_v1)
        assert code == EXIT_OK, out
        assert (
            "@v1) for trusted_broken_policy_tool on 324 gold label(s) (synthetic, from the sample "
            "policy rule)"
        ) in out
        assert "human gold" not in out
        assert "in cell label_round=1: CERTIFIED" in out
        assert "TPR 0.951 (95% lower bound 0.914, n=162 FAIL)" in out
        assert "TNR 0.988 (95% lower bound 0.963, n=162 PASS)" in out
        assert "rate-unsafe" not in out

        code, out = _run(capsys, root, *bias_v1)
        assert code == EXIT_DOMAIN_ERROR, out
        assert "BIASED, must-pass bias exceeded: length, sycophancy" in out
        # The placebos (the trial's first input asked again) never disagreed on these two, so the
        # flips are not the judge's sampling noise.
        assert "length (must-pass): 3 of 12 trial(s) disagreed, 0 of 12 placebo(s) did" in out
        assert "sycophancy (must-pass): 2 of 12 trial(s) disagreed, 0 of 12 placebo(s) did" in out

        code, out = _run(capsys, root, *calibrate_v2)
        assert code == EXIT_OK, out
        assert "in cell label_round=2: CERTIFIED" in out
        assert "TPR 0.969 (95% lower bound 0.938, n=162 FAIL)" in out
        assert "TNR 1.000 (95% lower bound 0.984, n=162 PASS)" in out
        # Perfect on the PASS class: the boundary floor (11 per class at 1.0) is met, so the record
        # can back a corrected rate.
        assert "rate-unsafe" not in out

        code, out = _run(capsys, root, *bias_v2)
        assert code == EXIT_OK, out
        assert "@v2): CLEAR" in out
        assert "length (must-pass): 0 of 30 trial(s) disagreed, 0 of 30 placebo(s) did" in out
        assert "sycophancy (must-pass): 0 of 30 trial(s) disagreed, 0 of 30 placebo(s) did" in out
        assert (
            "self_preference (must-pass): 2 of 30 trial(s) disagreed, 1 of 30 placebo(s) did" in out
        )

        code, out = _run(capsys, root, *judge_v2)
        assert code == EXIT_OK, out
        assert "judge 'orderbot-refund-policy' admitted" in out
        assert re.search(
            r"wrote 200 verdict\(s\) for trusted_broken_policy_tool on 200 stored trace\(s\) "
            r"in the production-representative frame: 38 FAIL, 162 PASS",
            out,
        ), out

        code, out = _run(capsys, root, *run_v2)
        # 162 of 200 verdicts pass, whose 95% lower bound is under the 0.80 gate.
        assert code == EXIT_DOMAIN_ERROR
        assert "CI gate over 200 judge verdicts" in out
        assert "FAILED, 1 of 1 mode(s) did not: trusted_broken_policy_tool" in out
        # The corrected rate lands on the gold rule's 39 of 200 (0.195).
        assert "corrected 0.196, 95% CI [0.142, 0.258]" in out
        assert "raw 0.190, 95% CI [0.142, 0.250]" in out

    def test_a_second_v1_calibration_on_round_one_is_refused(self, capsys, tmp_path):
        """The single-use rule the two rounds exist for: the same gold cannot be reused."""
        root = tmp_path / "store"
        assert _run(capsys, root, "quickstart", "--certify")[0] == EXIT_OK
        calibrate_v1 = documented_commands(root)[0]
        assert _run(capsys, root, *calibrate_v1)[0] == EXIT_OK
        code, out = _run(capsys, root, *calibrate_v1)
        assert code != EXIT_OK
        assert "this gold set was already used to test judge" in out


class TestJudgeFrameScope:
    def test_frame_scores_only_the_production_sample(self, capsys, tmp_path):
        """`calibrated-agent-evaluation judge --frame` leaves the hunted traces of both rounds
        unscored."""
        root = tmp_path / "store"
        assert _run(capsys, root, "quickstart", "--certify")[0] == EXIT_OK
        _, _, calibrate_v2, bias_v2, judge_v2, _ = documented_commands(root)
        assert _run(capsys, root, *calibrate_v2)[0] == EXIT_OK
        assert _run(capsys, root, *bias_v2)[0] == EXIT_OK
        assert _run(capsys, root, *judge_v2)[0] == EXIT_OK
        with LocalBackend(root=root) as backend:
            frames = {t.trace_id: t.frame for t in backend.query()}
            scored = {
                tid for tid in frames for lb in backend.get_scores(tid) if lb.kind == "llm-judge"
            }
        assert len(scored) == 200
        assert {frames[t] for t in scored} == {"production-representative"}
        assert (root / KIT_DIR / JUDGE_FILES["v2"]).exists()
