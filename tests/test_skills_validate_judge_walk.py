# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""The validate-judge skill runs end to end on the `calibrated-agent-evaluation` commands it names.

The commands are read out of the skill's own fenced blocks, so a change to the skill is a change
to what runs here. Placeholders become the fixture files, and each command gets ``--root`` (and
``--cassette``, so no judge model is called). Two walks:

  * the fixture corpus, where the OrderBot judge has one gold label per class: calibrate is
    REFUSED and stops bias and judge, and every command the refusals tell the user to run is one
    the parser accepts;
  * a generated store of 162 FAIL and 162 PASS hunted gold that the fixture clean judge gets
    right on 160 of each, the rate-safe size, plus 160 production-representative traces it has
    not seen: CERTIFIED with no rate-unsafe line, bias CLEAR, judge admitted, and
    `calibrated-agent-evaluation run` reports a corrected rate for the mode from the production
    traces.
"""

from __future__ import annotations

import re
import shlex
from pathlib import Path

from calibrated_agent_evaluation.cli import main
from calibrated_agent_evaluation.cli.dispatch import EXIT_DOMAIN_ERROR, EXIT_OK
from calibrated_agent_evaluation.cli.parser import build_parser
from calibrated_agent_evaluation.cli.records import load_records_of
from test_cli_calibrate_bias import (
    BIAS_CASSETTE,
    CLEAN,
    ORDERBOT,
    ORDERBOT_CASSETTE,
    PROBES,
    THRESHOLDS,
    _gold_store,
    _replies,
    _verdict_cassette,
)
from test_skills_cli_surface import check_command

_SKILL = Path(__file__).resolve().parent.parent / "skills" / "validate-judge" / "SKILL.md"


def _skill_commands() -> list[list[str]]:
    """The `calibrated-agent-evaluation ...` lines of the skill's fenced bash blocks, in order,
    without `calibrated-agent-evaluation`."""
    text = _SKILL.read_text(encoding="utf-8")
    blocks = re.findall(r"```bash\n(.*?)```", text, re.DOTALL)
    lines = [ln.strip() for b in blocks for ln in b.splitlines()]
    return [shlex.split(ln)[1:] for ln in lines if ln.startswith("calibrated-agent-evaluation ")]


def _walk(root: Path, judge: Path, judge_cassette: Path, capsys) -> list[tuple[str, int, str]]:
    swap = {"judge.json": str(judge), "PROBES_DIR": str(PROBES), "THRESHOLDS_FILE": str(THRESHOLDS)}
    results = []
    for argv in _skill_commands():
        argv = [swap.get(a, a) for a in argv]
        stage = argv[0]
        extra = ["--cassette", str(BIAS_CASSETTE if stage == "bias" else judge_cassette)]
        if stage == "run":
            extra = ["--n-resamples", "199"]
        code = main(["--root", str(root), *argv, *extra])
        out = capsys.readouterr()
        results.append((stage, code, out.out + out.err))
    return results


def test_the_skill_names_the_four_stages_in_order():
    assert [a[0] for a in _skill_commands()] == ["calibrate", "bias", "judge", "run"]
    assert "--rate-safe" in _skill_commands()[0]


def test_on_the_fixture_corpus_the_walk_stops_at_an_honest_refusal(seeded_local_root, capsys):
    results = _walk(seeded_local_root, ORDERBOT, ORDERBOT_CASSETTE, capsys)
    stages = {stage: (code, text) for stage, code, text in results}
    code, text = stages["calibrate"]
    assert code == EXIT_DOMAIN_ERROR
    assert ": REFUSED" in text
    assert "rate-safe plan at expected accuracy 0.90: 162 gold label(s) per class" in text
    (record,) = load_records_of(seeded_local_root, "calibration")
    assert record.result == "REFUSED"
    for stage in ("bias", "judge"):
        assert stages[stage][0] == EXIT_DOMAIN_ERROR, stages[stage][1]
    assert load_records_of(seeded_local_root, "bias") == []
    # Every command a refusal tells the user to run next must be one the parser accepts.
    parser = build_parser()
    hints = [
        h
        for _, _, text in results
        for h in re.findall(r"`(calibrated-agent-evaluation [^`]+)`", text)
    ]
    assert hints
    for hint in hints:
        assert check_command(shlex.split(hint), parser) == [], hint


def test_on_a_rate_safe_gold_set_the_walk_certifies_clears_scores_and_corrects(tmp_path, capsys):
    from calibrated_agent_evaluation.backends.local import LocalBackend
    from cli_store_helpers import trace

    root = tmp_path / "store"
    traces = _gold_store(root, 162, 162)
    # The gold is hunted, and the rate is measured on traces the judge was not calibrated on: a
    # rate sample that reuses calibration gold is refused.
    production = [trace(f"P{i:03d}") for i in range(160)]
    with LocalBackend(root=root) as backend:
        for t in traces:
            backend.put_trace(t.model_copy(update={"frame": "discovery"}))
        for t in production:
            backend.put_trace(t)
    replies = [
        *_replies(traces, 162, 2),
        *((t, "fail" if i % 2 else "pass") for i, t in enumerate(production)),
    ]
    cassette = _verdict_cassette(tmp_path / "judge.cassette.json", CLEAN, replies)
    results = _walk(root, CLEAN, cassette, capsys)
    stages = {stage: (code, text) for stage, code, text in results}

    code, text = stages["calibrate"]
    assert code == EXIT_OK, text
    assert ": CERTIFIED" in text
    assert "this set has 162 FAIL and 162 PASS, enough" in text
    assert "rate-unsafe" not in text
    for hint in re.findall(r"`(calibrated-agent-evaluation [^`]+)`", text):
        assert check_command(shlex.split(hint), build_parser()) == [], hint

    code, text = stages["bias"]
    assert code == EXIT_OK, text
    assert ": CLEAR" in text

    code, text = stages["judge"]
    assert code == EXIT_OK, text
    assert "admitted" in text

    code, text = stages["run"]
    assert "RateUnsafeError" not in text
    assert "wrong_refund_decision" in text
    assert "corrected" in text
    assert len(load_records_of(root, "run")) == 1
    # Half the production sample is FAIL, so the gate's pass-rate bound is under 0.80 and --ci
    # blocks.
    assert code == EXIT_DOMAIN_ERROR
