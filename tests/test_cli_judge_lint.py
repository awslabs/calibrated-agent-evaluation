# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""A judge file is linted when it is loaded: refused scoring shapes are typed, plain refusals.

Every command that reads a judge file (`judge --check`, `calibrate`, `bias`, `judge`, `run
--judge`) goes through the same loader, so the checks run against `judge --check`, which reads no
store and calls no model, and once against `calibrate` to show the refusal comes before any work.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from calibrated_agent_evaluation.cli import main
from calibrated_agent_evaluation.cli.dispatch import EXIT_DOMAIN_ERROR, EXIT_OK
from calibrated_agent_evaluation.cli.errors import CliError
from calibrated_agent_evaluation.cli.judgefile import load_judge_file
from calibrated_agent_evaluation.enforce import (
    CritiqueOrderError,
    EnforcementError,
    LikertNotAllowedError,
    MultiAxisLikertError,
    WeightedCollapseError,
)

_REPO = Path(__file__).resolve().parent.parent
JUDGE_FILE = _REPO / "fixtures" / "judge-files" / "orderbot-refund-resolved.json"


def _write(tmp_path: Path, **change: object) -> Path:
    doc = {**json.loads(JUDGE_FILE.read_text(encoding="utf-8")), **change}
    path = tmp_path / "judge.json"
    path.write_text(json.dumps(doc), encoding="utf-8")
    return path


@pytest.mark.parametrize(
    ("change", "error"),
    [
        ({"scale": "1-5"}, LikertNotAllowedError),
        ({"scale": "likert"}, LikertNotAllowedError),
        ({"axes": [{"name": "helpful", "scale": "likert"}]}, LikertNotAllowedError),
        (
            {"axes": [{"name": "helpful", "scale": "1-5"}, {"name": "polite", "scale": "1-5"}]},
            MultiAxisLikertError,
        ),
        ({"weighted_collapse": True}, WeightedCollapseError),
        ({"weights": {"a": 0.5, "b": 0.5}}, WeightedCollapseError),
        (
            {"checklist": [{"name": "a", "critical": True, "weight": 0.7}, {"name": "b"}]},
            WeightedCollapseError,
        ),
        ({"output_fields": ["verdict", "critique"]}, CritiqueOrderError),
        ({"output_fields": ["verdict"]}, CritiqueOrderError),
    ],
)
def test_each_refused_shape_raises_its_own_enforcement_error(tmp_path, change, error):
    with pytest.raises(error) as info:
        load_judge_file(_write(tmp_path, **change))
    assert isinstance(info.value, EnforcementError)
    assert info.value.decision is not None
    assert info.value.decision.tier == "hard_refusal"


def test_an_anti_pattern_is_named_ahead_of_the_unknown_field_refusal(tmp_path):
    """`scale` is not a judge-file field, but the author learns it is Likert, not only that it is
    unknown."""
    with pytest.raises(LikertNotAllowedError):
        load_judge_file(_write(tmp_path, scale="1-5", rubric="x"))


def test_an_anti_pattern_is_named_ahead_of_a_missing_field(tmp_path):
    with pytest.raises(LikertNotAllowedError):
        load_judge_file(_write(tmp_path, scale="1-5", prompt_version=""))


def test_a_binary_scale_key_is_still_not_a_judge_file_field(tmp_path):
    with pytest.raises(CliError, match="one binary"):
        load_judge_file(_write(tmp_path, scale="binary"))


def test_several_binary_axes_are_refused_with_the_multi_criteria_nudge(tmp_path):
    """Several binary axes are only a warning to the library, but one judge file runs one."""
    with pytest.raises(CliError) as info:
        load_judge_file(_write(tmp_path, axes=[{"name": "a"}, {"name": "b"}]))
    assert not isinstance(info.value, EnforcementError)
    assert "one judge file per axis" in str(info.value)
    assert "single-axis" in str(info.value)


def test_an_unweighted_checklist_is_refused_as_inexpressible(tmp_path):
    with pytest.raises(CliError, match="checklist"):
        load_judge_file(_write(tmp_path, checklist=[{"name": "a", "critical": True}]))


def test_a_malformed_axes_value_is_a_typed_refusal(tmp_path):
    with pytest.raises(CliError, match="axes must be a list"):
        load_judge_file(_write(tmp_path, axes="helpful"))


def test_the_committed_judge_file_still_loads():
    assert load_judge_file(JUDGE_FILE).failure_mode


def test_the_cli_prints_the_plain_refusal_and_the_rule_under_explain(tmp_path, capsys, monkeypatch):
    monkeypatch.delenv("CALIBRATED_AGENT_EVALUATION_EXPLAIN", raising=False)
    path = _write(tmp_path, scale="1-5")
    assert main(["judge", "--check", str(path)]) == EXIT_DOMAIN_ERROR
    err = capsys.readouterr().err
    assert err.startswith("calibrated-agent-evaluation judge: The judge file scores on a Likert")
    assert "Next: rewrite the scale as one binary question" in err
    assert "(Run with --explain for the details.)" in err
    assert "LikertNotAllowedError" not in err

    assert main(["--explain", "judge", "--check", str(path)]) == EXIT_DOMAIN_ERROR
    explained = capsys.readouterr().err
    assert "LikertNotAllowedError: judge 'orderbot-refund-resolved' scores" in explained
    assert "ENF-" not in explained


def test_there_is_no_likert_opt_in_on_the_cli(tmp_path):
    path = _write(tmp_path, scale="1-5")
    with pytest.raises(SystemExit) as info:
        main(["judge", "--check", str(path), "--allow-likert"])
    assert info.value.code == 2


def test_calibrate_refuses_a_likert_file_before_reading_the_store(tmp_path, capsys):
    path = _write(tmp_path, scale="1-5")
    root = tmp_path / "root"
    rc = main(["--root", str(root), "calibrate", "--judge", str(path)])
    assert rc == EXIT_DOMAIN_ERROR
    assert "Likert" in capsys.readouterr().err
    assert not (root / "records").exists()


def test_a_clean_file_passes_the_check(capsys):
    assert main(["judge", "--check", str(JUDGE_FILE)]) == EXIT_OK
