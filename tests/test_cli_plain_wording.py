# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Plain CLI wording that used to mislead: absolute rate differences, synthetic gold, quickstart.

Each case here printed something an engineer could misread: "10%" for a ten-point drop, "human"
for synthetic gold, a quickstart that claimed to run stages it never ran, a rate refusal with no
next step, and a refusal that showed a CERTIFIED record while saying "not certified".
"""

from __future__ import annotations

import io
import subprocess
import sys
from pathlib import Path

from calibrated_agent_evaluation.cli import main, plain
from calibrated_agent_evaluation.cli.errors import GoldSetReusedError, JudgeNotCertifiedError
from calibrated_agent_evaluation.cli.present import Presenter

_REPO = Path(__file__).resolve().parent.parent


def test_an_absolute_reduction_reads_as_percentage_points() -> None:
    assert plain.points(0.10) == "10 percentage points"
    assert plain.points(0.02) == "2 percentage points"
    assert plain.points(0.025) == "2.5 percentage points"
    assert plain.points(0.01) == "1 percentage point"


def test_synthetic_fail_labels_are_not_called_human() -> None:
    assert plain.labeled_note(["synthetic-rule-oracle"] * 3, "FAIL label(s)") == (
        "3 FAIL label(s) (synthetic, from the sample policy rule)"
    )
    assert plain.labeled_note(["rater-a", "rater-b"], "FAIL label(s)") == "2 human FAIL label(s)"


def test_quickstart_does_not_claim_stages_it_did_not_run(tmp_path, capsys) -> None:
    assert main(["--root", str(tmp_path / "store"), "quickstart"]) == 0
    out = capsys.readouterr().out
    assert "probe -> define" not in out
    assert "nothing has been\njudged yet" in out


def test_a_too_small_rate_sample_names_the_next_step() -> None:
    reason = (
        "rate sample too small: the rate sample has 40 trace(s); a corrected rate needs at least "
        "150, so score 110 more production-representative trace(s) and rerun."
    )
    assert (
        "score 110 more production trace(s) with `calibrated-agent-evaluation judge`"
        in plain._refusal(reason)
    )


def _printed(exc: BaseException, capsys) -> str:
    out, err = io.StringIO(), io.StringIO()
    Presenter(out=out, err=err, explainers=plain.default_error_explainers()).error("bias", exc)
    return err.getvalue()


def test_a_judge_certified_for_another_version_says_so_plainly(capsys) -> None:
    exc = JudgeNotCertifiedError(
        "raw detail",
        reason="the newest certified record 'calib-1' is for m@v2, not this judge file",
        judge_path="judge-v1.json",
    )
    err = _printed(exc, capsys)
    assert (
        "This judge isn't certified yet: the newest certified record 'calib-1' is for m@v2" in err
    )
    assert "calibrated-agent-evaluation calibrate --judge judge-v1.json" in err
    assert "CliError:" not in err


def test_a_reused_gold_set_prints_a_plain_refusal(capsys) -> None:
    err = _printed(GoldSetReusedError("raw detail", judge_id="j"), capsys)
    assert "These labels were already used to test a judge for this failure mode" in err
    assert "Next: label fresh traces" in err


def test_a_reused_gold_set_refused_before_says_used_not_certified(capsys) -> None:
    err = _printed(GoldSetReusedError("raw", judge_id="j", prior_result="REFUSED"), capsys)
    assert "already used" in err
    assert "it refused that judge" in err
    assert "certified on" not in err


def test_a_partly_reused_gold_set_names_the_overlap(capsys) -> None:
    err = _printed(GoldSetReusedError("raw", judge_id="j", n_overlap=3), capsys)
    assert "3 of these labels were already used" in err


def test_the_id_guard_scans_the_review_ui_static_assets(tmp_path) -> None:
    js = "// see review-ui-integrity/06\n"  # allow-id: guard fixture
    css = "/* FR-005 */\n"  # allow-id: guard fixture
    (tmp_path / "app.js").write_text(js, encoding="utf-8")
    (tmp_path / "theme.css").write_text(css, encoding="utf-8")
    result = subprocess.run(
        [sys.executable, str(_REPO / "scripts" / "check_no_ids.py"), str(tmp_path)],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 1
    assert "app.js:1" in result.stdout
    assert "theme.css:1" in result.stdout


def test_a_rate_near_the_ends_is_not_rounded_to_never_or_always() -> None:
    assert plain.pct(0.11) == "11%"
    assert plain.pct(0.997) == "99.7%"
    assert plain.pct(0.9996) == "99.96%"
    assert plain.pct(0.004) == "0.4%"
    assert plain.pct(1.0) == "100%"
    assert plain.pct(0.0) == "0%"


def test_judge_output_counts_verdicts_and_the_traces_they_cover() -> None:
    whole = plain.judge(n_fail=2, n_pass=3, n_traces=5, mode="m", judge_path="j.json")
    assert whole[0] == "Wrote 5 verdict(s) for m on 5 trace(s): 2 FAIL, 3 PASS."
    spans = plain.judge(n_fail=2, n_pass=4, n_traces=2, mode="a/m", judge_path="j.json")
    assert spans[0] == "Wrote 6 verdict(s) for a/m on spans of 2 trace(s): 2 FAIL, 4 PASS."


class _Probe:
    def __init__(self, bias: str, status: str) -> None:
        self.bias, self.status, self.is_must_pass = bias, status, True


class _Battery:
    verdict = "CLEAR"
    failed_biases = ()

    def __init__(self, *probes: _Probe) -> None:
        self.results = probes


def test_a_clear_bias_check_names_must_pass_checks_that_did_not_run() -> None:
    battery = _Battery(_Probe("length", "run"), _Probe("self_preference", "not_applicable"))
    (head, _next) = plain.bias(battery, judge_path="j.json")
    assert "on the 1 must-pass check(s) that ran" in head
    assert "self_preference didn't run" in head
    assert "every must-pass check" not in head


class _Relocation:
    verdict = "relocation"


class _Panel:
    regressed = ()
    relocation = _Relocation()


class _Verdict:
    verdict = "rejected"
    panel = _Panel()

    class prereg:
        success_threshold = 0.10


def test_a_crew_fix_rejected_for_relocation_says_the_failures_moved() -> None:
    (head, _next) = plain.improve(_Verdict())
    assert "the failures moved to another step" in head
    assert "doesn't show the failure rate dropping" not in head
