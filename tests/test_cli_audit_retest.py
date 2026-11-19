# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""``calibrated-agent-evaluation audit retest`` and ``report``: the improve verdict on a held-out
replay, and the claim.

The replay roots are built from their JSON descriptions per test (``_replay_fixture``). Claim
records are stored by ``calibrated-agent-evaluation audit abtest`` and ``recommendation`` over
``Stubber``; an arms read is stored from the stamped fixture window through a trace-source double.
No request leaves the process.
"""

from __future__ import annotations

import json
import shutil
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest

from _agentcore_capture import captured_run

pytest.importorskip("boto3")
pytest.importorskip("statsmodels")

from _arms_fixture import AB_TEST_ARN, MAIN, VERSIONS, Double, Window, deliver
from _rec_fixture import queue_recommendation, rec
from _replay_fixture import (
    GUARD,
    GUARD_JUDGE,
    H1,
    H1_JUDGED,
    H2,
    TARGET,
    TARGET_JUDGE,
    THIRD,
    as_run_manifest,
    build_records,
    build_root,
    copy_root,
)
from calibrated_agent_evaluation.agentcore import optimization
from calibrated_agent_evaluation.cli import main
from calibrated_agent_evaluation.cli.dispatch import EXIT_DOMAIN_ERROR, EXIT_OK, EXIT_USAGE
from calibrated_agent_evaluation.cli.records import append_records, load_records_of
from test_cli_audit import _claims, _run_abtest

_REPO = Path(__file__).resolve().parents[1]
_GOLDEN = _REPO / "fixtures" / "expected" / "audit"
_TEST_ID = "fixture-abtest-stopped"
_REC_ID = "fixture-rec-system-prompt"
# Before the stopped test's startedAt (2026-10-28T19:00Z) and the recommendations' createdAt.
_EARLY = datetime(2026, 10, 1, 9, 0, tzinfo=UTC)


@pytest.fixture(autouse=True)
def env(monkeypatch: pytest.MonkeyPatch) -> None:
    for key, value in {
        "AWS_ACCESS_KEY_ID": "testing",
        "AWS_SECRET_ACCESS_KEY": "testing",
        "AWS_DEFAULT_REGION": "us-west-2",
        "CALIBRATED_AGENT_EVALUATION_AGENTCORE_SETTLE_SECONDS": "0",
    }.items():
        monkeypatch.setenv(key, value)
    monkeypatch.delenv("AWS_PROFILE", raising=False)
    monkeypatch.delenv("CALIBRATED_AGENT_EVALUATION_LOCAL_ROOT", raising=False)


def _oracle(name: str) -> dict[str, Any]:
    return json.loads((_GOLDEN / name).read_text(encoding="utf-8"))


def _register(root: Path, frozen: datetime = _EARLY, metric: str = TARGET) -> str:
    from calibrated_agent_evaluation.cli.records import load_ledger, save_ledger
    from calibrated_agent_evaluation.improve import PreRegEntry, register_prereg

    entry = PreRegEntry(prereg_id=f"p-{metric}", target_metric=metric, success_threshold=0.10)
    ledger = register_prereg(load_ledger(root), entry, now=frozen)
    save_ledger(root, ledger)
    return ledger.entries[-1].frozen_at


def _store_arms(root: Path) -> str:
    from calibrated_agent_evaluation.agentcore.arms import read_arms

    read = read_arms(
        Double(deliver(MAIN)),
        ab_test_arn=AB_TEST_ARN,
        runtime="fixture-runtime",
        window=Window(),
        bundle_versions=VERSIONS,
    )
    append_records(root, "agentcore-arms", list(read.assignments))
    return read.read_id


class _H1:
    """The H1 roots, built once per test under ``tmp``."""

    def __init__(self, tmp: Path, name: str = H1) -> None:
        self.before = build_root(f"{name}/control", tmp / f"{name.replace('/', '-')}-control")
        self.after = build_root(f"{name}/treatment", tmp / f"{name.replace('/', '-')}-treatment")
        self.development = build_root(f"{H1}/development", tmp / "h1-development")

    def args(self) -> list[str]:
        return [
            "--before-root",
            str(self.before),
            "--after-root",
            str(self.after),
            "--development-root",
            str(self.development),
        ]


def _ab_root(tmp: Path, monkeypatch: pytest.MonkeyPatch, name: str = "root") -> Path:
    root = tmp / name
    assert _run_abtest(root, monkeypatch) == EXIT_OK
    _register(root)
    _store_arms(root)
    return root


def _retest(root: Path, roots: list[str], *extra: str, subject: list[str] | None = None) -> int:
    return main(
        [
            "--root",
            str(root),
            "audit",
            "retest",
            TARGET,
            *(subject or ["--ab-test-id", _TEST_ID]),
            *roots,
            *extra,
        ]
    )


def _improves(root: Path) -> list[Any]:
    return load_records_of(root, "improve")


def _audits(root: Path) -> list[Any]:
    return load_records_of(root, "agentcore-audit")


def _same_verdict(record: Any, oracle: dict[str, Any]) -> None:
    assert record.verdict == oracle["verdict"]
    assert record.paired.method == oracle["mcnemar"]["method"]
    assert record.paired.n_discordant == oracle["mcnemar"]["n_discordant"]
    assert record.paired.p_value == pytest.approx(oracle["mcnemar"]["p_value"], rel=1e-9)
    for key in ("point", "lo", "hi"):
        assert getattr(record.paired.effect, key) == pytest.approx(
            oracle["mcnemar"]["effect"][key], abs=1e-9
        )
    assert record.panel.method == "holm"
    assert record.panel.m == oracle["panel"]["m"]
    assert [s.name for s in record.panel.slices] == [s["name"] for s in oracle["panel"]["slices"]]
    assert [s.regression_p for s in record.panel.slices] == pytest.approx(
        [s["regression_p"] for s in oracle["panel"]["slices"]]
    )
    assert list(record.panel.regressed) == oracle["panel"]["regressed"]


# --- the paired re-test --------------------------------------------------------------------------


def test_audit_retest_records_the_improve_verdict_and_its_inputs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = _ab_root(tmp_path, monkeypatch)
    h1 = _H1(tmp_path)

    assert _retest(root, h1.args(), "--verdicts", "human") == EXIT_OK

    [improve] = _improves(root)
    _same_verdict(improve, _oracle("retest_improve.json"))
    [audit] = _audits(root)
    [claim] = _claims(root)
    assert audit.improve_id == improve.improve_id
    assert audit.claim_record_id == claim["record_id"]
    assert (audit.before_arm.version_id, audit.after_arm.version_id) == (
        VERSIONS["C"],
        VERSIONS["T1"],
    )
    assert len(audit.confirmation_prompt_digests) == 40
    assert audit.development_digest_count == 10
    assert audit.arms_read_id is not None
    assert audit.arm_counts["C.traces"] == 5
    assert audit.arm_counts["T1.traces"] == 4
    assert audit.verdict_source == "human"
    assert audit.target_success_threshold == pytest.approx(0.10)
    assert audit.target_rater == "human"
    assert audit.guard_modes == {GUARD: "human"}
    assert audit.unguarded_modes == (THIRD,)
    assert audit.before_provenance_sha256 != audit.after_provenance_sha256
    assert audit.confirmation_slice_id == improve.confirmation_slice_id


def test_audit_retest_reads_the_run_manifest_the_executor_leaves_in_a_replay_root(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from calibrated_agent_evaluation.agentcore.replay import RuntimeRunManifest, provenance_digest

    root = _ab_root(tmp_path, monkeypatch)
    h1 = _H1(tmp_path)
    before_log, after_log = as_run_manifest(h1.before), as_run_manifest(h1.after)

    assert _retest(root, h1.args(), "--verdicts", "human") == EXIT_OK

    [improve] = _improves(root)
    _same_verdict(improve, _oracle("retest_improve.json"))
    [audit] = _audits(root)
    assert (audit.before_provenance_sha256, audit.after_provenance_sha256) == (
        provenance_digest(RuntimeRunManifest(before_log).provenance()),
        provenance_digest(RuntimeRunManifest(after_log).provenance()),
    )


def test_a_failed_audit_write_names_the_stored_improve_record_and_the_decided_slice(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    import calibrated_agent_evaluation.cli.records as records

    root = _ab_root(tmp_path, monkeypatch)
    real = records.append_records

    def refuse_the_audit(where: Path, kind: str, new: Any, **kwargs: Any) -> int:
        if kind == "agentcore-audit":
            raise OSError("disk full")
        return real(where, kind, new, **kwargs)

    monkeypatch.setattr(records, "append_records", refuse_the_audit)
    capsys.readouterr()

    assert _retest(root, _H1(tmp_path).args(), "--verdicts", "human") == 1

    [improve] = _improves(root)
    assert _audits(root) == []
    err = capsys.readouterr().err
    assert f"improve record {improve.improve_id} is stored" in err
    assert "disk full" in err
    assert "now counts as decided" in err


def test_audit_retest_without_verdicts_reads_judge_verdicts_as_improve_does(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    root = _ab_root(tmp_path, monkeypatch)
    capsys.readouterr()

    assert _retest(root, _H1(tmp_path).args()) == EXIT_DOMAIN_ERROR

    assert "no judge verdicts for the target" in capsys.readouterr().err
    assert _improves(root) == []
    assert _audits(root) == []


def test_audit_retest_reaches_the_same_verdict_as_improve_evaluate(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from calibrated_agent_evaluation.improve import IMPROVE_SCHEMA_VERSION

    audit_root = _ab_root(tmp_path, monkeypatch, "audit-root")
    h1 = _H1(tmp_path)
    assert _retest(audit_root, h1.args(), "--verdicts", "human") == EXIT_OK

    improve_root = tmp_path / "improve-root"
    _register(improve_root)
    dev = tmp_path / "dev.txt"
    dev.write_text("\n".join(f"h1-d{i:02d}" for i in range(10)) + "\n", encoding="utf-8")
    before = copy_root(h1.before, tmp_path / "copy-before")
    after = copy_root(h1.after, tmp_path / "copy-after")
    code = main(
        [
            "--root",
            str(improve_root),
            "improve",
            "--evaluate",
            TARGET,
            "--before-root",
            str(before),
            "--after-root",
            str(after),
            "--development",
            str(dev),
            "--verdicts",
            "human",
        ]
    )
    assert code == EXIT_OK

    [via_audit] = _improves(audit_root)
    [via_improve] = _improves(improve_root)
    assert via_audit.verdict == via_improve.verdict
    assert via_audit.paired == via_improve.paired
    assert via_audit.panel == via_improve.panel
    assert via_audit.schema_version == via_improve.schema_version == IMPROVE_SCHEMA_VERSION


def test_an_improve_refusal_leaves_no_audit_record(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    root = tmp_path / "root"
    assert _run_abtest(root, monkeypatch) == EXIT_OK
    capsys.readouterr()

    assert _retest(root, _H1(tmp_path).args(), "--verdicts", "human") == EXIT_DOMAIN_ERROR

    assert "pre-regist" in capsys.readouterr().err
    assert _improves(root) == []
    assert _audits(root) == []
    assert not list((root / "records" / "agentcore-audit").glob("development/*.txt"))


def test_an_ab_retest_needs_a_development_root_that_holds_traces(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    root = _ab_root(tmp_path, monkeypatch)
    assert _run_abtest(root, monkeypatch) == EXIT_OK  # a second claim record for the test
    h1 = _H1(tmp_path)
    capsys.readouterr()
    roots = ["--before-root", str(h1.before), "--after-root", str(h1.after)]

    assert _retest(root, roots, "--verdicts", "human") == EXIT_USAGE
    err = capsys.readouterr().err
    assert "--development-root" in err
    assert _TEST_ID in err
    assert "--claim-record" not in err

    [first, _] = _claims(root)
    empty = tmp_path / "empty"
    empty.mkdir()
    args = [*roots, "--development-root", str(empty), "--claim-record", first["record_id"]]
    assert _retest(root, args, "--verdicts", "human") == EXIT_DOMAIN_ERROR
    err = capsys.readouterr().err
    assert "AgentCoreDevelopmentInputMissing" in err
    assert str(empty) in err
    assert "--development " not in err
    assert "temporary" not in err
    assert _improves(root) == []
    assert _audits(root) == []


@pytest.mark.parametrize(
    ("frozen", "order"),
    [
        (datetime(2026, 10, 28, 18, 0, tzinfo=UTC), ("before", "before")),
        (datetime(2026, 10, 29, 12, 0, tzinfo=UTC), ("after", "before")),
    ],
    ids=["before-start", "between-start-and-analysis"],
)
def test_the_pre_registration_order_is_recorded_not_enforced_for_ab_tests(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, frozen: datetime, order: tuple[str, str]
) -> None:
    root = tmp_path / "root"
    assert _run_abtest(root, monkeypatch) == EXIT_OK
    _register(root, frozen)
    _store_arms(root)

    assert _retest(root, _H1(tmp_path).args(), "--verdicts", "human") == EXIT_OK

    [audit] = _audits(root)
    assert (
        audit.prereg_order["ab_test_started_at"],
        audit.prereg_order["analysis_timestamp"],
    ) == order


def test_a_target_variant_test_cannot_be_retested(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    root = tmp_path / "root"
    assert _run_abtest(root, monkeypatch, "abtest_target.json") == EXIT_OK
    _register(root)
    [claim] = _claims(root)
    capsys.readouterr()

    code = _retest(
        root,
        _H1(tmp_path).args(),
        "--verdicts",
        "human",
        subject=["--ab-test-id", claim["ab_test_id"]],
    )

    assert code == EXIT_DOMAIN_ERROR
    err = capsys.readouterr().err
    assert "AgentCoreNotReplayable" in err
    [target] = [v for v in claim["variants"] if v["target_name"]]
    assert target["target_name"] in err
    assert _improves(root) == []


def test_agentcore_figures_do_not_move_the_calibrated_agent_evaluation_verdict(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    first = _ab_root(tmp_path, monkeypatch, "first")
    second = _ab_root(tmp_path, monkeypatch, "second")
    stored = second / "records" / "agentcore-abtest" / "agentcore_abtest_record.json"
    docs = json.loads(stored.read_text(encoding="utf-8"))
    for claim in docs[0]["evaluator_claims"]:
        claim["control"]["mean"], claim["variants"][0]["mean"] = (
            claim["variants"][0]["mean"],
            claim["control"]["mean"],
        )
        for v in claim["variants"]:
            v["is_significant"] = not v["is_significant"]
            v["p_value"] = 0.001
    stored.write_text(json.dumps(docs), encoding="utf-8")
    (stored.with_suffix(".csv")).unlink(missing_ok=True)

    h1 = _H1(tmp_path)
    assert _retest(first, h1.args(), "--verdicts", "human") == EXIT_OK
    again = [
        "--before-root",
        str(copy_root(h1.before, tmp_path / "b2")),
        "--after-root",
        str(copy_root(h1.after, tmp_path / "a2")),
        "--development-root",
        str(h1.development),
    ]
    assert _retest(second, again, "--verdicts", "human") == EXIT_OK

    [a] = _improves(first)
    [b] = _improves(second)
    assert (a.verdict, a.paired, a.panel, a.prereg.success_threshold) == (
        b.verdict,
        b.paired,
        b.panel,
        b.prereg.success_threshold,
    )


def test_a_held_out_slice_decides_one_change(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    root = _ab_root(tmp_path, monkeypatch)
    h1 = _H1(tmp_path)
    assert _retest(root, h1.args(), "--verdicts", "human") == EXIT_OK
    [first] = _audits(root)
    capsys.readouterr()

    assert _retest(root, h1.args(), "--verdicts", "human") == EXIT_DOMAIN_ERROR
    err = capsys.readouterr().err
    assert "AgentCoreSliceReused" in err
    assert first.audit_id in err
    assert "by prompt digest" in err

    def new_ids(doc: dict[str, Any]) -> None:
        for item in (*doc["traces"], *doc["labels"], *doc["run_provenance"]):
            item["trace_id"] = item["trace_id"].replace("h1-c", "h1-n")

    renamed = [
        "--before-root",
        str(build_root(f"{H1}/control", tmp_path / "renamed-before", edit=new_ids)),
        "--after-root",
        str(build_root(f"{H1}/treatment", tmp_path / "renamed-after", edit=new_ids)),
        "--development-root",
        str(h1.development),
    ]
    assert _retest(root, renamed, "--verdicts", "human") == EXIT_DOMAIN_ERROR
    assert "by prompt digest" in capsys.readouterr().err
    assert len(_improves(root)) == 1

    # An improve evaluation of the same slice, in a root with no audit record.
    other = _ab_root(tmp_path, monkeypatch, "other")
    dev = tmp_path / "dev.txt"
    dev.write_text("h1-d00\n", encoding="utf-8")
    before = copy_root(h1.before, tmp_path / "eval-before")
    after = copy_root(h1.after, tmp_path / "eval-after")
    assert (
        main(
            [
                "--root",
                str(other),
                "improve",
                "--evaluate",
                TARGET,
                "--before-root",
                str(before),
                "--after-root",
                str(after),
                "--development",
                str(dev),
                "--verdicts",
                "human",
            ]
        )
        == EXIT_OK
    )
    [evaluated] = _improves(other)
    capsys.readouterr()
    args = ["--before-root", str(before), "--after-root", str(after)]
    args += ["--development-root", str(h1.development)]
    assert _retest(other, args, "--verdicts", "human") == EXIT_DOMAIN_ERROR
    err = capsys.readouterr().err
    assert evaluated.improve_id in err
    assert f"by slice id {evaluated.confirmation_slice_id}" in err
    assert len(_improves(other)) == 1

    h2 = [
        "--before-root",
        str(build_root(f"{H2}/input", tmp_path / "h2-in")),
        "--after-root",
        str(build_root(f"{H2}/result", tmp_path / "h2-out")),
        "--development-root",
        str(h1.development),
    ]
    # The H2 replay runs the same two bundle versions, so it can re-test the A/B change too.
    assert _retest(root, h2, "--verdicts", "human") == EXIT_DOMAIN_ERROR  # rejected, recorded
    assert len(_audits(root)) == 2


# --- judge verdicts ------------------------------------------------------------------------------


def _judged(tmp: Path, monkeypatch: pytest.MonkeyPatch, records: str | None) -> tuple[Path, _H1]:
    root = _ab_root(tmp, monkeypatch)
    if records is not None:
        build_records(f"{H1_JUDGED}/{records}", root)
    return root, _H1(tmp, H1_JUDGED)


def _relabel(src: Path, dest: Path, change: Any) -> Path:
    """A copy of a built root with its labels rewritten by ``change(label) -> label | None``."""
    from calibrated_agent_evaluation.backends.local import LocalBackend

    tmp = dest.with_name(dest.name + "-src")
    shutil.copytree(src, tmp)
    with LocalBackend(root=tmp) as backend:
        traces = list(backend.query())
        scores = {t.trace_id: backend.get_scores(t.trace_id) for t in traces}
    with LocalBackend(root=dest) as backend:
        for t in traces:
            backend.put_trace(t)
            for label in scores[t.trace_id]:
                for new in change(label):
                    backend.put_score(new)
    shutil.rmtree(dest / "records", ignore_errors=True)
    shutil.copytree(tmp / "records", dest / "records")
    shutil.rmtree(tmp)
    return dest


def _args(before: Path, after: Path, h1: _H1) -> list[str]:
    return [
        "--before-root",
        str(before),
        "--after-root",
        str(after),
        "--development-root",
        str(h1.development),
    ]


def test_judge_verdicts_need_the_records_calibrated_agent_evaluation_judge_admits_a_judge_on(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    root, h1 = _judged(tmp_path / "ok", monkeypatch, "records-certified")
    assert _retest(root, h1.args(), "--verdicts", "judge") == EXIT_OK
    [improve] = _improves(root)
    _same_verdict(improve, _oracle("retest_improve.json"))
    assert improve.inputs["n_adjudicated"] == "15"
    [audit] = _audits(root)
    assert audit.target_rater == TARGET_JUDGE
    assert audit.guard_modes == {GUARD: GUARD_JUDGE}

    def drop_guard_bias(doc: dict[str, Any]) -> None:
        doc["bias"] = [b for b in doc["bias"] if b["judge_id"] != "guard-judge"]

    def biased_later(doc: dict[str, Any]) -> None:
        doc["bias"].append(
            {
                **doc["bias"][1],
                "battery_id": "bias-late",
                "verdict": "BIASED",
                "run_at": "2026-10-31T09:00:00+00:00",
            }
        )

    def refused_same_second(doc: dict[str, Any]) -> None:
        doc["calibration"].append(
            {**doc["calibration"][1], "calib_id": "calib-late", "result": "REFUSED"}
        )

    def restamped(doc: dict[str, Any]) -> None:
        for kind in ("calibration", "bias"):
            for r in doc[kind]:
                if r["judge_id"] == "guard-judge":
                    r["judge_fingerprint"] = "0d9c8b7a6f5e4d3c"

    cases = {
        "no calibration record": ("records-uncertified-guard", None),
        "no bias record": ("records-certified", drop_guard_bias),
        "BIASED": ("records-certified", biased_later),
        "REFUSED": ("records-certified", refused_same_second),
        "0d9c8b7a6f5e4d3c": ("records-certified", restamped),
    }
    for i, (expected, (records, edit)) in enumerate(cases.items()):
        case_root = _ab_root(tmp_path / f"case-{i}", monkeypatch)
        build_records(f"{H1_JUDGED}/{records}", case_root, edit=edit)
        case = _H1(tmp_path / f"case-{i}", H1_JUDGED)
        capsys.readouterr()
        assert _retest(case_root, case.args(), "--verdicts", "judge") == EXIT_DOMAIN_ERROR
        err = capsys.readouterr().err
        assert "AgentCoreUncertifiedJudge" in err, expected
        assert GUARD in err
        assert GUARD_JUDGE in err
        assert expected in err
        assert _improves(case_root) == []
        assert _audits(case_root) == []

    # A newer CERTIFIED record for another fingerprint of the guard judge changes nothing.
    def other_fingerprint(doc: dict[str, Any]) -> None:
        doc["calibration"].append(
            {
                **doc["calibration"][1],
                "calib_id": "calib-other",
                "judge_fingerprint": "ffff",
                "run_at": "2026-11-01T09:00:00+00:00",
            }
        )

    still = _ab_root(tmp_path / "still", monkeypatch)
    build_records(f"{H1_JUDGED}/records-certified", still, edit=other_fingerprint)
    assert _retest(still, _H1(tmp_path / "still", H1_JUDGED).args(), "--verdicts", "judge") == (
        EXIT_OK
    )

    # A guard label whose rater names no judge@fingerprint.
    for rater in (None, "guard-judge"):
        bad = tmp_path / f"rater-{rater}"
        case_root = _ab_root(bad, monkeypatch)
        build_records(f"{H1_JUDGED}/records-certified", case_root)
        case = _H1(bad, H1_JUDGED)

        def unrated(label: Any, rater: str | None = rater) -> list[Any]:
            if label.failure_mode == GUARD and label.trace_id == "h1-c00":
                return [label.model_copy(update={"rater_id": rater})]
            return [label]

        before = _relabel(case.before, bad / "relabelled", unrated)
        capsys.readouterr()
        assert _retest(case_root, _args(before, case.after, case), "--verdicts", "judge") == (
            EXIT_DOMAIN_ERROR
        )
        err = capsys.readouterr().err
        assert "h1-c00" in err
        assert GUARD in err
        assert str(before) in err
        assert repr(rater or "none") in err


def test_a_failure_mode_judged_by_two_raters_is_refused(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    other = "0d9c8b7a6f5e4d3c"

    def certify_other(judge: str) -> Any:
        def edit(doc: dict[str, Any]) -> None:
            for kind, idx in (("calibration", "calib_id"), ("bias", "battery_id")):
                mine = next(r for r in doc[kind] if r["judge_id"] == judge)
                doc[kind].append({**mine, idx: f"{mine[idx]}-other", "judge_fingerprint": other})

        return edit

    # The treatment store re-judged by the guard judge under another fingerprint.
    root = _ab_root(tmp_path / "guard", monkeypatch)
    build_records(f"{H1_JUDGED}/records-certified", root, edit=certify_other("guard-judge"))
    h1 = _H1(tmp_path / "guard", H1_JUDGED)

    def rejudged(label: Any) -> list[Any]:
        if label.failure_mode == GUARD and label.kind == "llm-judge":
            return [label.model_copy(update={"rater_id": f"guard-judge@{other}"})]
        return [label]

    after = _relabel(h1.after, tmp_path / "guard" / "rejudged", rejudged)
    capsys.readouterr()
    assert _retest(root, _args(h1.before, after, h1), "--verdicts", "judge") == EXIT_DOMAIN_ERROR
    err = capsys.readouterr().err
    assert GUARD in err
    assert f"{GUARD_JUDGE} in {h1.before}" in err
    assert f"guard-judge@{other} in {after}" in err

    # The treatment store keeping the target judge's labels and a second target label per trace.
    root = _ab_root(tmp_path / "target", monkeypatch)
    build_records(f"{H1_JUDGED}/records-certified", root, edit=certify_other("target-judge"))
    h1 = _H1(tmp_path / "target", H1_JUDGED)

    def both(label: Any) -> list[Any]:
        if label.failure_mode == TARGET and label.kind == "llm-judge":
            return [label, label.model_copy(update={"rater_id": f"target-judge@{other}"})]
        return [label]

    after = _relabel(h1.after, tmp_path / "target" / "both", both)
    capsys.readouterr()
    assert _retest(root, _args(h1.before, after, h1), "--verdicts", "judge") == EXIT_DOMAIN_ERROR
    err = capsys.readouterr().err
    assert TARGET in err
    assert f"{TARGET_JUDGE} in {h1.before}, {after}" in err
    assert f"target-judge@{other} in {after}" in err
    assert _improves(root) == []


def test_a_retest_without_verdicts_checks_the_judges_improve_reads(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    root, h1 = _judged(tmp_path / "uncertified", monkeypatch, "records-uncertified-guard")
    capsys.readouterr()
    assert _retest(root, h1.args()) == EXIT_DOMAIN_ERROR
    err = capsys.readouterr().err
    assert "AgentCoreUncertifiedJudge" in err
    assert GUARD_JUDGE in err
    assert str(h1.before) in err
    assert _improves(root) == []

    # The stage called with a namespace that has no `verdicts` attribute at all.
    import argparse

    from calibrated_agent_evaluation.cli.errors import (
        CliError,  # noqa: F401  (the boundary the namespace skips)
    )
    from calibrated_agent_evaluation.cli.stages.audit import audit_handler

    namespace = argparse.Namespace(
        action="retest",
        metric=TARGET,
        ab_test_id=_TEST_ID,
        recommendation_id=None,
        claim_record=None,
        arms_read=None,
        audit_id=None,
        before_root=str(h1.before),
        after_root=str(h1.after),
        development_root=str(h1.development),
        region=None,
        profile=None,
        root=str(root),
    )
    from calibrated_agent_evaluation.agentcore.errors import AgentCoreUncertifiedJudge

    with pytest.raises(AgentCoreUncertifiedJudge):
        audit_handler(namespace)

    ok, h1_ok = _judged(tmp_path / "certified", monkeypatch, "records-certified")
    assert _retest(ok, h1_ok.args()) == EXIT_OK
    [audit] = _audits(ok)
    assert audit.verdict_source == "judge"


# --- the report ----------------------------------------------------------------------------------


def _report(root: Path, *extra: str, subject: list[str] | None = None) -> int:
    return main(
        ["--root", str(root), "audit", "report", *(subject or ["--ab-test-id", _TEST_ID]), *extra]
    )


def _golden_report(root: Path) -> str:
    [audit] = _audits(root)
    [claim] = _claims(root)
    text = (_GOLDEN / "report.txt").read_text(encoding="utf-8")
    for placeholder, value in {
        "<improve-id>": audit.improve_id,
        "<audit-id>": audit.audit_id,
        "<frozen-at>": audit.prereg_frozen_at,
        "<read-at>": claim["read_at"],
        "<claim-record>": claim["record_id"],
        "<arms-read>": str(audit.arms_read_id),
    }.items():
        text = text.replace(placeholder, value)
    return text


def test_the_report_matches_the_golden(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    root = _ab_root(tmp_path, monkeypatch)
    assert _retest(root, _H1(tmp_path).args(), "--verdicts", "human") == EXIT_OK
    capsys.readouterr()

    assert _report(root) == EXIT_OK

    assert capsys.readouterr().out == _golden_report(root)


def test_the_report_makes_no_aws_call(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    from calibrated_agent_evaluation.agentcore.session import AwsClients

    root = _ab_root(tmp_path, monkeypatch)
    assert _retest(root, _H1(tmp_path).args(), "--verdicts", "human") == EXIT_OK

    def refuse(*args: Any, **kwargs: Any) -> Any:
        raise AssertionError("the report built an AWS client")

    monkeypatch.setattr(AwsClients, "from_profile", refuse)
    monkeypatch.setattr(optimization, "audit_clients", refuse)
    for key in ("AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY", "AWS_DEFAULT_REGION"):
        monkeypatch.delenv(key, raising=False)
    capsys.readouterr()

    assert _report(root) == EXIT_OK
    assert "Calibrated Agent Evaluation re-test" in capsys.readouterr().out


def test_a_report_takes_its_arms_read_and_counts_from_its_audit_record(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    from calibrated_agent_evaluation.agentcore.arms import read_arms
    from calibrated_agent_evaluation.agentcore.sources import TimeWindow

    root = tmp_path / "root"
    assert _run_abtest(root, monkeypatch) == EXIT_OK
    _register(root)
    main_read = _store_arms(root)
    later = TimeWindow(start="2026-10-28T19:30:00+00:00", end="2026-10-28T20:30:00+00:00")
    second = read_arms(
        Double(deliver(MAIN[:4])),
        ab_test_arn=AB_TEST_ARN,
        runtime="fixture-runtime",
        window=later,
        bundle_versions=VERSIONS,
    )
    append_records(root, "agentcore-arms", list(second.assignments))
    capsys.readouterr()

    # No audit record yet: the report needs the read named, and says no re-test is stored.
    assert _report(root) == EXIT_DOMAIN_ERROR
    err = capsys.readouterr().err
    assert main_read in err
    assert second.read_id in err
    assert _report(root, "--arms-read", second.read_id) == EXIT_OK
    out = capsys.readouterr().out
    assert f"traffic in arms read {second.read_id}" in out
    assert "No re-test is stored" in out
    assert "calibrated-agent-evaluation audit retest" in out

    args = [*_H1(tmp_path).args(), "--arms-read", main_read]
    assert _retest(root, args, "--verdicts", "human") == EXIT_OK
    [audit] = _audits(root)
    capsys.readouterr()
    assert _report(root) == EXIT_OK
    assert f"traffic in arms read {main_read}" in capsys.readouterr().out
    assert _report(root, "--arms-read", main_read) == EXIT_OK
    capsys.readouterr()
    assert _report(root, "--arms-read", second.read_id) == EXIT_DOMAIN_ERROR
    err = capsys.readouterr().err
    assert audit.audit_id in err
    assert main_read in err
    assert second.read_id in err
    assert capsys.readouterr().out == ""

    # The report prints the counts the audit record keeps; it recounts nothing.
    stored = root / "records" / "agentcore-audit" / "agentcore_audit_record.json"
    docs = json.loads(stored.read_text(encoding="utf-8"))
    docs[0]["arm_counts"]["C.traces"] = 99
    stored.write_text(json.dumps(docs), encoding="utf-8")
    stored.with_suffix(".csv").unlink(missing_ok=True)
    assert _report(root) == EXIT_OK
    assert "C: 99 trace(s)" in capsys.readouterr().out


def test_a_command_names_the_claim_record_when_several_are_stored(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    root = tmp_path / "root"
    assert _run_abtest(root, monkeypatch) == EXIT_OK
    assert _run_abtest(root, monkeypatch, "abtest_stopped_while_running.json") == EXIT_OK
    _register(root)
    _store_arms(root)
    stopped, running = sorted(_claims(root), key=lambda c: c["execution_status"], reverse=True)
    h1 = _H1(tmp_path)
    capsys.readouterr()

    for run in (lambda: _retest(root, h1.args(), "--verdicts", "human"), lambda: _report(root)):
        assert run() == EXIT_DOMAIN_ERROR
        err = capsys.readouterr().err
        for claim in (stopped, running):
            assert claim["record_id"] in err
            assert claim["read_at"] in err
        assert "STOPPED" in err
        assert "RUNNING" in err
    assert _improves(root) == []

    assert _report(root, "--claim-record", "abtest-nope") == EXIT_DOMAIN_ERROR
    assert "calibrated-agent-evaluation audit abtest" in capsys.readouterr().err
    assert _report(root, "--claim-record", stopped["record_id"]) == EXIT_OK
    out = capsys.readouterr().out
    assert "No re-test is stored" in out
    assert "execution STOPPED" in out

    args = [*h1.args(), "--claim-record", stopped["record_id"]]
    assert _retest(root, args, "--verdicts", "human") == EXIT_OK
    [audit] = _audits(root)
    assert audit.claim_record_id == stopped["record_id"]
    capsys.readouterr()
    assert _report(root) == EXIT_OK
    assert "execution STOPPED" in capsys.readouterr().out
    assert _report(root, "--claim-record", running["record_id"]) == EXIT_DOMAIN_ERROR
    err = capsys.readouterr().err
    assert audit.audit_id in err
    assert stopped["record_id"] in err
    assert running["record_id"] in err


def test_a_report_names_its_audit_record_when_several_are_stored(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    root = _ab_root(tmp_path, monkeypatch)
    assert _retest(root, _H1(tmp_path).args(), "--verdicts", "human") == EXIT_OK
    [first] = _audits(root)
    [improve] = _improves(root)
    append_records(root, "improve", [improve.model_copy(update={"improve_id": "improve-second"})])
    append_records(
        root,
        "agentcore-audit",
        [first.model_copy(update={"audit_id": "audit-second", "improve_id": "improve-second"})],
    )
    capsys.readouterr()

    assert _report(root) == EXIT_DOMAIN_ERROR
    err = capsys.readouterr().err
    assert f"{first.audit_id} (improve {first.improve_id})" in err
    assert "audit-second (improve improve-second)" in err
    assert capsys.readouterr().out == ""
    for audit_id, improve_id in (
        (first.audit_id, first.improve_id),
        ("audit-second", "improve-second"),
    ):
        assert _report(root, "--audit-id", audit_id) == EXIT_OK
        assert f"improve record {improve_id}" in capsys.readouterr().out

    assert _report(root, "--audit-id", "audit-nope") == EXIT_DOMAIN_ERROR
    assert "audit-nope" in capsys.readouterr().err
    stored = root / "records" / "improve" / "improve_record.json"
    if stored.exists():
        docs = [
            d
            for d in json.loads(stored.read_text(encoding="utf-8"))
            if d["improve_id"] != "improve-second"
        ]
        stored.write_text(json.dumps(docs), encoding="utf-8")
        stored.with_suffix(".csv").unlink(missing_ok=True)
        assert _report(root, "--audit-id", "audit-second") == EXIT_DOMAIN_ERROR
        assert "improve-second" in capsys.readouterr().err


# --- a recommendation ----------------------------------------------------------------------------


def _store_recommendation(
    root: Path, monkeypatch: pytest.MonkeyPatch, name: str = "rec_system_prompt.json", **over: Any
) -> int:
    real = optimization.audit_clients
    stubs: list[Any] = []
    doc = rec(name, **over)

    def stubbed(region: str | None = None, profile: str | None = None) -> Any:
        made = real(region or "us-west-2", profile)
        stubs.extend(queue_recommendation(made, doc))
        for stub in stubs:
            stub.activate()
        return made

    monkeypatch.setattr(optimization, "audit_clients", stubbed)
    try:
        return main(
            [
                "--root",
                str(root),
                "audit",
                "recommendation",
                "--recommendation-id",
                doc["recommendationId"],
            ]
        )
    finally:
        for stub in stubs:
            stub.deactivate()
        monkeypatch.setattr(optimization, "audit_clients", real)


def _h2(tmp: Path, dev: bool = True) -> list[str]:
    args = [
        "--before-root",
        str(build_root(f"{H2}/input", tmp / "h2-input")),
        "--after-root",
        str(build_root(f"{H2}/result", tmp / "h2-result")),
    ]
    if dev:
        args += ["--development-root", str(build_root(f"{H2}/development", tmp / "h2-dev"))]
    return args


def _rec_subject(rec_id: str = _REC_ID) -> list[str]:
    return ["--recommendation-id", rec_id]


def test_audit_recommendation_stores_the_claim(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    assert _store_recommendation(tmp_path, monkeypatch) == EXIT_OK
    out = capsys.readouterr().out
    assert "AgentCore reported (vendor claim, not evidence)" in out
    [stored] = load_records_of(tmp_path, "agentcore-recommendation")
    assert stored.recommendation_id == _REC_ID
    path = tmp_path / "records" / "agentcore-recommendation"
    assert "F036-CANARY-PROMPT" not in "".join(p.read_text() for p in path.glob("*"))

    real = optimization.audit_clients
    stubs: list[Any] = []

    def denied(region: str | None = None, profile: str | None = None) -> Any:
        from botocore.stub import Stubber

        made = real(region or "us-west-2", profile)
        stub = Stubber(made.client("bedrock-agentcore"))
        stub.add_client_error("get_recommendation", service_error_code="AccessDeniedException")
        stub.activate()
        stubs.append(stub)
        return made

    monkeypatch.setattr(optimization, "audit_clients", denied)
    code = main(
        ["--root", str(tmp_path), "audit", "recommendation", "--recommendation-id", "rec-denied"]
    )
    for stub in stubs:
        stub.deactivate()
    assert code == EXIT_DOMAIN_ERROR
    err = capsys.readouterr().err
    assert "GetRecommendation" in err
    assert "rec-denied" in err


def test_a_recommendation_is_retested_result_against_input(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "root"
    _register(root)
    assert _store_recommendation(root, monkeypatch) == EXIT_OK

    code = _retest(root, _h2(tmp_path), "--verdicts", "human", subject=_rec_subject())

    assert code == EXIT_DOMAIN_ERROR
    [improve] = _improves(root)
    oracle = _oracle("retest_recommendation_improve.json")
    _same_verdict(improve, oracle)
    assert any("does not clear the pre-registered reduction" in r for r in improve.reasons)
    [audit] = _audits(root)
    assert audit.subject == "recommendation"
    assert audit.arms_read_id is None
    assert audit.before_arm.version_id == VERSIONS["C"]
    assert audit.after_arm.version_id == VERSIONS["T1"]
    assert audit.prereg_order == {"recommendation_created_at": "before"}


@pytest.mark.parametrize(
    ("created_at", "refused"),
    [
        ("2026-10-01T08:59:59+00:00", True),
        ("2026-10-01T09:00:00+00:00", True),
        ("2026-10-01T09:00:01+00:00", False),
    ],
    ids=["frozen-a-second-after", "frozen-at-created", "frozen-a-second-before"],
)
def test_a_pre_registration_after_the_recommendation_started_is_refused(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    created_at: str,
    refused: bool,
) -> None:
    root = tmp_path / "root"
    # The recommendation is created a second before, at, or a second after the registration.
    frozen = _register(root, datetime.fromisoformat("2026-10-01T09:00:00+00:00"))
    assert _store_recommendation(root, monkeypatch, createdAt=created_at) == EXIT_OK
    capsys.readouterr()

    code = _retest(root, _h2(tmp_path), "--verdicts", "human", subject=_rec_subject())

    err = capsys.readouterr().err
    if refused:
        assert code == EXIT_DOMAIN_ERROR
        assert frozen in err
        assert created_at in err
        assert _improves(root) == []
    else:
        assert "not before the recommendation was created" not in err
        assert len(_improves(root)) == 1


@pytest.mark.parametrize(
    ("name", "over", "reason"),
    [
        ("rec_failed.json", {}, "INSUFFICIENT_TRACES"),
        ("rec_completed_while_in_progress.json", {}, "IN_PROGRESS"),
        ("rec_failed.json", {"status": "COMPLETED"}, "INSUFFICIENT_TRACES"),
        ("rec_inline_text.json", {}, "inline-text input"),
        ("rec_tool_inline.json", {}, "inline-text input"),
    ],
    ids=["failed", "in-progress", "error-code", "inline-text", "no-result-bundle"],
)
def test_recommendations_with_nothing_to_replay_are_refused(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    name: str,
    over: dict[str, Any],
    reason: str,
) -> None:
    root = tmp_path / "root"
    _register(root)
    assert _store_recommendation(root, monkeypatch, name, **over) == EXIT_OK
    [claim] = load_records_of(root, "agentcore-recommendation")
    capsys.readouterr()

    code = _retest(
        root, _h2(tmp_path), "--verdicts", "human", subject=_rec_subject(claim.recommendation_id)
    )

    assert code == EXIT_DOMAIN_ERROR
    err = capsys.readouterr().err
    assert "AgentCoreNotReplayable" in err
    assert reason in err
    assert _improves(root) == []


def test_a_result_without_a_bundle_is_refused(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    doc = rec("rec_system_prompt.json")
    result = doc["recommendationResult"]["systemPromptRecommendationResult"]
    result.pop("configurationBundle")
    root = tmp_path / "root"
    _register(root)
    assert (
        _store_recommendation(root, monkeypatch, recommendationResult=doc["recommendationResult"])
        == EXIT_OK
    )
    capsys.readouterr()

    code = _retest(root, _h2(tmp_path), "--verdicts", "human", subject=_rec_subject())

    assert code == EXIT_DOMAIN_ERROR
    err = capsys.readouterr().err
    assert "AgentCoreNotReplayable" in err
    assert "no result bundle" in err


def test_a_session_span_retest_without_a_development_root_is_refused_first(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    root = tmp_path / "root"
    # Not registered and FAILED: neither is looked at before the missing flag.
    assert _store_recommendation(root, monkeypatch, status="FAILED") == EXIT_OK
    capsys.readouterr()

    code = _retest(root, _h2(tmp_path, dev=False), "--verdicts", "human", subject=_rec_subject())

    assert code == EXIT_USAGE
    err = capsys.readouterr().err
    assert "--development-root" in err
    assert _REC_ID in err
    assert "FAILED" not in err
    assert _improves(root) == []


def test_a_development_trace_id_without_an_input_is_refused(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    root = tmp_path / "root"
    _register(root)
    assert _store_recommendation(root, monkeypatch, "rec_spans_missing.json") == EXIT_OK
    capsys.readouterr()
    args = _h2(tmp_path)
    code = _retest(
        root, args, "--verdicts", "human", subject=_rec_subject("fixture-rec-spans-missing")
    )
    assert code == EXIT_DOMAIN_ERROR
    err = capsys.readouterr().err
    assert "AgentCoreDevelopmentInputMissing" in err
    assert "sessionSpans" in err
    assert _improves(root) == []

    # A development root holding the same traces under re-keyed session ids names all 10.
    def rekeyed(doc: dict[str, Any]) -> None:
        for item in (*doc["traces"], *doc["labels"]):
            item["trace_id"] = f"s-{item['trace_id'][:8]}"

    other = tmp_path / "other"
    _register(other)
    assert _store_recommendation(other, monkeypatch) == EXIT_OK
    dev = build_root(f"{H2}/development", tmp_path / "rekeyed-dev", edit=rekeyed)
    capsys.readouterr()
    code = _retest(
        other,
        [*args[:4], "--development-root", str(dev)],
        "--verdicts",
        "human",
        subject=_rec_subject(),
    )
    assert code == EXIT_DOMAIN_ERROR
    assert "10 of the recommendation's sessionSpans" in capsys.readouterr().err


@pytest.mark.parametrize("name", ["rec_cloudwatch.json", "rec_online.json"])
def test_a_window_sourced_recommendation_refuses_a_development_root(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], name: str
) -> None:
    root = tmp_path / "root"
    _register(root)
    assert _store_recommendation(root, monkeypatch, name) == EXIT_OK
    [claim] = load_records_of(root, "agentcore-recommendation")
    built: list[Any] = []
    monkeypatch.setattr(optimization, "audit_clients", lambda *a, **k: built.append(a))
    capsys.readouterr()

    code = _retest(
        root, _h2(tmp_path), "--verdicts", "human", subject=_rec_subject(claim.recommendation_id)
    )

    assert code == EXIT_USAGE
    err = capsys.readouterr().err
    assert "--development-root" in err
    assert claim.traces_source in err
    assert built == []
    assert _improves(root) == []


def test_a_development_window_is_read_on_the_logs_insights_path_with_its_groups(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from botocore.stub import Stubber

    from _arms_fixture import stub_insights
    from calibrated_agent_evaluation.cli.stages.audit import _window_read

    captured = captured_run()
    run, spans, content = captured["trace_id"], captured["span_records"], captured["log_records"]
    service = spans[0]["resource"]["attributes"]["service.name"]
    window = ("2026-11-06T18:00:00+00:00", "2026-11-06T19:00:00+00:00")
    group = spans[0]["resource"]["attributes"]["aws.log.group.names"]

    def read(rows: list[dict[str, Any]]) -> tuple[Any, list[Any]]:
        with optimization.audit_clients("us-west-2") as clients:
            stub = Stubber(clients.client("logs"))
            stub_insights(stub, rows, content)
            with stub:
                source = _window_read(clients, service, window)
                delivered = list(source.pull())
            return source, delivered

    source, delivered = read(spans)
    assert source.groups_read() == (group, "aws/spans")
    [trace] = delivered
    assert trace.otlp is not None

    # A span naming two groups adds both; a span naming none adds none.
    from calibrated_agent_evaluation.agentcore.arms import RawTraceLog
    from calibrated_agent_evaluation.agentcore.sources import RawTrace

    def kept(rows: list[dict[str, Any]]) -> tuple[str, ...]:
        class _Reader:
            def read_traces(self) -> Any:
                yield RawTrace(trace_id=run, span_records=rows, log_records=[], session_ids=())

            def read_scores(self) -> Any:
                return iter(())

        log = RawTraceLog(_Reader())
        list(log.read_traces())
        return log.groups_read()

    two = [json.loads(json.dumps(s)) for s in spans]
    two[0]["resource"]["attributes"]["aws.log.group.names"] = [group, "/aws/fixture-second"]
    assert kept(two) == tuple(sorted((group, "/aws/fixture-second", "aws/spans")))
    none = [json.loads(json.dumps(s)) for s in spans]
    for s in none:
        s["resource"]["attributes"].pop("aws.log.group.names", None)
    assert kept(none) == ("aws/spans",)
