# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""``calibrated-agent-evaluation dataset answers`` and ``build`` on the seeded fixture corpus."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

from _helpers import load_corpus_labels, load_corpus_trace
from calibrated_agent_evaluation.backends.local import LocalBackend
from calibrated_agent_evaluation.cli import main
from calibrated_agent_evaluation.cli.dispatch import EXIT_OK
from calibrated_agent_evaluation.scenarios.slicefile import read_slice, write_slice
from calibrated_agent_evaluation.scenarios.types import DatasetSlice, PublishedVersion
from calibrated_agent_evaluation.schema.types import Label
from cli_store_helpers import trace

FIXTURES = Path(__file__).resolve().parent.parent / "fixtures"
CORPUS = ("ORD-1001", "ORD-1002", "CLM-4471", "CLM-4473", "CLM-4475", "CLM-4477")
ANSWER_KEY = FIXTURES / "labels" / "answer-key.json"
ORACLE = FIXTURES / "expected" / "datasets" / "slice.json"


@pytest.fixture(autouse=True)
def _no_env_root(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("CALIBRATED_AGENT_EVALUATION_LOCAL_ROOT", raising=False)


@pytest.fixture
def corpus_root(tmp_path: Path) -> Path:
    """A local store holding the whole fixture corpus and its gold labels."""
    root = tmp_path / "store"
    with LocalBackend(root=root) as backend:
        for trace_id in CORPUS:
            backend.put_trace(load_corpus_trace(trace_id))
        for label in load_corpus_labels():
            backend.put_score(label)
    return root


def _dataset(root: Path, *argv: str) -> int:
    return main(["--root", str(root), "dataset", *argv])


def test_starter_key_lists_trace_level_failure_modes_only(corpus_root, tmp_path, capsys):
    out = tmp_path / "key.json"

    assert _dataset(corpus_root, "answers", "--out", str(out)) == EXIT_OK

    key = json.loads(out.read_text())
    assert list(key["by_failure_mode"]) == [
        "dropped_material_fact",
        "trusted_broken_policy_tool",
        "vacancy_threshold_margin_mishandled",
    ]
    assert all(
        entry == {"assertions": [], "expected_trajectory": []}
        for entry in key["by_failure_mode"].values()
    )
    assert key["by_trace"] == {}
    # misread_amount is only ever a span-level FAIL (CLM-4475's trace-level label is PASS).
    assert "misread_amount" not in key["by_failure_mode"]
    assert "Wrote an answer key for 3 failure mode(s)" in capsys.readouterr().out


def test_build_from_the_corpus_matches_the_oracle_slice(corpus_root, tmp_path, capsys):
    out = tmp_path / "slice.json"

    code = _dataset(corpus_root, "build", "--answers", str(ANSWER_KEY), "--out", str(out))

    assert code == EXIT_OK
    assert read_slice(out) == DatasetSlice.model_validate(json.loads(ORACLE.read_text()))
    ord_1001 = next(s for s in read_slice(out).scenarios if s.scenario_id == "ORD-1001")
    assert ord_1001.form == "predefined"
    gold = next(lb for lb in load_corpus_labels() if lb.trace_id == "ORD-1001")
    assert ord_1001.turns[0].input == load_corpus_trace("ORD-1001").input
    assert json.loads(ord_1001.turns[0].expected_response or "") == gold.expected
    printed = capsys.readouterr().out
    assert "Built 4 scenario(s)" in printed
    assert "2 with assertions, 2 with an expected tool order" in printed
    assert f"fill in their failure modes in {ANSWER_KEY}" in printed


def test_build_summary_reports_counts_and_left_out_traces(corpus_root, tmp_path, capsys):
    out = tmp_path / "slice.json"

    assert _dataset(corpus_root, "build", "--out", str(out)) == EXIT_OK

    printed = capsys.readouterr().out
    assert "0 with assertions, 0 with an expected tool order" in printed
    assert "4 scenario(s) have no assertions" in printed
    assert "calibrated-agent-evaluation dataset answers" in printed
    assert "Left out" not in printed


def test_build_summary_names_traces_it_left_out(corpus_root, tmp_path, capsys):
    with LocalBackend(root=corpus_root) as backend:
        backend.put_score(
            Label(
                trace_id="ORD-1001",
                span_id=None,
                kind="human",
                outcome="PASS",
                critique="A second reviewer thinks the refund was fine.",
                rater_id="second-reviewer",
            )
        )

    assert _dataset(corpus_root, "build", "--out", str(tmp_path / "s.json")) == EXIT_OK

    printed = capsys.readouterr().out
    assert "Built 3 scenario(s)" in printed
    assert "Left out 1 trace(s): 1 because their raters disagree on PASS or FAIL" in printed
    assert "ORD-1001" in printed


def test_failure_mode_filter_limits_scope(corpus_root, tmp_path):
    key_out, slice_out = tmp_path / "key.json", tmp_path / "slice.json"
    mode = "trusted_broken_policy_tool"

    assert _dataset(corpus_root, "answers", "--failure-mode", mode, "--out", str(key_out)) == 0
    assert _dataset(corpus_root, "build", "--failure-mode", mode, "--out", str(slice_out)) == 0

    assert list(json.loads(key_out.read_text())["by_failure_mode"]) == [mode]
    assert [s.scenario_id for s in read_slice(slice_out).scenarios] == ["CLM-4473", "ORD-1001"]


def test_filter_with_no_matching_failures_is_refused(corpus_root, tmp_path, capsys):
    out = tmp_path / "none.json"

    code = _dataset(corpus_root, "build", "--failure-mode", "misread_amount", "--out", str(out))

    assert code != EXIT_OK
    err = capsys.readouterr().err
    assert "'misread_amount'" in err
    assert "check the mode name" in err
    assert not out.exists()


def test_build_refuses_when_no_failure_is_in_scope(tmp_path, capsys):
    root = tmp_path / "store"
    with LocalBackend(root=root) as backend:
        backend.put_trace(trace("ORD-OK"))
        backend.put_score(
            Label(trace_id="ORD-OK", span_id=None, kind="human", outcome="PASS", critique="fine")
        )

    assert _dataset(root, "build", "--out", str(tmp_path / "s.json")) != EXIT_OK
    assert "nothing to build" in capsys.readouterr().err


@pytest.mark.parametrize(
    ("content", "named"),
    [('{"by_failure_mode": {}, "by_mode": {}}', "by_mode"), ('{"by_failure_mode": ', "line 1")],
)
def test_malformed_answer_keys_are_refused(corpus_root, tmp_path, capsys, content, named):
    key = tmp_path / "key.json"
    key.write_text(content)

    code = _dataset(corpus_root, "build", "--answers", str(key), "--out", str(tmp_path / "s.json"))

    assert code != EXIT_OK
    err = capsys.readouterr().err
    assert named in err
    assert str(key) in err


def test_existing_output_is_kept_unless_forced(corpus_root, tmp_path, capsys):
    out = tmp_path / "slice.json"
    out.write_text("keep me")

    assert _dataset(corpus_root, "answers", "--out", str(out)) != EXIT_OK
    assert _dataset(corpus_root, "build", "--out", str(out)) != EXIT_OK
    assert "already exists" in capsys.readouterr().err
    assert out.read_text() == "keep me"

    # A non-slice file is not silently replaced by `build --force` either...
    assert _dataset(corpus_root, "build", "--out", str(out), "--force") != EXIT_OK
    assert out.read_text() == "keep me"
    # ...but `answers --force` replaces it.
    assert _dataset(corpus_root, "answers", "--out", str(out), "--force") == EXIT_OK
    assert "by_failure_mode" in json.loads(out.read_text())


def test_rebuild_keeps_the_published_pointer(corpus_root, tmp_path):
    out = tmp_path / "slice.json"
    assert _dataset(corpus_root, "build", "--out", str(out)) == EXIT_OK
    pointer = PublishedVersion(
        dataset_arn="arn:aws:bedrock-agentcore:us-west-2:111122223333:dataset/orderbot-AbCdEf1234",
        dataset_id="orderbot-AbCdEf1234",
        dataset_name="orderbot_regressions",
        version="1",
        region="us-west-2",
        content_digest="0" * 64,
        published_at="2026-10-27T19:00:00Z",
    )
    published = read_slice(out).model_copy(update={"published": pointer})
    write_slice(out, published, force=True)

    code = _dataset(
        corpus_root, "build", "--answers", str(ANSWER_KEY), "--out", str(out), "--force"
    )

    assert code == EXIT_OK
    rebuilt = read_slice(out)
    assert rebuilt.published == pointer
    assert sum(1 for s in rebuilt.scenarios if s.assertions) == 2


@pytest.mark.parametrize(
    ("argv", "named"),
    [
        (["build", "--out", "s.json", "--yes"], "--yes"),
        (["answers", "--out", "k.json", "--answers", "k2.json"], "--answers"),
        (["build"], "--out"),
    ],
)
def test_flags_an_action_does_not_take_are_refused(corpus_root, capsys, argv, named):
    assert _dataset(corpus_root, *argv) != EXIT_OK
    assert named in capsys.readouterr().err


def test_offline_commands_run_without_the_agentcore_extra(corpus_root, tmp_path, monkeypatch):
    for module in ("boto3", "botocore", "bedrock_agentcore"):
        monkeypatch.setitem(sys.modules, module, None)

    assert _dataset(corpus_root, "answers", "--out", str(tmp_path / "k.json")) == EXIT_OK
    assert _dataset(corpus_root, "build", "--out", str(tmp_path / "s.json")) == EXIT_OK


def test_trace_flag_builds_only_that_trace(corpus_root, tmp_path):
    out = tmp_path / "slice.json"

    assert _dataset(corpus_root, "build", "--trace", "ORD-1001", "--out", str(out)) == EXIT_OK

    assert [s.scenario_id for s in read_slice(out).scenarios] == ["ORD-1001"]


# --- publish and pull, against a stubbed AgentCore account -----------------------------------


@pytest.fixture
def aws(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    """Route the CLI's AWS clients to a stubbed account, recording the profile and region."""
    from agentcore_dataset_stubs import Account

    for var in ("AWS_PROFILE", "AWS_REGION", "AWS_DEFAULT_REGION"):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setenv("AWS_CONFIG_FILE", str(tmp_path / "no-aws-config"))
    asked: dict[str, object] = {}
    holder: dict[str, Account] = {}

    def from_profile(cls, *, region=None, profile=None, handlers=(), **_kw):
        asked.update(region=region, profile=profile)
        account = holder.get("account") or Account(region=region)
        holder["account"] = account
        return account.clients

    monkeypatch.setattr(
        "calibrated_agent_evaluation.agentcore.session.AwsClients.from_profile",
        classmethod(from_profile),
    )

    class Handle:
        def __init__(self) -> None:
            self.asked = asked

        def account(self, region: str | None = "us-west-2") -> Account:
            holder["account"] = Account(region=region)
            return holder["account"]

    return Handle()


@pytest.fixture
def slice_file(tmp_path: Path) -> Path:
    from agentcore_dataset_stubs import a_slice

    path = tmp_path / "slice.json"
    write_slice(path, a_slice(2))
    return path


def test_plan_without_yes_writes_nothing(aws, slice_file, capsys):
    account = aws.account()
    account.no_datasets()
    before = slice_file.read_text()

    code = main(
        [
            "dataset",
            "publish",
            str(slice_file),
            "--name",
            "orderbot_regressions",
            "--region",
            "us-west-2",
        ]
    )

    assert code == EXIT_OK
    out = capsys.readouterr().out
    assert "creates a new dataset 'orderbot_regressions'" in out
    assert "dataset: orderbot_regressions in us-west-2" in out
    assert "scenarios: 2 (1 with assertions, 2 with an expected tool order)" in out
    assert "Nothing was written" in out
    assert slice_file.read_text() == before
    account.done()


def test_publish_with_yes_records_the_version_in_the_slice_file(aws, slice_file, capsys):
    account = aws.account()
    account.no_datasets().create().get("ACTIVE").version("1").get("ACTIVE").versions(1)

    argv = ["dataset", "publish", str(slice_file), "--name", "orderbot_regressions", "--yes"]
    assert main([*argv, "--region", "us-west-2"]) == EXIT_OK

    published = read_slice(slice_file).published
    assert published is not None
    assert (published.version, published.dataset_name) == ("1", "orderbot_regressions")
    assert "as version 1 of dataset orderbot_regressions" in capsys.readouterr().out
    account.done()


def test_profile_and_region_flags_choose_the_account(aws, slice_file, capsys):
    aws.account(region="eu-west-1").no_datasets()

    argv = [
        "dataset",
        "publish",
        str(slice_file),
        "--name",
        "n1",
        "--profile",
        "evals",
        "--region",
        "eu-west-1",
    ]
    assert main(argv) == EXIT_OK

    assert aws.asked == {"region": "eu-west-1", "profile": "evals"}
    assert "in eu-west-1" in capsys.readouterr().out


def test_missing_region_is_refused_not_defaulted(aws, slice_file, tmp_path, capsys):
    aws.account(region=None)

    publish_code = main(["dataset", "publish", str(slice_file), "--name", "n1"])
    pull_code = main(
        [
            "dataset",
            "pull",
            "--dataset-id",
            "d",
            "--version",
            "1",
            "--out",
            str(tmp_path / "p.json"),
        ]
    )

    assert publish_code != EXIT_OK
    assert pull_code != EXIT_OK
    err = capsys.readouterr().err
    assert err.count("--region or set AWS_REGION") == 2
    assert not (tmp_path / "p.json").exists()


@pytest.mark.parametrize(
    ("code", "status", "named"),
    [
        ("AccessDeniedException", 403, "bedrock-agentcore:ListDatasets"),
        ("ThrottlingException", 429, "throttled"),
    ],
)
def test_aws_failures_are_typed_errors_without_tracebacks(
    aws, slice_file, capsys, code, status, named
):
    aws.account().error("list_datasets", code, status, "nope")

    assert (
        main(["dataset", "publish", str(slice_file), "--name", "n1", "--region", "us-west-2"]) == 1
    )

    err = capsys.readouterr().err
    assert named in err
    assert "Traceback" not in err


def test_an_unreachable_endpoint_on_the_first_write_is_a_typed_error(
    monkeypatch, slice_file, capsys
):
    from botocore.exceptions import EndpointConnectionError

    from agentcore_dataset_stubs import Account

    account = Account(fail={"CreateDataset": EndpointConnectionError(endpoint_url="https://x")})
    account.no_datasets().create()
    monkeypatch.setattr(
        "calibrated_agent_evaluation.agentcore.session.AwsClients.from_profile",
        classmethod(lambda cls, **_kw: account.clients),
    )

    argv = ["dataset", "publish", str(slice_file), "--name", "n1", "--yes", "--region", "us-west-2"]
    assert main(argv) == 1

    err = capsys.readouterr().err
    assert "CreateDataset could not reach AgentCore" in err
    assert "Traceback" not in err
    assert read_slice(slice_file).published is None


def test_publish_and_pull_without_the_extra_print_the_install_command(
    slice_file, tmp_path, monkeypatch, capsys
):
    import calibrated_agent_evaluation.agentcore as agentcore

    for module in ("boto3", "botocore", "bedrock_agentcore"):
        monkeypatch.setitem(sys.modules, module, None)
    for name in ("AwsClients", "dataset_guard", "plan_publish", "publish", "pull"):
        monkeypatch.delitem(vars(agentcore), name, raising=False)
    monkeypatch.delitem(
        sys.modules, "calibrated_agent_evaluation.agentcore.datasets", raising=False
    )
    monkeypatch.delitem(sys.modules, "calibrated_agent_evaluation.agentcore.session", raising=False)

    assert main(["dataset", "publish", str(slice_file), "--name", "n1"]) == 1
    pull_argv = [
        "dataset",
        "pull",
        "--dataset-id",
        "d",
        "--version",
        "1",
        "--out",
        str(tmp_path / "p.json"),
    ]
    assert main(pull_argv) == 1

    err = capsys.readouterr().err
    assert err.count("agentcore") >= 2
    assert "Traceback" not in err


@pytest.mark.parametrize("version", ["DRAFT", "1.0"])
def test_only_published_versions_can_be_pulled(aws, tmp_path, capsys, version):
    account = aws.account()  # nothing queued: any call would fail the stub

    argv = [
        "dataset",
        "pull",
        "--dataset-id",
        "d",
        "--version",
        version,
        "--out",
        str(tmp_path / "p.json"),
    ]
    assert main([*argv, "--region", "us-west-2"]) == 1

    assert "only published versions" in capsys.readouterr().err
    account.done()


def test_a_publish_that_cannot_record_its_version_names_it(aws, slice_file, monkeypatch, capsys):
    from calibrated_agent_evaluation.scenarios.errors import ScenarioError

    account = aws.account()
    account.no_datasets().create().get("ACTIVE").version("1").get("ACTIVE").versions(1)

    def refuse_write(*_args, **_kwargs):
        raise ScenarioError("cannot write the slice file: read-only file system")

    monkeypatch.setattr("calibrated_agent_evaluation.scenarios.slicefile.write_slice", refuse_write)
    argv = ["dataset", "publish", str(slice_file), "--name", "orderbot_regressions", "--yes"]

    assert main([*argv, "--region", "us-west-2"]) == 1

    err = capsys.readouterr().err
    assert "published version 1 of orderbot_regressions" in err
    assert "arn:aws:bedrock-agentcore:us-west-2:111122223333:dataset/" in err
    assert "pull that version" in err
    assert read_slice(slice_file).published is None
