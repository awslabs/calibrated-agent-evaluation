# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""The arm read over a live A/B test's own span rows, redacted, served to both trace readers.

The rows are a capture of the integration account's A/B test (20 Gateway-routed requests),
redacted to the fixture placeholders. They are served through ``Stubber`` to the ingest's
Logs Insights and Omni readers, as ``calibrated-agent-evaluation audit arms`` builds them, and every
read must equal ``fixtures/expected/audit/arms.json.gz``, which a script with no Calibrated Agent
Evaluation code wrote from the same rows. The live agent writes its message content on span
attributes, which the ingest does not recover, so the ingest delivers every trace as an error; the
arm read still reads each trace's arm from its spans and marks it as having come back without
content.
"""

from __future__ import annotations

import gzip
import json
from pathlib import Path
from typing import Any

import pytest

pytest.importorskip("boto3")

from botocore.stub import Stubber

from _arms_fixture import insights_results
from _fixture_json import read_json
from calibrated_agent_evaluation.agentcore import optimization
from calibrated_agent_evaluation.cli import main
from calibrated_agent_evaluation.cli.dispatch import EXIT_DOMAIN_ERROR, EXIT_OK

_REPO = Path(__file__).resolve().parents[1]
_LIVE = _REPO / "fixtures" / "raw" / "agentcore" / "abtest" / "live"
_GOLDEN = read_json(_REPO / "fixtures" / "expected" / "audit" / "arms.json")
_TEST = json.loads((_LIVE / "abtest_live.json").read_text(encoding="utf-8"))
_FLAGS = {
    "no-flag": [],
    "attribute": ["--applied-bundle-attribute", "config.bundle_version"],
    "attribute-and-require": [
        "--applied-bundle-attribute",
        "config.bundle_version",
        "--applied-bundle-require",
        "config.source=bundle",
    ],
}


def _load(name: str) -> Any:
    path = _LIVE / name
    if path.suffix == ".gz":
        return json.loads(gzip.decompress(path.read_bytes()))
    return json.loads(path.read_text(encoding="utf-8"))


@pytest.fixture
def root(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    for key, value in {
        "AWS_ACCESS_KEY_ID": "testing",
        "AWS_SECRET_ACCESS_KEY": "testing",
        "AWS_DEFAULT_REGION": "us-west-2",
        "CALIBRATED_AGENT_EVALUATION_AGENTCORE_SETTLE_SECONDS": "0",
    }.items():
        monkeypatch.setenv(key, value)
    monkeypatch.delenv("AWS_PROFILE", raising=False)
    _record_claim(tmp_path, monkeypatch)
    return tmp_path


def _serve(monkeypatch: pytest.MonkeyPatch, queue: Any) -> list[Stubber]:
    real = optimization.audit_clients
    stubs: list[Stubber] = []

    def stubbed(region: str | None = None, profile: str | None = None) -> Any:
        made = real(region or "us-west-2", profile)
        stubs.extend(queue(made))
        for stub in stubs:
            stub.activate()
        return made

    monkeypatch.setattr(optimization, "audit_clients", stubbed)
    return stubs


def _record_claim(root: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    def queue(clients: Any) -> list[Stubber]:
        data = Stubber(clients.client("bedrock-agentcore"))
        control = Stubber(clients.client("bedrock-agentcore-control"))
        data.add_response("get_ab_test", _TEST, {"abTestId": _TEST["abTestId"]})
        for name in ("bundle_version_c.json", "bundle_version_t1.json"):
            version = _load(name)
            control.add_response(
                "get_configuration_bundle_version",
                version,
                {"bundleId": version["bundleId"], "versionId": version["versionId"]},
            )
        return [data, control]

    stubs = _serve(monkeypatch, queue)
    code = main(["--root", str(root), "audit", "abtest", "--ab-test-id", _TEST["abTestId"]])
    for stub in stubs:
        stub.assert_no_pending_responses()
        stub.deactivate()
    assert code == EXIT_OK


def _omni_round(stub: Stubber, rows: list[dict[str, Any]]) -> None:
    stub.add_response("start_telemetry_query_session", {"sessionId": "sess"})
    stub.add_response("start_telemetry_query", {"queryId": "q", "sessionId": "sess"})
    stub.add_response(
        "get_telemetry_query_results",
        {
            "status": "Complete",
            "rows": [dict(r, **{"@message": json.dumps(r["@message"])}) for r in rows],
            "statistics": {"partialResults": {"partialResultsDetected": False}},
        },
    )
    stub.add_response("stop_telemetry_query_session", {})


def _read(
    root: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    omni: bool,
    flags: list[str],
    rekey: bool = True,
    expect: int = EXIT_OK,
) -> int:
    def queue(clients: Any) -> list[Stubber]:
        stub = Stubber(clients.client("cloudwatchomni" if omni else "logs"))
        if omni:
            _omni_round(stub, _load("omni_spans.json.gz"))
            _omni_round(stub, _load("omni_content.json.gz"))
        else:
            stub.add_response("start_query", {"queryId": "q-spans"})
            stub.add_response(
                "get_query_results", insights_results(_load("insights_spans.json.gz"))
            )
            stub.add_response("start_query", {"queryId": "q-content"})
            stub.add_response(
                "get_query_results", insights_results(_load("insights_content.json.gz"))
            )
        return [stub]

    stubs = _serve(monkeypatch, queue)
    code = main(
        [
            "--root",
            str(root),
            "--explain",
            "audit",
            "arms",
            "--ab-test-id",
            _TEST["abTestId"],
            "--runtime",
            _GOLDEN["runtime"],
            "--since",
            _GOLDEN["window_start"],
            "--until",
            _GOLDEN["window_end"],
            *(["--rekey-sessions"] if rekey else []),
            *(["--omni"] if omni else []),
            *flags,
        ]
    )
    for stub in stubs:
        stub.assert_no_pending_responses()
        stub.deactivate()
    assert code == expect
    return code


def _stored(root: Path, read_id: str) -> dict[str, dict[str, Any]]:
    path = root / "records" / "agentcore-arms" / "agentcore_arms_record.json"
    docs = json.loads(path.read_text(encoding="utf-8"))
    return {d["trace_id"]: d for d in docs if d["read_id"] == read_id}


_FIELDS = (
    "status",
    "variant",
    "runtime_session_id",
    "spans_total",
    "spans_stamped",
    "spans_half_stamped",
    "content_missing",
)


@pytest.mark.parametrize("omni", [False, True], ids=["logs-insights", "omni"])
@pytest.mark.parametrize("setting", list(_FLAGS))
def test_a_live_windows_rows_give_the_golden_arms_on_both_read_paths(
    root: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    omni: bool,
    setting: str,
) -> None:
    [golden] = [r for r in _GOLDEN["reads"] if r["setting"] == setting and r["omni"] == omni]
    capsys.readouterr()

    _read(root, monkeypatch, omni=omni, flags=_FLAGS[setting])

    stored = _stored(root, golden["read_id"])
    expected = {a["trace_id"]: a for a in golden["assignments"]}
    assert set(stored) == set(expected)
    for trace_id, want in expected.items():
        assert {k: stored[trace_id][k] for k in _FIELDS} == {k: want[k] for k in _FIELDS}
        assert stored[trace_id]["omni"] is omni
        assert stored[trace_id]["rekey_sessions"] is True
    out = capsys.readouterr().out
    counts = golden["counts"]
    assert f"C: {counts['C.traces']} trace(s), {counts['C.runtime_sessions']} runtime" in out
    assert f"T1: {counts['T1.traces']} trace(s), {counts['T1.runtime_sessions']} runtime" in out
    assert "20 trace(s) came back without their message content" in out


@pytest.mark.parametrize("omni", [False, True], ids=["logs-insights", "omni"])
def test_without_rekeying_the_live_traces_that_lack_content_stay_unread_and_nothing_is_stored(
    root: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    omni: bool,
) -> None:
    # Each live trace carries two session ids and no content: read without re-keying, the ingest
    # refuses it for its sessions, so its content-free spans must not be read as an assignment.
    capsys.readouterr()

    _read(root, monkeypatch, omni=omni, flags=[], rekey=False, expect=EXIT_DOMAIN_ERROR)

    err = capsys.readouterr().err
    assert "AgentCoreArmsMissing" in err
    assert "came back unread" in err
    assert not (root / "records" / "agentcore-arms").exists()


def test_the_live_arm_counts_match_the_sample_sizes_agentcore_reported(
    root: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    [metric] = _TEST["results"]["evaluatorMetrics"]
    sizes = {metric["controlStats"]["variantName"]: metric["controlStats"]["sampleSize"]}
    sizes.update({v["variantName"]: v["sampleSize"] for v in metric["variantResults"]})
    [golden] = [r for r in _GOLDEN["reads"] if r["setting"] == "no-flag" and not r["omni"]]

    _read(root, monkeypatch, omni=False, flags=[])

    stored = _stored(root, golden["read_id"]).values()
    assert {arm: sum(a["variant"] == arm for a in stored) for arm in sizes} == sizes


def test_the_live_capture_names_only_placeholder_accounts_and_resource_ids() -> None:
    import re

    texts = {p.name: gzip.decompress(p.read_bytes()).decode("utf-8") for p in _LIVE.glob("*.gz")}
    texts |= {p.name: p.read_text(encoding="utf-8") for p in _LIVE.glob("*.json")}
    assert len(texts) >= 7
    for name, text in texts.items():
        accounts = set(re.findall(r"arn:aws:[a-z0-9-]+:[a-z0-9-]*:(\d{12}):", text))
        accounts |= set(re.findall(r'"(?:aws\.account\.id|cloud\.account\.id)":\s*"(\d+)"', text))
        assert accounts <= {"111122223333"}, (name, accounts)


def test_every_gzipped_audit_fixture_names_only_the_placeholder_account() -> None:
    import re

    fixtures = _REPO / "fixtures"
    paths = [
        *fixtures.glob("raw/agentcore/abtest/**/*.json.gz"),
        *fixtures.glob("raw/agentcore/recommendations/**/*.json.gz"),
        *fixtures.glob("expected/audit/*.json.gz"),
    ]
    assert len(paths) >= 16
    for path in paths:
        text = gzip.decompress(path.read_bytes()).decode("utf-8")
        accounts = set(re.findall(r"arn:aws:[a-z0-9-]+:[a-z0-9-]*:(\d{12}):", text))
        accounts |= set(re.findall(r'"(?:aws\.account\.id|cloud\.account\.id)":\s*"(\d+)"', text))
        assert accounts <= {"111122223333"}, (path.name, accounts)
