# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""The `calibrated-agent-evaluation ingest --source agentcore` CLI surface."""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any

import pytest

pytest.importorskip("boto3")  # the handler builds AwsClients, so skip on a bare install

from calibrated_agent_evaluation.backends.base import SourceTrace
from calibrated_agent_evaluation.backends.local import LocalBackend
from calibrated_agent_evaluation.cli.errors import CliError
from calibrated_agent_evaluation.cli.parser import build_parser
from calibrated_agent_evaluation.cli.stages.ingest import ingest_handler
from calibrated_agent_evaluation.schema.types import Span, Trace


def _trace(trace_id: str) -> Trace:
    return Trace(
        trace_id=trace_id,
        frame="production-representative",
        ownership="owned",
        input={"q": "x"},
        output={"a": "y"},
        taxonomy_cell={},
        spans=[
            Span(
                span_id="sp-1",
                trace_id=trace_id,
                agent="agent",
                role="specialist",
                parent=None,
                seq=0,
                status="completed",
                input={"q": "x"},
                output={"a": "y"},
            )
        ],
    )


class _FakeSource:
    def __init__(self, scores: list[dict] | None = None) -> None:
        self.rekeyed: list[tuple[str, str]] = []
        self._scores = scores or []

    def pull(self) -> Iterator[SourceTrace]:
        yield SourceTrace(source_id="t-1", trace=_trace("t-1"))

    def read_scores(self) -> Iterator[dict]:
        yield from self._scores


_SCORE_RESULT = {
    "evaluatorArn": "arn:aws:bedrock-agentcore:us-east-1:111122223333:evaluator/correctness",
    "evaluatorName": "correctness",
    "value": 1.0,
    "label": "correct",
    "explanation": "the answer is grounded",
    "context": {"spanContext": {"traceId": "t-1"}},
}


def _args(tmp_path: Any, argv: list[str]) -> Any:
    full = ["--backend", "local", "--root", str(tmp_path / "store"), "ingest", *argv]
    return build_parser().parse_args(full)


def test_backend_agentcore_is_rejected_naming_the_real_stores() -> None:
    with pytest.raises(SystemExit):
        build_parser().parse_args(["--backend", "agentcore", "ingest", "--source", "agentcore"])


def test_source_agentcore_without_runtime_is_refused(tmp_path: Any) -> None:
    args = _args(
        tmp_path,
        [
            "--source",
            "agentcore",
            "--since",
            "2026-10-01T00:00:00+00:00",
            "--until",
            "2026-10-01T01:00:00+00:00",
        ],
    )
    with pytest.raises(CliError, match=r"service\.name.*\.DEFAULT"):
        ingest_handler(args)


def test_the_runtime_help_says_it_takes_the_service_name(
    capsys: pytest.CaptureFixture[str],
) -> None:
    with pytest.raises(SystemExit):
        build_parser().parse_args(["ingest", "--help"])
    help_text = " ".join(capsys.readouterr().out.split())
    assert "service.name" in help_text
    assert ".DEFAULT" in help_text


def test_source_agentcore_without_window_is_refused(tmp_path: Any) -> None:
    args = _args(tmp_path, ["--source", "agentcore", "--runtime", "nl_sql_agent"])
    with pytest.raises(CliError, match="since"):
        ingest_handler(args)


def test_source_agentcore_rejects_an_unsupported_flag_loudly(tmp_path: Any) -> None:
    # --overwrite is a file/Langfuse-path flag; the AgentCore pull cannot honor it, and mainline's
    # own refusal messages tell users to pass it, so it is rejected before any AWS call rather than
    # silently ignored.
    args = _args(
        tmp_path,
        [
            "--source",
            "agentcore",
            "--runtime",
            "nl_sql_agent",
            "--since",
            "2026-10-01T00:00:00+00:00",
            "--until",
            "2026-10-01T01:00:00+00:00",
            "--overwrite",
        ],
    )
    with pytest.raises(CliError, match="does not support"):
        ingest_handler(args)


def test_source_agentcore_ingests_the_pulled_trace(tmp_path: Any, monkeypatch: Any) -> None:
    import calibrated_agent_evaluation.agentcore as agentcore

    monkeypatch.setattr(agentcore, "agentcore_source", lambda _clients, _config: _FakeSource())
    args = _args(
        tmp_path,
        [
            "--source",
            "agentcore",
            "--runtime",
            "nl_sql_agent",
            "--omni",
            "--since",
            "2026-10-01T00:00:00+00:00",
            "--until",
            "2026-10-01T01:00:00+00:00",
        ],
    )
    outcome = ingest_handler(args)
    assert outcome.exit_code == 0
    assert LocalBackend(root=str(tmp_path / "store")).get_trace("t-1").trace_id == "t-1"


def _scores_args(tmp_path: Any, *, with_scores: bool) -> Any:
    argv = [
        "--source",
        "agentcore",
        "--runtime",
        "nl_sql_agent",
        "--omni",
        "--since",
        "2026-10-01T00:00:00+00:00",
        "--until",
        "2026-10-01T01:00:00+00:00",
    ]
    if with_scores:
        argv.append("--with-scores")
    return _args(tmp_path, argv)


def test_with_scores_stores_the_advisory_label_alongside_the_trace(
    tmp_path: Any, monkeypatch: Any
) -> None:
    import calibrated_agent_evaluation.agentcore as agentcore

    monkeypatch.setattr(
        agentcore, "agentcore_source", lambda _clients, _config: _FakeSource([_SCORE_RESULT])
    )
    ingest_handler(_scores_args(tmp_path, with_scores=True))
    labels = LocalBackend(root=str(tmp_path / "store")).get_scores("t-1")
    assert len(labels) == 1
    assert labels[0].kind == "llm-judge"
    # The raw verdict is preserved in the critique for later calibration.
    assert "label='correct'" in labels[0].critique


def test_without_the_scores_flag_no_label_is_stored(tmp_path: Any, monkeypatch: Any) -> None:
    import calibrated_agent_evaluation.agentcore as agentcore

    monkeypatch.setattr(
        agentcore, "agentcore_source", lambda _clients, _config: _FakeSource([_SCORE_RESULT])
    )
    ingest_handler(_scores_args(tmp_path, with_scores=False))
    assert LocalBackend(root=str(tmp_path / "store")).get_scores("t-1") == []
