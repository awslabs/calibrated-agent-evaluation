# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for calibrated_agent_evaluation.execute.inspect_executor."""

from __future__ import annotations

from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from calibrated_agent_evaluation.execute.cassette import CassetteMiss, Mode
from calibrated_agent_evaluation.execute.errors import ExecutionError
from calibrated_agent_evaluation.execute.inspect_executor import InspectExecutor
from calibrated_agent_evaluation.schema.types import Trace


@pytest.fixture
def tmp_log_dir(tmp_path: Path) -> Path:
    d = tmp_path / "logs"
    d.mkdir()
    return d


@pytest.fixture
def sample_eval_log_path(tmp_path: Path) -> Path:
    """Create a minimal .eval file that the normalizer can process.

    We use a real fixture if available, otherwise create a marker file for mocking.
    """
    return tmp_path / "test.eval"


def _make_executor(
    model: str = "cassette/orderbot",
    log_dir: Path | None = None,
    cassette_path: Path | None = None,
    cassette_mode: Mode = Mode.REPLAY,
) -> InspectExecutor:
    return InspectExecutor(
        model=model,
        log_dir=log_dir,
        cassette_path=cassette_path,
        cassette_mode=cassette_mode,
    )


class TestLazyImport:
    """Importing calibrated_agent_evaluation.execute succeeds; run raises if inspect-ai absent."""

    def test_import_succeeds_without_inspect(self) -> None:
        from calibrated_agent_evaluation.execute import Cassette, ExecutionError, ExecutorBackend

        assert ExecutorBackend is not None
        assert ExecutionError is not None
        assert Cassette is not None

    def test_run_raises_execution_error_when_inspect_missing(self, tmp_log_dir: Path) -> None:
        executor = _make_executor(log_dir=tmp_log_dir)
        with (
            patch.dict("sys.modules", {"inspect_ai": None, "inspect_ai.log": None}),
            patch(
                "calibrated_agent_evaluation.execute.inspect_executor.InspectExecutor._load_inspect",
                side_effect=ExecutionError(
                    "inspect-ai is not installed. Install via: pip install "
                    "calibrated-agent-evaluation[inspect]"
                ),
            ),
            pytest.raises(ExecutionError, match="calibrated-agent-evaluation\\[inspect\\]"),
        ):
            executor.run("some_task")


class TestErrorWrapping:
    """Eval exceptions are wrapped in ExecutionError."""

    def test_eval_exception_wrapped_as_execution_error(self, tmp_log_dir: Path) -> None:
        executor = _make_executor(model="bedrock/claude", log_dir=tmp_log_dir)

        mock_eval = MagicMock(side_effect=RuntimeError("model connection refused"))
        with patch.object(executor, "_load_inspect", return_value=(mock_eval, None)):
            with pytest.raises(ExecutionError, match="eval failed.*model connection refused"):
                executor.run("task")

    def test_wrapped_error_chains_original(self, tmp_log_dir: Path) -> None:
        executor = _make_executor(model="bedrock/claude", log_dir=tmp_log_dir)
        original = RuntimeError("auth failed")
        mock_eval = MagicMock(side_effect=original)

        with patch.object(executor, "_load_inspect", return_value=(mock_eval, None)):
            with pytest.raises(ExecutionError) as exc_info:
                executor.run("task")
            assert exc_info.value.__cause__ is original


class TestReplayMiss:
    """REPLAY miss raises ExecutionError (not raw CassetteMiss)."""

    def test_cassette_miss_wrapped_as_execution_error(self, tmp_log_dir: Path) -> None:
        executor = _make_executor(model="bedrock/claude", log_dir=tmp_log_dir)
        mock_eval = MagicMock(side_effect=CassetteMiss("no recorded response for fp abc123"))

        with patch.object(executor, "_load_inspect", return_value=(mock_eval, None)):
            with pytest.raises(ExecutionError, match="cassette error during eval"):
                executor.run("task")


class TestEvalStatus:
    """EvalLog with status != 'success' raises ExecutionError."""

    def test_eval_error_status_raises(self, tmp_log_dir: Path) -> None:
        executor = _make_executor(model="bedrock/claude", log_dir=tmp_log_dir)

        mock_log = MagicMock()
        mock_log.status = "error"
        mock_log.error = "out of memory"
        mock_eval = MagicMock(return_value=[mock_log])

        with patch.object(executor, "_load_inspect", return_value=(mock_eval, None)):
            with pytest.raises(ExecutionError, match="eval did not succeed.*status=error"):
                executor.run("task")

    def test_eval_cancelled_status_raises(self, tmp_log_dir: Path) -> None:
        executor = _make_executor(model="bedrock/claude", log_dir=tmp_log_dir)

        mock_log = MagicMock()
        mock_log.status = "cancelled"
        mock_log.error = None
        mock_eval = MagicMock(return_value=[mock_log])

        with patch.object(executor, "_load_inspect", return_value=(mock_eval, None)):
            with pytest.raises(ExecutionError, match="eval did not succeed.*status=cancelled"):
                executor.run("task")


class TestLogPersistence:
    """.eval log persisted to log_dir (explicit or default temp)."""

    def test_explicit_log_dir_used(self, tmp_log_dir: Path) -> None:
        executor = _make_executor(log_dir=tmp_log_dir)
        assert executor.log_dir == tmp_log_dir

    def test_default_temp_dir_created(self) -> None:
        executor = _make_executor()
        log_dir = executor.log_dir
        assert log_dir.exists()
        assert "calibrated_agent_evaluation_eval_" in str(log_dir)

    def test_context_manager_cleans_temp_dir(self) -> None:
        with InspectExecutor(model="cassette/x") as executor:
            temp_dir = executor.log_dir
            assert temp_dir.exists()
        assert not temp_dir.exists()


class TestValidation:
    """Returned Traces must pass validate; invalid traces rejected."""

    def test_invalid_trace_from_normalize_raises(self, tmp_log_dir: Path) -> None:
        executor = _make_executor(model="bedrock/claude", log_dir=tmp_log_dir)

        mock_log = MagicMock()
        mock_log.status = "success"
        mock_log.location = None
        mock_eval = MagicMock(return_value=[mock_log])

        eval_file = tmp_log_dir / "test.eval"
        eval_file.touch()

        def mock_normalize_log(self_inner: Any, log_path: Path) -> Trace:
            raise ExecutionError(
                "normalized trace failed validation: SchemaError: duplicate span_id 'dup'"
            )

        with patch.object(executor, "_load_inspect", return_value=(mock_eval, None)):
            with patch.object(InspectExecutor, "_normalize_log", mock_normalize_log):
                with pytest.raises(ExecutionError, match="normalized trace failed validation"):
                    executor.run("task")


class TestUnreachable:
    """Unreachable model raises ExecutionError within bounded time."""

    def test_connection_failure_raises_execution_error(self, tmp_log_dir: Path) -> None:
        executor = _make_executor(model="bedrock/unreachable", log_dir=tmp_log_dir)
        mock_eval = MagicMock(side_effect=ConnectionError("connection refused"))

        with patch.object(executor, "_load_inspect", return_value=(mock_eval, None)):
            with pytest.raises(ExecutionError, match="eval failed.*connection refused"):
                executor.run("task")


class TestSageMaker:
    """Sagemaker/ model uses SageMaker path; non-sagemaker does not."""

    def test_non_sagemaker_model_does_not_invoke_sagemaker(self, tmp_log_dir: Path) -> None:
        executor = _make_executor(model="bedrock/claude-sonnet-4.6", log_dir=tmp_log_dir)
        assert not executor._model.startswith("sagemaker/")

    def test_sagemaker_model_identifier_preserved(self, tmp_log_dir: Path) -> None:
        executor = _make_executor(model="sagemaker/my-endpoint", log_dir=tmp_log_dir)
        assert executor._model == "sagemaker/my-endpoint"


class TestCorruptCassetteThroughRun:
    """A corrupt cassette surfaces as ExecutionError from run, never a raw CassetteError.

    Regression for the review finding: the Cassette was constructed OUTSIDE run's try
    block, so a corrupt-file CassetteError leaked to the caller (contradicting the typed-error
    contract).
    """

    def test_corrupt_cassette_wrapped_as_execution_error(self, tmp_log_dir: Path) -> None:
        corrupt = tmp_log_dir / "bad.cassette.json"
        corrupt.write_text('{"version": 1, "entries": [  ', encoding="utf-8")  # truncated JSON

        executor = _make_executor(
            model="cassette/orderbot",
            log_dir=tmp_log_dir,
            cassette_path=corrupt,
            cassette_mode=Mode.REPLAY,
        )
        with pytest.raises(ExecutionError, match="cassette error during eval"):
            executor.run("orderbot_fail")

    def test_missing_version_cassette_wrapped(self, tmp_log_dir: Path) -> None:
        import json

        noversion = tmp_log_dir / "noversion.cassette.json"
        noversion.write_text(json.dumps({"entries": []}), encoding="utf-8")

        executor = _make_executor(
            model="cassette/orderbot",
            log_dir=tmp_log_dir,
            cassette_path=noversion,
            cassette_mode=Mode.REPLAY,
        )
        with pytest.raises(ExecutionError, match="cassette error during eval"):
            executor.run("orderbot_fail")


class TestCassettePathAsString:
    """A ``str`` cassette path reaches the cassette the same as a ``Path``."""

    def test_string_path_reaches_the_cassette(self, tmp_log_dir: Path) -> None:
        corrupt = tmp_log_dir / "bad.cassette.json"
        corrupt.write_text('{"version": 1, "entries": [  ', encoding="utf-8")

        executor = InspectExecutor(
            model="cassette/orderbot",
            log_dir=tmp_log_dir,
            cassette_path=str(corrupt),
            cassette_mode=Mode.REPLAY,
        )
        with pytest.raises(ExecutionError, match="cassette error during eval"):
            executor.run("orderbot_fail")


class TestCassettePathRequired:
    """A cassette/ model with no cassette_path raises a clear typed error (no silent no-op).

    Regression for the review finding: the executor used to `except ImportError: pass`,
    silently skipping provider configuration so run() failed later with a confusing eval error.
    """

    def test_cassette_model_without_path_raises_execution_error(self, tmp_log_dir: Path) -> None:
        executor = InspectExecutor(
            model="cassette/orderbot",
            log_dir=tmp_log_dir,
            cassette_path=None,
            cassette_mode=Mode.REPLAY,
        )
        with pytest.raises(ExecutionError, match="needs a cassette but no cassette_path"):
            executor.run("orderbot_fail")


class TestRecordInnerModel:
    """/ review finding: the live model recorded against is an explicit param, not derived.

    `cassette/orderbot`.replace("cassette/", "bedrock/") == `bedrock/orderbot` (an invalid Bedrock
    id). The executor must pass the explicit `cassette_inner_model` (or the provider default) to
    the provider, never the broken derivation.
    """

    def test_explicit_inner_model_forwarded_to_provider(self, tmp_log_dir: Path) -> None:
        import json

        cassette_path = tmp_log_dir / "seed.cassette.json"
        cassette_path.write_text(json.dumps({"version": 1, "entries": []}), encoding="utf-8")

        executor = _make_executor(
            model="cassette/orderbot",
            log_dir=tmp_log_dir,
            cassette_path=cassette_path,
            cassette_mode=Mode.AUTO,
        )
        executor._cassette_inner_model = "bedrock/us.anthropic.claude-sonnet-4-6-v1"

        captured: dict[str, Any] = {}

        def fake_configure(cassette: Any, inner_model: str) -> None:
            captured["inner_model"] = inner_model

        import calibrated_agent_evaluation.execute.cassette_provider as provider

        with patch.object(provider, "configure", fake_configure):
            executor._configure_cassette_provider()

        assert captured["inner_model"] == "bedrock/us.anthropic.claude-sonnet-4-6-v1"
        assert "orderbot" not in captured["inner_model"].replace("cassette/", "")

    def test_defaults_to_provider_default_inner_model(self, tmp_log_dir: Path) -> None:
        import json

        cassette_path = tmp_log_dir / "seed.cassette.json"
        cassette_path.write_text(json.dumps({"version": 1, "entries": []}), encoding="utf-8")

        executor = _make_executor(
            model="cassette/orderbot",
            log_dir=tmp_log_dir,
            cassette_path=cassette_path,
            cassette_mode=Mode.AUTO,
        )

        captured: dict[str, Any] = {}
        import calibrated_agent_evaluation.execute.cassette_provider as provider

        def fake_configure(cassette: Any, inner_model: str) -> None:
            captured["inner_model"] = inner_model

        with patch.object(provider, "configure", fake_configure):
            executor._configure_cassette_provider()

        assert captured["inner_model"] == provider.DEFAULT_INNER_MODEL


class TestProviderShipped:
    """Review finding: the cassette provider is shipped in src/, importable without corpus/."""

    def test_provider_importable_from_shipped_package(self) -> None:
        from calibrated_agent_evaluation.execute import cassette_provider

        assert hasattr(cassette_provider, "configure")
        assert hasattr(cassette_provider, "CassetteModelAPI")
        assert cassette_provider.DEFAULT_INNER_MODEL.startswith("bedrock/")

    def test_examples_shim_reexports_shipped_provider(self) -> None:
        # corpus/ lives at the repo root (not under src/), so it is not on sys.path for the
        # top-level test suite — add it (same pattern as tests/corpus/conftest) so this actually
        # runs rather than silently skipping (which would read as "shim verified" when it wasn't).
        import sys
        from pathlib import Path

        repo_root = Path(__file__).resolve().parent.parent
        if str(repo_root) not in sys.path:
            sys.path.insert(0, str(repo_root))

        import corpus.agents.inspect_cassette_provider as shim

        from calibrated_agent_evaluation.execute import cassette_provider

        assert shim.CassetteModelAPI is cassette_provider.CassetteModelAPI
        assert shim.configure is cassette_provider.configure


class TestFramePropagation:
    """Route-through fix: the executor preserves the caller's dataset frame tag.

    Regression for the manual-acceptance finding — a synthesized *discovery* input must not come
    back tagged 'production-representative' (which would make the propagation check
    reject the real executor's output). The executor threads its ``frame`` to ``InspectEvalLog``.
    """

    def _run_capturing_source(self, frame_arg: dict[str, Any], tmp_log_dir: Path) -> Any:
        executor = InspectExecutor(model="bedrock/claude", log_dir=tmp_log_dir, **frame_arg)
        mock_log = MagicMock()
        mock_log.status = "success"
        mock_log.location = None
        mock_eval = MagicMock(return_value=[mock_log])
        (tmp_log_dir / "test.eval").touch()

        captured: dict[str, Any] = {}

        def fake_normalize_all(source: Any) -> list[Trace]:
            captured["frame"] = source.frame
            return [
                Trace(
                    trace_id="T",
                    frame=source.frame,
                    ownership="probed",
                    input={},
                    output={},
                    taxonomy_cell={},
                )
            ]

        with patch.object(executor, "_load_inspect", return_value=(mock_eval, None)):
            with patch("calibrated_agent_evaluation.normalize.normalize_all", fake_normalize_all):
                with patch("calibrated_agent_evaluation.schema.validate.validate", lambda t: None):
                    traces = executor.run("task")
        return captured["frame"], traces[0].frame

    def test_frame_defaults_to_production_representative(self, tmp_log_dir: Path) -> None:
        src_frame, trace_frame = self._run_capturing_source({}, tmp_log_dir)
        assert src_frame == "production-representative"
        assert trace_frame == "production-representative"

    def test_frame_discovery_is_threaded_to_the_trace(self, tmp_log_dir: Path) -> None:
        src_frame, trace_frame = self._run_capturing_source({"frame": "discovery"}, tmp_log_dir)
        assert src_frame == "discovery"
        assert trace_frame == "discovery"


class TestBoundary:
    """ExecutorBackend has no vendor types; execute/ doesn't import methodology."""

    def test_executor_backend_protocol_has_no_vendor_types(self) -> None:
        import typing

        from calibrated_agent_evaluation.execute.base import ExecutorBackend

        hints = typing.get_type_hints(ExecutorBackend.run)
        for name, hint in hints.items():
            hint_str = str(hint)
            assert "inspect_ai" not in hint_str, f"vendor type in {name}: {hint_str}"
            assert "boto3" not in hint_str, f"vendor type in {name}: {hint_str}"
            assert "langfuse" not in hint_str, f"vendor type in {name}: {hint_str}"
