# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""StrandsExecutor: runs Strands agents/crews and returns normalized Traces.

An ``ExecutorBackend`` over Strands: it runs a ``StrandsTask`` (a builder callable
that returns a ``strands.Agent``/``Graph``/``Swarm`` + the input), captures the crew's genuine OTel
span tree via :class:`~calibrated_agent_evaluation.execute.strands_capture.SpanCapture`, serializes
it to canonical OTLP-JSON, and normalizes it through the ``StrandsSession`` adapter; the structural
twin of ``InspectExecutor``. Strands is lazy-imported so
``import calibrated_agent_evaluation.execute`` succeeds without the ``[strands]`` extra; calling the
executor without it raises a typed ``ExecutionError``. Every vendor exception is wrapped at this
boundary; no raw ``strands``/``boto3`` escapes. Written against ``strands-agents==1.45.0``.
"""

from __future__ import annotations

import tempfile
import uuid
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

from calibrated_agent_evaluation._install import install_hint
from calibrated_agent_evaluation.execute.cassette import (
    Cassette,
    CassetteError,
    CassetteMiss,
    Mode,
    mode_from_env,
)
from calibrated_agent_evaluation.execute.errors import ExecutionError
from calibrated_agent_evaluation.schema.types import Frame, Trace


def _install_hint() -> str:
    return f"strands-agents is not installed. Install via: {install_hint('strands')}"


_DEFAULT_MODEL_ID = "global.anthropic.claude-sonnet-4-6"
_DEFAULT_REGION = "us-west-2"
_DEFAULT_TIMEOUT_SECONDS = 60.0
#: Kept as the default so every existing caller behaves exactly as before. Pass ``None`` to send no
#: decoding parameter at all; required for models that reject an explicit ``temperature`` (see
#: :class:`StrandsExecutor`).
_DEFAULT_TEMPERATURE: float | None = 0.0


@dataclass(frozen=True)
class StrandsTask:
    """A runnable Strands target for :meth:`StrandsExecutor.run`.

    ``build`` receives the executor's cassette-backed model and returns a runnable Strands object
    (an ``Agent``/``Graph``/``Swarm``; all callable on the input). The executor injects the model
    so it owns replay/capture wiring; the caller only assembles topology. ``mode``/
    ``trace_id``/``taxonomy_cell`` are passed through to the ``StrandsSession`` normalizer so the
    produced ``Trace`` carries the right GenAI-convention branch and headers.
    """

    build: Callable[[Any], Any]
    input: str | list[dict[str, Any]]
    mode: Literal["legacy", "latest"] = "latest"
    trace_id: str | None = None
    taxonomy_cell: dict[str, str] = field(default_factory=dict)


class StrandsExecutor:
    """An ``ExecutorBackend`` over Strands. Context manager for the temp log dir.

    ``temperature`` defaults to ``0.0`` (what this executor has always sent) and accepts ``None``
    to send **no** decoding parameter. That escape hatch exists because it is not a preference: the
    newest Anthropic generations reject an explicit ``temperature`` as deprecated and fail the
    request outright, so a hard-coded value silently restricted this executor to older models, and a
    caller whose own agent sends no decoding parameters could not reproduce its own subject here.
    Determinism for those models comes from the prompt and a structured output schema instead.

    Args:
        log_dir: Where captured spans are written; a temporary directory when omitted.
        cassette_path: A cassette to record or replay model calls through.
        cassette_mode: Whether the cassette replays, records, or both.
        model_id: The Bedrock model the agent runs on.
        region: The Bedrock region.
        timeout_seconds: The per-request timeout.
        frame: The sampling frame the produced traces are stamped with.
        temperature: The decoding temperature, or ``None`` to send none.
    """

    def __init__(
        self,
        *,
        log_dir: Path | None = None,
        cassette_path: str | Path | None = None,
        cassette_mode: Mode | None = None,
        model_id: str = _DEFAULT_MODEL_ID,
        region: str = _DEFAULT_REGION,
        timeout_seconds: float = _DEFAULT_TIMEOUT_SECONDS,
        frame: Frame = "production-representative",
        temperature: float | None = _DEFAULT_TEMPERATURE,
    ) -> None:
        self._explicit_log_dir = log_dir
        self._cassette_path = None if cassette_path is None else Path(cassette_path)
        self._cassette_mode = cassette_mode or mode_from_env(default=Mode.AUTO)
        self._model_id = model_id
        self._region = region
        self._temperature = temperature
        # Explicit bound on the live Bedrock connect/read so an unreachable peer raises within a
        # known window rather than hanging on a shifting SDK default (every network call is bounded
        # by an explicit timeout).
        self._timeout_seconds = timeout_seconds
        # The dataset frame the produced traces carry; the caller (synthesis of a discovery
        # frame) passes frame="discovery" so the tag survives execution (mirrors InspectExecutor).
        self._frame: Frame = frame
        self._tmpdir: tempfile.TemporaryDirectory[str] | None = None

    @property
    def log_dir(self) -> Path:
        """Where the captured OTLP-JSON export is persisted."""
        if self._explicit_log_dir is not None:
            return self._explicit_log_dir
        if self._tmpdir is None:
            self._tmpdir = tempfile.TemporaryDirectory(
                prefix="calibrated_agent_evaluation_strands_"
            )
        return Path(self._tmpdir.name)

    def __enter__(self) -> StrandsExecutor:
        return self

    def __exit__(self, *_exc: object) -> None:
        if self._tmpdir is not None:
            self._tmpdir.cleanup()
            self._tmpdir = None

    def run(self, task: Any, *, log_dir: Path | None = None) -> list[Trace]:
        """Run a ``StrandsTask`` and return one normalized, validated ``Trace``.

        Raises ``ExecutionError`` on any failure; missing extra, cassette error, unreachable model,
        empty capture, or invalid trace. Never returns a partial Trace.
        """
        if not isinstance(task, StrandsTask):
            raise ExecutionError(
                f"StrandsExecutor.run expects a StrandsTask, got {type(task).__name__}"
            )
        span_capture_cls, bedrock_model_cls, cassette_model_cls = self._load_strands()
        effective_log_dir = log_dir or self.log_dir
        effective_log_dir.mkdir(parents=True, exist_ok=True)

        # Build+validate the cassette FIRST (stdlib); a corrupt file raises CassetteError here,
        # before any Strands work, and is wrapped as ExecutionError.
        model = self._build_model(bedrock_model_cls, cassette_model_cls)

        capture = span_capture_cls("calibrated_agent_evaluation.strands")
        try:
            with capture:
                capture.register_global()
                runnable = task.build(model)
                runnable(task.input)
            spans = capture.finished_spans
        except (CassetteMiss, CassetteError) as exc:
            raise ExecutionError(f"cassette error during strands run: {exc}") from exc
        except ExecutionError:
            raise
        except Exception as exc:
            raise ExecutionError(f"strands run failed: {type(exc).__name__}: {exc}") from exc

        if not spans:
            raise ExecutionError(
                "no spans captured for this run; the Strands run emitted nothing "
                "(was the capture provider installed / did the agent actually run?)"
            )

        log_path = self._write_export(spans, effective_log_dir, task)
        return [self._normalize(log_path, task)]

    def _build_model(self, bedrock_model_cls: Any, cassette_model_cls: Any) -> Any:
        """Return the model the task runs on: a cassette-wrapped model, or the live one.

        With a cassette, the real ``BedrockModel`` is built only when a cassette miss needs a live
        call, so a replay never resolves AWS credentials or a profile.
        """
        # `temperature=None` omits the key entirely rather than passing None through: the models
        # that reject an explicit temperature reject a null one too.
        decoding = {} if self._temperature is None else {"temperature": self._temperature}

        def build_live() -> Any:
            try:
                from botocore.config import Config as BotocoreConfig

                # Bound the live connect/read so an unreachable Bedrock peer raises within the
                # window instead of hanging on a shifting SDK default.
                client_config = BotocoreConfig(
                    connect_timeout=self._timeout_seconds, read_timeout=self._timeout_seconds
                )
                return bedrock_model_cls(
                    model_id=self._model_id,
                    region_name=self._region,
                    boto_client_config=client_config,
                    **decoding,
                )
            except Exception as exc:
                raise ExecutionError(f"failed to construct BedrockModel: {exc}") from exc

        if self._cassette_path is None:
            # No cassette: live-only run (record/replay disabled). The caller opted into live calls.
            return build_live()
        # Building the Cassette loads+validates the file, so a corrupt cassette raises here.
        try:
            cassette = Cassette(self._cassette_path, mode=self._cassette_mode)
        except CassetteError as exc:
            raise ExecutionError(f"cassette {self._cassette_path} is corrupt: {exc}") from exc
        from calibrated_agent_evaluation.execute.strands_cassette_model import DeferredModel

        inner = DeferredModel(build_live, {"model_id": self._model_id, **decoding})
        return cassette_model_cls(inner, cassette)

    def _write_export(self, spans: Any, log_dir: Path, task: StrandsTask) -> Path:
        """Serialize the captured spans to OTLP-JSON in ``log_dir`` and return the path."""
        from calibrated_agent_evaluation.execute.strands_capture import write_otlp_json

        # `task.trace_id or "strands-run"` meant two untitled runs into one log_dir silently
        # overwrote each other's raw export -- and a run cannot be repeated to recover it. An
        # untitled run gets a unique suffix; a titled one keeps its deterministic name, which is
        # what makes the frozen fixtures reproducible.
        if task.trace_id:
            log_path = log_dir / f"{task.trace_id}.otlp.json"
        else:
            log_path = log_dir / f"strands-run-{uuid.uuid4().hex[:12]}.otlp.json"
        if log_path.exists():
            raise ExecutionError(
                f"OTLP-JSON export {log_path} already exists; refusing to overwrite a previous "
                "run's raw export (a run cannot be repeated to recover it). Use a distinct "
                "task.trace_id, or a fresh log_dir"
            )
        try:
            write_otlp_json(spans, log_path)
        except OSError as exc:
            raise ExecutionError(f"failed to write OTLP-JSON export to {log_path}: {exc}") from exc
        return log_path

    def _normalize(self, log_path: Path, task: StrandsTask) -> Trace:
        """Normalize the OTLP-JSON export into a validated Trace."""
        from calibrated_agent_evaluation.normalize import normalize
        from calibrated_agent_evaluation.normalize.sources import StrandsSession
        from calibrated_agent_evaluation.schema.validate import validate

        try:
            trace = normalize(
                StrandsSession(
                    path=log_path,
                    mode=task.mode,
                    trace_id=task.trace_id,
                    frame=self._frame,
                    taxonomy_cell=task.taxonomy_cell,
                )
            )
        except Exception as exc:
            raise ExecutionError(
                f"normalization failed for {log_path}: {type(exc).__name__}: {exc}"
            ) from exc

        try:
            validate(trace)
        except Exception as exc:
            raise ExecutionError(
                f"normalized trace failed validation: {type(exc).__name__}: {exc}"
            ) from exc

        return trace

    @staticmethod
    def _load_strands() -> tuple[Any, Any, Any]:
        """Lazy-import the Strands surface; raise ExecutionError if the extra is absent."""
        try:
            from strands.models.bedrock import BedrockModel

            from calibrated_agent_evaluation.execute.strands_capture import SpanCapture
            from calibrated_agent_evaluation.execute.strands_cassette_model import CassetteModel
        except ImportError as exc:
            raise ExecutionError(_install_hint()) from exc
        return SpanCapture, BedrockModel, CassetteModel
