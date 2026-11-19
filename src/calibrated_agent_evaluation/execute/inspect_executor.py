# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""InspectExecutor: runs Inspect AI tasks and returns normalized Traces.

Wraps Inspect's ``eval()`` with the cassette model provider for deterministic replay, normalizes
the resulting ``.eval`` log via the ``InspectEvalLog`` adapter (one Trace per sample), and
validates every trace
before returning. Inspect is imported lazily so importing this module without the
``[inspect]`` extra raises a clear ``ExecutionError`` at call time, not at import time.

Written against ``inspect-ai==0.3.242``. The verified surface:
``eval()``, ``ModelAPI.generate()``, ``@modelapi``, ``read_eval_log()``, ``EvalLog.status``,
``EvalLog.location``.
"""

from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Any

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


class InspectExecutor:
    """An ``ExecutorBackend`` over Inspect AI's ``eval()``.

    Context manager: ``__enter__``/``__exit__`` manage the temp log directory (if no explicit
    ``log_dir`` was provided) and ensure resources are cleaned up.

    Args:
        model: The Inspect model the agent runs on.
        log_dir: Where Inspect writes its logs; a temporary directory when omitted.
        cassette_path: A cassette to record or replay model calls through.
        cassette_mode: Whether the cassette replays, records, or both.
        cassette_inner_model: The real model a recording cassette calls on a miss.
        frame: The sampling frame the produced traces are stamped with.
    """

    def __init__(
        self,
        model: str = "cassette/orderbot",
        *,
        log_dir: Path | None = None,
        cassette_path: str | Path | None = None,
        cassette_mode: Mode | None = None,
        cassette_inner_model: str | None = None,
        frame: Frame = "production-representative",
    ) -> None:
        self._model = model
        self._explicit_log_dir = log_dir
        self._cassette_path = None if cassette_path is None else Path(cassette_path)
        self._cassette_mode = cassette_mode or mode_from_env(default=Mode.AUTO)
        # The dataset frame the produced traces carry. Defaults to "production-representative"
        # (the InspectEvalLog default), but a caller synthesizing a discovery-frame dataset
        # passes frame="discovery" so the frame tag survives execution; the normalizer reads it
        # from the source, not the .eval log (frame propagation for the synthesis route-through).
        self._frame: Frame = frame
        # The real model a `cassette/...` run records against on a RECORD-mode miss. Explicit
        # (not derived from the cassette model string): `cassette/orderbot` is a logical replay
        # handle, not a Bedrock id, so `.replace("cassette/", "bedrock/")` would produce the
        # invalid `bedrock/orderbot`. Defaults to the provider's DEFAULT_INNER_MODEL when unset.
        self._cassette_inner_model = cassette_inner_model
        self._tmpdir: tempfile.TemporaryDirectory[str] | None = None

    @property
    def log_dir(self) -> Path:
        """The directory where ``.eval`` logs are persisted."""
        if self._explicit_log_dir is not None:
            return self._explicit_log_dir
        if self._tmpdir is None:
            self._tmpdir = tempfile.TemporaryDirectory(prefix="calibrated_agent_evaluation_eval_")
        return Path(self._tmpdir.name)

    def __enter__(self) -> InspectExecutor:
        return self

    def __exit__(self, *exc: object) -> None:
        if self._tmpdir is not None:
            self._tmpdir.cleanup()
            self._tmpdir = None

    def run(self, task: Any, *, log_dir: Path | None = None) -> list[Trace]:
        """Run an Inspect task and return normalized, validated Traces.

        Raises ``ExecutionError`` on any failure; eval error, cassette miss, unreachable model,
        or invalid trace from normalization.
        """
        eval_fn, _read_eval_log = self._load_inspect()
        effective_log_dir = log_dir or self.log_dir
        effective_log_dir.mkdir(parents=True, exist_ok=True)

        try:
            # Inside the try: building the Cassette loads (and validates) the file, so a corrupt
            # cassette raises CassetteError HERE; it must be wrapped as ExecutionError, never leak
            # to the caller (cassette error types are implementation details).
            self._configure_cassette_provider()
            logs = eval_fn(
                task,
                model=self._model,
                log_dir=str(effective_log_dir),
                score=False,
            )
        except (CassetteMiss, CassetteError) as exc:
            raise ExecutionError(f"cassette error during eval: {exc}") from exc
        except ExecutionError:
            raise
        except Exception as exc:
            raise ExecutionError(f"eval failed: {type(exc).__name__}: {exc}") from exc

        traces: list[Trace] = []
        for eval_log in logs:
            if eval_log.status != "success":
                error_detail = getattr(eval_log, "error", None)
                raise ExecutionError(
                    f"eval did not succeed: status={eval_log.status}, error={error_detail}"
                )
            log_path = self._resolve_log_path(eval_log, effective_log_dir)
            traces.extend(self._normalize_log(log_path))

        return traces

    def _configure_cassette_provider(self) -> None:
        """Wire the cassette into the Inspect model provider registry if using a cassette model.

        Imports the shipped ``cassette_provider`` (promoted from ``examples/`` so replay works for
        installed users). The import side-effect registers the ``@modelapi(name="cassette")``
        provider with Inspect. Constructing the ``Cassette`` here loads+validates the file, so a
        corrupt cassette raises inside ``run()``'s try block (wrapped as ``ExecutionError``).
        """
        if not self._model.startswith("cassette/"):
            return
        if self._cassette_path is None:
            raise ExecutionError(
                f"model {self._model!r} needs a cassette but no cassette_path was provided; "
                "pass cassette_path=... to InspectExecutor(...)"
            )
        # Build+validate the cassette (stdlib) FIRST; a corrupt file raises CassetteError here,
        # before the vendor provider import, so it fails fast and without needing [inspect].
        cassette = Cassette(self._cassette_path, mode=self._cassette_mode)
        from calibrated_agent_evaluation.execute import cassette_provider

        cassette_provider.configure(
            cassette,
            inner_model=self._cassette_inner_model or cassette_provider.DEFAULT_INNER_MODEL,
        )

    def _resolve_log_path(self, eval_log: Any, log_dir: Path) -> Path:
        """Find the .eval file Inspect wrote.

        The `location` the EvalLog reports is authoritative. The mtime fallback exists for an
        EvalLog that carries none, and it is only safe when the directory holds exactly one log:
        with several, `max(..., key=mtime)` can resolve a DIFFERENT run's log (two runs sharing a
        log_dir, or a directory reused across runs), and the caller would normalize the wrong
        transcript with no error. Ambiguity is refused rather than guessed.
        """
        location = getattr(eval_log, "location", None)
        if location and Path(location).exists():
            return Path(location)
        evals = list(log_dir.glob("*.eval"))
        if len(evals) == 1:
            return evals[0]
        if evals:
            raise ExecutionError(
                f"the eval log reported no location and {log_dir} holds {len(evals)} .eval files, "
                "so which one this run wrote is ambiguous; picking the newest could normalize a "
                "different run's transcript. Use a per-run log_dir"
            )
        raise ExecutionError(f"eval completed but no .eval log found in {log_dir}")

    def _normalize_log(self, log_path: Path) -> list[Trace]:
        """Normalize an .eval log into validated Traces: ONE PER SAMPLE.

        `normalize_all`, not `normalize`: a Task's dataset has one Sample per row, so an eval over N
        rows writes a log with N samples. This used to call `normalize`, which read the first sample
        and drained the rest, so `run()` returned 1 trace for an N-row eval and every rate computed
        downstream came from 1/N of the data with no error anywhere.
        """
        from calibrated_agent_evaluation.normalize import normalize_all
        from calibrated_agent_evaluation.normalize.sources import InspectEvalLog
        from calibrated_agent_evaluation.schema.validate import validate

        try:
            traces = normalize_all(InspectEvalLog(path=log_path, frame=self._frame))
        except Exception as exc:
            raise ExecutionError(
                f"normalization failed for {log_path}: {type(exc).__name__}: {exc}"
            ) from exc

        for trace in traces:
            try:
                validate(trace)
            except Exception as exc:
                raise ExecutionError(
                    f"normalized trace {trace.trace_id!r} failed validation: "
                    f"{type(exc).__name__}: {exc}"
                ) from exc

        return traces

    @staticmethod
    def _load_inspect() -> tuple[Any, Any]:
        """Lazy-import Inspect AI; raise ExecutionError if not installed."""
        try:
            from inspect_ai import eval as eval_fn
            from inspect_ai.log import read_eval_log
        except ImportError as exc:
            raise ExecutionError(
                f"inspect-ai is not installed. Install via: {install_hint('inspect')}"
            ) from exc
        return eval_fn, read_eval_log
