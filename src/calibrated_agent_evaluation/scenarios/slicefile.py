# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Read and write slice files and answer keys: plain JSON, written atomically."""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import TypeVar

from calibrated_agent_evaluation._model import CalibratedAgentEvaluationModel
from calibrated_agent_evaluation._recordstore import atomic_write_text
from calibrated_agent_evaluation.scenarios.errors import ScenarioError
from calibrated_agent_evaluation.scenarios.types import (
    SLICE_SCHEMA_VERSION,
    AnswerKey,
    DatasetSlice,
)

__all__ = ["read_answer_key", "read_slice", "write_json_model", "write_slice"]

M = TypeVar("M", bound=CalibratedAgentEvaluationModel)


def read_slice(path: str | Path) -> DatasetSlice:
    """The slice in the file at ``path``; a missing, unparsable or invalid file is refused.

    The schema version is checked first, so a file from another version is named as such rather
    than reported through whichever field changed shape.
    """
    return _read(Path(path), DatasetSlice, "slice file", version=SLICE_SCHEMA_VERSION)


def read_answer_key(path: str | Path) -> AnswerKey:
    """The answer key in the file at ``path``; unknown fields and bad JSON are refused."""
    return _read(Path(path), AnswerKey, "answer key")


def write_slice(path: str | Path, slice_: DatasetSlice, *, force: bool = False) -> None:
    """Write ``slice_`` to ``path``, refusing to replace an existing file unless ``force``."""
    write_json_model(Path(path), slice_, force=force)


def write_json_model(
    path: Path, model: CalibratedAgentEvaluationModel, *, force: bool = False
) -> None:
    """Write ``model`` as indented JSON to ``path`` atomically; an existing file needs ``force``."""
    text = json.dumps(model.model_dump(mode="json"), indent=2, ensure_ascii=False) + "\n"
    try:
        if force:
            atomic_write_text(path, text)
        else:
            _create_atomically(path, text)
    except FileExistsError:
        raise ScenarioError(f"{path} already exists; pass --force to replace it") from None
    except OSError as exc:
        raise ScenarioError(f"cannot write {path}: {exc}") from exc


def _create_atomically(path: Path, text: str) -> None:
    """Write a complete temp sibling, then link it into place only if ``path`` is still absent.

    A reader never sees a partial file, and a file that appears after any earlier check is kept
    (the link raises ``FileExistsError``).
    """
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="") as fh:
            fh.write(text)
        os.link(tmp, path)
    finally:
        Path(tmp).unlink(missing_ok=True)


def _read(path: Path, model: type[M], what: str, *, version: str | None = None) -> M:
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise ScenarioError(f"cannot read the {what} {path}: {exc}") from exc
    try:
        data = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ScenarioError(
            f"the {what} {path} is not valid JSON "
            f"(line {exc.lineno}, column {exc.colno}): {exc.msg}"
        ) from exc
    if not isinstance(data, dict):
        raise ScenarioError(f"the {what} {path} must hold a JSON object, got {type(data).__name__}")
    found = data.get("schema_version", version)
    if version is not None and found != version:
        raise ScenarioError(
            f"the {what} {path} has schema_version {found!r}; this Calibrated Agent Evaluation "
            f"reads {version!r}"
        )
    try:
        return model.model_validate(data)
    except ScenarioError as exc:
        raise ScenarioError(f"the {what} {path} is invalid: {exc}") from exc
