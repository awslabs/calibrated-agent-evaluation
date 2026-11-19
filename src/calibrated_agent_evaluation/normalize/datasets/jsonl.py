# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""JSON-lines dataset reader.

One JSON object per line → one record dict (→ one ``Trace``). Uses the stdlib ``json`` only (zero
dependency; no extra needed). Blank/whitespace-only lines are skipped; a malformed JSON line is
surfaced as a typed :class:`~calibrated_agent_evaluation.normalize.errors.NormalizationError`
chained from the underlying ``JSONDecodeError``, naming the 1-based line number; never a raw
exception.
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path
from typing import Any

from calibrated_agent_evaluation.normalize.errors import NormalizationError


def iter_rows(path: Path) -> Iterator[dict[str, Any]]:
    """Yield one record dict per non-blank JSONL line; typed on a bad line."""
    try:
        handle = path.open(encoding="utf-8")
    except OSError as exc:
        raise NormalizationError(f"could not open JSONL dataset {str(path)!r}: {exc}") from exc
    with handle:
        for lineno, line in enumerate(handle, start=1):
            if not line.strip():
                continue  # skip blank / whitespace-only lines
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise NormalizationError(
                    f"{str(path)!r} line {lineno} is not valid JSON: {exc}"
                ) from exc
            if not isinstance(record, dict):
                kind = type(record).__name__
                raise NormalizationError(
                    f"{str(path)!r} line {lineno} is not a JSON object (got {kind})"
                )
            yield record
