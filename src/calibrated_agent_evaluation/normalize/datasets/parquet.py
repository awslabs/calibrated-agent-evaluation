# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Streaming parquet dataset reader.

Reads a parquet dataset row-by-row via ``pyarrow.parquet.ParquetFile.iter_batches``; walking
**every** row group (never only the first; the silent-truncation guard), yielding one record dict
per row within a bounded working set (a multi-hundred-MB dataset is streamed, not loaded whole).
``pyarrow`` is imported lazily so a missing ``[datasets]`` extra surfaces as a typed
:class:`~calibrated_agent_evaluation.normalize.errors.NormalizationError` naming the installable
string, never a bare ``ImportError``; any read/decode error is wrapped typed and chained.
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path
from typing import Any

from calibrated_agent_evaluation._install import install_hint
from calibrated_agent_evaluation.normalize.errors import NormalizationError

_BATCH_ROWS = 64  # small batches keep the working set bounded while streaming


def iter_rows(path: Path) -> Iterator[dict[str, Any]]:
    """Yield one record dict per parquet row, across every row group.

    Raises :class:`NormalizationError` naming the ``[datasets]`` extra if ``pyarrow`` is missing,
    or wrapping any pyarrow/OS read error; never a raw exception escapes.
    """
    pq = _import_pyarrow_parquet()
    try:
        parquet_file = pq.ParquetFile(str(path))
    except FileNotFoundError as exc:
        raise NormalizationError(f"parquet dataset not found: {str(path)!r}: {exc}") from exc
    except OSError as exc:
        raise NormalizationError(f"could not open parquet dataset {str(path)!r}: {exc}") from exc
    except Exception as exc:  # pyarrow raises ArrowInvalid for a non-parquet file
        raise NormalizationError(f"{str(path)!r} is not a valid parquet dataset: {exc}") from exc

    try:
        for batch in parquet_file.iter_batches(batch_size=_BATCH_ROWS):
            yield from batch.to_pylist()
    except Exception as exc:  # a decode error mid-stream (corrupt row group)
        raise NormalizationError(f"failed reading parquet dataset {str(path)!r}: {exc}") from exc


def _import_pyarrow_parquet() -> Any:
    """Import ``pyarrow.parquet`` or raise a typed error naming the missing extra."""
    try:
        import pyarrow.parquet as pq
    except ImportError as exc:
        raise NormalizationError(
            "reading a parquet dataset requires the optional extra; install it with "
            f"`{install_hint('datasets')}`"
        ) from exc
    return pq
