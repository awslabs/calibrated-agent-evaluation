# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""CSV+JSON round-trip for the FlagshipReport.

``dump_flagship_reports`` writes ``flagship_report.csv`` and ``flagship_report.json`` under a
directory and ``load_flagship_reports`` reads them back with every stage outcome, oracle check,
gate reference and world-parity result rebuilt. A stored ``schema_version`` other than the running
``FLAGSHIP_SCHEMA_VERSION``, and every read or parse failure, raises :class:`FlagshipStoreError`.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

from calibrated_agent_evaluation._recordstore import RecordStore
from calibrated_agent_evaluation.flagship.errors import FlagshipStoreError
from calibrated_agent_evaluation.flagship.report import FLAGSHIP_SCHEMA_VERSION, FlagshipReport

__all__ = ["dump_flagship_reports", "load_flagship_reports"]

_STORE = RecordStore(
    FlagshipReport,
    stem="flagship_report",
    schema_version=FLAGSHIP_SCHEMA_VERSION,
    error=FlagshipStoreError,
    noun="flagship report",
    rerun="the flagship proof",
)


def dump_flagship_reports(
    reports: Sequence[FlagshipReport], path: str | Path, fmt: str = "csv"
) -> None:
    """Write ``reports`` to ``flagship_report.{csv,json}`` under ``path``.

    Args:
        reports: The flagship reports to persist.
        path: The directory to write into.
        fmt: Which file a later :func:`load_flagship_reports` reads back (both are always written).

    Raises:
        FlagshipStoreError: A bad ``fmt`` or any filesystem failure.
    """
    _STORE.dump(reports, path, fmt)


def load_flagship_reports(path: str | Path, fmt: str = "csv") -> list[FlagshipReport]:
    """Read flagship reports back from ``path``; a missing file is an empty store.

    Args:
        path: The directory :func:`dump_flagship_reports` wrote.
        fmt: ``"csv"`` or ``"json"``.

    Returns:
        The stored reports.

    Raises:
        FlagshipStoreError: A schema-version mismatch, or any read, parse or validation failure.
    """
    return _STORE.load(path, fmt)
