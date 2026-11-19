# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""CSV+JSON round-trip for the BiasBatteryRecord.

A bias-battery record is not a ``TraceStoreBackend`` object, so the backend-driven ``dump()``
(which collects only Trace/Span/Label) cannot carry it. ``dump_battery`` writes
``bias_battery_record.{csv,json}`` under a directory and ``load_battery`` reads them back; a
stored ``schema_version`` other than the running one, and every read or parse failure, raises
:class:`BiasStoreError`.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

from calibrated_agent_evaluation._recordstore import RecordStore
from calibrated_agent_evaluation.bias.errors import BiasStoreError
from calibrated_agent_evaluation.bias.record import BiasBatteryRecord
from calibrated_agent_evaluation.bias.version import BIAS_SCHEMA_VERSION

__all__ = ["dump_battery", "load_battery"]

_STORE = RecordStore(
    BiasBatteryRecord,
    stem="bias_battery_record",
    schema_version=BIAS_SCHEMA_VERSION,
    error=BiasStoreError,
    noun="bias battery record",
    rerun="calibrated-agent-evaluation bias",
)


def dump_battery(records: Sequence[BiasBatteryRecord], path: str | Path, fmt: str = "csv") -> None:
    """Write ``records`` to ``bias_battery_record.{csv,json}`` under ``path``.

    Args:
        records: The battery records to persist.
        path: The directory to write into.
        fmt: Which file a later :func:`load_battery` reads back (both are always written).

    Raises:
        BiasStoreError: A bad ``fmt`` or any filesystem failure.
    """
    _STORE.dump(records, path, fmt)


def load_battery(path: str | Path, fmt: str = "csv") -> list[BiasBatteryRecord]:
    """Read bias battery records back from ``path``; a missing file is an empty store.

    Args:
        path: The directory :func:`dump_battery` wrote.
        fmt: ``"csv"`` or ``"json"``.

    Returns:
        The stored records.

    Raises:
        BiasStoreError: A schema-version mismatch, or any read, parse or validation failure.
    """
    return _STORE.load(path, fmt)
