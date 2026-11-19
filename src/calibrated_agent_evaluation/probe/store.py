# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""CSV+JSON round-trip for the probe battery record.

A battery record is not a ``TraceStoreBackend`` object, so the backend-driven ``dump()`` cannot
carry it. ``dump_probes`` writes ``probe_record.{csv,json}`` under a directory and ``load_probes``
reads them back; a stored ``schema_version`` other than the running one, and every read or parse
failure, raises :class:`~calibrated_agent_evaluation.probe.errors.ProbeError`.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

from calibrated_agent_evaluation._recordstore import RecordStore
from calibrated_agent_evaluation.probe.errors import ProbeError
from calibrated_agent_evaluation.probe.record import ProbeBatteryRecord
from calibrated_agent_evaluation.probe.version import PROBE_SCHEMA_VERSION

__all__ = ["dump_probes", "load_probes"]

_STORE: RecordStore[ProbeBatteryRecord] = RecordStore(
    ProbeBatteryRecord,
    stem="probe_record",
    schema_version=PROBE_SCHEMA_VERSION,
    error=ProbeError,
    noun="probe battery record",
    rerun="calibrated-agent-evaluation probe --battery",
)


def dump_probes(records: Sequence[ProbeBatteryRecord], path: str | Path, fmt: str = "csv") -> None:
    """Write ``records`` to ``probe_record.{csv,json}`` under ``path``.

    Args:
        records: The battery records to persist.
        path: The directory to write into.
        fmt: Which file a later :func:`load_probes` reads back (both are always written).

    Raises:
        ProbeError: A bad ``fmt`` or any filesystem failure.
    """
    _STORE.dump(records, path, fmt)


def load_probes(path: str | Path, fmt: str = "csv") -> list[ProbeBatteryRecord]:
    """Read probe battery records back from ``path``; a missing file is an empty store.

    Args:
        path: The directory :func:`dump_probes` wrote.
        fmt: ``"csv"`` or ``"json"``.

    Returns:
        The stored records.

    Raises:
        ProbeError: A schema-version mismatch, or any read, parse or validation failure.
    """
    return _STORE.load(path, fmt)
