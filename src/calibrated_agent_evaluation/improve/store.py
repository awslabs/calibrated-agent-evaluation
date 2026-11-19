# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""CSV+JSON round-trip for the ImproveRecord, and the pre-registration ledger file.

``dump_improve_records`` writes ``improve_record.csv`` and ``improve_record.json`` under a
directory and ``load_improve_records`` reads them back with every nested part (pre-registration
entry, McNemar result, guard panel and its relocation result) rebuilt. A stored ``schema_version``
other than the running ``IMPROVE_SCHEMA_VERSION``, and every read or parse failure, raises
:class:`ImproveStoreError`.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path

from pydantic import TypeAdapter, ValidationError

from calibrated_agent_evaluation._model import plain_error
from calibrated_agent_evaluation._recordstore import RecordStore, atomic_write_text
from calibrated_agent_evaluation.improve.errors import (
    ImproveError,
    ImproveStoreError,
    LedgerIntegrityError,
)
from calibrated_agent_evaluation.improve.report import (
    IMPROVE_SCHEMA_VERSION,
    ImproveRecord,
    PreRegEntry,
    PreRegLedger,
)

__all__ = [
    "dump_improve_records",
    "dump_prereg_ledger",
    "load_improve_records",
    "load_prereg_ledger",
]

_STORE = RecordStore(
    ImproveRecord,
    stem="improve_record",
    schema_version=IMPROVE_SCHEMA_VERSION,
    error=ImproveStoreError,
    noun="improve record",
    rerun="calibrated-agent-evaluation improve",
)
_LEDGER_NAME = "prereg_ledger.json"
_ENTRIES = TypeAdapter(tuple[PreRegEntry, ...])


def dump_improve_records(
    records: Sequence[ImproveRecord], path: str | Path, fmt: str = "csv"
) -> None:
    """Write ``records`` to ``improve_record.{csv,json}`` under ``path``.

    Args:
        records: The improve records to persist.
        path: The directory to write into.
        fmt: Which file a later :func:`load_improve_records` reads back (both are always written).

    Raises:
        ImproveStoreError: A bad ``fmt`` or any filesystem failure.
    """
    _STORE.dump(records, path, fmt)


def load_improve_records(path: str | Path, fmt: str = "csv") -> list[ImproveRecord]:
    """Read improve records back from ``path``; a missing file is an empty store.

    Args:
        path: The directory :func:`dump_improve_records` wrote.
        fmt: ``"csv"`` or ``"json"``.

    Returns:
        The stored records.

    Raises:
        ImproveStoreError: A schema-version mismatch, or any read, parse or validation failure.
    """
    return _STORE.load(path, fmt)


def dump_prereg_ledger(ledger: PreRegLedger, path: str | Path) -> None:
    """Write ``ledger`` to ``prereg_ledger.json`` under ``path``, entries in ledger order.

    Every entry keeps its ``content_hash``, ``frozen_at`` and ``prev_hash`` as stamped, so
    :func:`load_prereg_ledger` returns a ledger
    :func:`~calibrated_agent_evaluation.improve.verify_ledger` accepts iff the original did. The
    ledger is append-only on disk too: a ledger that changes or drops an entry already in the file
    is refused, which is how an edited and re-frozen last entry would otherwise be saved.

    Args:
        ledger: The ledger to persist.
        path: The directory to write into.

    Raises:
        LedgerIntegrityError: ``ledger`` does not extend the ledger already in ``path``.
        ImproveStoreError: Any filesystem failure, or a stored ledger that cannot be read.
    """
    out = Path(path)
    stored = load_prereg_ledger(out).entries
    if ledger.entries[: len(stored)] != stored:
        changed = next(
            (i for i, e in enumerate(stored) if i >= len(ledger.entries) or ledger.entries[i] != e),
            len(stored),
        )
        raise LedgerIntegrityError(
            f"refusing to write the pre-registration ledger to {out}: entry {changed} "
            f"({stored[changed].target_metric!r}) is already frozen there and this ledger "
            f"{'drops' if changed >= len(ledger.entries) else 'changes'} it. The ledger is "
            "append-only; register a new metric instead of editing a frozen one."
        )
    try:
        out.mkdir(parents=True, exist_ok=True)
        docs = _ENTRIES.dump_python(ledger.entries, mode="json")
        atomic_write_text(out / _LEDGER_NAME, json.dumps(docs, indent=2, sort_keys=True))
    except OSError as exc:
        raise ImproveStoreError(
            f"failed to write the pre-registration ledger to {out}: {exc}"
        ) from exc


def load_prereg_ledger(path: str | Path) -> PreRegLedger:
    """Read the ledger written by :func:`dump_prereg_ledger`; an absent file is an empty ledger.

    Returns the entries exactly as stored and does not verify them: the improve gate's
    :func:`~calibrated_agent_evaluation.improve.require_prereg` does, so an edited file is refused
    there with the entry it broke.

    Args:
        path: The directory :func:`dump_prereg_ledger` wrote.

    Returns:
        The stored ledger.

    Raises:
        ImproveStoreError: Any read, parse or validation failure.
    """
    file = Path(path) / _LEDGER_NAME
    if not file.exists():
        return PreRegLedger()
    try:
        return PreRegLedger(entries=_ENTRIES.validate_json(file.read_bytes()))
    except OSError as exc:
        raise ImproveStoreError(
            f"failed to read the pre-registration ledger {file}: {exc}"
        ) from exc
    except ValidationError as exc:
        kind = "is not valid JSON" if _is_json_error(exc) else "is malformed"
        raise ImproveStoreError(
            f"pre-registration ledger {file} {kind}: {plain_error(exc)}"
        ) from exc
    except ImproveError as exc:
        raise ImproveStoreError(f"pre-registration ledger {file} is malformed: {exc}") from exc


def _is_json_error(exc: ValidationError) -> bool:
    return any(err["type"] == "json_invalid" for err in exc.errors())
