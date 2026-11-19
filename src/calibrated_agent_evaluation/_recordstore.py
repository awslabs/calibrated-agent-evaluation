# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""One CSV+JSON round-trip shared by every record store.

Each record family (calibration, bias battery, run, improve, enforcement, flagship, judge run)
persists the same way: under a directory, ``<name>.json`` holds the records as a readable,
key-sorted list, and ``<name>.csv`` holds one row per record with one JSON-encoded value per
top-level field. Both files are always written; the caller picks which one to read back.

Loading refuses a stored ``schema_version`` that is missing or differs from the running one before
any record is returned, and wraps every filesystem, JSON, CSV and validation failure in the
family's own typed error, so no raw ``OSError`` or ``ValidationError`` reaches a caller.

Each file is written to a temporary sibling and moved into place, so a crash mid-write leaves the
previous file whole rather than a truncated one. The two files are two renames, so a crash between
them leaves them disagreeing; the file a load reads is written last, and a load that finds the two
disagreeing refuses rather than silently reading the one that missed the write. A caller that reads
the stored records, decides on them and writes them back holds :func:`directory_lock` across all
three steps.
"""

from __future__ import annotations

import csv
import io
import json
import os
import sys
import tempfile
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Generic, TypeVar

from pydantic import BaseModel, ValidationError

from calibrated_agent_evaluation._errors import CalibratedAgentEvaluationDataError
from calibrated_agent_evaluation._model import plain_error

__all__ = ["RecordStore", "atomic_write_text", "directory_lock", "widen_csv_field_limit"]

M = TypeVar("M", bound=BaseModel)


def atomic_write_text(path: Path, text: str) -> None:
    """Write ``text`` to ``path`` through a temporary sibling and an atomic rename.

    A reader sees either the old file or the complete new one, never a partial write.

    Args:
        path: The file to write; its directory must exist.
        text: The content.

    Raises:
        OSError: The write or the rename failed; the temporary file is removed.
    """
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp")
    try:
        with os.fdopen(fd, "w", newline="", encoding="utf-8") as fh:
            # mkstemp creates the file owner-only; give it the mode a plain open() would have.
            Path(tmp).chmod(0o666 & ~_UMASK)
            fh.write(text)
            fh.flush()
            os.fsync(fh.fileno())
        Path(tmp).replace(path)
    except BaseException:
        Path(tmp).unlink(missing_ok=True)
        raise


def widen_csv_field_limit() -> None:
    """Lift the csv module's 128 KiB per-field cap, which a single large tool result exceeds.

    Process-wide by the csv module's design. Called at import by every module that reads CSV, so it
    is write-once-at-startup state; the value only ever rises, so a second call changes nothing.
    """
    limit = sys.maxsize
    while True:
        try:
            csv.field_size_limit(limit)
        except OverflowError:  # the cap is a C long, narrower than sys.maxsize on some platforms
            limit //= 2
        else:
            return


@contextmanager
def directory_lock(directory: Path) -> Iterator[None]:
    """Hold an exclusive lock on ``directory`` for the ``with`` block.

    The lock is advisory, on ``<directory>/.lock`` (created if missing): ``fcntl.flock`` on POSIX,
    where another process waits until the holder leaves the block, and ``msvcrt.locking`` on
    Windows, which tries about once a second for about 10 seconds and then raises ``OSError``, so
    a longer holder makes the other process fail instead of wait. It is not reentrant: asking
    again inside the block waits forever on POSIX and fails after those 10 seconds on Windows.

    Args:
        directory: The directory to lock; it must exist.

    Raises:
        OSError: The lock file can't be opened (the directory is missing or not a directory, or
            can't be written), or the lock can't be taken (on Windows, another process held it
            for those 10 seconds). The caller wraps it in its own error.
    """
    with (directory / ".lock").open("a+b") as fh:
        _lock(fh.fileno())
        try:
            yield
        finally:
            _unlock(fh.fileno())


if os.name == "nt":  # pragma: no cover - exercised on Windows only
    import msvcrt

    def _lock(fd: int) -> None:
        os.lseek(fd, 0, os.SEEK_SET)
        msvcrt.locking(fd, msvcrt.LK_LOCK, 1)

    def _unlock(fd: int) -> None:
        os.lseek(fd, 0, os.SEEK_SET)
        msvcrt.locking(fd, msvcrt.LK_UNLCK, 1)

else:
    import fcntl

    def _lock(fd: int) -> None:
        fcntl.flock(fd, fcntl.LOCK_EX)

    def _unlock(fd: int) -> None:
        fcntl.flock(fd, fcntl.LOCK_UN)


def _read_umask() -> int:
    current = os.umask(0)
    os.umask(current)
    return current


# Read once at import: os.umask can only be read by setting it, which is not thread safe.
_UMASK = _read_umask()
widen_csv_field_limit()


class RecordStore(Generic[M]):
    """Dump and load one record model to ``<stem>.csv`` and ``<stem>.json`` under a directory.

    Args:
        model: The pydantic model persisted by this store.
        stem: File name stem, e.g. ``"calibration_record"``.
        schema_version: The running schema version every stored record must carry.
        error: The family's typed store error, raised for every failure.
        noun: How messages name a record, e.g. ``"calibration record"``.
        version_error: A narrower error for a schema-version mismatch; defaults to ``error``.
        rerun: The command that rebuilds these records, named in a version refusal.
    """

    def __init__(
        self,
        model: type[M],
        *,
        stem: str,
        schema_version: str,
        error: type[Exception],
        noun: str,
        version_error: type[Exception] | None = None,
        rerun: str | None = None,
    ) -> None:
        self._model = model
        self._csv = f"{stem}.csv"
        self._json = f"{stem}.json"
        self._version = schema_version
        self._error = error
        self._version_error = version_error or error
        self._noun = noun
        self._rerun = rerun

    def dump(self, records: Sequence[M], path: str | Path, fmt: str = "csv") -> None:
        """Write ``records`` under ``path``; both files are always written.

        Args:
            records: The records to persist.
            path: The directory to write into; created if missing.
            fmt: ``"csv"`` or ``"json"``, the format a later load reads back.

        Raises:
            Exception: The store's typed error for a bad ``fmt`` or any filesystem failure.
        """
        if fmt not in {"csv", "json"}:
            raise self._error(f"fmt must be 'csv' or 'json', got {fmt!r}")
        docs = [r.model_dump(mode="json") for r in records]
        out = Path(path)
        rows = io.StringIO(newline="")
        writer = csv.DictWriter(rows, fieldnames=list(self._model.model_fields))
        writer.writeheader()
        writer.writerows({k: json.dumps(v) for k, v in doc.items()} for doc in docs)
        files = {
            self._json: json.dumps(docs, indent=2, sort_keys=True),
            self._csv: rows.getvalue(),
        }
        read_back = self._json if fmt == "json" else self._csv
        try:
            out.mkdir(parents=True, exist_ok=True)
            for name in sorted(files, key=lambda n: n == read_back):
                atomic_write_text(out / name, files[name])
        except OSError as exc:
            raise self._error(f"failed to write {self._noun}s to {out}: {exc}") from exc

    def load(self, path: str | Path, fmt: str = "csv") -> list[M]:
        """Read the records under ``path``; a missing file is an empty store.

        Args:
            path: The directory a previous :meth:`dump` wrote.
            fmt: ``"csv"`` or ``"json"``.

        Returns:
            The records, in stored order.

        Raises:
            Exception: The store's version error for a stored ``schema_version`` that differs from
                the running one, or its typed error for a missing one and for any read, parse or
                validation failure.
        """
        src = Path(path)
        docs = self._read(src, fmt)
        for doc in docs:
            if not isinstance(doc, dict):
                raise self._error(
                    f"{self._noun} in {src} decoded to a {type(doc).__name__}, not an object; "
                    "refusing to load a malformed artifact"
                )
            if "schema_version" not in doc:
                raise self._error(
                    f"{self._noun} in {src} carries no schema_version; refusing to load a record "
                    "whose format cannot be checked (no coercion)"
                )
            stored = doc["schema_version"]
            if stored != self._version:
                raise self._version_error(self._version_message(src, stored))
        try:
            records = [self._model.model_validate(doc) for doc in docs]
        except ValidationError as exc:
            raise self._error(
                f"malformed {self._noun} in {src} ({plain_error(exc)}); "
                "refusing to load a partial record"
            ) from exc
        except CalibratedAgentEvaluationDataError as exc:
            if isinstance(exc, self._error):
                raise
            raise self._error(
                f"malformed {self._noun} in {src} ({exc}); refusing to load a partial record"
            ) from exc
        self._refuse_desync(src, fmt, docs)
        return records

    def _refuse_desync(self, src: Path, fmt: str, docs: list[dict[str, Any]]) -> None:
        """Refuse when the other copy exists and holds different records (an interrupted write)."""
        other = "csv" if fmt == "json" else "json"
        if not (src / (self._json if other == "json" else self._csv)).exists():
            return
        if self._read(src, other) == docs:
            return
        raise self._error(
            f"the {self._csv} and {self._json} copies of the {self._noun}s in {src} disagree, "
            "which an interrupted write leaves behind; refusing to read either as the truth. "
            "Inspect both, keep the one that holds the latest records, and delete the other"
        )

    def _version_message(self, src: Path, stored: object) -> str:
        rerun = f"`{self._rerun}`" if self._rerun else "the stage that wrote them"
        return (
            f"the {self._noun}s in {src} were written by a different version (schema_version "
            f"{stored!r}); this version reads "
            f"schema_version {self._version!r} and does not convert them (no coercion). "
            f"Move {src} aside and re-run {rerun} to rebuild them."
        )

    def _read(self, src: Path, fmt: str) -> list[dict[str, Any]]:
        if fmt not in {"csv", "json"}:
            raise self._error(f"fmt must be 'csv' or 'json', got {fmt!r}")
        file = src / (self._json if fmt == "json" else self._csv)
        try:
            if not file.exists():
                return []
            if fmt == "json":
                loaded = json.loads(file.read_text(encoding="utf-8"))
            else:
                with file.open(newline="", encoding="utf-8") as fh:
                    reader = csv.DictReader(fh)
                    return [self._row(file, reader.line_num, row) for row in reader]
        except self._error:
            # A malformed row's refusal, whatever family the store's error belongs to.
            raise
        except OSError as exc:
            raise self._error(f"failed to read {self._noun}s from {src}: {exc}") from exc
        except (ValueError, TypeError, RecursionError) as exc:
            # ValueError covers JSONDecodeError and UnicodeDecodeError, and a number too long to
            # read; RecursionError, a document nested deeper than the decoder can go.
            raise self._error(f"{self._noun} file in {src} is not valid JSON: {exc}") from exc
        except csv.Error as exc:
            raise self._error(f"{self._noun} CSV in {src} is malformed: {exc}") from exc
        if not isinstance(loaded, list):
            raise self._error(f"{self._noun} file {file} does not hold a list of records")
        return loaded

    def _row(self, file: Path, line: int, row: dict[Any, Any]) -> dict[str, Any]:
        """One CSV row decoded; a row with too many or too few cells is refused."""
        if None in row or None in row.values():
            raise self._error(
                f"{self._noun} CSV {file} line {line} has a different number of cells than its "
                "header; refusing to load a malformed record"
            )
        return {k: json.loads(v) for k, v in row.items()}
