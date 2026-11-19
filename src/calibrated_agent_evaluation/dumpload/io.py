# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""dump / load: the lossless CSV+JSON round-trip that proves no lock-in.

``dump(backend, path, fmt)`` writes every artifact of every **registered** type to
``<name>.csv`` and ``<name>.json`` under ``path``; ``load(backend, path)`` reads them back
and re-persists them into a backend. A dump→wipe→load cycle reconstructs the workspace
field-for-field. This layer ships the mechanism + the core ``Trace`` / ``Span`` / ``Label``
codecs; downstream features register their own types, which then participate
automatically (the registry is iterated, never a hardcoded type list).

Round-trip model: a ``Trace`` carries its own span tree, so persisting a trace also persists
its spans (``put_trace``). ``Span`` is still a registered type; its rows are dumped (so the
registry covers it and its codec round-trips) and on load each span is verified
to belong to a loaded trace; spans are **not** re-``put`` independently (that would orphan
them), they ride back in via their trace. Labels are dumped and re-``put`` directly.

A dump is sealed by ``dump-manifest.json``, written last: the dump format version and, for every
data file the dump wrote, its record count and sha256. ``load`` reads nothing from a directory
whose files do not match it, so an interrupted dump (no manifest), a dump overwritten in part by
another (a digest that differs) and a stray data file (one the manifest does not list) are each a
typed :class:`CorruptDumpError`. Data files are the ``.csv`` and ``.json`` files outside hidden
directories; lock files, temporary files and a README beside a committed dump are not part of it.

``load`` checks each artifact's ``schema_version`` against the running ``SCHEMA_VERSION`` and
raises :class:`SchemaVersionError` *before* writing anything on a mismatch. An
unregistered type at dump/load is :class:`UnregisteredArtifactError`, never a silent skip.
"""

from __future__ import annotations

import csv
import hashlib
import io
import json
from collections.abc import Callable
from pathlib import Path
from typing import Any, ClassVar, Protocol

from pydantic import ConfigDict, Field

from calibrated_agent_evaluation._model import CalibratedAgentEvaluationModel
from calibrated_agent_evaluation._recordstore import atomic_write_text, widen_csv_field_limit
from calibrated_agent_evaluation.dumpload.registry import (
    Codec,
    codec_for,
    is_registered,
    register,
    registered_types,
)
from calibrated_agent_evaluation.schema.errors import (
    CalibratedAgentEvaluationDataError,
    SchemaVersionError,
    UnregisteredArtifactError,
)
from calibrated_agent_evaluation.schema.serde import (
    label_from_dict,
    span_from_dict,
    to_dict,
    trace_from_dict,
)
from calibrated_agent_evaluation.schema.types import SCHEMA_VERSION, Label, Span, Trace
from calibrated_agent_evaluation.schema.validate import validate


class CorruptDumpError(CalibratedAgentEvaluationDataError):
    """The ``load`` path is not a loadable dump.

    A file is not valid JSON, the path is missing, is not a directory, or holds no dump file, or
    the directory does not match its manifest (the subclasses below name which way).

    Raised instead of letting a raw ``json.JSONDecodeError`` escape, or a wrong path load nothing:
    a corrupt dump is a
    data-plane failure and must surface inside the :class:`CalibratedAgentEvaluationDataError`
    family so callers can catch the whole family at the plane boundary ("no silent failure").
    """


class IncompleteDumpError(CorruptDumpError):
    """The directory holds dump files but no manifest.

    ``dump`` writes the manifest after every data file, so a dump that was interrupted, or is still
    being written, reads this way. So does a dump written before dumps carried a manifest.
    """


class DumpMismatchError(CorruptDumpError):
    """A file the manifest lists is missing, or its contents differ from what the manifest records.

    A dump overwritten in part by another, or a file edited after the dump, reads this way.
    ``path`` is the file, relative to the dump directory.
    """

    def __init__(self, message: str, *, path: str) -> None:
        super().__init__(message)
        self.path = path


class UnlistedDumpFileError(CorruptDumpError):
    """The directory holds files the manifest does not list; ``paths`` names them."""

    def __init__(self, message: str, *, paths: tuple[str, ...]) -> None:
        super().__init__(message)
        self.paths = paths


MANIFEST_NAME = "dump-manifest.json"
DUMP_FORMAT_VERSION = "1"


class DumpFileEntry(CalibratedAgentEvaluationModel):
    """One file of a dump: how many records it holds and its sha256."""

    model_config = ConfigDict(frozen=True, extra="forbid")
    invalid_error: ClassVar[type[Exception]] = CorruptDumpError

    records: int = Field(ge=0)
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")


class DumpManifest(CalibratedAgentEvaluationModel):
    """The manifest that seals a dump, keyed by each file's POSIX path relative to the dump."""

    model_config = ConfigDict(frozen=True, extra="forbid")
    invalid_error: ClassVar[type[Exception]] = CorruptDumpError

    format_version: str = DUMP_FORMAT_VERSION
    files: dict[str, DumpFileEntry]


class _RoundTripBackend(Protocol):
    """The backend surface dump/load needs (the LocalBackend provides it)."""

    def all_traces(self) -> list[Trace]: ...
    def all_scores(self) -> list[Label]: ...
    def put_trace(self, trace: Trace) -> None: ...
    def put_score(self, label: Label) -> None: ...


# --- built-in codecs for the core artifact types -----------------------------------------
#
# A row is a flat dict[str, str]: each top-level field is JSON-encoded into one cell, so
# nested objects (spans, tool_calls, taxonomy_cell, expected) survive a CSV round-trip.


def _to_row(obj: Trace | Span | Label) -> dict[str, str]:
    return {k: json.dumps(v) for k, v in to_dict(obj).items()}


def _row_to_doc(row: dict[str, str]) -> dict[str, Any]:
    # Each cell is a JSON-encoded string (CSV flattens one level). A corrupt cell must surface as
    # a typed CorruptDumpError, never a raw json.JSONDecodeError escaping the
    # CalibratedAgentEvaluationDataError family; the JSON load path already wraps; the CSV path
    # decodes here and must match it.
    doc: dict[str, Any] = {}
    for k, v in row.items():
        if v == "":
            # Every written cell is JSON, never empty; an empty cell is a field this row left
            # out (an unset trace session_id), so the model's default applies. A required field
            # left empty is still refused, by the model.
            continue
        try:
            doc[k] = json.loads(v)
        except json.JSONDecodeError as exc:
            raise CorruptDumpError(f"dump cell {k!r} is not valid JSON: {exc}") from exc
    return doc


def _trace_to_row(t: Trace) -> dict[str, str]:
    return _to_row(t)


def _trace_from_row(row: dict[str, str]) -> Trace:
    return trace_from_dict(_row_to_doc(row))


def _span_to_row(s: Span) -> dict[str, str]:
    return _to_row(s)


def _span_from_row(row: dict[str, str]) -> Span:
    return span_from_dict(_row_to_doc(row))


def _label_to_row(label: Label) -> dict[str, str]:
    return _to_row(label)


def _label_from_row(row: dict[str, str]) -> Label:
    return label_from_dict(_row_to_doc(row))


def register_core_codecs() -> None:
    """Register the built-in ``Trace`` / ``Span`` / ``Label`` codecs (idempotent)."""
    if not is_registered(Trace):
        register(Trace, _trace_to_row, _trace_from_row, name="trace")
    if not is_registered(Span):
        register(Span, _span_to_row, _span_from_row, name="span")
    if not is_registered(Label):
        register(Label, _label_to_row, _label_from_row, name="label")


def _collect(backend: _RoundTripBackend, artifact_type: type) -> list[Any]:
    """Return every artifact of ``artifact_type`` currently in the backend (dump source)."""
    if artifact_type is Trace:
        return list(backend.all_traces())
    if artifact_type is Span:
        return [span for trace in backend.all_traces() for span in trace.spans]
    if artifact_type is Label:
        return list(backend.all_scores())
    # A registered downstream type with no backend collector cannot be dumped from the store
    # generically; downstream features that add a type also extend collection. Surface loudly.
    raise UnregisteredArtifactError(
        f"no dump collector for registered type {artifact_type.__name__!r}; "
        "the owning feature must provide one"
    )


# --- dump ---------------------------------------------------------------------------------


def dump(
    backend: _RoundTripBackend,
    path: str | Path,
    fmt: str = "csv",
    *,
    extra: Callable[[Path], object] | None = None,
) -> DumpManifest:
    """Write every registered artifact type to ``<name>.csv`` and ``<name>.json`` under ``path``.

    Both formats are always written (the JSON is the lossless source on load; the CSV is the
    human-inspectable mirror). ``fmt`` selects which one ``load`` reads back. ``extra``, when
    given, is called with the dump directory after the artifact files, to write more files into
    the dump (the CLI writes the stage records); the manifest seals those too.

    Every file is written atomically, and the manifest last, so a dump that does not finish has
    no manifest and ``load`` refuses it. An earlier dump in ``path`` is removed first: its
    manifest and the files it lists, and nothing else.

    Returns:
        The manifest written to ``dump-manifest.json``.
    """
    if fmt not in {"csv", "json"}:
        raise ValueError(f"fmt must be 'csv' or 'json', got {fmt!r}")
    register_core_codecs()
    out = Path(path)
    out.mkdir(parents=True, exist_ok=True)
    _remove_previous_dump(out)
    # Each writer replaces its file by rename, so a file this dump wrote has a new stamp. A data
    # file it did not write stays out of the manifest, and load refuses the directory over it.
    untouched = {rel: _stamp(out / rel) for rel in _dump_files(out)}
    for artifact_type in registered_types():
        codec = codec_for(artifact_type)  # raises UnregisteredArtifactError if missing
        rows = [codec.to_row(a) for a in _collect(backend, artifact_type)]
        _write_json(out / f"{codec.name}.json", rows)
        _write_csv(out / f"{codec.name}.csv", rows, codec)
    if extra is not None:
        extra(out)
    written = [rel for rel in _dump_files(out) if untouched.get(rel) != _stamp(out / rel)]
    manifest = DumpManifest(files={rel: _entry(rel, (out / rel).read_bytes()) for rel in written})
    _write_manifest(out, manifest)
    return manifest


def _write_manifest(out: Path, manifest: DumpManifest) -> None:
    atomic_write_text(
        out / MANIFEST_NAME,
        json.dumps(manifest.model_dump(mode="json"), indent=2, sort_keys=True) + "\n",
    )


def _remove_previous_dump(out: Path) -> None:
    """Remove an earlier dump in ``out``: its manifest, then each file the manifest lists.

    Left in place, a file the earlier dump wrote and this one does not (a record kind the store no
    longer has) would make this dump unloadable. Files the manifest does not list are kept.
    """
    try:
        previous = _read_manifest(out)
    except CorruptDumpError:
        return
    if previous is None:
        return
    (out / MANIFEST_NAME).unlink()
    root = out.resolve()
    for rel in previous.files:
        target = (out / rel).resolve()
        if target.is_relative_to(root) and target != root and target.is_file():
            target.unlink()


def _dump_files(src: Path) -> list[str]:
    """Every data file under ``src``, as sorted POSIX paths relative to it.

    A data file is a ``.csv`` or ``.json`` file outside any hidden directory; the manifest, hidden
    files and other files (a README beside a committed dump) are not.
    """
    found = []
    for p in src.rglob("*"):
        rel = p.relative_to(src)
        if any(part.startswith(".") for part in rel.parts) or rel.as_posix() == MANIFEST_NAME:
            continue
        if p.suffix in _DATA_SUFFIXES and p.is_file():
            found.append(rel.as_posix())
    return sorted(found)


_DATA_SUFFIXES = frozenset({".csv", ".json"})


def _stamp(path: Path) -> tuple[int, int, int]:
    st = path.stat()
    return (st.st_ino, st.st_mtime_ns, st.st_size)


def _entry(rel: str, data: bytes) -> DumpFileEntry:
    return DumpFileEntry(records=_count_records(rel, data), sha256=hashlib.sha256(data).hexdigest())


def _count_records(rel: str, data: bytes) -> int:
    """Rows in a CSV; items in a JSON list, or one for any other JSON value."""
    text = data.decode("utf-8")
    if rel.endswith(".csv"):
        return sum(1 for _ in csv.DictReader(io.StringIO(text, newline="")))
    doc = json.loads(text)
    return len(doc) if isinstance(doc, list) else 1


def _write_json(path: Path, rows: list[dict[str, str]]) -> None:
    # JSON cells hold already-JSON-encoded strings; decode one level so the file is readable
    # and itself a faithful object dump.
    decoded = [{k: json.loads(v) for k, v in row.items()} for row in rows]
    atomic_write_text(path, json.dumps(decoded, indent=2, sort_keys=True))


def _write_csv(path: Path, rows: list[dict[str, str]], codec: Codec) -> None:
    # Every key any row has, in first-seen order: a field left out when unset (a trace's
    # session_id) is in some rows only. A row without it gets an empty cell, read back as absent.
    fieldnames = list(dict.fromkeys(k for row in rows for k in row)) or _codec_fieldnames(codec)
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(rows)
    atomic_write_text(path, buffer.getvalue())


def _codec_fieldnames(codec: Codec) -> list[str]:
    # An empty store still produces a loadable (header-only) CSV. Derive the columns from a
    # zero-value instance of the core types so the file is well-formed.
    sample = _SAMPLE_ROWS.get(codec.name)
    return list(sample.keys()) if sample else []


# Header templates for empty dumps (edge case: dump of an empty store stays loadable).
_SAMPLE_ROWS: dict[str, dict[str, str]] = {}


def _init_sample_rows() -> None:
    if _SAMPLE_ROWS:
        return
    empty_trace = Trace(
        trace_id="x",
        frame="discovery",
        ownership="probed",
        input={},
        output={},
        taxonomy_cell={},
    )
    empty_label = Label(trace_id="x", span_id=None, kind="human", outcome="PASS", critique="x")
    empty_span = Span(
        span_id="x",
        trace_id="x",
        agent="x",
        role="tool",
        parent=None,
        seq=0,
        status="completed",
        input={},
    )
    _SAMPLE_ROWS["trace"] = _to_row(empty_trace)
    _SAMPLE_ROWS["span"] = _to_row(empty_span)
    _SAMPLE_ROWS["label"] = _to_row(empty_label)


# --- load ---------------------------------------------------------------------------------


def load(backend: _RoundTripBackend, path: str | Path, fmt: str = "csv") -> None:
    """Read a dumped workspace back and re-persist it into ``backend``.

    The directory is first checked against its manifest (:func:`verify_dump`), then every file is
    parsed, version-checked, membership-checked and validated *before* any ``put``, so a refusal
    for any of those reasons writes nothing. Traces (with their spans) and labels are re-``put``;
    the separately-dumped spans are verified to belong to a loaded trace but not re-``put``
    independently (they ride in via their trace, avoiding orphans). A label whose trace is not in
    the dump is loaded as it was dumped, since a store accepts labels for traces it does not hold.
    """
    register_core_codecs()
    src = Path(path)
    if fmt not in {"csv", "json"}:
        raise CorruptDumpError(f"fmt must be 'csv' or 'json', got {fmt!r}")
    manifest = verify_dump(src)
    parsed: dict[type, list[Any]] = {}
    for artifact_type in registered_types():
        codec = codec_for(artifact_type)
        rows = _read_rows(src, codec.name, fmt)
        _check_count(src, f"{codec.name}.{fmt}", len(rows), manifest)
        _check_versions(rows, artifact_type)  # before any put
        parsed[artifact_type] = [codec.from_row(row) for row in rows]

    loaded_span_ids = {(s.trace_id, s.span_id) for t in parsed.get(Trace, []) for s in t.spans}
    for span in parsed.get(Span, []):
        if (span.trace_id, span.span_id) not in loaded_span_ids:
            raise UnregisteredArtifactError(
                f"dumped span {span.span_id!r} has no parent trace in the dump (orphan span)"
            )
    # Validated up front so a bad artifact late in the dump cannot leave the store half-loaded:
    # each put validates too, but by then the artifacts before it are already written.
    for artifact in (*parsed.get(Trace, []), *parsed.get(Label, [])):
        validate(artifact)

    # Persist traces first (spans ride in), then labels.
    for trace in parsed.get(Trace, []):
        backend.put_trace(trace)
    for label in parsed.get(Label, []):
        backend.put_score(label)


def verify_dump(path: str | Path) -> DumpManifest:
    """Check that ``path`` is one whole dump, as ``dump`` sealed it, and return its manifest.

    Raises:
        CorruptDumpError: ``path`` is missing, is not a directory, holds no dump, or has a manifest
            that is not valid or is of another dump format version.
        IncompleteDumpError: ``path`` holds dump files but no manifest.
        DumpMismatchError: A file the manifest lists is missing or differs from its entry.
        UnlistedDumpFileError: ``path`` holds files the manifest does not list.
    """
    src = Path(path)
    if not src.exists():
        raise CorruptDumpError(f"dump directory {src} does not exist")
    if not src.is_dir():
        raise CorruptDumpError(f"dump path {src} is not a directory")
    manifest = _read_manifest(src)
    present = _dump_files(src)
    if manifest is None:
        if not present:
            raise CorruptDumpError(
                f"{src} holds no dump file; it is not a calibrated-agent-evaluation dump directory"
            )
        raise IncompleteDumpError(
            f"{src} holds dump files but no {MANIFEST_NAME}, so nothing was loaded. dump writes "
            "the manifest after every other file, so this dump was interrupted or is still being "
            "written (or was written before dumps carried a manifest); dump the source store "
            "again."
        )
    if manifest.format_version != DUMP_FORMAT_VERSION:
        raise CorruptDumpError(
            f"{src / MANIFEST_NAME} is dump format {manifest.format_version!r}; this version reads "
            f"format {DUMP_FORMAT_VERSION!r}. Load it with the release that wrote it."
        )
    for rel, expected in sorted(manifest.files.items()):
        _check_entry(src, rel, expected)
    unlisted = tuple(rel for rel in present if rel not in manifest.files)
    if unlisted:
        more = f" and {len(unlisted) - 5} more" if len(unlisted) > 5 else ""
        shown = ", ".join(unlisted[:5]) + more
        raise UnlistedDumpFileError(
            f"{src} holds {len(unlisted)} file(s) its {MANIFEST_NAME} does not list ({shown}), so "
            "nothing was loaded. They were not written by the dump the manifest seals; remove "
            "them, or dump the source store into an empty directory.",
            paths=unlisted,
        )
    return manifest


def _read_manifest(src: Path) -> DumpManifest | None:
    path = src / MANIFEST_NAME
    if not path.is_file():
        return None
    try:
        text = path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError) as exc:
        raise CorruptDumpError(f"cannot read {path}: {exc}") from exc
    try:
        doc = json.loads(text)
    except json.JSONDecodeError as exc:
        raise CorruptDumpError(f"{path} is not valid JSON: {exc}") from exc
    try:
        return DumpManifest.model_validate(doc)
    except CorruptDumpError as exc:
        raise CorruptDumpError(f"{path} is not a dump manifest: {exc}") from exc


def _check_entry(src: Path, rel: str, expected: DumpFileEntry) -> None:
    path = src / rel
    if not path.is_file():
        raise DumpMismatchError(
            f"{rel}, listed in {src / MANIFEST_NAME}, is missing from {src}, so nothing was "
            "loaded; dump the source store again.",
            path=rel,
        )
    try:
        data = path.read_bytes()
    except OSError as exc:
        raise CorruptDumpError(f"cannot read dump file {path}: {exc}") from exc
    digest = hashlib.sha256(data).hexdigest()
    if digest == expected.sha256:
        return
    try:
        held = f"{_count_records(rel, data)} record(s)"
    except (ValueError, csv.Error):
        held = "content that does not parse"
    raise DumpMismatchError(
        f"{rel} in {src} differs from {src / MANIFEST_NAME}, so nothing was loaded: the manifest "
        f"records {expected.records} record(s) with sha256 {expected.sha256[:12]}, the file holds "
        f"{held} with sha256 {digest[:12]}. The file was changed after the dump, or another "
        "dump overwrote part of this one; dump the source store again.",
        path=rel,
    )


def _check_count(src: Path, rel: str, read: int, manifest: DumpManifest) -> None:
    entry = manifest.files.get(rel)
    expected = entry.records if entry is not None else 0
    if read != expected:
        raise DumpMismatchError(
            f"{rel} in {src} reads as {read} record(s) and {src / MANIFEST_NAME} records "
            f"{expected}, so nothing was loaded; dump the source store again.",
            path=rel,
        )


def _read_rows(src: Path, name: str, fmt: str) -> list[dict[str, str]]:
    if fmt == "json":
        path = src / f"{name}.json"
        if not path.exists():
            return []
        try:
            decoded = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise CorruptDumpError(f"dump file {path} is not valid JSON: {exc}") from exc
        # re-encode each cell to the canonical JSON-per-cell row shape from_row expects.
        return [{k: json.dumps(v) for k, v in obj.items()} for obj in decoded]
    path = src / f"{name}.csv"
    if not path.exists():
        return []
    try:
        with path.open(newline="", encoding="utf-8") as fh:
            return list(csv.DictReader(fh))
    except csv.Error as exc:
        raise CorruptDumpError(f"dump file {path} is not valid CSV: {exc}") from exc
    except (OSError, UnicodeDecodeError) as exc:
        raise CorruptDumpError(f"cannot read dump file {path}: {exc}") from exc


def _check_versions(rows: list[dict[str, str]], artifact_type: type) -> None:
    # Only Trace carries schema_version; check it before any write.
    if artifact_type is not Trace:
        return
    for row in rows:
        raw = row.get("schema_version")
        try:
            version = json.loads(raw) if raw is not None else SCHEMA_VERSION
        except json.JSONDecodeError as exc:
            raise CorruptDumpError(f"dump schema_version cell is not valid JSON: {exc}") from exc
        if version != SCHEMA_VERSION:
            raise SchemaVersionError(
                f"dump schema_version {version!r} != running SCHEMA_VERSION {SCHEMA_VERSION!r}; "
                "refusing to load (no coercion)"
            )


_init_sample_rows()
widen_csv_field_limit()
