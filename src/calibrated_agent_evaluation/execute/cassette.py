# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Record/replay cassette for deterministic agent execution.

A framework-agnostic store that intercepts model calls and either records or replays them keyed
by a stable content fingerprint (SHA-256). Three modes:

  * **RECORD**: call the live provider, persist ``(fingerprint → response)`` atomically.
  * **REPLAY**: never call the provider; a miss raises ``CassetteMiss``.
  * **AUTO**: replay on hit, record on miss.

The cassette is committed alongside ``fixtures/raw/cassettes/`` so a fresh clone (and CI) replays
with no cloud credentials. Secrets never enter a cassette: only request *content* (messages, tool
configs) and response content are stored (secrets stay in the process).

A cassette file is standard JSON: an object with a ``version`` and a list of ``entries``, each an
object with a string ``fingerprint``, ``request`` and ``response`` objects, and an optional
``model`` stamp. Reading refuses any other shape as a :class:`CassetteError` naming the file.
Reading and writing both refuse a NaN or infinite number (a token, or a literal such as
``1e999``) and an entry's request, response or model stamp nested more than 256 levels deep,
counted from that part, so what one Python version writes every other one reads.

Promoted from ``corpus/generate/cassette.py`` into the shipped package so
the ``InspectExecutor`` can use it at runtime.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from threading import Lock
from typing import Any

from calibrated_agent_evaluation._filelock import exclusive_lock
from calibrated_agent_evaluation._finite import (
    MAX_JSON_DEPTH,
    NestedTooDeepError,
    NonFiniteNumberError,
    refuse_json_constant,
    require_finite,
)
from calibrated_agent_evaluation._recordstore import atomic_write_text

logger = logging.getLogger("calibrated_agent_evaluation.execute.cassette")


class Mode(StrEnum):
    """How the cassette behaves for a given run."""

    RECORD = "record"
    REPLAY = "replay"
    AUTO = "auto"


MODE_ENV_VAR = "CALIBRATED_AGENT_EVALUATION_CASSETTE_MODE"
"""The environment variable :func:`mode_from_env` reads the cassette mode from."""


def mode_from_env(default: Mode = Mode.AUTO) -> Mode:
    """The cassette mode the environment names (record|replay|auto).

    The variable is ``CALIBRATED_AGENT_EVALUATION_CASSETTE_MODE``.
    """
    raw = os.environ.get(MODE_ENV_VAR)
    if raw is None:
        return default
    try:
        return Mode(raw.strip().lower())
    except ValueError as exc:
        valid = ", ".join(m.value for m in Mode)
        raise ValueError(
            f"{MODE_ENV_VAR}={raw!r} is not a valid cassette mode (expected one of: {valid})"
        ) from exc


class CassetteMiss(RuntimeError):
    """Raised in REPLAY mode when a request fingerprint has no recorded response."""


class CassetteError(RuntimeError):
    """A cassette could not be read or written (wraps the underlying OS/JSON error)."""


class CassetteModelMismatch(CassetteError):
    """A cassette entry was recorded with a different model or decoding config than this run's.

    Replaying it would attribute one model's answers to another. Entries record the model that
    produced them; an entry with no record of its model (a cassette recorded before entries carried
    one) is replayed as before, unless the caller requires the record, which refuses it too.
    """


def model_stamp(
    model_id: str | None, config: dict[str, Any] | None = None
) -> dict[str, Any] | None:
    """The model identity an entry is recorded under: its id and the decoding config it ran with.

    Returns ``None`` when the model id is unknown, which records and checks nothing.
    """
    if model_id is None:
        return None
    return _canonical({"id": model_id, "config": dict(config or {})})


def _canonical(value: dict[str, Any]) -> dict[str, Any]:
    # Round-trip through JSON so a stamp built in this run compares equal to one read from disk.
    return json.loads(json.dumps(value, sort_keys=True, default=str))


def fingerprint(request: dict[str, Any]) -> str:
    """Stable content hash of a model request (sorted-key canonical JSON -> SHA-256, 16 hex chars).

    Default values that vary run-to-run but not semantically (e.g. a streaming flag) should be
    stripped by the caller before fingerprinting; this function hashes exactly what it is given.
    """
    canonical = json.dumps(request, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:16]


@dataclass
class _Entry:
    fingerprint: str
    request: dict[str, Any]
    response: dict[str, Any]
    model: dict[str, Any] | None = None


# The parts of an entry the cassette keeps, each checked from its own root.
_ENTRY_PARTS = ("request", "response", "model")


class Cassette:
    """A JSON-file-backed record/replay store for one logical capture run.

    Thread-safe: the in-memory index and the append are guarded by a lock. Safe across processes
    too: each append takes an advisory lock on a sidecar ``.<name>.lock`` file, merges in what
    other recorders wrote since, and rewrites the file atomically through a uniquely named
    temporary, so a crashed capture leaves a valid (if partial) cassette and two recorders on one
    file keep both their entries.
    """

    def __init__(self, path: str | Path, mode: Mode = Mode.AUTO) -> None:
        self._path = Path(path)
        self._mode = mode
        self._lock = Lock()
        self._entries: dict[str, _Entry] = {}
        self._load()

    @property
    def mode(self) -> Mode:
        """Whether the cassette replays, records, or does both."""
        return self._mode

    @property
    def path(self) -> Path:
        """The JSON file the cassette reads and writes."""
        return self._path

    def _load(self) -> None:
        self._entries.update(self._read_disk())

    def _read_disk(self) -> dict[str, _Entry]:
        # Read rather than check `exists()` first: before Python 3.14 that check raises when a
        # directory on the way can't be searched, and the read reports it the same way anyway.
        try:
            raw = json.loads(
                self._path.read_text(encoding="utf-8"), parse_constant=refuse_json_constant
            )
            _require_standard_document(raw)
        except (FileNotFoundError, NotADirectoryError):
            # Not recorded yet: an empty cassette, which the first recording creates.
            return {}
        except UnicodeDecodeError as exc:
            # Not its own message, which quotes the bytes that didn't decode.
            raise CassetteError(
                f"failed to read cassette {self._path}: it is not UTF-8 text"
            ) from exc
        except (NestedTooDeepError, RecursionError) as exc:
            # The decoder's own RecursionError comes only from nesting far past the limit, and on
            # some Python versions before the depth check can run, so it gets the same answer.
            raise CassetteError(
                f"failed to read cassette {self._path}: it nests more than {MAX_JSON_DEPTH} "
                "levels deep, deeper than a cassette may"
            ) from exc
        except (OSError, ValueError) as exc:
            # ValueError: not JSON, a NaN or infinite number, or a number too long to convert.
            raise CassetteError(f"failed to read cassette {self._path}: {exc}") from exc
        if not isinstance(raw, dict) or "version" not in raw:
            raise CassetteError(f"cassette {self._path} is corrupt: missing 'version' key")
        entries = raw.get("entries", [])
        if not isinstance(entries, list):
            raise CassetteError(
                f"cassette {self._path} is corrupt: 'entries' is {_json_kind(entries)}, not a list"
            )
        read: dict[str, _Entry] = {}
        for position, item in enumerate(entries):
            entry = self._entry_from(position, item)
            read[entry.fingerprint] = entry
        return read

    def _lock_path(self) -> Path:
        return self._path.with_name(f".{self._path.name}.lock")

    def _record_and_flush(self, entry: _Entry) -> None:
        """Merge ``entry`` into what is on disk now and rewrite the file, under the file lock.

        Another process may have recorded into the same cassette since this one loaded it. Writing
        this instance's in-memory view would drop those entries, and each one is a paid live call
        the next replay would miss and pay for again.
        """
        try:
            with exclusive_lock(self._lock_path()):
                self._entries.update(self._read_disk())
                previous = self._entries.get(entry.fingerprint)
                self._entries[entry.fingerprint] = entry
                try:
                    self._flush()
                except CassetteError:
                    # The file holds what it held before; memory goes back to match, so no later
                    # lookup serves, and no later write saves, an unsaved response.
                    if previous is None:
                        del self._entries[entry.fingerprint]
                    else:
                        self._entries[entry.fingerprint] = previous
                    raise
        except OSError as exc:
            raise CassetteError(
                f"failed to write cassette {self._path} (could not take its lock): {exc}"
            ) from exc

    def _entry_from(self, position: int, item: object) -> _Entry:
        # Each fault names the entry's position and the key, never a recorded value.
        where = f"cassette {self._path} is corrupt: entry {position}"
        if not isinstance(item, dict):
            raise CassetteError(f"{where} is {_json_kind(item)}, not an object")
        missing = [key for key in ("fingerprint", "request", "response") if key not in item]
        if missing:
            raise CassetteError(f"{where} has no {', '.join(repr(key) for key in missing)}")
        fp = item["fingerprint"]
        if not isinstance(fp, str):
            raise CassetteError(
                f"{where} has a 'fingerprint' that is {_json_kind(fp)}, not a string"
            )
        for key in ("request", "response"):
            if not isinstance(item[key], dict):
                raise CassetteError(
                    f"{where} has a {key!r} that is {_json_kind(item[key])}, not an object"
                )
        model = item.get("model")
        if model is not None and not isinstance(model, dict):
            raise CassetteError(f"{where} has a 'model' that is {_json_kind(model)}, not an object")
        return _Entry(fp, item["request"], item["response"], model)

    def _flush(self) -> None:
        payload = {
            "version": 1,
            "entries": [_entry_json(e) for e in self._entries.values()],
        }
        # `default=str` matches `fingerprint`'s encoder. Without it a recordable-but-not-JSON
        # -serializable response raised a raw TypeError AFTER the paid live call had happened, and
        # the response was lost -- never written to the cassette, so the next run pays again.
        # Checked first, so the writer never writes a file the reader refuses, and every failure
        # (a RecursionError from an object's text included) is a CassetteError `record` rolls back.
        try:
            for entry in self._entries.values():
                _require_standard_entry(entry)
            encoded = json.dumps(payload, indent=2, sort_keys=True, default=str, allow_nan=False)
        except (NonFiniteNumberError, NestedTooDeepError) as exc:
            # Their messages name the rule and the part, never a value.
            raise CassetteError(
                f"cassette {self._path} could not encode a recorded response: {exc}"
            ) from exc
        except (TypeError, ValueError, RecursionError) as exc:
            # Named by type only: an object's own text, reached through default=str, can carry
            # what was recorded.
            raise CassetteError(
                f"cassette {self._path} could not encode a recorded response ({type(exc).__name__})"
            ) from exc
        # `_load`'s read path already wraps OSError; the write path did not, so a full disk or a
        # read-only directory escaped as a raw OSError past the typed boundary.
        try:
            self._path.parent.mkdir(parents=True, exist_ok=True)
            atomic_write_text(self._path, encoded)
        except OSError as exc:
            raise CassetteError(f"failed to write cassette {self._path}: {exc}") from exc

    def lookup(
        self, fp: str, *, model: dict[str, Any] | None = None, require_model: bool = False
    ) -> dict[str, Any] | None:
        """Return a copy of the recorded response for fingerprint ``fp``, or ``None`` if absent.

        ``model`` is this run's :func:`model_stamp`. An entry recorded under a different stamp
        raises :class:`CassetteModelMismatch`. An entry recorded with no stamp is returned, unless
        ``require_model`` is set: then it raises :class:`CassetteModelMismatch` too, since nothing
        shows it came from ``model``.
        """
        with self._lock:
            entry = self._entries.get(fp)
            if entry is None:
                return None
            if model is not None and entry.model is None and require_model:
                raise CassetteModelMismatch(
                    f"cassette {self._path.name}: the recorded response for fingerprint {fp} has "
                    f"no record of the model it came from, so it can't be credited to "
                    f"{_describe(model)}. Re-record the cassette with this model"
                )
            if model is not None and entry.model is not None and entry.model != model:
                raise CassetteModelMismatch(
                    f"cassette {self._path.name}: the recorded response for fingerprint {fp} came "
                    f"from {_describe(entry.model)}, but this run uses {_describe(model)}; "
                    "replaying it would credit one model's answer to another. Re-record the "
                    "cassette with this model, or run with the model it was recorded with"
                )
            return json.loads(json.dumps(entry.response))

    def record(
        self,
        fp: str,
        request: dict[str, Any],
        response: dict[str, Any],
        *,
        model: dict[str, Any] | None = None,
    ) -> None:
        """Record ``response`` for ``fp`` under ``model`` and write the cassette file.

        Raises:
            CassetteError: The file could not be written. The cassette is left as it was, in
                memory and on file: the response is not recorded, and an entry it was replacing
                stays.
        """
        with self._lock:
            self._record_and_flush(_Entry(fp, request, response, model))

    def play(
        self,
        request: dict[str, Any],
        live_call: Any,
        *,
        model: dict[str, Any] | None = None,
        require_model: bool = False,
    ) -> dict[str, Any]:
        """Return the response for ``request``, honoring the cassette mode.

        ``live_call`` is a zero-arg callable that performs the real provider request and returns its
        response dict; it is invoked only in RECORD mode or on an AUTO miss. In REPLAY mode a miss
        raises :class:`CassetteMiss`. ``model`` is the run's :func:`model_stamp`, checked against
        a hit and recorded with a new entry; ``require_model`` refuses a hit with no stamp, as
        :meth:`lookup` does.
        """
        fp = fingerprint(request)
        if self._mode is Mode.REPLAY:
            hit = self.lookup(fp, model=model, require_model=require_model)
            if hit is None:
                raise CassetteMiss(
                    f"cassette {self._path.name}: no recorded response for fingerprint {fp} "
                    f"(REPLAY mode does not fall through to a live call)"
                )
            logger.debug("cassette replay hit fp=%s", fp)
            return hit
        if self._mode is Mode.AUTO:
            hit = self.lookup(fp, model=model, require_model=require_model)
            if hit is not None:
                logger.debug("cassette auto hit fp=%s", fp)
                return hit
        logger.info("cassette recording live call fp=%s -> %s", fp, self._path.name)
        response = live_call()
        self.record(fp, request, response, model=model)
        return response

    async def aplay(
        self,
        request: dict[str, Any],
        live_call: Any,
        *,
        model: dict[str, Any] | None = None,
        require_model: bool = False,
    ) -> dict[str, Any]:
        """Async twin of :meth:`play`; ``live_call`` is an async (awaitable) callable."""
        fp = fingerprint(request)
        if self._mode is Mode.REPLAY:
            hit = self.lookup(fp, model=model, require_model=require_model)
            if hit is None:
                raise CassetteMiss(
                    f"cassette {self._path.name}: no recorded response for fingerprint {fp} "
                    f"(REPLAY mode does not fall through to a live call)"
                )
            return hit
        if self._mode is Mode.AUTO:
            hit = self.lookup(fp, model=model, require_model=require_model)
            if hit is not None:
                return hit
        logger.info("cassette recording live call fp=%s -> %s", fp, self._path.name)
        response = await live_call()
        self.record(fp, request, response, model=model)
        return response


def _require_standard_document(raw: object) -> None:
    # Each entry's parts are counted from their own roots; everything else in the file, the
    # top-level object and each entry's other keys, from the top of the file.
    entries = raw.get("entries") if isinstance(raw, dict) else None
    if not isinstance(raw, dict) or not isinstance(entries, list):
        require_finite(raw, where="the cassette file")
        return
    rest: list[object] = []
    for item in entries:
        if isinstance(item, dict):
            for part in _ENTRY_PARTS:
                require_finite(item.get(part), where=f"an entry's {part}")
            rest.append({key: value for key, value in item.items() if key not in _ENTRY_PARTS})
        else:
            rest.append(item)
    require_finite({**raw, "entries": rest}, where="the cassette file")


def _json_kind(value: object) -> str:
    # How a message names the kind of a decoded JSON value, never the value itself.
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "a boolean"
    if isinstance(value, (int, float)):
        return "a number"
    if isinstance(value, str):
        return "a string"
    if isinstance(value, list):
        return "a list"
    return "an object"


def _require_standard_entry(entry: _Entry) -> None:
    require_finite(entry.request, where="an entry's request")
    require_finite(entry.response, where="an entry's response")
    require_finite(entry.model, where="an entry's model")


def _entry_json(entry: _Entry) -> dict[str, Any]:
    # An entry with no model stamp is written exactly as before, so rewriting an old cassette
    # (recording one new entry into it) leaves its existing entries byte-for-byte unchanged.
    out: dict[str, Any] = {
        "fingerprint": entry.fingerprint,
        "request": entry.request,
        "response": entry.response,
    }
    if entry.model is not None:
        out["model"] = entry.model
    return out


def _describe(stamp: dict[str, Any]) -> str:
    config = stamp.get("config") or {}
    settings = ", ".join(f"{k}={v}" for k, v in sorted(config.items()))
    return f"model {stamp.get('id')!r}" + (f" ({settings})" if settings else "")
