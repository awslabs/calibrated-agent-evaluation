# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for calibrated_agent_evaluation.execute.cassette."""

from __future__ import annotations

import json
import os
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import pytest

from calibrated_agent_evaluation._finite import NestedTooDeepError, NonFiniteNumberError
from calibrated_agent_evaluation.execute.cassette import (
    Cassette,
    CassetteError,
    CassetteMiss,
    Mode,
    fingerprint,
)

_REPO = Path(__file__).resolve().parents[1]
_SENTINEL = "SENTINEL-5e0c"
_NOT_A_NUMBER = "a number JSON cannot represent"
_TOO_DEEP = "nests more than 256 levels"


@pytest.fixture
def cassette_dir(tmp_path: Path) -> Path:
    return tmp_path / "cassettes"


def _nested_lists(levels: int) -> list[Any]:
    """Lists ``levels`` deep, built with a loop."""
    value: list[Any] = []
    for _ in range(levels - 1):
        value = [value]
    return value


def _with_extra(text: str) -> str:
    """A cassette with no entries and ``text`` under a key it never reads."""
    return '{"version": 1, "entries": [], "x": ' + text + "}"


def _with_response(text: str) -> str:
    """A cassette whose one entry's response is ``text``."""
    doc = {"version": 1, "entries": [{"fingerprint": "fp", "request": {}, "response": "_"}]}
    return json.dumps(doc).replace('"_"', text)


def _entry(fp: str, response: object, **parts: Any) -> dict[str, Any]:
    return {"fingerprint": fp, "request": {"prompt": fp}, "response": response, **parts}


@contextmanager
def _permissions(directory: Path, permissions: int) -> Iterator[Path]:
    """Give ``directory`` ``permissions`` for the ``with`` block, then make it usable again."""
    if hasattr(os, "geteuid") and os.geteuid() == 0:
        pytest.skip("the superuser reads and writes through any permission")
    directory.mkdir(parents=True, exist_ok=True)
    directory.chmod(permissions)
    try:
        yield directory
    finally:
        directory.chmod(0o700)


@pytest.fixture
def read_only_dir(cassette_dir: Path) -> Iterator[Path]:
    with _permissions(cassette_dir, 0o500) as directory:
        yield directory


@pytest.fixture
def unsearchable_dir(cassette_dir: Path) -> Iterator[Path]:
    with _permissions(cassette_dir, 0o000) as directory:
        yield directory


def _make_cassette(path: Path, mode: Mode = Mode.AUTO) -> Cassette:
    return Cassette(path, mode=mode)


def _seed_cassette(path: Path, entries: list[dict]) -> None:
    """Write a valid cassette file to disk."""
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"version": 1, "entries": entries}
    path.write_text(json.dumps(payload), encoding="utf-8")


class TestReplayMode:
    """REPLAY returns cached response; raises CassetteMiss on miss."""

    def test_replay_returns_cached_response(self, cassette_dir: Path) -> None:
        path = cassette_dir / "test.cassette.json"
        request = {"messages": [{"role": "user", "content": "hello"}]}
        fp = fingerprint(request)
        entry = {"fingerprint": fp, "request": request, "response": {"text": "world"}}
        _seed_cassette(path, [entry])

        cassette = _make_cassette(path, Mode.REPLAY)
        result = cassette.play(
            request, live_call=lambda: pytest.fail("live_call must not be invoked")
        )
        assert result == {"text": "world"}

    def test_replay_from_a_string_path(self, cassette_dir: Path) -> None:
        path = cassette_dir / "test.cassette.json"
        request = {"messages": [{"role": "user", "content": "hello"}]}
        entry = {"fingerprint": fingerprint(request), "request": request, "response": {"t": "w"}}
        _seed_cassette(path, [entry])

        cassette = Cassette(str(path), mode=Mode.REPLAY)
        assert cassette.path == path
        assert cassette.play(request, live_call=lambda: pytest.fail("no live call")) == {"t": "w"}

    def test_replay_raises_on_miss(self, cassette_dir: Path) -> None:
        path = cassette_dir / "empty.cassette.json"
        _seed_cassette(path, [])

        cassette = _make_cassette(path, Mode.REPLAY)
        request = {"messages": [{"role": "user", "content": "unknown"}]}
        with pytest.raises(CassetteMiss, match="no recorded response for fingerprint"):
            cassette.play(request, live_call=lambda: {})


class TestRecordMode:
    """RECORD calls live_call, persists entry, returns response."""

    def test_record_calls_live_and_persists(self, cassette_dir: Path) -> None:
        path = cassette_dir / "rec.cassette.json"
        request = {"messages": [{"role": "user", "content": "record me"}]}
        live_response = {"text": "recorded"}

        cassette = _make_cassette(path, Mode.RECORD)
        result = cassette.play(request, live_call=lambda: live_response)

        assert result == live_response
        assert path.exists()

        reloaded = Cassette(path, Mode.REPLAY)
        replayed = reloaded.play(request, live_call=lambda: pytest.fail("should not call live"))
        assert replayed == live_response

    def test_record_overwrites_existing_entry(self, cassette_dir: Path) -> None:
        path = cassette_dir / "overwrite.cassette.json"
        request = {"messages": [{"role": "user", "content": "same"}]}
        fp = fingerprint(request)
        _seed_cassette(path, [{"fingerprint": fp, "request": request, "response": {"v": 1}}])

        cassette = _make_cassette(path, Mode.RECORD)
        cassette.play(request, live_call=lambda: {"v": 2})

        reloaded = Cassette(path, Mode.REPLAY)
        assert reloaded.play(request, live_call=lambda: pytest.fail("no")) == {"v": 2}


class TestFailedWrite:
    """A response whose write fails is left out of the cassette, in memory and on file."""

    def test_a_response_whose_write_failed_is_not_found_or_saved_by_a_later_write(
        self, read_only_dir: Path
    ) -> None:
        path = read_only_dir / "c.json"
        cassette = _make_cassette(path, Mode.RECORD)
        with pytest.raises(CassetteError, match="failed to write cassette"):
            cassette.record("fp1", {"req": 1}, {"ok": 1})
        assert cassette.lookup("fp1") is None

        read_only_dir.chmod(0o700)
        cassette.record("fp2", {"req": 2}, {"ok": 2})

        saved = json.loads(path.read_text(encoding="utf-8"))
        assert [entry["fingerprint"] for entry in saved["entries"]] == ["fp2"]

    def test_an_auto_repeat_of_a_request_whose_write_failed_calls_live_again(
        self, read_only_dir: Path
    ) -> None:
        cassette = _make_cassette(read_only_dir / "c.json", Mode.AUTO)
        request = {"messages": [{"role": "user", "content": "unsaved"}]}
        calls: list[int] = []

        def live() -> dict[str, int]:
            calls.append(1)
            return {"call": len(calls)}

        with pytest.raises(CassetteError):
            cassette.play(request, live_call=live)
        read_only_dir.chmod(0o700)

        assert cassette.play(request, live_call=live) == {"call": 2}
        assert len(calls) == 2

    def test_a_failed_overwrite_keeps_the_response_already_recorded(
        self, cassette_dir: Path
    ) -> None:
        path = cassette_dir / "c.json"
        request = {"messages": [{"role": "user", "content": "same"}]}
        fp = fingerprint(request)
        _seed_cassette(path, [{"fingerprint": fp, "request": request, "response": {"v": 1}}])
        cassette = _make_cassette(path, Mode.RECORD)

        with _permissions(cassette_dir, 0o500), pytest.raises(CassetteError):
            cassette.play(request, live_call=lambda: {"v": 2})

        assert cassette.lookup(fp) == {"v": 1}
        assert Cassette(path, Mode.REPLAY).lookup(fp) == {"v": 1}


class TestCorruptCassette:
    """Corrupt cassette raises CassetteError with diagnostic message."""

    @pytest.mark.parametrize("mode", list(Mode), ids=[m.value for m in Mode])
    @pytest.mark.parametrize(
        ("text", "refusal", "quoted"),
        [
            (_with_extra('"caf\xe9"'), "not UTF-8 text", "xe9"),
            (_with_extra("9" * 5000), "failed to read cassette", "9" * 10),
            # Read by the decoder on every supported Python, then refused by the depth check.
            (_with_response(json.dumps({"v": _nested_lists(256)})), _TOO_DEEP, "[["),
            # Deep enough for the decoder itself to raise RecursionError on most versions.
            (_with_response('{"v": ' + "[" * 100_000 + "]" * 100_000 + "}"), _TOO_DEEP, "[["),
            # Under a key the cassette never reads, counted from the top of the file.
            (_with_extra(json.dumps(_nested_lists(257))), _TOO_DEEP, "[["),
        ],
        ids=[
            "not-utf-8",
            "number-too-long-to-read",
            "response-257-levels",
            "response-100000-levels",
            "ignored-key-257-levels",
        ],
    )
    def test_a_cassette_that_cannot_be_read_is_refused_naming_its_path_without_quoting_it(
        self, cassette_dir: Path, text: str, refusal: str, quoted: str, mode: Mode
    ) -> None:
        path = cassette_dir / "unreadable.cassette.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        # One byte per character, so the first file holds a byte that is not UTF-8.
        path.write_bytes(text.encode("latin-1"))

        with pytest.raises(CassetteError) as refused:
            Cassette(path, mode)

        message = str(refused.value)
        assert str(path) in message
        assert refusal in message
        assert quoted not in message

    @pytest.mark.parametrize("mode", list(Mode), ids=[m.value for m in Mode])
    def test_a_cassette_in_a_directory_that_cannot_be_searched_is_refused(
        self, unsearchable_dir: Path, mode: Mode
    ) -> None:
        path = unsearchable_dir / "c.json"

        with pytest.raises(CassetteError, match="failed to read cassette") as refused:
            Cassette(path, mode)

        assert str(path) in str(refused.value)

    def test_a_cassette_under_a_file_is_a_cassette_not_recorded_yet(self, tmp_path: Path) -> None:
        blocker = tmp_path / "blocked"
        blocker.write_text("not a directory", encoding="utf-8")

        assert Cassette(blocker / "c.json", Mode.AUTO).lookup("fp") is None

    def test_corrupt_json_raises_cassette_error(self, cassette_dir: Path) -> None:
        path = cassette_dir / "corrupt.cassette.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("not valid json {{{{", encoding="utf-8")

        with pytest.raises(CassetteError, match="failed to read cassette"):
            Cassette(path, Mode.REPLAY)

    def test_missing_version_key_raises_cassette_error(self, cassette_dir: Path) -> None:
        path = cassette_dir / "noversion.cassette.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({"entries": []}), encoding="utf-8")

        with pytest.raises(CassetteError, match="missing 'version' key"):
            Cassette(path, Mode.REPLAY)

    @pytest.mark.parametrize(
        ("entries", "fault"),
        [
            ({"fp": {"request": {}, "response": {"text": _SENTINEL}}}, "'entries' is an object"),
            (None, "'entries' is null"),
            ([_SENTINEL], "entry 0 is a string, not an object"),
            ([{"request": {}, "response": {"text": _SENTINEL}}], "entry 0 has no 'fingerprint'"),
            ([{"fingerprint": "fp", "response": {}}], "entry 0 has no 'request'"),
            ([{"fingerprint": "fp", "request": {}}], "entry 0 has no 'response'"),
            ([_entry("fp", {}, fingerprint=7)], "'fingerprint' that is a number, not a string"),
            ([_entry("fp", {}, fingerprint=None)], "'fingerprint' that is null, not a string"),
            ([_entry("fp", {}, request=[_SENTINEL])], "'request' that is a list, not an object"),
            ([_entry("fp", _SENTINEL)], "'response' that is a string, not an object"),
            ([_entry("fp", {}, model=_SENTINEL)], "'model' that is a string, not an object"),
        ],
        ids=[
            "entries-as-an-object",
            "entries-as-null",
            "entry-as-a-string",
            "no-fingerprint",
            "no-request",
            "no-response",
            "fingerprint-as-a-number",
            "fingerprint-as-null",
            "request-as-a-list",
            "response-as-a-string",
            "model-as-a-string",
        ],
    )
    def test_a_malformed_entry_is_refused_naming_the_path_and_the_fault(
        self, cassette_dir: Path, entries: object, fault: str
    ) -> None:
        path = cassette_dir / "malformed.cassette.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({"version": 1, "entries": entries}), encoding="utf-8")

        with pytest.raises(CassetteError) as refused:
            Cassette(path, Mode.AUTO)

        message = str(refused.value)
        assert f"cassette {path} is corrupt" in message
        assert fault in message
        assert _SENTINEL not in message


class TestAutoMode:
    """AUTO replays on hit, records on miss."""

    def test_auto_replays_on_hit(self, cassette_dir: Path) -> None:
        path = cassette_dir / "auto.cassette.json"
        request = {"messages": [{"role": "user", "content": "cached"}]}
        fp = fingerprint(request)
        entry = {"fingerprint": fp, "request": request, "response": {"cached": True}}
        _seed_cassette(path, [entry])

        cassette = _make_cassette(path, Mode.AUTO)
        result = cassette.play(
            request, live_call=lambda: pytest.fail("should replay, not call live")
        )
        assert result == {"cached": True}

    def test_auto_records_on_miss(self, cassette_dir: Path) -> None:
        path = cassette_dir / "auto_miss.cassette.json"
        _seed_cassette(path, [])

        cassette = _make_cassette(path, Mode.AUTO)
        request = {"messages": [{"role": "user", "content": "new"}]}
        result = cassette.play(request, live_call=lambda: {"fresh": True})

        assert result == {"fresh": True}
        reloaded = Cassette(path, Mode.REPLAY)
        assert reloaded.play(request, live_call=lambda: pytest.fail("no")) == {"fresh": True}


class TestFingerprint:
    """Fingerprint stability and determinism."""

    def test_same_content_same_fingerprint(self) -> None:
        req = {"messages": [{"role": "user", "content": "hi"}], "tools": []}
        assert fingerprint(req) == fingerprint(req)

    def test_key_order_does_not_affect_fingerprint(self) -> None:
        a = {"z": 1, "a": 2}
        b = {"a": 2, "z": 1}
        assert fingerprint(a) == fingerprint(b)

    def test_different_content_different_fingerprint(self) -> None:
        a = {"messages": [{"role": "user", "content": "hello"}]}
        b = {"messages": [{"role": "user", "content": "world"}]}
        assert fingerprint(a) != fingerprint(b)


class TestConcurrentRecorders:
    """Two recorders on one cassette file keep every recorded (paid) entry."""

    def test_two_instances_on_one_file_keep_both_entries(self, cassette_dir: Path) -> None:
        path = cassette_dir / "shared.cassette.json"
        first, second = Cassette(path, Mode.AUTO), Cassette(path, Mode.AUTO)
        first.play({"q": 1}, lambda: {"paid": "call 1"})
        second.play({"q": 2}, lambda: {"paid": "call 2"})

        entries = json.loads(path.read_text(encoding="utf-8"))["entries"]
        assert sorted(e["response"]["paid"] for e in entries) == ["call 1", "call 2"]
        replay = Cassette(path, Mode.REPLAY)
        assert replay.play({"q": 1}, _never) == {"paid": "call 1"}
        assert replay.play({"q": 2}, _never) == {"paid": "call 2"}

    def test_parallel_recorders_lose_nothing(self, cassette_dir: Path) -> None:
        from concurrent.futures import ProcessPoolExecutor

        path = cassette_dir / "parallel.cassette.json"
        with ProcessPoolExecutor(max_workers=4) as pool:
            list(pool.map(_record_one, [str(path)] * 12, range(12)))

        entries = json.loads(path.read_text(encoding="utf-8"))["entries"]
        assert len(entries) == 12
        assert not list(cassette_dir.glob("*.tmp"))

    def test_an_entry_missing_its_fields_is_a_typed_error(self, cassette_dir: Path) -> None:
        path = cassette_dir / "partial.cassette.json"
        _seed_cassette(path, [{"fingerprint": "abc", "request": {}}])
        with pytest.raises(CassetteError, match="entry 0 has no 'response'"):
            Cassette(path, Mode.REPLAY)


def _never() -> dict:
    raise AssertionError("replay must not call the live provider")


def _record_one(path: str, index: int) -> None:
    Cassette(path, Mode.AUTO).play({"q": index}, lambda: {"paid": f"call {index}"})


def _with_text_at(place: str, text: str) -> str:
    """A cassette's JSON text with ``text`` at ``place``: an entry's request, response or model
    stamp, a key of an entry that the cassette ignores, or a top-level key it ignores."""
    marker = '"__MARKER__"'
    entry = _entry("fp", {"answer": _SENTINEL})
    if place in ("request", "response"):
        entry[place] = {"note": _SENTINEL, "value": "__MARKER__"}
    elif place == "model":
        entry["model"] = {"id": "m", "config": {"temperature": "__MARKER__"}}
    elif place == "entry-key":
        entry["extra"] = ["__MARKER__"]
    doc: dict[str, Any] = {"version": 1, "entries": [entry]}
    if place == "top-level-key":
        doc["extra"] = {"value": "__MARKER__"}
    return json.dumps(doc).replace(marker, text)


_PLACES = ["request", "response", "model", "entry-key", "top-level-key"]


class _EndlessText:
    # Reached through the writer's default=str, it recurses until the stack runs out.
    def __str__(self) -> str:
        return str(self)


class TestStandardJsonOnly:
    """A cassette holds standard JSON: no NaN or infinite number, and no part nested more than
    256 levels deep. One that holds either is refused when it is read and when it is written."""

    @pytest.mark.parametrize("place", _PLACES)
    @pytest.mark.parametrize(
        ("number", "mode"),
        [
            ("NaN", Mode.REPLAY),
            ("Infinity", Mode.REPLAY),
            ("-Infinity", Mode.REPLAY),
            # The decoder reads these as infinities without a token, so only the check after
            # decoding can refuse them.
            ("1e999", Mode.AUTO),
            ("-1e999", Mode.AUTO),
        ],
        ids=["NaN", "Infinity", "-Infinity", "1e999", "-1e999"],
    )
    def test_a_cassette_holding_nan_infinity_or_a_number_too_large_for_a_float_is_refused(
        self, cassette_dir: Path, number: str, mode: Mode, place: str
    ) -> None:
        path = cassette_dir / "numbers.cassette.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(_with_text_at(place, number), encoding="utf-8")

        with pytest.raises(CassetteError) as refused:
            Cassette(path, mode)

        message = str(refused.value)
        assert str(path) in message
        assert _NOT_A_NUMBER in message
        assert _SENTINEL not in message
        # A token is named, but a literal is never quoted back.
        assert "1e999" not in message

    @pytest.mark.parametrize(
        ("response", "refusal", "cause"),
        [
            ({"score": float("nan")}, _NOT_A_NUMBER, NonFiniteNumberError),
            ({"scores": [1.0, float("inf")]}, _NOT_A_NUMBER, NonFiniteNumberError),
            ({"t": {"x": -float("inf")}}, _NOT_A_NUMBER, NonFiniteNumberError),
            # The response is the first level, so its value adds 256 more.
            ({"v": _nested_lists(256)}, _TOO_DEEP, NestedTooDeepError),
            ({"value": _EndlessText()}, "(RecursionError)", RecursionError),
        ],
        ids=["nan", "infinity-in-a-list", "-infinity-nested", "257-levels", "endless-text"],
    )
    def test_recording_a_response_that_is_not_standard_json_is_refused_and_rolled_back(
        self, cassette_dir: Path, response: dict[str, Any], refusal: str, cause: type[Exception]
    ) -> None:
        path = cassette_dir / "c.json"
        _seed_cassette(path, [_entry("kept", {"v": 1}), _entry("replaced", {"v": 2})])
        before = path.read_bytes()
        cassette = _make_cassette(path, Mode.RECORD)

        for fp in ("new", "replaced"):
            with pytest.raises(CassetteError) as refused:
                cassette.record(fp, {"prompt": fp}, response)

            assert str(path) in str(refused.value)
            assert refusal in str(refused.value)
            assert isinstance(refused.value.__cause__, cause)
        assert cassette.lookup("new") is None
        assert cassette.lookup("replaced") == {"v": 2}
        # Not written as null, nor as a NaN token: the file is untouched.
        assert path.read_bytes() == before

    def test_a_response_nested_256_levels_records_and_replays_equal(
        self, cassette_dir: Path
    ) -> None:
        path = cassette_dir / "c.json"
        # The response is the first level, so its value adds 255 more.
        response = {"v": _nested_lists(255)}
        cassette = _make_cassette(path, Mode.RECORD)

        cassette.record("deep", {"prompt": "deep"}, response)

        assert Cassette(path, Mode.REPLAY).lookup("deep") == response

    def test_every_committed_cassette_still_loads(self) -> None:
        top = ("fixtures", "tests")
        committed = sorted(path for d in top for path in (_REPO / d).rglob("*.cassette.json"))

        assert committed, "no committed cassette was found to check"
        for path in committed:
            entries = json.loads(path.read_text(encoding="utf-8"))["entries"]
            cassette = Cassette(path, Mode.REPLAY)
            for entry in entries:
                assert cassette.lookup(entry["fingerprint"]) is not None, path.name
