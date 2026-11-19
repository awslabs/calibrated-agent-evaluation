# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Rewrite a retired name out of committed recordings, deterministically and in place.

Recorded traces, cassettes and AWS read-backs keep the names they were captured under: AWS resource
names (runtimes, evaluators, online configs, gateways, Policy engines), span attribute keys,
instrumentation scope names, session ids and metadata keys. When the project renames one of those
after recording, a re-recording costs live calls and can change the model's replies, so this script
rewrites the recordings instead. It is a pure text rewrite of every spelling of one word, plus any
exact literal replacements, and it never touches a recorded value that does not contain them.

The spellings of ``--old WORD`` that are rewritten, given ``--new "first second third"``:

- ``WORD_x``, ``x_WORD``, ``WORD.x`` and a bare ``WORD`` become ``first_second_third...``;
- ``WORD-x`` becomes ``first-second-third-x`` (hyphenated resource names and session ids);
- ``WORDx`` (the word glued to a lowercase suffix) becomes ``first_second_third_x``;
- ``.WORD`` as a whole path segment becomes ``.first-second-third``;
- ``WORD_UPPER`` becomes ``FIRST_SECOND_THIRD``, ``Word`` inside an identifier
  ``FirstSecondThird``, and a standalone ``Word`` ``First Second Third``.

A cassette entry is keyed by the fingerprint of its request, so when a rewrite changes a request
the entry's fingerprint is recomputed with the library's own
:func:`calibrated_agent_evaluation.execute.cassette.fingerprint`, and only where the stored key was
that function's output before the rewrite. An entry keyed some other way whose request changed is
refused rather than left unreplayable.

``.gz`` and ``.xz`` files are rewritten inside and recompressed the way the generators write them
(gzip level 9 with a zero mtime, xz preset 9), so a second run over its own output changes nothing.

Usage::

    uv run python scripts/rewrite_recordings.py --old WORD --new "new name words" \\
        [--replace OLD=NEW ...] [--check] PATH [PATH ...]

A directory is walked for ``.json``, ``.jsonl``, ``.txt``, ``.gz`` and ``.xz`` files; a file path
is rewritten whatever its suffix. ``--check`` writes nothing and exits 1 when a file would change.
"""

from __future__ import annotations

import argparse
import gzip
import json
import lzma
import re
import sys
from collections.abc import Callable, Iterable, Iterator
from pathlib import Path
from typing import Any

from calibrated_agent_evaluation.execute.cassette import fingerprint

_WALKED_SUFFIXES = (".json", ".jsonl", ".txt", ".gz", ".xz")


def build_rewriter(old: str, new_words: list[str]) -> Callable[[str], str]:
    """The text rewrite for every spelling of ``old`` (one lowercase word) into ``new_words``."""
    if not re.fullmatch(r"[a-z][a-z0-9]*", old):
        raise SystemExit(f"--old must be one lowercase word, got {old!r}")
    if not new_words or not all(re.fullmatch(r"[a-z][a-z0-9]*", w) for w in new_words):
        raise SystemExit(f"--new must be lowercase words separated by spaces, got {new_words!r}")
    snake = "_".join(new_words)
    kebab = "-".join(new_words)
    upper = snake.upper()
    camel = "".join(w.capitalize() for w in new_words)
    prose = " ".join(w.capitalize() for w in new_words)
    title = old.capitalize()
    rules: list[tuple[str, str]] = [
        (rf"(?<![A-Za-z]){old.upper()}", upper),
        (rf"(?<=[A-Za-z0-9_]){title}|{title}(?=[A-Z])", camel),
        (rf"\b{title}\b", prose),
        (rf"(?<![A-Za-z0-9_])\.{old}(?![A-Za-z0-9_])", f".{kebab}"),
        (rf"(?<![A-Za-z0-9]){old}(?=-)", kebab),
        (rf"(?<![A-Za-z0-9]){old}(?=[a-z0-9])", f"{snake}_"),
        (rf"(?<![A-Za-z0-9]){old}", snake),
    ]
    combined = re.compile("|".join(f"({pattern})" for pattern, _ in rules))

    def rewrite(text: str) -> str:
        def sub(match: re.Match[str]) -> str:
            return next(new for (_, new), hit in zip(rules, match.groups(), strict=True) if hit)

        return combined.sub(sub, text)

    return rewrite


def _cassette_entries(doc: Any) -> list[dict[str, Any]] | None:
    if not isinstance(doc, dict) or not isinstance(doc.get("entries"), list):
        return None
    entries = doc["entries"]
    if not all(isinstance(e, dict) and "fingerprint" in e and "request" in e for e in entries):
        return None
    return entries


def refingerprint(before: str, after: str, where: Path) -> str:
    """``after`` with each rewritten cassette entry's fingerprint recomputed from its request."""
    try:
        old_doc, new_doc = json.loads(before), json.loads(after)
    except json.JSONDecodeError:
        return after
    old_entries, new_entries = _cassette_entries(old_doc), _cassette_entries(new_doc)
    if old_entries is None or new_entries is None:
        return after
    for position, (old_entry, new_entry) in enumerate(zip(old_entries, new_entries, strict=True)):
        if old_entry["request"] == new_entry["request"]:
            continue
        stored = old_entry["fingerprint"]
        if stored != fingerprint(old_entry["request"]):
            raise SystemExit(
                f"{where}: entry {position}'s request changed, but its fingerprint is not the "
                "cassette fingerprint of its request, so it cannot be recomputed"
            )
        recomputed = fingerprint(new_entry["request"])
        after = after.replace(f'"fingerprint": "{stored}"', f'"fingerprint": "{recomputed}"')
    return after


def _read(path: Path) -> str:
    raw = path.read_bytes()
    if path.suffix == ".gz":
        raw = gzip.decompress(raw)
    elif path.suffix == ".xz":
        raw = lzma.decompress(raw)
    return raw.decode("utf-8")


def _write(path: Path, text: str) -> None:
    raw = text.encode("utf-8")
    if path.suffix == ".gz":
        raw = gzip.compress(raw, compresslevel=9, mtime=0)
    elif path.suffix == ".xz":
        raw = lzma.compress(raw, preset=9)
    path.write_bytes(raw)


def _files(paths: Iterable[Path]) -> Iterator[Path]:
    for path in paths:
        if path.is_dir():
            yield from sorted(p for p in path.rglob("*") if p.suffix in _WALKED_SUFFIXES)
        elif path.is_file():
            yield path
        else:
            raise SystemExit(f"{path} does not exist")


def _literal(pair: str) -> tuple[str, str]:
    old, sep, new = pair.partition("=")
    if not sep or not old:
        raise argparse.ArgumentTypeError(f"expected OLD=NEW, got {pair!r}")
    return old, new


def main(argv: list[str] | None = None) -> int:
    """Rewrite the named recordings; return 1 under ``--check`` when any would change."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--old", required=True, help="the retired word, lowercase")
    parser.add_argument("--new", required=True, help='the new name as lowercase words, "a b c"')
    parser.add_argument(
        "--replace",
        type=_literal,
        action="append",
        default=[],
        metavar="OLD=NEW",
        help="an exact literal to replace as well (repeatable)",
    )
    parser.add_argument("--check", action="store_true", help="write nothing; exit 1 on a change")
    parser.add_argument("paths", nargs="+", type=Path)
    args = parser.parse_args(argv)

    rewrite = build_rewriter(args.old, args.new.split())
    changed = 0
    for path in _files(args.paths):
        before = _read(path)
        after = rewrite(before)
        for old, new in args.replace:
            after = after.replace(old, new)
        if after == before:
            continue
        after = refingerprint(before, after, path)
        changed += 1
        print(f"{'would rewrite' if args.check else 'rewrote'} {path}")
        if not args.check:
            _write(path, after)
    print(f"{changed} file(s) {'need rewriting' if args.check else 'rewritten'}")
    return 1 if args.check and changed else 0


if __name__ == "__main__":
    sys.exit(main())
