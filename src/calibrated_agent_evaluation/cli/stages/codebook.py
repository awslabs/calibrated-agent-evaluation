# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Which codebook version a review session stamps, and which failure modes its form offers.

Source order, first match wins:

1. ``--codebook FILE``: a hand-written list. Its ``version`` is stamped; a ``--codebook-version``
   that names another version is refused.
2. ``--codebook-version V`` with stored ``codebook`` records for ``V`` (written by
   ``calibrated-agent-evaluation cluster --ratify``): the union of their modes, in ratification
   order.
3. Otherwise no list: an open-coding session, where a FAIL saves with its critique alone and the
   modes are named later by ``calibrated-agent-evaluation cluster``. ``--codebook-version`` (or,
   without it, ``None``, the unversioned bucket) is still stamped on every label.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from calibrated_agent_evaluation.cli.errors import CliError, UsageError


@dataclass(frozen=True)
class SessionCodebook:
    """The version a session stamps, the modes it offers, and the line that says so."""

    version: str | None
    modes: tuple[Any, ...]
    note: str


def _version_flag(args: argparse.Namespace) -> str | None:
    raw = getattr(args, "codebook_version", None)
    if raw is None:
        return None
    if not raw.strip() or raw != raw.strip():
        raise UsageError(f"--codebook-version {raw!r} must be non-empty with no surrounding spaces")
    return raw


def stored_codebooks(root: Path) -> list[Any]:
    """Every stored ``codebook`` record under ``root``, oldest ratification first."""
    from calibrated_agent_evaluation.cli.records import load_records_of

    return sorted(load_records_of(root, "codebook"), key=lambda r: r.ratified_at)


def modes_for(records: list[Any], version: str) -> tuple[Any, ...]:
    """The union of the modes ratified under ``version``, first ratification first."""
    seen: dict[str, Any] = {}
    for record in records:
        if record.version == version:
            for mode in record.modes:
                seen.setdefault(mode.name, mode)
    return tuple(seen.values())


def resolve_session_codebook(args: argparse.Namespace, root: Path) -> SessionCodebook:
    """Resolve the session's codebook from the flags and the stored records (see module doc)."""
    version = _version_flag(args)
    path = getattr(args, "codebook_file", None)
    if path:
        from calibrated_agent_evaluation.review import load_codebook_file

        book = load_codebook_file(path)
        if version is not None and version != book.version:
            raise CliError(
                f"--codebook-version {version!r} does not match the version in {path} "
                f"({book.version!r}); drop the flag or fix the file"
            )
        return SessionCodebook(
            book.version,
            tuple(book.modes),
            f"failure modes: {len(book.modes)} from {path}, codebook version {book.version!r}",
        )
    records = stored_codebooks(root)
    if version is not None:
        modes = modes_for(records, version)
        if modes:
            return SessionCodebook(
                version,
                modes,
                f"failure modes: {len(modes)} ratified under codebook version {version!r}",
            )
        return SessionCodebook(
            version,
            (),
            f"open coding under codebook version {version!r}: no failure-mode list for it, so a "
            "FAIL saves with its critique alone (pass --codebook FILE, or ratify modes with "
            "`calibrated-agent-evaluation cluster --ratify MAP`)",
        )
    hint = ""
    if records:
        newest = records[-1]
        hint = (
            f"; codebook version {newest.version!r} has "
            f"{len(modes_for(records, newest.version))} ratified mode(s): pass "
            f"--codebook-version {newest.version} to label with them"
        )
    return SessionCodebook(
        None,
        (),
        "open coding, unversioned: a FAIL saves with its critique alone and the labels carry no "
        f"codebook version{hint}",
    )
