# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Hand-written docs pages carry no merge-conflict markers and no em dashes in their prose.

The strict MkDocs build renders a leftover conflict marker as ordinary text, so it has to be
caught here. Em dashes are banned by the docs style guide; captured command output inside fenced
blocks is exempt, since it is whatever the program printed.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

CONTENT = Path(__file__).resolve().parents[2] / "docs" / "site" / "content"
PAGES = sorted(CONTENT.rglob("*.md"))
_CONFLICT = re.compile(r"^(<{7}|>{7})( |$)|^={7}$")
_FENCE = re.compile(r"^\s*(`{3,}|~{3,})")


def _prose_lines(text: str) -> list[tuple[int, str]]:
    """The page's lines outside fenced code blocks, numbered from 1."""
    out: list[tuple[int, str]] = []
    fence: str | None = None
    for number, line in enumerate(text.splitlines(), start=1):
        match = _FENCE.match(line)
        if match:
            marker = match.group(1)
            if fence is None:
                fence = marker[0] * len(marker)
            elif marker.startswith(fence):
                fence = None
            continue
        if fence is None:
            out.append((number, line))
    return out


def test_pages_were_found() -> None:
    assert len(PAGES) > 20


@pytest.mark.parametrize("page", PAGES, ids=lambda p: str(p.relative_to(CONTENT)))
def test_no_conflict_markers(page: Path) -> None:
    lines = page.read_text(encoding="utf-8").splitlines()
    hits = [n for n, line in enumerate(lines, start=1) if _CONFLICT.match(line)]
    assert not hits, f"{page.relative_to(CONTENT)} has merge-conflict markers on lines {hits}"


@pytest.mark.parametrize("page", PAGES, ids=lambda p: str(p.relative_to(CONTENT)))
def test_no_em_dashes_in_prose(page: Path) -> None:
    hits = [n for n, line in _prose_lines(page.read_text(encoding="utf-8")) if "—" in line]
    assert not hits, f"{page.relative_to(CONTENT)} has em dashes on lines {hits}"


def test_the_guards_catch_what_they_guard() -> None:
    assert _CONFLICT.match("<<<<<<< HEAD")
    assert _CONFLICT.match(">>>>>>> branch")
    assert _CONFLICT.match("=======")
    assert not _CONFLICT.match("======== heading underline")
    sample = "prose — here\n```text\noutput — kept\n```\nafter"
    assert [n for n, line in _prose_lines(sample) if "—" in line] == [1]
