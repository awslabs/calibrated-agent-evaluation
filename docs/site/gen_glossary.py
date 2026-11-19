# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Generate the glossary page from the abbreviations file that drives the site's hover tooltips.

``content/includes/abbreviations.md`` is the single source: every page gets its terms as tooltips
through ``pymdownx.snippets`` ``auto_append``, and this script renders the same entries as
``concepts/glossary.md``. Plural and spelling variants (same definition as an earlier entry) are
listed once, under the first spelling. HTML comments in the source carry what a tooltip cannot:
``group:`` starts a glossary section, ``see:`` links the entry above it to the page that explains
it, and ``glossary-only:`` adds a row with no tooltip, for words too common to underline site-wide.
"""

from __future__ import annotations

import posixpath
import re
from pathlib import Path

import mkdocs_gen_files

_ABBREVIATIONS = Path(__file__).parent / "content" / "includes" / "abbreviations.md"
_PAGE = "concepts/glossary.md"
_ENTRY = re.compile(r"^\*\[(?P<term>[^\]]+)\]:\s*(?P<definition>.+)$")
_GLOSSARY_ONLY = re.compile(
    r"^<!--\s*glossary-only:\s*(?P<term>[^:]+):\s*(?P<definition>.+?)\s*-->$"
)
_GROUP = re.compile(r"^<!--\s*group:\s*(?P<name>.+?)\s*-->$")
_SEE = re.compile(r"^<!--\s*see:\s*(?P<path>\S+)\s*-->$")


class Entry:
    """One glossary row: a term, its plain definition and the page that explains it, if any."""

    def __init__(self, term: str, definition: str) -> None:
        self.term = term
        self.definition = definition
        self.see: str | None = None


def glossary_groups(text: str) -> list[tuple[str, list[Entry]]]:
    """Return ``(group name, entries)`` in file order, skipping repeated definitions."""
    groups: list[tuple[str, list[Entry]]] = []
    seen: set[str] = set()
    last: Entry | None = None
    for line in (raw.strip() for raw in text.splitlines()):
        if group := _GROUP.match(line):
            groups.append((group["name"], []))
            last = None
            continue
        if see := _SEE.match(line):
            if last is not None:
                last.see = see["path"]
            continue
        match = _ENTRY.match(line) or _GLOSSARY_ONLY.match(line)
        if match is None:
            continue
        definition = match["definition"].strip()
        if definition in seen:
            continue
        if not groups:
            raise SystemExit(f"gen_glossary: entry {match['term']!r} comes before any group: line")
        seen.add(definition)
        last = Entry(match["term"].strip(), definition)
        groups[-1][1].append(last)
    return groups


def render(groups: list[tuple[str, list[Entry]]]) -> str:
    """The glossary page body: one definition list per group, sorted case-insensitively by term."""
    lines = [
        "# Glossary\n",
        "Terms the documentation uses with a specific meaning. Most of them show the same "
        "definition when you hover them on another page.\n",
    ]
    for name, entries in groups:
        lines.append(f"## {name}\n")
        for entry in sorted(entries, key=lambda e: e.term.casefold()):
            body = entry.definition
            if entry.see is not None:
                link = posixpath.relpath(entry.see, posixpath.dirname(_PAGE))
                body += f" [More]({link})"
            lines.append(f"{entry.term}\n:   {body}\n")
    return "\n".join(lines)


with mkdocs_gen_files.open(_PAGE, "w") as fd:
    fd.write(render(glossary_groups(_ABBREVIATIONS.read_text(encoding="utf-8"))))
mkdocs_gen_files.set_edit_path(_PAGE, "docs/site/content/includes/abbreviations.md")
