# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Fail if a requirement or ticket id appears in shipped code, the tests or the tooling around them.

Users read the library's comments, docstrings and messages in the API reference and the CLI, and
they can't resolve a planning id. The same holds for the scripts, the corpus agents and the shipped
skill plugin: someone reading them has no spec open. Tests are named and described by the behavior
they verify, so test source carries no ids either. In test files the guard also checks test
function names (``test_fr012_...`` or ``test_FR012_...``).

Two forms are allowed, and only these:

- A string literal that is exactly a decision-record rule code (``"ENF-2"``, ``"PRIN-3"``,
  ``"BIAS-1"``, ``"RUN-4"``). Records carry these as stable identifiers, like a linter's rule
  codes, and code compares against them. Any other id is flagged wherever it appears, including
  inside a literal.
- A line containing ``# allow-id: <reason>``. Use it only where changing the text would change
  behavior, such as model-facing schema text that recorded cassettes key on, or the guard's own
  pattern below.

A short one-letter id with one or two digits (a feature or acceptance-row number) is flagged in
comments, docstrings and non-Python files only, because the same shape is an ordinary fixture id
in test data.

Usage: ``python scripts/check_no_ids.py [path ...]``. With no paths it scans every root below,
wherever it is run from, and prints a count per root: the library and the tests, scripts, the
corpus agents, fixture prose, notes and generators, the docker stack, the skill plugin, the docs
site, the root Markdown files, and the build and hook config. ``--all-roots`` is accepted for older
callers and means the same. With paths it scans only those, using the Python, web and Markdown
suffixes. A path that does not exist is an error, and a file path scans that file.
"""

from __future__ import annotations

import argparse
import io
import re
import sys
import tokenize
from collections import Counter
from pathlib import Path

ID_PATTERN = re.compile(
    r"\b(?:"
    r"(?:N?FR|SC|US|ADR|ENF|PRIN|DEF|IMP|LBL|JDG|PROBE|BIAS|STAT|RUN|GEN)-\d+[a-z]?"
    r"|(?i:n?fr|sc|adr)-\d+[a-z]?"  # lowercase spellings
    r"|(?:N?FR|SC|REQ)\d{2,}"  # undashed spellings
    r"|REQ-[A-Z]+(?:-\d+)?"
    r"|[FPTD]-\d{3}"  # one-letter feature, principle, task and decision ids
    r")\b"
    r"|(?i:\bticket\s+#?\d+\b)"
    r"|Decision #\d+"
    r"|\.scratch/"  # allow-id: the pattern itself
    r"|\b[a-z]+(?:-[a-z]+)+/\d{2}\b"
)
# A short one-letter id is also an everyday fixture id in test data (a trace named by a letter and
# a number), so it is flagged only in prose: comments, docstrings, and non-Python files.
SHORT_ID = re.compile(r"\b[FPTDA]-\d{1,2}\b")
# Literals that are data, not ids: a whole decision-record rule code. Anything else in a literal
# is still checked.
DATA_LITERAL = re.compile(r'"(?:ENF|PRIN|BIAS|RUN)-\d+"')
# An id folded into a test function name: test_fr012_..., test_FR012_..., test_x_p006.
TEST_NAME_ID = re.compile(
    r"\bdef test_\w*?(?<=_)(?i:n?fr\d+[a-z]?|sc\d+[a-z]?|us\d+[a-z]?|req\d+[a-z]?|p\d{3}|f\d{3})"
    r"(?=_|\b)"
)
# The same id folded into a test class name: TestFR010LazyImport, TestF017DropIn.
TEST_CLASS_ID = re.compile(r"\bclass Test(?:N?FR|SC|US|REQ|F|P)\d{3}")
ESCAPE = re.compile(r"\\[ntr]")
REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ROOTS = ("src/calibrated_agent_evaluation", "tests")
# Python plus the review UI's shipped static assets and the docs that ship inside the package.
SCANNED_SUFFIXES = frozenset({".py", ".js", ".html", ".css", ".md"})
_SOURCE_SUFFIXES = SCANNED_SUFFIXES | {".toml", ".yaml", ".yml", ".sh"}
# Root -> the suffixes scanned under it. Fixtures are scanned for their prose, generators and the
# notes in hand-written fixture data; recorded traces, cassettes and goldens are captured data and
# keep whatever they recorded (see _EXCLUDED). The docs site and the root Markdown files are read
# by users, so they are scanned too.
_ROOT_DOCS = ("README.md", "CONTRIBUTING.md", "SECURITY.md", "CODE_OF_CONDUCT.md", "AGENTS.md")
EXTRA_ROOTS: dict[str, frozenset[str]] = {
    "scripts": _SOURCE_SUFFIXES,
    "corpus": _SOURCE_SUFFIXES,
    "fixtures": frozenset({".py", ".md", ".json", ".jsonl", ".txt", ".yaml", ".yml"}),
    "docker": frozenset({".md", ".yml", ".yaml", ".sh", ".example"}),
    "skills": _SOURCE_SUFFIXES | {".json"},
    ".claude-plugin": frozenset({".json"}),
    "docs/site": _SOURCE_SUFFIXES | {".txt"},
    "pyproject.toml": frozenset({".toml"}),
    ".pre-commit-config.yaml": frozenset({".yaml"}),
    "justfile": frozenset({""}),
    "noxfile.py": frozenset({".py"}),
    "CHANGELOG.md": frozenset({".md"}),
    "THIRD-PARTY-LICENSES": frozenset({""}),
    **{name: frozenset({".md"}) for name in _ROOT_DOCS},
}
# Under a scanned root, but captured data, vendored code or an internal-only tree rather than
# source: the docs goldens (captured command output), the recorded cassettes, the human gold labels
# that recorded model requests quote, the vendored minified scripts and the built site. Paths the
# repository marks ``export-ignore`` in .gitattributes are skipped too: they never leave the repo.
_EXCLUDED = (
    "fixtures/expected/docs/",
    "fixtures/raw/cassettes/",
    "fixtures/labels/",
    "docs/site/content/assets/javascripts/",
    "docs/site/_site/",
)
# Recorded data under fixtures/raw/ is captured output; its README and PROVENANCE prose and its
# generators are still scanned.
_RECORDED = "fixtures/raw/"
_RECORDED_SCANNED_SUFFIXES = frozenset({".md", ".py"})


def _export_ignored() -> tuple[str, ...]:
    attributes = REPO_ROOT / ".gitattributes"
    if not attributes.is_file():
        return ()
    prefixes = []
    for line in attributes.read_text(encoding="utf-8").splitlines():
        fields = line.split()
        if len(fields) >= 2 and "export-ignore" in fields[1:]:
            prefixes.append(fields[0].lstrip("/").removesuffix("/**").rstrip("/"))
    return tuple(prefixes)


_EXPORT_IGNORED = _export_ignored()


def _skipped(path: Path) -> bool:
    # Installed or vendored third-party trees under a scanned root are not project source.
    if any(part in {".venv", "node_modules", "__pycache__"} for part in path.parts):
        return True
    try:
        rel = path.resolve().relative_to(REPO_ROOT).as_posix()
    except ValueError:
        return False
    if rel.startswith(_RECORDED) and path.suffix not in _RECORDED_SCANNED_SUFFIXES:
        return True
    if any(rel == p or rel.startswith(f"{p}/") for p in _EXPORT_IGNORED):
        return True
    return rel.startswith(_EXCLUDED)


def _python_prose(text: str) -> dict[int, str]:
    """The comment and docstring text of each line of a Python source."""
    prose: dict[int, list[str]] = {}
    statement_start = True
    try:
        for tok in tokenize.generate_tokens(io.StringIO(text).readline):
            if tok.type == tokenize.COMMENT:
                prose.setdefault(tok.start[0], []).append(tok.string)
            elif tok.type == tokenize.STRING and statement_start:
                for offset, line in enumerate(tok.string.splitlines()):
                    prose.setdefault(tok.start[0] + offset, []).append(line)
            if tok.type in (tokenize.NEWLINE, tokenize.NL, tokenize.INDENT, tokenize.DEDENT):
                statement_start = True
            elif tok.type != tokenize.COMMENT:
                statement_start = False
    except (tokenize.TokenError, SyntaxError):
        return dict(enumerate(text.splitlines(), 1))
    return {n: " ".join(parts) for n, parts in prose.items()}


def violations(path: Path) -> list[str]:
    text = path.read_text(encoding="utf-8")
    lines = text.splitlines()
    prose = _python_prose(text) if path.suffix == ".py" else dict(enumerate(lines, 1))
    found = []
    for lineno, line in enumerate(lines, 1):
        if "# allow-id:" in line:
            continue
        # An escape such as \\n glues its letter to the id, which would hide it from \\b.
        text_line = ESCAPE.sub(" ", line)
        if (
            ID_PATTERN.search(DATA_LITERAL.sub('""', text_line))
            or TEST_NAME_ID.search(line)
            or TEST_CLASS_ID.search(line)
            or SHORT_ID.search(ESCAPE.sub(" ", prose.get(lineno, "")))
        ):
            found.append(f"{path}:{lineno}: {line.strip()}")
    return found


def _files(root: Path, suffixes: frozenset[str] = SCANNED_SUFFIXES) -> list[Path]:
    if root.is_file():
        return [root]
    return [
        p
        for p in sorted(root.rglob("*"))
        if p.suffix in suffixes and p.is_file() and not _skipped(p)
    ]


def _parse(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Fail if a requirement or ticket id appears in shipped code, the tests or the "
            "tooling around them. With no paths, scans every root."
        )
    )
    parser.add_argument("paths", nargs="*", type=Path, help="files or directories to scan")
    parser.add_argument(
        "--all-roots",
        action="store_true",
        help="scan every root (the default with no paths; kept for older callers)",
    )
    args = parser.parse_args(argv)
    if args.all_roots and args.paths:
        parser.error("--all-roots scans the repository roots; it takes no paths")
    return args


def main(argv: list[str]) -> int:
    args = _parse(argv)
    full_scan = not args.paths
    if full_scan:
        roots = dict.fromkeys((REPO_ROOT / r for r in DEFAULT_ROOTS), SCANNED_SUFFIXES)
        roots |= {REPO_ROOT / r: suffixes for r, suffixes in EXTRA_ROOTS.items()}
    else:
        roots = dict.fromkeys(args.paths, SCANNED_SUFFIXES)
    missing = [r for r in roots if not r.exists()]
    if missing:
        print(f"no such path: {', '.join(map(str, missing))}", file=sys.stderr)
        return 2
    per_root: Counter[Path] = Counter()
    found = []
    for root, suffixes in roots.items():
        for path in _files(root, suffixes):
            hits = violations(path)
            per_root[root] += len(hits)
            found.extend(hits)
    for v in found:
        print(v)
    if full_scan:
        for root in roots:
            print(f"{per_root[root]:6d}  {root.relative_to(REPO_ROOT)}", file=sys.stderr)
    if found:
        print(
            f"\n{len(found)} line(s) carry a requirement or ticket id. Say the rule in plain words "
            "instead, and name tests for the behavior they verify; ids belong in planning docs.",
            file=sys.stderr,
        )
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
