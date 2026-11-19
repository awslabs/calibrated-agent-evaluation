# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Fail if a source file does not start with the project's copyright and SPDX license header.

The header is two lines in the file type's comment syntax (``HEADERS``), at the top of the file or
right after a shebang line. Python, JavaScript and CSS files are checked; vendored files (any path
under a ``vendor/`` directory, and ``*.min.js``) keep their upstream headers and are skipped.

Usage: ``python scripts/check_license_header.py [--fix] path ...``. With ``--fix`` a missing
header is inserted instead of reported. The pre-commit hook passes the staged files.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_COPYRIGHT = "Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved."
_SPDX = "SPDX-License-Identifier: Apache-2.0"

HEADERS = {
    ".py": f"# {_COPYRIGHT}\n# {_SPDX}\n",
    ".js": f"// {_COPYRIGHT}\n// {_SPDX}\n",
    ".css": f"/* {_COPYRIGHT} */\n/* {_SPDX} */\n",
}
HEADER = HEADERS[".py"]


def is_vendored(path: Path) -> bool:
    """Whether ``path`` is a vendored file that keeps its upstream header."""
    return "vendor" in path.parts or path.name.endswith(".min.js")


def header_for(path: Path) -> str | None:
    """The header ``path`` must carry, or ``None`` when the file is not checked."""
    if is_vendored(path):
        return None
    return HEADERS.get(path.suffix)


def _split_shebang(text: str) -> tuple[str, str]:
    if text.startswith("#!"):
        first, sep, rest = text.partition("\n")
        return first + sep, rest
    return "", text


def has_header(text: str, header: str = HEADER) -> bool:
    """Whether ``text`` carries ``header`` at the top, after an optional shebang line."""
    return _split_shebang(text)[1].startswith(header)


def with_header(text: str, header: str = HEADER) -> str:
    """``text`` with ``header`` inserted after any shebang, followed by a blank line."""
    shebang, body = _split_shebang(text)
    if not body:
        return shebang + header
    spacer = "" if body.startswith("\n") else "\n"
    return shebang + header + spacer + body


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description="Check the copyright and SPDX header.")
    parser.add_argument("--fix", action="store_true", help="insert a missing header")
    parser.add_argument("paths", nargs="*", type=Path)
    args = parser.parse_args(argv)

    missing: list[tuple[Path, str]] = []
    for path in args.paths:
        header = header_for(path)
        if header is None:
            continue
        text = path.read_text(encoding="utf-8")
        if has_header(text, header):
            continue
        if args.fix:
            path.write_text(with_header(text, header), encoding="utf-8")
        else:
            missing.append((path, header))

    for path, header in missing:
        print(f"{path}: missing the copyright and SPDX header:\n{header}", file=sys.stderr)
    if missing:
        print(
            "Add the header at the top (after any shebang), or run "
            "`uv run python scripts/check_license_header.py --fix <file>`.",
            file=sys.stderr,
        )
    return 1 if missing else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
