#!/usr/bin/env python3
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Fail when a built wheel's PyPI description has a reference PyPI cannot resolve.

The README uses repository-relative paths, and the build rewrites them to URLs pinned to the
release tag (see ``[tool.hatch.metadata.hooks.fancy-pypi-readme]`` in pyproject.toml). This reads
the description out of the wheel's METADATA and checks that nothing relative survived, that every
image is a raw URL at ``v<Version>``, and that every pinned path exists in the checkout.

    uv run python scripts/check_pypi_readme.py dist/*.whl
"""

from __future__ import annotations

import argparse
import re
import sys
import zipfile
from email.parser import Parser
from pathlib import Path

REPO_SLUG = "awslabs/calibrated-agent-evaluation"
RAW = f"https://raw.githubusercontent.com/{REPO_SLUG}/"
BLOB = f"https://github.com/{REPO_SLUG}/blob/"

_IMAGE_ATTR = re.compile(r'\b(?:src|srcset)="([^"]+)"')
_HREF = re.compile(r'\bhref="([^"]+)"')
_MD_IMAGE = re.compile(r"!\[[^\]]*\]\(([^)\s]+)\)")
_MD_LINK = re.compile(r"(?<!!)\[[^\]]*\]\(([^)\s]+)\)")
_ABSOLUTE = re.compile(r"^(?:https?://|#|mailto:)")
_FENCE = re.compile(r"^```.*?^```", re.DOTALL | re.MULTILINE)


def read_description(wheel: Path) -> tuple[str, str]:
    """Return the wheel's version and long description."""
    with zipfile.ZipFile(wheel) as zf:
        name = next(n for n in zf.namelist() if n.endswith(".dist-info/METADATA"))
        message = Parser().parsestr(zf.read(name).decode("utf-8"))
    version = message["Version"]
    body = message.get_payload()
    if not isinstance(body, str) or not body.strip():
        raise ValueError(f"{wheel.name} has no long description")
    return version, body


def problems(description: str, version: str, repo: Path) -> list[str]:
    """Every reference in ``description`` that PyPI would show broken, or that points at the wrong
    release."""
    text = _FENCE.sub("", description)
    found: list[str] = []
    images = _IMAGE_ATTR.findall(text) + _MD_IMAGE.findall(text)
    links = _HREF.findall(text) + _MD_LINK.findall(text)
    for ref in images + links:
        if not _ABSOLUTE.match(ref):
            found.append(f"relative reference left in the description: {ref}")
    raw_prefix = f"{RAW}v{version}/"
    for ref in images:
        if not ref.startswith(raw_prefix):
            if _ABSOLUTE.match(ref) and not ref.startswith(RAW):
                continue
            found.append(f"image not pinned to {raw_prefix}: {ref}")
        elif not (repo / ref.removeprefix(raw_prefix)).is_file():
            found.append(f"image path missing from the checkout: {ref}")
    blob_prefix = f"{BLOB}v{version}/"
    for ref in links:
        if ref.startswith(BLOB):
            if not ref.startswith(blob_prefix):
                found.append(f"link not pinned to {blob_prefix}: {ref}")
            elif not (repo / ref.removeprefix(blob_prefix).split("#")[0]).exists():
                found.append(f"link path missing from the checkout: {ref}")
    return found


def main() -> int:
    parser = argparse.ArgumentParser(description=(__doc__ or "").splitlines()[0])
    parser.add_argument("wheel", type=Path)
    parser.add_argument("--repo", type=Path, default=Path.cwd())
    args = parser.parse_args()
    version, description = read_description(args.wheel)
    found = problems(description, version, args.repo)
    for line in found:
        print(f"{args.wheel.name}: {line}", file=sys.stderr)
    if not found:
        print(f"{args.wheel.name}: PyPI description resolves every image and link at v{version}")
    return 1 if found else 0


if __name__ == "__main__":
    sys.exit(main())
