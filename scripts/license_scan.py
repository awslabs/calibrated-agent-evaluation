#!/usr/bin/env python3
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""License scan over the resolved dependency tree.

Checks the shipped dependency tree, the default install **and each optional extra** (`ui`,
`langfuse`, `bedrock`, …), and fails on:

* any copyleft (GPL/LGPL/AGPL) node;
* any weak-copyleft (MPL, EPL, CDDL) node that is not in the reviewed list below;
* any node whose license metadata is missing or ``UNKNOWN`` and that is not in the reviewed list.

Reviewed nodes are pinned to the license string they were reviewed under, so a release that
changes its license needs review again. Reviewed weak-copyleft nodes are printed on every run. The
scanner is `pip-licenses` (MIT); this script drives it per extra and classifies the results.

The classification is a pure function (:func:`classify`) so it is unit-testable on a fixture
manifest without resolving a real environment; :func:`scan` wires it to a live `uv`-resolved env
per extra. `just license-scan` calls :func:`main`.

Usage:
    uv run python scripts/license_scan.py            # scan default + every extra, exit non-zero
    on copyleft
    uv run python scripts/license_scan.py --extra bedrock   # scan one extra
    uv run python scripts/license_scan.py --list-scopes     # print the scopes a full scan covers
    uv run python scripts/license_scan.py --manifest ui=licenses.json   # classify a saved
    pip-licenses JSON for a scope instead of resolving it (repeatable)
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import tempfile
import tomllib
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
_PYPROJECT = _REPO_ROOT / "pyproject.toml"

# Copyleft license families forbidden in the shipped tree. Matched case-insensitively as
# substrings of the reported license string, with permissive carve-outs handled below.
_COPYLEFT_MARKERS = ("gpl", "lgpl", "agpl", "gnu general public", "gnu lesser", "gnu affero")

# Permissive licenses whose SPDX text contains a forbidden substring by coincidence. "LGPLLREF"
# never occurs; the real trap is a dual-license string like "MPL-2.0 OR LGPL-2.1" — we treat any
# node whose license names a copyleft family as an offender (conservative — flag for review).
_PERMISSIVE_EXCEPTIONS: frozenset[str] = frozenset()

# Word-bounded so a full license text ("... IMPLIED ...") does not read as MPL.
_WEAK_COPYLEFT = re.compile(
    r"\bmpl\b|mozilla public|\bepl\b|eclipse public|\bcddl\b|common development and distribution",
    re.IGNORECASE,
)

# Weak-copyleft nodes reviewed for the shipped tree, with the license string they were reviewed
# under. Each is used as an unmodified dependency; MPL-2.0 obligations apply only to modified
# files of the covered package.
REVIEWED_WEAK_COPYLEFT: dict[str, tuple[str, str]] = {
    "certifi": (
        "Mozilla Public License 2.0 (MPL 2.0)",
        "Mozilla's CA bundle, required by httpx and requests; used unmodified",
    ),
    "tqdm": (
        "MPL-2.0 AND MIT",
        "progress bars, required by huggingface-hub and datasets; unmodified",
    ),
}

# Nodes whose wheel declares no license metadata, with the license read from the LICENSE file the
# wheel ships, at the version named.
REVIEWED_UNKNOWN: dict[str, tuple[str, str]] = {
    "agent-client-protocol": ("UNKNOWN", "Apache-2.0 per its bundled LICENSE (checked at 0.12.1)"),
    "autoevals": ("UNKNOWN", "MIT per its bundled LICENSE (checked at 0.3.0)"),
    "polyleven": ("UNKNOWN", "MIT per its bundled LICENSE (checked at 0.11.0)"),
}


@dataclass
class Findings:
    """One scope's classification: what fails the scan, and the reviewed nodes it reports."""

    copyleft: list[tuple[str, str]] = field(default_factory=list)
    weak_copyleft: list[tuple[str, str]] = field(default_factory=list)
    unknown: list[tuple[str, str]] = field(default_factory=list)
    reviewed_weak_copyleft: list[tuple[str, str]] = field(default_factory=list)

    @property
    def failures(self) -> int:
        return len(self.copyleft) + len(self.weak_copyleft) + len(self.unknown)


def _is_reviewed(reviewed: dict[str, tuple[str, str]], name: str, license_str: str) -> bool:
    entry = reviewed.get(name)
    return entry is not None and entry[0] == license_str


def classify(packages: Iterable[dict[str, str]]) -> Findings:
    """Classify pip-licenses-shaped records (``{"Name", "License"}``) for one scope."""
    packages = list(packages)
    findings = Findings()
    copyleft = {name for name, _ in copyleft_offenders(packages)}
    for pkg in packages:
        name = pkg.get("Name", "?")
        license_str = (pkg.get("License", "") or "").strip()
        entry = (name, license_str)
        if name in copyleft:
            findings.copyleft.append(entry)
        elif license_str.upper() in ("", "UNKNOWN"):
            if not _is_reviewed(REVIEWED_UNKNOWN, name, license_str or "UNKNOWN"):
                findings.unknown.append(entry)
        elif _WEAK_COPYLEFT.search(license_str):
            if _is_reviewed(REVIEWED_WEAK_COPYLEFT, name, license_str):
                findings.reviewed_weak_copyleft.append(entry)
            else:
                findings.weak_copyleft.append(entry)
    return findings


def copyleft_offenders(packages: Iterable[dict[str, str]]) -> list[tuple[str, str]]:
    """Return ``(name, license)`` for every package whose license names a copyleft family.

    Pure classification over pip-licenses-shaped records (``{"Name", "License"}``) — the unit-
    testable core. Conservative: any copyleft substring flags the node for review, because no
    copyleft node may ship.
    """
    offenders: list[tuple[str, str]] = []
    for pkg in packages:
        name = pkg.get("Name", "?")
        license_str = (pkg.get("License", "") or "").lower()
        if name in _PERMISSIVE_EXCEPTIONS:
            continue
        if any(marker in license_str for marker in _COPYLEFT_MARKERS):
            offenders.append((name, pkg.get("License", "")))
    return offenders


def _all_extras() -> list[str]:
    """Every optional extra declared in pyproject (the scan iterates each one)."""
    data = tomllib.loads(_PYPROJECT.read_text(encoding="utf-8"))
    return sorted(data.get("project", {}).get("optional-dependencies", {}).keys())


def _resolve_and_list(extra: str | None) -> list[dict[str, str]]:
    """Resolve the env for the default install (extra=None) or one extra, and pip-licenses it.

    Uses ``uv run --no-project --isolated`` from a temporary directory, with the package + extra
    installed into an ephemeral env so the resolved tree matches what a user would get, then runs
    ``pip-licenses --format=json`` (which leaves out pip-licenses and its own dependencies).
    """
    target = f"{_REPO_ROOT}[{extra}]" if extra else str(_REPO_ROOT)
    cmd = [
        "uv",
        "run",
        "--no-project",
        "--isolated",
        "--with",
        "pip-licenses",
        "--with",
        target,
        "pip-licenses",
        "--format=json",
    ]
    # Run outside the checkout: inside it, uv picks the project's .venv as the base interpreter and
    # layers the --with packages over it, so every scope would list the whole dev environment.
    with tempfile.TemporaryDirectory(prefix="license-scan-") as cwd:
        result = subprocess.run(  # noqa: S603
            cmd, cwd=cwd, capture_output=True, text=True, check=True
        )
    return json.loads(result.stdout)


def scan(extras: list[str] | None = None) -> dict[str, Findings]:
    """Scan the default tree + each extra; return ``{scope: findings}``."""
    scopes: list[str | None] = [None, *(extras if extras is not None else _all_extras())]
    return {extra or "(default)": classify(_resolve_and_list(extra)) for extra in scopes}


def _scan_manifests(manifests: list[str]) -> dict[str, Findings]:
    """Classify saved pip-licenses JSON files given as ``SCOPE=FILE``, without resolving."""
    findings: dict[str, Findings] = {}
    for spec in manifests:
        scope, sep, path = spec.partition("=")
        if not sep or not scope or not path:
            raise SystemExit(f"--manifest takes SCOPE=FILE, got {spec!r}")
        try:
            packages = json.loads(Path(path).read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise SystemExit(f"cannot read the manifest for {scope!r} at {path}: {exc}") from exc
        findings[scope] = classify(packages)
    return findings


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Dependency license scan.")
    parser.add_argument("--extra", default=None, help="scan only this extra (default: all + base)")
    parser.add_argument(
        "--manifest",
        action="append",
        default=None,
        metavar="SCOPE=FILE",
        help="classify a saved pip-licenses JSON for SCOPE instead of resolving (repeatable)",
    )
    parser.add_argument(
        "--list-scopes", action="store_true", help="print the scopes a full scan covers and exit"
    )
    args = parser.parse_args(argv)

    if args.list_scopes:
        print("\n".join(["(default)", *_all_extras()]))
        return 0
    if args.manifest:
        findings = _scan_manifests(args.manifest)
    else:
        findings = scan([args.extra] if args.extra else None)

    for scope, found in findings.items():
        for name, lic in found.reviewed_weak_copyleft:
            reason = REVIEWED_WEAK_COPYLEFT[name][1]
            print(f"note: reviewed weak copyleft in {scope}: {name} ({lic}): {reason}")
        for name, lic in found.copyleft:
            print(f"COPYLEFT in {scope}: {name} ({lic})", file=sys.stderr)
        for name, lic in found.weak_copyleft:
            print(f"WEAK COPYLEFT in {scope}: {name} ({lic}): not reviewed", file=sys.stderr)
        for name, lic in found.unknown:
            print(f"UNKNOWN LICENSE in {scope}: {name} ({lic or 'none declared'})", file=sys.stderr)

    copyleft = sum(len(f.copyleft) for f in findings.values())
    failures = sum(f.failures for f in findings.values())
    if failures:
        print(
            f"license-scan FAILED: {copyleft} copyleft node(s) in the shipped tree (GPL, LGPL and "
            f"AGPL are not allowed) and {failures - copyleft} weak-copyleft or unknown-license "
            "node(s) that need review; add a reviewed node to REVIEWED_WEAK_COPYLEFT or "
            "REVIEWED_UNKNOWN in scripts/license_scan.py with its license and reason.",
            file=sys.stderr,
        )
        return 1
    print(
        "license-scan OK: no copyleft node, and no unreviewed weak-copyleft or unknown-license "
        "node, in the default install or any optional extra."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
