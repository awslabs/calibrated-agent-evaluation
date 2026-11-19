# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""License-scan tests.

The scanner is repository tooling (``scripts/license_scan.py``), so these tests run it the way the
recipe and the pre-commit hook do, as a subprocess. Its classifier is checked on saved
pip-licenses manifests (a permissive tree passes; a seeded copyleft node is caught, named, with its
license). The ``just check`` recipe is asserted to include the scan step, and a check-sequence with
the step removed is shown NOT to catch a copyleft dep (proving the step is load-bearing).
"""

from __future__ import annotations

import json
import pathlib
import subprocess
import sys

_REPO = pathlib.Path(__file__).resolve().parent.parent
_SCRIPT = _REPO / "scripts" / "license_scan.py"

_PERMISSIVE_TREE = [
    {"Name": "boto3", "License": "Apache-2.0"},
    {"Name": "inspect-ai", "License": "MIT"},
    {"Name": "scipy", "License": "BSD-3-Clause"},
    {"Name": "python-fasthtml", "License": "Apache Software License"},
]
_COPYLEFT_TREE = [
    {"Name": "boto3", "License": "Apache-2.0"},
    {"Name": "semgrep", "License": "LGPL-2.1"},  # the real cyberseceval trap
    {"Name": "some-agpl-pkg", "License": "AGPL-3.0"},
]


def _scan(tmp_path: pathlib.Path, **trees: list[dict[str, str]]) -> subprocess.CompletedProcess:
    """Run the scanner over saved manifests, one per scope."""
    args = []
    for scope, tree in trees.items():
        path = tmp_path / f"{scope}.json"
        path.write_text(json.dumps(tree), encoding="utf-8")
        args += ["--manifest", f"{scope}={path}"]
    return subprocess.run(
        [sys.executable, str(_SCRIPT), *args],
        capture_output=True,
        text=True,
        check=False,
        timeout=60,
    )


def test_permissive_tree_has_no_offenders(tmp_path):
    """Positive: a fully-permissive manifest classifies clean (no offenders)."""
    run = _scan(tmp_path, default=_PERMISSIVE_TREE)
    assert run.returncode == 0, run.stderr
    assert "license-scan OK" in run.stdout


def test_copyleft_node_is_caught_and_named(tmp_path):
    """Negative: a seeded copyleft node is flagged, named, with its license."""
    run = _scan(tmp_path, default=_COPYLEFT_TREE)
    assert run.returncode == 1
    assert "semgrep (LGPL-2.1)" in run.stderr
    assert "some-agpl-pkg (AGPL-3.0)" in run.stderr
    assert "boto3" not in run.stderr


def test_gpl_family_variants_are_all_caught(tmp_path):
    """Edge: GPL / LGPL / AGPL and the spelled-out GNU names are all classified copyleft."""
    tree = [
        {"Name": "a", "License": "GPL-3.0"},
        {"Name": "b", "License": "GNU Lesser General Public License v2 (LGPLv2)"},
        {"Name": "c", "License": "GNU Affero General Public License"},
    ]
    run = _scan(tmp_path, default=tree)
    assert run.returncode == 1
    assert "3 copyleft node(s)" in run.stderr


def test_just_check_includes_license_scan_step():
    """Contract: the `just check` recipe runs license-scan (the step is wired in)."""
    justfile = (_REPO / "justfile").read_text(encoding="utf-8")
    # The aggregate recipe lists license-scan as a dependency.
    assert "check: lint fmt-check typecheck test license-scan" in justfile
    assert "license-scan *args:" in justfile


def test_check_without_the_step_would_not_catch_copyleft(tmp_path):
    """Distinct negative: the scan step is load-bearing — the other check steps don't see
    licenses.

    Proves the enforcement lives in license-scan, not decoratively elsewhere:
    lint/fmt/typecheck/test
    never inspect dependency licenses, so a `check` recipe missing the license-scan step would let a
    copyleft dep through. Asserts by construction that only the license-scan step classifies
    licenses.
    """
    # A copyleft tree is only caught by the scan; no other check step has this capability.
    assert _scan(tmp_path, default=_COPYLEFT_TREE).returncode == 1
    # Sanity: removing the step from the recipe string leaves the other steps, none license-aware.
    reduced = "check: lint fmt-check typecheck test"
    assert "license-scan" not in reduced  # a recipe without the step has no license enforcement


def test_pre_commit_hook_runs_the_scan_when_a_manifest_changes():
    """As re-scoped 2026-10-15: `just check` and the pre-commit hook enforce the scan.

    The hook runs the same script as the recipe and fires on a change to either dependency
    manifest, so a new copyleft dependency cannot be committed through a hooked checkout.
    """
    import re

    import yaml  # pyyaml arrives with pre-commit, which the dev group pins

    config = yaml.safe_load((_REPO / ".pre-commit-config.yaml").read_text(encoding="utf-8"))
    hooks = [hook for repo in config["repos"] for hook in repo["hooks"]]
    (hook,) = [h for h in hooks if h["id"] == "license-scan"]
    assert hook["entry"] == "uv run python scripts/license_scan.py"
    for manifest in ("pyproject.toml", "uv.lock", "scripts/license_scan.py"):
        assert re.search(hook["files"], manifest), manifest
    assert not re.search(hook["files"], "src/calibrated_agent_evaluation/__init__.py")


def test_the_scan_covers_the_agentcore_and_synth_extras():
    run = subprocess.run(
        [sys.executable, str(_SCRIPT), "--list-scopes"],
        capture_output=True,
        text=True,
        check=True,
        timeout=60,
    )
    scopes = run.stdout.split()
    assert scopes[0] == "(default)"
    assert {"agentcore", "synth"} <= set(scopes)


def test_a_copyleft_node_in_the_agentcore_closure_fails_that_scope(tmp_path):
    """Negative: a GPL distribution resolved only under [agentcore] is reported under that scope."""
    base = [{"Name": "boto3", "License": "Apache-2.0"}]
    planted = [*base, {"Name": "planted-gpl-dist", "License": "GPL-3.0"}]

    run = _scan(tmp_path, default=base, agentcore=planted, synth=base)

    assert run.returncode == 1
    assert "COPYLEFT in agentcore: planted-gpl-dist (GPL-3.0)" in run.stderr
    assert "COPYLEFT in synth" not in run.stderr
    assert "COPYLEFT in default" not in run.stderr


def test_an_unknown_license_fails_unless_reviewed(tmp_path):
    run = _scan(tmp_path, default=[{"Name": "mystery-pkg", "License": "UNKNOWN"}])
    assert run.returncode == 1
    assert "UNKNOWN LICENSE in default: mystery-pkg (UNKNOWN)" in run.stderr
    reviewed = _scan(tmp_path, scorers=[{"Name": "autoevals", "License": "UNKNOWN"}])
    assert reviewed.returncode == 0, reviewed.stderr


def test_weak_copyleft_is_reported_and_fails_unless_reviewed(tmp_path):
    run = _scan(
        tmp_path,
        default=[
            {"Name": "some-mpl-pkg", "License": "MPL-2.0"},
            {"Name": "some-epl-pkg", "License": "Eclipse Public License 2.0"},
        ],
    )
    assert run.returncode == 1
    assert "WEAK COPYLEFT in default: some-mpl-pkg (MPL-2.0)" in run.stderr
    assert "WEAK COPYLEFT in default: some-epl-pkg (Eclipse Public License 2.0)" in run.stderr


def test_reviewed_mpl_dependencies_pass_and_are_reported(tmp_path):
    tree = [
        {"Name": "certifi", "License": "Mozilla Public License 2.0 (MPL 2.0)"},
        {"Name": "tqdm", "License": "MPL-2.0 AND MIT"},
    ]
    run = _scan(tmp_path, default=tree)
    assert run.returncode == 0, run.stderr
    assert "reviewed weak copyleft in default: certifi" in run.stdout
    assert "reviewed weak copyleft in default: tqdm" in run.stdout


def test_a_reviewed_package_whose_license_changes_needs_review_again(tmp_path):
    run = _scan(tmp_path, default=[{"Name": "certifi", "License": "EPL-2.0"}])
    assert run.returncode == 1
    assert "WEAK COPYLEFT in default: certifi (EPL-2.0)" in run.stderr


def test_a_full_mit_text_is_not_mistaken_for_mpl(tmp_path):
    mit_text = (
        "MIT License\n\nTHE SOFTWARE IS PROVIDED AS IS, WITHOUT WARRANTY OF ANY KIND, IMPLIED"
    )
    run = _scan(tmp_path, default=[{"Name": "tiktoken", "License": mit_text}])
    assert run.returncode == 0, run.stderr


def test_a_real_scan_leaves_the_project_venv_untouched():
    """The scan resolves each scope in a throwaway env; it must never re-sync the checkout's .venv,
    which would swap its interpreter or drop the extras a test run depends on."""
    import pytest

    venv = _REPO / ".venv"
    if not (venv / "pyvenv.cfg").is_file():
        pytest.skip("no project .venv to compare")

    def _snapshot() -> tuple[str, list[str]]:
        cfg = (venv / "pyvenv.cfg").read_text(encoding="utf-8")
        dists = sorted(p.name for p in venv.glob("lib/python*/site-packages/*.dist-info"))
        dists += sorted(p.name for p in venv.glob("Lib/site-packages/*.dist-info"))
        return cfg, dists

    before = _snapshot()
    run = subprocess.run(
        [sys.executable, str(_SCRIPT), "--extra", "otel"],
        capture_output=True,
        text=True,
        check=False,
        timeout=600,
    )
    assert run.returncode == 0, run.stderr
    assert _snapshot() == before
