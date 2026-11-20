# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""The PyPI description resolves every README image and link at the release tag."""

from __future__ import annotations

import importlib.util
import os
import re
import shutil
import subprocess
import sys
import tomllib
from pathlib import Path

import pytest

_REPO = Path(__file__).resolve().parents[1]
_SCRIPT = _REPO / "scripts" / "check_pypi_readme.py"
_spec = importlib.util.spec_from_file_location("check_pypi_readme", _SCRIPT)
assert _spec is not None
assert _spec.loader is not None
check = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(check)

_VERSION = "9.9.9"
_RAW = f"{check.RAW}v{_VERSION}/"


def _rewrite(text: str, version: str) -> str:
    """Apply pyproject's substitutions the way hatch-fancy-pypi-readme does: each pattern in order
    with re.sub, then $HFPR_VERSION."""
    config = tomllib.loads((_REPO / "pyproject.toml").read_text(encoding="utf-8"))
    hook = config["tool"]["hatch"]["metadata"]["hooks"]["fancy-pypi-readme"]
    for sub in hook["substitutions"]:
        text = re.sub(sub["pattern"], sub["replacement"], text)
    return text.replace("$HFPR_VERSION", version)


def test_the_rewritten_readme_has_nothing_pypi_cannot_resolve():
    readme = (_REPO / "README.md").read_text(encoding="utf-8")
    assert check.problems(_rewrite(readme, _VERSION), _VERSION, _REPO) == []


def test_the_readme_itself_uses_relative_image_paths():
    """Absolute raw URLs would break the images on a branch, in a pull request and in an editor."""
    readme = (_REPO / "README.md").read_text(encoding="utf-8")
    assert check.RAW not in readme


def test_a_relative_image_or_link_is_flagged():
    text = '<img src="docs/a.svg"> [guide](CONTRIBUTING.md) ![x](docs/b.png)'
    found = check.problems(text, _VERSION, _REPO)
    assert any("docs/a.svg" in f for f in found)
    assert any("CONTRIBUTING.md" in f for f in found)
    assert any("docs/b.png" in f for f in found)


def test_an_image_pinned_to_main_is_flagged():
    text = f'<img src="{check.RAW}main/docs/site/content/assets/readme/banner-light.svg">'
    assert any("not pinned" in f for f in check.problems(text, _VERSION, _REPO))


def test_a_pinned_path_missing_from_the_checkout_is_flagged():
    text = f'<img src="{_RAW}docs/no-such.svg"> [x]({check.BLOB}v{_VERSION}/NO-SUCH.md)'
    found = check.problems(text, _VERSION, _REPO)
    assert any("image path missing" in f for f in found)
    assert any("link path missing" in f for f in found)


def test_third_party_images_and_code_fences_are_left_alone():
    text = '<img src="https://img.shields.io/pypi/v/x">\n```text\n[not](a-link)\n```\n'
    assert check.problems(text, _VERSION, _REPO) == []


@pytest.mark.wheel
def test_the_built_wheel_passes(tmp_path: Path):
    uv = shutil.which("uv")
    if uv is None:
        pytest.skip("uv is not on PATH")
    env = {k: v for k, v in os.environ.items() if k != "VIRTUAL_ENV"}
    subprocess.run(
        [uv, "build", "--wheel", "--out-dir", str(tmp_path)], cwd=_REPO, env=env, check=True
    )
    (wheel,) = tmp_path.glob("*.whl")
    result = subprocess.run(
        [sys.executable, str(_SCRIPT), str(wheel), "--repo", str(_REPO)],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
