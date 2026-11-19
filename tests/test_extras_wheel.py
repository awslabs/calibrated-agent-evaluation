# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Each optional extra, installed alone from the built wheel, imports every module it enables.

``test_extras_closure.py`` checks the same map quickly by hiding distributions inside the dev
environment. This file checks it the way a user meets it: build the wheel, install
``calibrated-agent-evaluation[<extra>]`` into a fresh virtual environment, and import from there. It
needs ``uv`` and a package index or a warm uv cache, so it is opt-in: ``just test-extras``.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

from _extras import REPO, declared_extras, modules_for

pytestmark = pytest.mark.wheel

# Each module imports with every calibrated_agent_evaluation module dropped from sys.modules first,
# so a package whose __init__ failed halfway cannot leave submodules behind that make a later import
# look fine.
_IMPORT_EACH = textwrap.dedent(
    """
    import importlib
    import json
    import sys

    failures = {}
    for name in sys.argv[1:]:
        for loaded in [m for m in sys.modules if m == "calibrated_agent_evaluation" or
            m.startswith("calibrated_agent_evaluation.")]: del sys.modules[loaded]
        try:
            importlib.import_module(name)
        except Exception as exc:
            failures[name] = f"{type(exc).__name__}: {exc}"
    print(json.dumps(failures))
    """
)


def _uv() -> str:
    uv = shutil.which("uv")
    if uv is None:
        pytest.skip("uv is not on PATH")
    return uv


def _clean_env() -> dict[str, str]:
    drop = {"VIRTUAL_ENV", "UV_PROJECT_ENVIRONMENT", "PYTHONPATH", "PYTHONHOME"}
    return {k: v for k, v in os.environ.items() if k not in drop}


@pytest.fixture(scope="session")
def wheel(tmp_path_factory: pytest.TempPathFactory) -> Path:
    out = tmp_path_factory.mktemp("wheel")
    subprocess.run(
        [_uv(), "build", "--wheel", "--out-dir", str(out), str(REPO)],
        check=True,
        capture_output=True,
        env=_clean_env(),
        timeout=300,
    )
    (built,) = out.glob("calibrated_agent_evaluation-*.whl")
    return built


def _fresh_install(wheel: Path, venv: Path, extra: str) -> Path:
    """Install the wheel with ``extra`` (``""`` for none) into a new venv; return its python."""
    env = _clean_env()
    python = venv / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
    subprocess.run(
        [_uv(), "venv", "--quiet", "--python", sys.executable, str(venv)],
        check=True,
        env=env,
        timeout=120,
    )
    requirement = (
        f"calibrated-agent-evaluation[{extra}]" if extra else "calibrated-agent-evaluation"
    )
    subprocess.run(
        [
            _uv(),
            "pip",
            "install",
            "--quiet",
            "--python",
            str(python),
            f"{requirement} @ {wheel.as_uri()}",
        ],
        check=True,
        env=env,
        timeout=600,
    )
    return python


def _import_each(python: Path, cwd: Path, modules: tuple[str, ...]) -> dict[str, str]:
    result = subprocess.run(
        [str(python), "-I", "-c", _IMPORT_EACH, *modules],
        capture_output=True,
        text=True,
        check=False,
        cwd=cwd,
        env=_clean_env(),
        timeout=600,
    )
    assert result.returncode == 0, result.stderr[-3000:]
    return json.loads(result.stdout.strip().splitlines()[-1])


@pytest.mark.parametrize("extra", ["", *declared_extras()], ids=lambda e: e or "bare")
def test_a_fresh_install_of_one_extra_imports_every_module_it_enables(
    wheel: Path, tmp_path: Path, extra: str
) -> None:
    python = _fresh_install(wheel, tmp_path / "venv", extra)
    failures = _import_each(python, tmp_path, modules_for(extra))
    assert failures == {}, json.dumps(failures, indent=2)


def test_the_check_has_teeth(wheel: Path, tmp_path: Path) -> None:
    """A bare install cannot import a module listed under [stats]."""
    python = _fresh_install(wheel, tmp_path / "venv", "")
    failures = _import_each(python, tmp_path, ("calibrated_agent_evaluation.stats.rogan_gladen",))
    assert "ModuleNotFoundError" in failures["calibrated_agent_evaluation.stats.rogan_gladen"]
