# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Install hints name a uv command that works for how Calibrated Agent Evaluation was installed,
since the package is not on a public index yet and ``calibrated-agent-evaluation[x]`` from an index
fails from a checkout."""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from calibrated_agent_evaluation import _install
from calibrated_agent_evaluation._install import InstallSource, detect_install_source, hint_for

REPO = Path(__file__).resolve().parents[1]
WHEEL = "/tmp/dist/calibrated_agent_evaluation-0.1.0-py3-none-any.whl"


class _FakeDist:
    def __init__(self, direct_url: dict[str, object] | None, installer: str | None = None) -> None:
        self._direct = direct_url
        self._installer = installer

    def read_text(self, filename: str) -> str | None:
        if filename == "direct_url.json" and self._direct is not None:
            return json.dumps(self._direct)
        if filename == "INSTALLER":
            return self._installer
        return None


def _receipt(prefix: Path, extras: list[str], path: str) -> None:
    extras_toml = ", ".join(f'"{e}"' for e in extras)
    (prefix / "uv-receipt.toml").write_text(
        "[tool]\n"
        f'requirements = [{{ name = "calibrated-agent-evaluation", extras = [{extras_toml}], path '
        f'= "{path}" }}]\n'
    )


def test_the_checkouts_own_venv_is_a_project_install(tmp_path):
    venv = tmp_path / ".venv"
    venv.mkdir()
    dist = _FakeDist({"url": f"file://{tmp_path}", "dir_info": {"editable": True}})
    assert detect_install_source(dist, venv).kind == "project"


def test_an_editable_install_elsewhere_is_not_a_project_install(tmp_path):
    other = tmp_path / "elsewhere"
    other.mkdir()
    dist = _FakeDist({"url": f"file://{tmp_path}", "dir_info": {"editable": True}})
    assert detect_install_source(dist, other).kind == "editable"


def test_a_source_tree_that_was_never_installed_is_editable(tmp_path):
    assert detect_install_source(None, tmp_path).kind == "editable"


def test_a_uv_tool_env_reads_its_extras_and_wheel_from_the_receipt(tmp_path):
    _receipt(tmp_path, ["stats"], WHEEL)
    source = detect_install_source(_FakeDist({"url": f"file://{WHEEL}"}), tmp_path)
    assert source == InstallSource(kind="tool", path=WHEEL, tool_extras=("stats",))


def test_a_local_wheel_and_an_index_install_are_told_apart(tmp_path):
    assert detect_install_source(_FakeDist({"url": f"file://{WHEEL}"}), tmp_path).kind == "local"
    assert detect_install_source(_FakeDist(None), tmp_path).kind == "index"


def test_a_malformed_direct_url_reads_as_an_index_install(tmp_path):
    class Broken:
        def read_text(self, filename: str) -> str | None:
            return "{not json" if filename == "direct_url.json" else None

    assert detect_install_source(Broken(), tmp_path).kind == "index"


def test_a_project_sync_keeps_the_extras_already_installed():
    project = InstallSource(kind="project")
    assert hint_for(project, ("stats",)) == "uv sync --inexact --extra stats"
    assert (
        hint_for(project, ("sampling", "stats"))
        == "uv sync --inexact --extra sampling --extra stats"
    )


def test_another_editable_install_is_told_to_reinstall_with_the_extra():
    assert hint_for(InstallSource(kind="editable"), ("ui",)) == "uv pip install -e '.[ui]'"


def test_a_tool_reinstall_repeats_the_extras_it_already_has():
    tool = InstallSource(kind="tool", path=WHEEL, tool_extras=("stats",))
    assert (
        hint_for(tool, ("sampling",))
        == f"uv tool install 'calibrated-agent-evaluation[stats,sampling] @ {WHEEL}'"
    )
    assert (
        hint_for(tool, ("stats",))
        == f"uv tool install 'calibrated-agent-evaluation[stats] @ {WHEEL}'"
    )


def test_a_local_wheel_is_named_as_a_direct_reference():
    local = InstallSource(kind="local", path=WHEEL)
    assert (
        hint_for(local, ("stats",))
        == f"uv pip install 'calibrated-agent-evaluation[stats] @ {WHEEL}'"
    )


def test_an_index_install_names_the_distribution():
    assert (
        hint_for(InstallSource(kind="index"), ("stats", "ui"))
        == "uv pip install 'calibrated-agent-evaluation[stats,ui]'"
    )


def test_a_uv_install_targets_the_running_interpreter():
    """A bare `uv pip install` writes into whichever environment uv finds first."""
    index = InstallSource(kind="index", python="/opt/venv/bin/python")
    assert (
        hint_for(index, ("stats",))
        == "uv pip install --python /opt/venv/bin/python 'calibrated-agent-evaluation[stats]'"
    )


def test_a_pip_install_is_told_to_use_pip_from_its_own_interpreter(tmp_path):
    source = detect_install_source(_FakeDist(None, installer="pip\n"), tmp_path, "/opt/py/python")
    assert (source.installer, source.python) == ("pip", "/opt/py/python")
    assert (
        hint_for(source, ("stats",))
        == "/opt/py/python -m pip install 'calibrated-agent-evaluation[stats]'"
    )
    editable = InstallSource(kind="editable", installer="pip", python="/opt/py/python")
    assert hint_for(editable, ("ui",)) == "/opt/py/python -m pip install -e '.[ui]'"


@pytest.mark.skipif(sys.platform == "win32", reason="POSIX shell quoting")
def test_paths_with_spaces_and_quotes_survive_the_shell():
    import shlex

    wheel = "/tmp/it's here/dist/calibrated_agent_evaluation-0.1.0-py3-none-any.whl"
    python = "/Users/a b/venv/bin/python"
    hint = hint_for(InstallSource(kind="local", path=wheel, python=python), ("stats",))
    assert shlex.split(hint) == [
        *("uv", "pip", "install", "--python", python),
        f"calibrated-agent-evaluation[stats] @ {wheel}",
    ]
    pip = hint_for(InstallSource(kind="index", installer="pip", python=python), ("ui",))
    assert shlex.split(pip) == [python, "-m", "pip", "install", "calibrated-agent-evaluation[ui]"]


def test_the_running_install_is_detected_without_error():
    _install._running_source.cache_clear()
    assert _install._running_source().kind in {"project", "editable", "tool", "local", "index"}


def _shell(cmd: str, env: dict[str, str]) -> None:
    subprocess.run(cmd, shell=True, check=True, env=env, capture_output=True, text=True)


@pytest.mark.integration
@pytest.mark.skipif(shutil.which("uv") is None, reason="needs uv on PATH")
def test_the_printed_commands_really_add_the_extra(tmp_path):
    """Build the wheel, install it the two non-checkout ways, and run the hint each one prints."""
    import os

    env = {
        **os.environ,
        "UV_TOOL_DIR": str(tmp_path / "tools"),
        "UV_TOOL_BIN_DIR": str(tmp_path / "bin"),
    }
    env.pop("VIRTUAL_ENV", None)
    subprocess.run(
        ["uv", "build", "--wheel", "-o", str(tmp_path / "dist"), str(REPO)],
        check=True,
        capture_output=True,
    )
    wheel = str(next((tmp_path / "dist").glob("*.whl")))
    probe = (
        "from calibrated_agent_evaluation._install import install_hint; import sys; "
        "print(install_hint(sys.argv[1]))"
    )

    _shell(f'uv tool install "calibrated-agent-evaluation[ui] @ {wheel}"', env)
    tool_python = tmp_path / "tools" / "calibrated-agent-evaluation" / "bin" / "python"
    hint = subprocess.run(
        [tool_python, "-c", probe, "stats"], check=True, capture_output=True, text=True
    ).stdout.strip()
    assert hint == f"uv tool install 'calibrated-agent-evaluation[ui,stats] @ {wheel}'"
    _shell(hint, env)
    subprocess.run([tool_python, "-c", "import scipy, fasthtml"], check=True)

    venv = tmp_path / "venv"
    subprocess.run(["uv", "venv", "-q", str(venv)], check=True)
    venv_env = {**env, "VIRTUAL_ENV": str(venv)}
    _shell(f'uv pip install -q "{wheel}"', venv_env)
    venv_python = venv / ("Scripts" if sys.platform == "win32" else "bin") / "python"
    hint = subprocess.run(
        [venv_python, "-c", probe, "stats"], check=True, capture_output=True, text=True
    ).stdout.strip()
    assert hint == (
        f"uv pip install --python {venv_python} 'calibrated-agent-evaluation[stats] @ {wheel}'"
    )
    _shell(hint, venv_env)
    subprocess.run([venv_python, "-c", "import scipy"], check=True)
