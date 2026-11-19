# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""The command that adds an optional extra to the running installation.

Every "this needs an extra" message renders its install command here, so the command matches how
the package was installed: the checkout's ``uv sync``, a ``uv tool`` reinstall, or an install into
the running interpreter's environment, with pip when pip installed it and uv otherwise. Arguments
are quoted for the platform's shell.
"""

from __future__ import annotations

import json
import os
import shlex
import subprocess
import sys
import tomllib
from functools import cache
from importlib import metadata
from pathlib import Path
from typing import Literal, Protocol
from urllib.parse import urlparse
from urllib.request import url2pathname

from pydantic import BaseModel, ConfigDict

__all__ = ["InstallSource", "detect_install_source", "hint_for", "install_hint"]

_DISTRIBUTION = "calibrated-agent-evaluation"


class _Distribution(Protocol):
    def read_text(self, filename: str) -> str | None: ...


class InstallSource(BaseModel):
    """How the running Calibrated Agent Evaluation was installed."""

    model_config = ConfigDict(frozen=True)

    kind: Literal["project", "editable", "tool", "local", "index"]
    path: str | None = None
    """The wheel or directory a ``tool`` or ``local`` install came from."""
    tool_extras: tuple[str, ...] = ()
    """Extras already in a ``uv tool`` environment, which a reinstall must repeat to keep."""
    installer: str | None = None
    """The tool the distribution's ``INSTALLER`` file names (``pip``, ``uv``), when it has one."""
    python: str | None = None
    """The interpreter of the environment, which an install command has to target."""


def _direct_url(dist: _Distribution) -> dict[str, object]:
    raw = dist.read_text("direct_url.json")
    if raw is None:
        return {}
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        return {}
    return parsed if isinstance(parsed, dict) else {}


def _tool_receipt(prefix: Path) -> tuple[tuple[str, ...], str | None] | None:
    """The extras and source path recorded in a ``uv tool`` environment, or None outside one."""
    receipt = prefix / "uv-receipt.toml"
    if not receipt.is_file():
        return None
    try:
        doc = tomllib.loads(receipt.read_text(encoding="utf-8"))
    except (OSError, tomllib.TOMLDecodeError):
        return (), None
    for req in doc.get("tool", {}).get("requirements", []):
        if isinstance(req, dict) and req.get("name") == _DISTRIBUTION:
            return tuple(req.get("extras", ())), req.get("path")
    return (), None


def detect_install_source(
    dist: _Distribution | None, prefix: Path, python: str | None = None
) -> InstallSource:
    """Classify an installation from its distribution metadata and environment prefix.

    Args:
        dist: The installed ``calibrated-agent-evaluation`` distribution, or None when it was
            imported from a source tree that was never installed.
        prefix: The environment's ``sys.prefix``.
        python: The environment's interpreter, which the install command targets.
    """
    if dist is None:
        return InstallSource(kind="editable", python=python)
    installer = (dist.read_text("INSTALLER") or "").strip().lower() or None
    source = _classify(dist, prefix)
    return source.model_copy(update={"installer": installer, "python": python})


def _classify(dist: _Distribution, prefix: Path) -> InstallSource:
    direct = _direct_url(dist)
    url = str(direct.get("url", ""))
    # A file URL is percent-encoded: a checkout path with a space arrives as `%20`.
    local_path = url2pathname(urlparse(url).path) if url.startswith("file://") else None
    dir_info = direct.get("dir_info")
    if isinstance(dir_info, dict) and dir_info.get("editable"):
        # `uv sync` installs the checkout into <checkout>/.venv; any other editable install was
        # made into some other environment by hand.
        if local_path and prefix.resolve() == (Path(local_path) / ".venv").resolve():
            return InstallSource(kind="project")
        return InstallSource(kind="editable")
    receipt = _tool_receipt(prefix)
    if receipt is not None:
        extras, receipt_path = receipt
        return InstallSource(kind="tool", path=receipt_path or local_path, tool_extras=extras)
    if local_path:
        return InstallSource(kind="local", path=local_path)
    return InstallSource(kind="index")


def _quote(arg: str) -> str:
    """``arg`` as one word of the platform's shell."""
    return subprocess.list2cmdline([arg]) if os.name == "nt" else shlex.quote(arg)


def hint_for(source: InstallSource, extras: tuple[str, ...]) -> str:
    """The command that adds ``extras`` to an installation of the given kind.

    ``uv sync`` removes packages it was not asked for, so the project form passes ``--inexact``.
    ``uv tool install`` replaces the tool's extras, so the tool form repeats the ones it has.
    A path with extras has to be written as a direct reference (``name[x] @ path``); uv rejects
    ``path[x]``. Any other install is into the running interpreter's environment: a bare
    ``uv pip install`` targets whichever environment uv finds first, so the uv form names the
    interpreter with ``--python``, and an install pip made gets ``<python> -m pip install``.
    """
    names = ",".join(extras)
    if source.kind == "project":
        return "uv sync --inexact " + " ".join(f"--extra {e}" for e in extras)
    if source.kind == "tool":
        merged = ",".join(dict.fromkeys((*source.tool_extras, *extras)))
        spec = f"{_DISTRIBUTION}[{merged}]"
        return "uv tool install " + _quote(f"{spec} @ {source.path}" if source.path else spec)
    if source.kind == "editable":
        target = ["-e", _quote(f".[{names}]")]
    elif source.kind == "local" and source.path:
        target = [_quote(f"{_DISTRIBUTION}[{names}] @ {source.path}")]
    else:
        target = [_quote(f"{_DISTRIBUTION}[{names}]")]
    if source.installer == "pip":
        return " ".join([_quote(source.python or "python"), "-m", "pip", "install", *target])
    python = ["--python", _quote(source.python)] if source.python else []
    return " ".join(["uv", "pip", "install", *python, *target])


@cache
def _running_source() -> InstallSource:
    try:
        dist: _Distribution | None = metadata.distribution(_DISTRIBUTION)
    except metadata.PackageNotFoundError:
        dist = None
    return detect_install_source(dist, Path(sys.prefix), sys.executable)


def install_hint(*extras: str) -> str:
    """The shell command that installs ``extras`` into this installation.

    Args:
        extras: The optional extras to add, by their ``pyproject`` names.

    Returns:
        One command: ``uv sync --inexact --extra X`` in the checkout's own environment,
        ``uv tool install`` for a uv tool, and otherwise ``uv pip install --python PYTHON`` (or
        ``PYTHON -m pip install`` when pip installed it) with ``-e '.[X]'`` for another editable
        install or the distribution, as a direct reference to a local wheel or from an index.
    """
    return hint_for(_running_source(), extras)
