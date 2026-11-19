# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Sandboxed code execution: the ``SandboxBackend`` seam and the stdlib local default.

The execution scorers (``calibrated_agent_evaluation.scorers``) run agent-authored code through this
seam and never see a vendor type. Only CODE execution is pluggable; SQL execution is always local
``sqlite3`` and is deliberately not on this protocol.

``LocalSubprocessSandbox`` is the default: an isolated-mode (``python -I``) subprocess under a
wall-clock timeout, in a per-execution temporary working directory, with a minimal environment that
inherits none of the caller's variables, Python-level socket creation disabled inside the child and
POSIX CPU/memory limits applied where the platform offers them. It is best-effort resource and
accident isolation for benchmark code and is not a security boundary: the network block and the
working directory are Python-level measures that code can bypass (it can restore the socket module,
call the OS directly, or write to any absolute path the user can write). Untrusted model-written
code should run on the managed AgentCore Code Interpreter backend instead. On a platform without
POSIX ``resource`` limits the run still proceeds (timeout + network-off) and records that the
CPU/memory limits were not applied, rather than refusing.
"""

from __future__ import annotations

import contextlib
import math
import os
import signal
import subprocess
import sys
import tempfile
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol, runtime_checkable

from calibrated_agent_evaluation.execute.errors import SandboxError

# Source that disables Python-level socket creation in the process that runs it. The local sandbox
# prepends it to every program; a program that runs further Python children (the code scorer's
# harness does) must apply it in them too, since a fresh interpreter starts unpatched. Best-effort
# only: code can undo it. Patching ``socket.socket.__init__`` keeps the class intact, so modules
# that subclass it (``ssl``) still import.
NETWORK_BLOCK_SOURCE = (
    "import socket as _cae_socket, _socket as _cae_csocket\n"
    "def _cae_blocked(*a, **k):\n"
    "    raise OSError('network disabled in the sandbox')\n"
    "_cae_socket.socket.__init__ = _cae_blocked\n"
    "for _cae_mod, _cae_name in ((_cae_socket, 'create_connection'), (_cae_socket, 'socketpair'),\n"
    "        (_cae_socket, 'fromfd'), (_cae_csocket, 'socket'), (_cae_csocket, 'socketpair')):\n"
    "    if hasattr(_cae_mod, _cae_name):\n"
    "        setattr(_cae_mod, _cae_name, _cae_blocked)\n"
    "del _cae_socket, _cae_csocket, _cae_mod, _cae_name\n"
)

# ``python -I`` leaves the script's directory off ``sys.path``; put the sandbox dir back so seeded
# files stay importable.
_PREAMBLE = "import os as _cae_os, sys as _cae_sys\n_cae_sys.path.insert(0, _cae_os.getcwd())\n" + (
    NETWORK_BLOCK_SOURCE
)

# 2 GiB address-space cap: generous enough not to kill CPython itself.
_DEFAULT_MAX_MEMORY_BYTES = 2 * 1024 * 1024 * 1024
# 1 MiB of stdout/stderr each; a runaway printer is truncated (with a marker, never silently).
_OUTPUT_CAP_BYTES = 1024 * 1024
# 128 MiB cap on any single file the child writes (its captured stdout/stderr included): bounds the
# disk a runaway writer can consume, while staying generous enough for a legitimate output file.
_MAX_FILE_BYTES = 128 * 1024 * 1024


@dataclass(frozen=True)
class SandboxResult:
    """The normalized outcome of one sandboxed execution, free of any vendor type.

    ``exit_code`` is 0 on clean completion; ``timed_out`` is True when the wall-clock limit fired
    (``exit_code`` is then non-zero and undefined). ``limits_applied`` is False when the platform
    offered no POSIX resource limits, so a run on a limit-less host is auditable rather than
    silently unbounded.
    """

    stdout: str
    stderr: str
    exit_code: int
    timed_out: bool
    limits_applied: bool
    duration_s: float


@runtime_checkable
class SandboxBackend(Protocol):
    """Run a snippet of code and return a normalized :class:`SandboxResult`.

    The methodology plane programs against this; it never imports a vendor SDK type.
    Implementations raise :class:`SandboxError` when the execution could not be carried out (as
    opposed to a program that ran and exited non-zero, which is a normal ``SandboxResult``).
    """

    def run_code(
        self,
        code: str,
        *,
        language: str = "python",
        timeout_s: float,
        files: dict[str, str] | None = None,
    ) -> SandboxResult:
        """Execute ``code`` (seeded with ``files`` in its working directory) and return the result.

        Raises :class:`SandboxError` if the execution could not be carried out.
        """
        ...


def _posix_limit_setter(timeout_s: float, max_memory_bytes: int) -> Callable[[], None] | None:
    """Return a ``preexec_fn`` applying POSIX CPU/address-space limits.

    Returns ``None`` when the platform has no ``resource`` module (e.g. Windows), so the caller
    records that CPU/memory limits were not applied rather than claiming an isolation it lacks.
    """
    try:
        import resource
    except ImportError:
        return None

    cpu_seconds = max(1, math.ceil(timeout_s) + 1)

    def _apply() -> None:  # pragma: no cover - runs in the forked child, not measurable by coverage
        resource.setrlimit(resource.RLIMIT_CPU, (cpu_seconds, cpu_seconds))
        # Some platforms refuse an address-space cap; the CPU limit still bounds a runaway.
        with contextlib.suppress(ValueError, OSError):
            resource.setrlimit(resource.RLIMIT_AS, (max_memory_bytes, max_memory_bytes))
        # Bound the bytes the child can write to any file (its captured stdout/stderr too), so a
        # runaway printer fills disk no further than this before SIGXFSZ stops it.
        with contextlib.suppress(ValueError, OSError):
            resource.setrlimit(resource.RLIMIT_FSIZE, (_MAX_FILE_BYTES, _MAX_FILE_BYTES))

    return _apply


class LocalSubprocessSandbox:
    """The default, dependency-free sandbox: a timed, isolated-mode subprocess with a minimal env.

    See the module docstring for the threat model: this is best-effort resource and accident
    isolation, not a security boundary. ``max_memory_bytes`` caps the child's address space where
    POSIX allows it.
    """

    def __init__(self, *, max_memory_bytes: int = _DEFAULT_MAX_MEMORY_BYTES) -> None:
        self._max_memory_bytes = max_memory_bytes

    def run_code(
        self,
        code: str,
        *,
        language: str = "python",
        timeout_s: float,
        files: dict[str, str] | None = None,
    ) -> SandboxResult:
        """Run ``code`` as a Python program in a temp cwd seeded with ``files``.

        Raises :class:`SandboxError` for an unsupported language, a ``files`` name that resolves
        outside the sandbox directory (absolute, or escaping through ``..``), or if the child
        cannot be started.
        """
        if language != "python":
            raise SandboxError(
                f"the local sandbox runs Python only, got language={language!r}; use the "
                "AgentCore Code Interpreter backend for other languages"
            )
        setter = _posix_limit_setter(timeout_s, self._max_memory_bytes)
        limits_applied = setter is not None
        program = _PREAMBLE + code

        with tempfile.TemporaryDirectory(prefix="calibrated_agent_evaluation_sandbox_") as tmp:
            cwd = Path(tmp).resolve()
            seeded = [
                (_resolve_inside(cwd, name), content) for name, content in (files or {}).items()
            ]
            for dest, content in seeded:
                dest.parent.mkdir(parents=True, exist_ok=True)
                dest.write_text(content, encoding="utf-8")
            main = cwd / "_calibrated_agent_evaluation_main.py"
            main.write_text(program, encoding="utf-8")
            out_path = cwd / "_calibrated_agent_evaluation_stdout"
            err_path = cwd / "_calibrated_agent_evaluation_stderr"
            started = time.monotonic()
            # Stream the child's output straight to files rather than PIPEs: communicate() would
            # buffer an unbounded amount in THIS process's memory, so a candidate stuck in a print
            # loop could exhaust the scorer's RAM long before the timeout fired. The files are read
            # back capped, and RLIMIT_FSIZE bounds what the child can write to them.
            try:
                with out_path.open("wb") as out_f, err_path.open("wb") as err_f:
                    proc = subprocess.Popen(  # noqa: S603 - executing benchmark code is the feature
                        [sys.executable, "-I", str(main)],
                        cwd=str(cwd),
                        env=_child_env(cwd),
                        stdout=out_f,
                        stderr=err_f,
                        preexec_fn=setter,
                        # Own session/process group, so a timeout can kill grandchildren too, not
                        # just the direct child (PID == PGID on POSIX).
                        start_new_session=True,
                    )
                    try:
                        proc.wait(timeout=timeout_s)
                        timed_out = False
                    except subprocess.TimeoutExpired:
                        _kill_process_group(proc)
                        proc.wait()
                        timed_out = True
            except OSError as exc:
                raise SandboxError(f"could not start the sandbox subprocess: {exc}") from exc
            return SandboxResult(
                stdout=_read_capped(out_path),
                stderr=_read_capped(err_path),
                exit_code=-1 if timed_out else proc.returncode,
                timed_out=timed_out,
                limits_applied=limits_applied,
                duration_s=time.monotonic() - started,
            )


def _resolve_inside(root: Path, name: str) -> Path:
    """Resolve a seeded file name under ``root``, refusing one that would land outside it."""
    dest = (root / name).resolve()
    if not dest.is_relative_to(root) or dest == root:
        raise SandboxError(
            f"seeded file name {name!r} resolves outside the sandbox directory; use a relative "
            "path without '..'"
        )
    return dest


def _child_env(workdir: Path) -> dict[str, str]:
    """The child's whole environment: nothing inherited but PATH (and SYSTEMROOT on Windows).

    Credentials such as AWS_* or LANGFUSE_* never reach the program. HOME and the temp-dir
    variables point at the sandbox directory so well-behaved code writes there.
    """
    env = {"PATH": os.environ.get("PATH", os.defpath)}
    for key in ("HOME", "TMPDIR", "TEMP", "TMP"):
        env[key] = str(workdir)
    if sys.platform == "win32" and "SYSTEMROOT" in os.environ:
        env["SYSTEMROOT"] = os.environ["SYSTEMROOT"]
    return env


def _kill_process_group(proc: subprocess.Popen[bytes]) -> None:
    """SIGKILL the child's whole process group so grandchildren die too; fall back to the child.

    ``start_new_session=True`` made the child a session/group leader (PID == PGID on POSIX), so the
    group can be signalled directly. On a platform without ``killpg`` (Windows) only the child is
    killed, which matches the pre-existing single-process behavior there.
    """
    try:
        os.killpg(proc.pid, signal.SIGKILL)
    except (AttributeError, ProcessLookupError, PermissionError, OSError):
        proc.kill()


def _read_capped(path: Path) -> str:
    """Read back a captured-output file, truncated to the cap and marked (never a silent drop).

    Reads at most one byte past the cap, so a huge file never pulls more than ~1 MiB into memory
    here regardless of how much the child wrote to disk.
    """
    try:
        with path.open("rb") as f:
            raw = f.read(_OUTPUT_CAP_BYTES + 1)
    except OSError:
        return ""
    decoded = raw.decode("utf-8", "replace")
    if len(decoded) <= _OUTPUT_CAP_BYTES:
        return decoded
    return decoded[:_OUTPUT_CAP_BYTES] + "\n...[output truncated at 1 MiB]"
