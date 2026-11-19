# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Resolve the ``--backend`` / ``--root`` flags to a concrete trace-store backend.

``local`` (the default) constructs the offline ``LocalBackend`` -- zero third-party deps, the laptop
path -- rooted at the local root: ``--root``, else ``CALIBRATED_AGENT_EVALUATION_LOCAL_ROOT``, else
``./.calibrated-agent-evaluation``. ``langfuse`` constructs the ``LangfuseBackend`` from environment
credentials; its optional SDK is imported **lazily** here so the offline path never loads it, and an
absent ``[langfuse]`` extra surfaces as a typed ``CliError`` with the install hint, not a raw
``ImportError``.

The local root matters in the Langfuse world too: the records the stages write (calibration, run,
improve, bias, flagship) live under ``<local_root>/records/`` whichever trace store is selected.
"""

from __future__ import annotations

import argparse
import os
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

from calibrated_agent_evaluation.backends import DEFAULT_LOCAL_ROOT, TraceStoreBackend
from calibrated_agent_evaluation.cli.config import probe_extra, resolve_backend
from calibrated_agent_evaluation.cli.errors import CliError, UsageError

__all__ = [
    "LOCAL_ROOT_ENV",
    "backend_from_args",
    "backend_from_flag",
    "explicit_local_root",
    "langfuse_credentials",
    "langfuse_from_args",
    "local_root",
    "open_backend",
]

# The one environment variable that selects the local root, for a shell that runs many stages.
LOCAL_ROOT_ENV = "CALIBRATED_AGENT_EVALUATION_LOCAL_ROOT"


def explicit_local_root(args: argparse.Namespace) -> Path | None:
    """The local root the user chose, else ``None``.

    ``--root`` wins over ``CALIBRATED_AGENT_EVALUATION_LOCAL_ROOT``.
    """
    flag = getattr(args, "root", None)
    if flag:
        return Path(flag)
    env = os.environ.get(LOCAL_ROOT_ENV)
    if env:
        return Path(env)
    return None


def local_root(args: argparse.Namespace) -> Path:
    """The local root every stage reads and writes.

    The explicit choice, else ``./.calibrated-agent-evaluation``.
    """
    return explicit_local_root(args) or Path(DEFAULT_LOCAL_ROOT)


def backend_from_args(args: argparse.Namespace) -> TraceStoreBackend:
    """Return the backend the parsed ``--backend`` / ``--root`` flags select.

    The Langfuse backend comes wrapped in
    :class:`~calibrated_agent_evaluation.cli.consistency.ConsistentBackend`: Langfuse ingests
    asynchronously, so a command's reads first wait for the writes the earlier commands on this root
    recorded. The local backend is synchronous.
    """
    flag = getattr(args, "backend", "local")
    if flag != "langfuse":
        return backend_from_flag(flag, local_root=local_root(args))
    return langfuse_from_args(args)


def langfuse_from_args(args: argparse.Namespace) -> TraceStoreBackend:
    """The Langfuse backend from ``LANGFUSE_*``, its reads consistent with this root's writes.

    What ``--backend langfuse`` selects, and the source
    ``calibrated-agent-evaluation ingest --from-langfuse`` pulls from whichever destination is
    selected.
    """
    from calibrated_agent_evaluation.cli.consistency import (
        ConsistentBackend,
        WriteManifest,
        manifest_path,
    )

    backend = backend_from_flag("langfuse")
    host, public_key = _langfuse_identity()
    manifest = WriteManifest(manifest_path(local_root(args), f"langfuse|{host}|{public_key}"))
    return ConsistentBackend(backend, manifest)


@contextmanager
def open_backend(args: argparse.Namespace) -> Iterator[TraceStoreBackend]:
    """:func:`backend_from_args` bound to a block; the backend is closed when the block exits."""
    backend = backend_from_args(args)
    try:
        yield backend
    finally:
        close = getattr(backend, "close", None)
        if callable(close):
            close()


def backend_from_flag(flag: str, *, local_root: str | Path | None = None) -> TraceStoreBackend:
    """Return the backend named by ``--backend`` (``local`` | ``langfuse``).

    ``local_root`` roots the local backend; ``None`` means the library default
    (``./.calibrated-agent-evaluation``).
    """
    if flag == "local":
        return resolve_backend(None, local_root=local_root)
    if flag == "langfuse":
        return _langfuse_backend()
    raise UsageError(f"unknown --backend {flag!r} (expected 'local' or 'langfuse')")


def _langfuse_backend() -> TraceStoreBackend:
    """Construct the Langfuse backend from env creds; typed refusal if the extra/env is missing."""
    probe = probe_extra("langfuse")
    if not probe.available:
        raise CliError(f"the langfuse backend needs the optional extra: {probe.install_hint}")
    from calibrated_agent_evaluation.backends import LangfuseBackend

    host, public_key = _langfuse_identity()
    secret_key = os.environ.get("LANGFUSE_SECRET_KEY")
    if not secret_key:
        raise CliError(_MISSING_KEYS)
    return LangfuseBackend.from_credentials(host=host, public_key=public_key, secret_key=secret_key)


# The docs page, not the checkout's docker/langfuse/.env.example: an installed package has no
# checkout to look in.
_MISSING_KEYS = (
    "the langfuse backend needs LANGFUSE_PUBLIC_KEY / LANGFUSE_SECRET_KEY in the environment "
    "(see https://awslabs.github.io/calibrated-agent-evaluation/integrate/langfuse/)"
)


def langfuse_credentials() -> tuple[str, str, str]:
    """``(host, public_key, secret_key)`` from ``LANGFUSE_*``; a typed refusal when a key is unset.

    For a Langfuse write that is not the trace store (the run dashboard), so it is read whatever
    ``--backend`` selects. Refuses without the ``[langfuse]`` extra too.
    """
    probe = probe_extra("langfuse")
    if not probe.available:
        raise CliError(f"publishing to Langfuse needs the optional extra: {probe.install_hint}")
    host, public_key = _langfuse_identity()
    secret_key = os.environ.get("LANGFUSE_SECRET_KEY")
    if not secret_key:
        raise CliError(_MISSING_KEYS)
    return host, public_key, secret_key


def _langfuse_identity() -> tuple[str, str]:
    """The host and public key that name the Langfuse project (the secret never leaves here)."""
    host = os.environ.get("LANGFUSE_HOST", "http://localhost:3000")
    public_key = os.environ.get("LANGFUSE_PUBLIC_KEY")
    if not public_key:
        raise CliError(_MISSING_KEYS)
    return host, public_key
