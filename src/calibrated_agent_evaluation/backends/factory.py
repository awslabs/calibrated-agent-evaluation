# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Backend selection.

``get_backend`` returns the configured ``TraceStoreBackend``. When nothing is configured it
returns a :class:`LocalBackend` rooted at a local path; the offline, no-service default an
Owner gets on a laptop. An explicitly-passed backend is returned as-is
and is never overridden by the local default. The Langfuse backend is
constructed only when explicitly requested, so importing the optional SDK never happens on
the default path.

``get_backend`` hands ownership of the returned backend to the caller; ``backend_session`` binds it
to a block instead, closing only a backend it created itself.
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

from calibrated_agent_evaluation.backends.base import TraceStoreBackend
from calibrated_agent_evaluation.backends.local import LocalBackend

DEFAULT_LOCAL_ROOT = ".calibrated-agent-evaluation"


def get_backend(
    configured: TraceStoreBackend | None = None,
    *,
    local_root: str | Path = DEFAULT_LOCAL_ROOT,
) -> TraceStoreBackend:
    """Return the backend to use.

    If ``configured`` is given it is returned unchanged (no silent fallback to local).
    Otherwise a ``LocalBackend`` rooted at ``local_root`` is returned. Constructing
    the ``LocalBackend`` against an unwritable root raises (the filesystem error surfaces)
    rather than silently no-op'ing.

    **The caller owns the returned backend's lifetime.** A default ``LocalBackend`` holds an open
    ``sqlite3`` connection, so a caller that neither uses it as a context manager nor calls
    ``close()`` leaks one per call. Use :func:`backend_session` where the scope is a block; this
    function stays available for the callers that hand the backend onward.

    Args:
        configured: A backend the caller already built; returned as is.
        local_root: The directory of the default local store.

    Returns:
        ``configured``, or a new ``LocalBackend`` at ``local_root``.
    """
    if configured is not None:
        return configured
    return LocalBackend(root=local_root)


@contextmanager
def backend_session(
    configured: TraceStoreBackend | None = None,
    *,
    local_root: str | Path = DEFAULT_LOCAL_ROOT,
) -> Iterator[TraceStoreBackend]:
    """:func:`get_backend` with the lifetime bound to the block.

    Closes only what it created. A ``configured`` backend was opened by the caller and is left
    open; closing someone else's handle is the mirror-image bug of leaking your own.

    Args:
        configured: A backend the caller already built; yielded and left open.
        local_root: The directory of the default local store.

    Yields:
        The backend to use inside the block.
    """
    backend = get_backend(configured, local_root=local_root)
    if configured is not None:
        yield backend
        return
    closer = getattr(backend, "close", None)
    try:
        yield backend
    finally:
        if callable(closer):
            closer()
