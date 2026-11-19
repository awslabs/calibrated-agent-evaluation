# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Data-plane backends for Calibrated Agent Evaluation.

The ``TraceStoreBackend`` / ``LabelBackend`` protocols are defined against the
normalized schema only; no vendor type appears in any method signature.
``LocalBackend`` (stdlib sqlite3 + CSV/JSON) is the no-config offline default;
``LangfuseBackend`` is the opt-in both-worlds adapter (the ``langfuse`` extra).

``LangfuseBackend`` and the dashboard sinks are served **lazily** via ``__getattr__`` (mirroring
``execute/__init__``) so ``import calibrated_agent_evaluation.backends`` on a bare offline install
pulls neither the Langfuse vendor SDK nor the ``run``/``calibrate`` stats chain the dashboard sinks
import (``scipy``/``pyarrow``). This keeps the offline CLI path zero-third-party at import time
the light stdlib-only ``LocalBackend`` + ``get_backend`` are still eager.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from calibrated_agent_evaluation.backends.base import (
    LabelBackend,
    ScanningBackend,
    SkippedTrace,
    SourceTrace,
    TraceScan,
    TraceSource,
    TraceStoreBackend,
    scan_traces,
)
from calibrated_agent_evaluation.backends.factory import (
    DEFAULT_LOCAL_ROOT,
    backend_session,
    get_backend,
)
from calibrated_agent_evaluation.backends.local import LocalBackend

if TYPE_CHECKING:
    # Statically visible; served lazily at runtime by __getattr__ so the offline import path stays
    # free of the vendor SDK (langfuse) and the dashboard sinks' run/calibrate → scipy/pyarrow
    # chain.
    from calibrated_agent_evaluation.backends.dashboard import (
        DashboardSink,
        LangfuseDashboardSink,
        LocalDashboardSink,
    )
    from calibrated_agent_evaluation.backends.langfuse import LangfuseBackend

__all__ = [
    "DEFAULT_LOCAL_ROOT",
    "DashboardSink",
    "LabelBackend",
    "LangfuseBackend",
    "LangfuseDashboardSink",
    "LocalBackend",
    "LocalDashboardSink",
    "ScanningBackend",
    "SkippedTrace",
    "SourceTrace",
    "TraceScan",
    "TraceSource",
    "TraceStoreBackend",
    "backend_session",
    "get_backend",
    "scan_traces",
]


def __getattr__(name: str) -> object:
    if name == "LangfuseBackend":
        from calibrated_agent_evaluation.backends.langfuse import LangfuseBackend

        return LangfuseBackend
    if name in ("DashboardSink", "LangfuseDashboardSink", "LocalDashboardSink"):
        from calibrated_agent_evaluation.backends import dashboard

        return getattr(dashboard, name)
    raise AttributeError(f"module 'calibrated_agent_evaluation.backends' has no attribute {name!r}")
