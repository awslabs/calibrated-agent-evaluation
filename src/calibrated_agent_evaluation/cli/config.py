# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""CLI config resolution + graceful degradation.

Resolves the three things a stage needs before it runs; the trace-store backend, the model
provider, and whether an optional extra is installed; and, when a capability is absent, produces
a typed :class:`Degradation` carrying the exact install command for the extra rather
than letting a raw ``ImportError`` escape. The offline defaults (local backend, template
renderer) are the laptop path: no network, no service, zero third-party deps.

This module imports **no** vendor SDK at load time; extra availability is probed with
``importlib.util.find_spec`` (no import side-effect), so the offline dispatch path stays clean.
"""

from __future__ import annotations

import importlib.util
from dataclasses import dataclass
from pathlib import Path

from calibrated_agent_evaluation._install import install_hint
from calibrated_agent_evaluation.backends import TraceStoreBackend, get_backend

__all__ = [
    "Degradation",
    "extra_for_import",
    "extra_for_missing_module",
    "probe_extra",
    "require_extras",
    "resolve_backend",
]

# Maps a top-level importable module to the calibrated-agent-evaluation extra that provides it. Used
# to turn a would-be ImportError into an actionable install-hint. Only the modules the CLI stages
# reach for live here; unknown modules fall back to a generic hint.
_MODULE_TO_EXTRA: dict[str, str] = {
    "boto3": "bedrock",
    "bedrock_agentcore": "agentcore",
    "inspect_ai": "inspect",
    "strands": "strands",
    "fasthtml": "ui",
    "uvicorn": "ui",
    "langfuse": "langfuse",
    "autoevals": "scorers",
    "scipy": "stats",
    "numpy": "stats",
    "statsmodels": "stats",
    "allpairspy": "sampling",
    "pyarrow": "datasets",
    "huggingface_hub": "hf",
    "websockets": "probe-ws",
    "opentelemetry": "otel",
    "inspect_evals": "probe",
    "strands_evals": "synth",
}


def require_extras(*modules: str) -> None:
    """Raise a typed CliError if any of ``modules`` is not importable.

    The methodology-plane engines a stage reaches (``calibrated_agent_evaluation.sampling`` needs
    allpairspy/scipy, ``calibrated_agent_evaluation.run`` needs scipy/statsmodels) are optional
    extras, NOT in the default install; so a bare install must get a typed install-hint, never a raw
    ``ModuleNotFoundError`` traceback. Call this in a handler before importing the engine.
    """
    from calibrated_agent_evaluation.cli.errors import CliError

    # Every missing extra in one message, so a bare install is not fixed one re-run at a time.
    extras = list(
        dict.fromkeys(extra_for_import(m) for m in modules if not probe_extra(m).available)
    )
    if len(extras) == 1:
        raise CliError(
            f"this stage needs an engine that isn't installed: {install_hint(extras[0])}"
        )
    if extras:
        raise CliError(
            "this stage needs engines that aren't installed: "
            f"{'; '.join(install_hint(e) for e in extras)} "
            f"(or all at once: {install_hint(*extras)})"
        )


@dataclass(frozen=True)
class Degradation:
    """Whether an optional capability is available, and how to enable it if not.

    ``capability`` is the importable module the stage needs (e.g. ``"boto3"``). ``available`` is
    whether it resolves. ``install_hint`` is the install command for the extra when it is absent
    (see :mod:`calibrated_agent_evaluation._install`), else ``None``. The CLI turns an unavailable
    capability into a typed ``CliError(install_hint)``; never a raw ``ImportError``.
    """

    capability: str
    available: bool
    install_hint: str | None


def extra_for_import(module: str) -> str:
    """Return the optional extra that provides ``module`` (a best-effort mapping)."""
    return _MODULE_TO_EXTRA.get(module, module)


def extra_for_missing_module(name: str | None) -> str | None:
    """The extra whose absence a ``ModuleNotFoundError`` for ``name`` means, or ``None``.

    Only a module this map knows counts: any other missing module is a bug, not a missing extra.
    """
    return _MODULE_TO_EXTRA.get((name or "").partition(".")[0])


def probe_extra(module: str) -> Degradation:
    """Probe whether an optional ``module`` is importable, without importing it.

    Uses ``importlib.util.find_spec`` so a mere availability check carries no import side-effect;
    the offline path never loads a vendor SDK just to discover it is present.
    """
    try:
        spec = importlib.util.find_spec(module)
    except (ImportError, ValueError):
        spec = None
    if spec is not None:
        return Degradation(capability=module, available=True, install_hint=None)
    extra = extra_for_import(module)
    return Degradation(
        capability=module,
        available=False,
        install_hint=install_hint(extra),
    )


def resolve_backend(
    configured: TraceStoreBackend | None = None,
    *,
    local_root: str | Path | None = None,
) -> TraceStoreBackend:
    """Resolve the trace-store backend, defaulting to the offline local backend.

    Delegates to ``get_backend`` (reuse, not re-implement): an explicitly-configured
    backend is returned unchanged (never silently overridden), otherwise a ``LocalBackend`` on the
    local root. Passing ``local_root=None`` uses the ``DEFAULT_LOCAL_ROOT``.
    """
    if local_root is None:
        return get_backend(configured)
    return get_backend(configured, local_root=local_root)
