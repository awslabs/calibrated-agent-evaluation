# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""``doctor``: report the install so a user knows what works and how to unlock the rest.

The library's default install has an empty runtime dependency tree; every capability beyond the
offline local backend is an optional extra. That is great for footprint but means "it didn't work"
usually means "the extra isn't installed." ``doctor`` makes that legible: it prints the Python
version, each optional extra as installed / not-installed with the exact install command to enable
it, and checks the configured store: the local root ``--root`` /
``CALIBRATED_AGENT_EVALUATION_LOCAL_ROOT`` resolves to is writable, and under ``--backend langfuse``
the keys are set and the server accepts them. Offline it imports no vendor SDK: the extras and what
each needs are read from the installed package metadata, so ``calibrated-agent-evaluation doctor``
is safe and fast offline and cannot drift from ``pyproject.toml``.
"""

from __future__ import annotations

import argparse
import os
import re
import sys
import tempfile
from importlib.metadata import PackageNotFoundError, distribution, metadata
from pathlib import Path

from calibrated_agent_evaluation._install import install_hint
from calibrated_agent_evaluation.cli.backends import local_root
from calibrated_agent_evaluation.cli.dispatch import EXIT_DOMAIN_ERROR, EXIT_OK, StageOutcome

# What each extra unlocks, so the report reads as capabilities rather than package names. The
# extras themselves, and the distributions each needs, come from the installed package metadata,
# so a new or changed extra in pyproject.toml is reported without touching this table.
_UNLOCKS: dict[str, str] = {
    "stats": (
        "the judge-calibration engine (lower-bound gates, prevalence correction) and "
        "failure-mode clustering (calibrated-agent-evaluation cluster)"
    ),
    "ui": "the trace-review web UI (calibrated-agent-evaluation label --serve)",
    "langfuse": "the Langfuse trace-store backend (--backend langfuse)",
    "sampling": "covering-array discovery frames (the define and generate stages)",
    "scorers": "structured-target scorers (exact-match, JSON, numeric)",
    "inspect": "read Inspect AI .eval logs; run Inspect agents and judge models",
    "strands": "run Strands agents programmatically",
    "otel": "read OTLP protobuf trace exports",
    "datasets": "ingest local parquet datasets",
    "hf": "resolve a HuggingFace-hub dataset by id",
    "synth": "multi-turn input synthesis via strands-agents-evals",
    "probe": "the agent-security benchmark battery",
    "probe-ws": "probe WebSocket (ws://, wss://) endpoints",
    "bedrock": "the Bedrock LLM renderer + Titan embedder (the generate and cluster stages)",
    "agentcore": "the Amazon Bedrock AgentCore plug-in (calibrated_agent_evaluation.agentcore)",
}

_DIST = re.compile(r"^\s*([A-Za-z0-9][A-Za-z0-9._-]*)")
_EXTRA = re.compile(r"""extra\s*==\s*['"]([^'"]+)['"]""")


def extras_and_requirements() -> dict[str, list[str]]:
    """Each extra the installed distribution declares, mapped to the distributions it needs.

    Empty when the package metadata is unavailable (a source tree that was never installed).
    """
    try:
        md = metadata("calibrated-agent-evaluation")
    except PackageNotFoundError:
        return {}
    extras: dict[str, list[str]] = {e: [] for e in md.get_all("Provides-Extra") or []}
    for req in md.get_all("Requires-Dist") or []:
        marker = _EXTRA.search(req)
        name = _DIST.match(req)
        if marker and name and marker.group(1) in extras:
            extras[marker.group(1)].append(name.group(1))
    return extras


def _installed(dist: str) -> bool:
    try:
        distribution(dist)
    except PackageNotFoundError:
        return False
    return True


def _check_line(installed: bool, extra: str, unlocks: str, missing: list[str]) -> str:
    """One report row: a check/cross, the extra, what it unlocks, and the fix if absent."""
    if installed:
        return f"  [x] [{extra}]: {unlocks}"
    return (
        f"  [ ] [{extra}]: {unlocks}\n"
        f"        missing: {', '.join(missing)}; enable: {install_hint(extra)}"
    )


_AGENTCORE_ONLY = ("region", "profile", "runtime", "omni_domain")


def doctor_handler(args: argparse.Namespace) -> StageOutcome:
    """Report Python, per-extra availability, and the configured store's health.

    A missing extra is information, not an error, so it never fails the command. An extra counts
    as installed when every distribution it requires is installed, which is read from package
    metadata without importing anything. The store is different: a local root that cannot be
    written, or a Langfuse backend with missing or rejected keys, exits 1, because every stage
    after this one would fail on it. ``--agentcore`` instead checks an AWS account with read-only
    calls and exits 1 unless every check passes.
    """
    from calibrated_agent_evaluation.cli.errors import UsageError

    if getattr(args, "agentcore", False):
        return agentcore_doctor(args)
    stray = [name for name in _AGENTCORE_ONLY if getattr(args, name, None) is not None]
    if stray:
        flags = ", ".join("--" + name.replace("_", "-") for name in stray)
        raise UsageError(
            f"{flags} only apply with --agentcore (they configure the AWS account check); add "
            "--agentcore, or drop them"
        )
    py = sys.version_info
    lines: list[str] = [
        "calibrated-agent-evaluation doctor: Calibrated Agent Evaluation install report",
        "",
        f"Python: {py.major}.{py.minor}.{py.micro}  (Calibrated Agent Evaluation needs >= 3.11)",
        "Default runtime: offline local backend (no extras needed, always available)",
        "",
        "Optional extras:",
    ]
    extras = extras_and_requirements()
    installed_count = 0
    for extra in sorted(extras):
        missing = [d for d in extras[extra] if not _installed(d)]
        if not missing:
            installed_count += 1
        lines.append(_check_line(not missing, extra, _UNLOCKS.get(extra, ""), missing))

    root_ok, root_line = _local_root_status(local_root(args))
    lines.append("")
    lines.append(root_line)
    store_ok = root_ok
    if getattr(args, "backend", "local") == "langfuse":
        langfuse_ok, langfuse_line = _langfuse_status()
        lines.append(langfuse_line)
        store_ok = store_ok and langfuse_ok
    lines.append("")
    if extras:
        lines.append(
            f"{installed_count}/{len(extras)} optional extras installed. "
            "Install only what your workflow needs; the offline core needs none."
        )
    else:
        lines.append(
            "Optional extras: unknown (no installed calibrated-agent-evaluation metadata to read)"
        )
    return StageOutcome(
        stage="doctor",
        exit_code=EXIT_OK if store_ok else EXIT_DOMAIN_ERROR,
        summary="\n".join(lines),
    )


def _local_root_status(root: Path) -> tuple[bool, str]:
    """Whether the local root can be written, and the report line saying so.

    It used to open an in-memory store, which proved nothing about ``--root``: a read-only or
    impossible root reported "writable". A root that exists gets a temporary file written and
    removed in it; one that does not exist yet is checked through its nearest existing parent,
    without creating anything, since the first stage that writes creates it.
    """
    where = str(root)
    if root.exists() and not root.is_dir():
        return False, f"Local backend: NOT usable; {where} exists and is not a directory"
    if root.is_dir():
        try:
            with tempfile.NamedTemporaryFile(
                dir=root, prefix=".calibrated-agent-evaluation-doctor-"
            ):
                pass
        except OSError as exc:
            return False, f"Local backend: NOT writable at {where} ({exc.strerror or exc})"
        return True, f"Local backend: writable at {where} (offline path ready)"
    parent = next((p for p in root.absolute().parents if p.exists()), None)
    if parent is None or not parent.is_dir() or not os.access(parent, os.W_OK | os.X_OK):
        return False, (
            f"Local backend: NOT writable; {where} does not exist and cannot be created under "
            f"{parent or 'any existing directory'}"
        )
    return True, f"Local backend: {where} will be created on first write (offline path ready)"


def _langfuse_status() -> tuple[bool, str]:
    """Whether the Langfuse backend's keys are set and accepted, and the report line saying so."""
    from calibrated_agent_evaluation.cli.backends import backend_from_flag
    from calibrated_agent_evaluation.cli.errors import CliError
    from calibrated_agent_evaluation.schema.errors import BackendError

    try:
        backend = backend_from_flag("langfuse")
    except CliError as exc:
        return False, f"Langfuse backend: NOT ready; {exc}"
    try:
        _project_id, name = backend.project()  # type: ignore[attr-defined]
    except BackendError as exc:
        return False, f"Langfuse backend: NOT reachable; {exc}"
    finally:
        close = getattr(backend, "close", None)
        if callable(close):
            close()
    return True, f"Langfuse backend: reachable, keys accepted (project {name!r})"


def agentcore_doctor(args: argparse.Namespace) -> StageOutcome:
    """Run the AgentCore readiness checks; exit 0 only when every one passes."""
    # Each access below raises AgentCoreExtraMissing without the [agentcore] extra.
    import calibrated_agent_evaluation.agentcore as agentcore
    from calibrated_agent_evaluation.cli.errors import CliError

    checks = agentcore.DEFAULT_CHECKS
    clients = agentcore.AwsClients.from_profile(
        region=args.region, profile=args.profile, handlers=[agentcore.read_only_guard(checks)]
    )
    with clients:
        if clients.region is None:
            raise CliError("no AWS region configured; pass --region or set AWS_REGION")
        target = agentcore.DoctorTarget(
            region=clients.region,
            profile=args.profile,
            runtime_name=args.runtime,
            omni_domain=args.omni_domain,
        )
        report = agentcore.run_checks(checks, clients, target)

    passed = sum(r.status == "pass" for r in report.results)
    lines = [
        f"AgentCore readiness for {target.region} as {report.identity}",
        "",
        *(
            f"[{r.status}] {r.name}" + ("" if r.status == "pass" else f": {r.next_step}")
            for r in report.results
        ),
        "",
        f"{passed}/{len(report.results)} checks passed. Every call was read-only.",
    ]
    detail = "\n".join(f"{r.name}: {r.observed}" for r in report.results)
    return StageOutcome(
        stage="doctor",
        exit_code=EXIT_OK if report.ok else EXIT_DOMAIN_ERROR,
        summary="\n".join(lines),
        detail=detail,
    )
