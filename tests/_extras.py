# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Which optional extra each ``calibrated_agent_evaluation`` module needs, for the per-extra import
checks.

A module needs the extra named by the longest entry in ``NEEDS`` that is the module itself or one
of its parent packages. A module no entry covers must import on a bare install. So a new module is
held to the bare-install bar until it is listed here, and an extra is held to importing every
bare module plus every module listed under it.
"""

from __future__ import annotations

import tomllib
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
SRC = REPO / "src" / "calibrated_agent_evaluation"

NEEDS: dict[str, str] = {
    # Runs the bias battery, which needs the stats extra; it imports no AWS SDK at load time.
    "calibrated_agent_evaluation.agentcore.bias": "stats",
    "calibrated_agent_evaluation.agentcore.code_interpreter": "agentcore",
    "calibrated_agent_evaluation.agentcore.datasets": "agentcore",
    "calibrated_agent_evaluation.agentcore.detectors": "agentcore",
    "calibrated_agent_evaluation.agentcore.doctor": "agentcore",
    "calibrated_agent_evaluation.agentcore.runtime": "agentcore",
    "calibrated_agent_evaluation.agentcore.runtime_boundary": "agentcore",
    "calibrated_agent_evaluation.agentcore.runtime_cassette": "agentcore",
    "calibrated_agent_evaluation.agentcore.runtime_executor": "agentcore",
    "calibrated_agent_evaluation.agentcore.runtime_probe": "agentcore",
    "calibrated_agent_evaluation.agentcore.session": "agentcore",
    "calibrated_agent_evaluation.attribution": "stats",
    "calibrated_agent_evaluation.bias": "stats",
    "calibrated_agent_evaluation.calibrate.engine": "stats",
    "calibrated_agent_evaluation.calibrate.plan": "stats",
    "calibrated_agent_evaluation.dumpload.results": "stats",
    "calibrated_agent_evaluation.enforce": "stats",
    "calibrated_agent_evaluation.execute.cassette_provider": "inspect",
    "calibrated_agent_evaluation.execute.probe_bridge": "inspect",
    "calibrated_agent_evaluation.execute.strands_capture": "strands",
    "calibrated_agent_evaluation.execute.strands_cassette_model": "strands",
    "calibrated_agent_evaluation.flagship": "stats",
    "calibrated_agent_evaluation.improve": "stats",
    "calibrated_agent_evaluation.judges": "stats",
    "calibrated_agent_evaluation.review": "ui",
    "calibrated_agent_evaluation.review.agreement": "stats",
    "calibrated_agent_evaluation.review.overlap": "stats",
    "calibrated_agent_evaluation.run.drift_report": "stats",
    "calibrated_agent_evaluation.run.flags": "stats",
    "calibrated_agent_evaluation.run.gate": "stats",
    "calibrated_agent_evaluation.run.multi": "stats",
    "calibrated_agent_evaluation.run.staleness": "stats",
    "calibrated_agent_evaluation.run.trend": "stats",
    "calibrated_agent_evaluation.stats": "stats",
}


def declared_extras() -> tuple[str, ...]:
    """The optional extras ``pyproject.toml`` declares."""
    project = tomllib.loads((REPO / "pyproject.toml").read_text())["project"]
    return tuple(project["optional-dependencies"])


def all_modules() -> tuple[str, ...]:
    """Every importable module in the source tree, packages included."""
    names = []
    for path in sorted(SRC.rglob("*.py")):
        parts = list(path.relative_to(SRC.parent).with_suffix("").parts)
        if parts[-1] == "__main__":
            continue
        if parts[-1] == "__init__":
            parts.pop()
        names.append(".".join(parts))
    return tuple(names)


def extra_of(module: str) -> str:
    """The extra ``module`` needs, or ``""`` when a bare install must import it."""
    parts = module.split(".")
    for end in range(len(parts), 0, -1):
        extra = NEEDS.get(".".join(parts[:end]))
        if extra is not None:
            return extra
    return ""


def modules_for(extra: str) -> tuple[str, ...]:
    """The modules an install with only ``extra`` (``""`` for none) must import."""
    return tuple(m for m in all_modules() if extra_of(m) in ("", extra))
