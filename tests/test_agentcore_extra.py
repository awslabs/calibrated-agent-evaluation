# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""The AgentCore and synthesis extras declare exactly the verified pins, each with a major cap."""

from __future__ import annotations

import tomllib
from importlib import metadata
from pathlib import Path

from packaging.requirements import Requirement
from packaging.utils import canonicalize_name

_PYPROJECT = Path(__file__).resolve().parents[1] / "pyproject.toml"

_AGENTCORE_PINS = {
    "boto3": ">=1.43.103,<2",
    "botocore": ">=1.43.103,<2",
    "bedrock-agentcore": ">=1.24.2,<2",
    "jinja2": ">=3.1.6,<4",
    "strands-agents-evals": ">=1.6.0,<2",
    # Declared because the runtime executor imports urllib3's errors. The floor, 1.26.5, imports on
    # Python 3.11 to 3.13, while 1.26.4 fails on 3.12 and 3.13; 3.14 is not checked yet.
    "urllib3": ">=1.26.5,!=2.2.0,<3",
    "strands-agents": ">=1.45,<2",
}
_STRANDS_EVALS_PIN = ">=1.6.0,<2"


def _declared_for_extra(extra: str) -> list[Requirement]:
    reqs = [Requirement(raw) for raw in metadata.requires("calibrated-agent-evaluation") or []]
    return [r for r in reqs if r.marker is not None and r.marker.evaluate({"extra": extra})]


def _pin_problems(requirements: list[Requirement], expected: dict[str, str]) -> list[str]:
    """Every way ``requirements`` departs from ``expected`` (name -> specifier), as messages."""
    declared = {canonicalize_name(r.name): r for r in requirements}
    problems = [f"missing {name}" for name in expected if name not in declared]
    for name, req in declared.items():
        if name not in expected:
            problems.append(f"unexpected requirement {req}")
            continue
        if req.specifier != Requirement(f"{name}{expected[name]}").specifier:
            problems.append(f"{name} is {req.specifier}, expected {expected[name]}")
        if not any(spec.operator == "<" for spec in req.specifier):
            problems.append(f"{name} has no major cap")
    return problems


def test_the_agentcore_extra_declares_exactly_the_verified_pins() -> None:
    assert _pin_problems(_declared_for_extra("agentcore"), _AGENTCORE_PINS) == []


def test_an_extra_requirement_or_an_uncapped_pin_is_reported() -> None:
    drifted = [
        Requirement("boto3>=1.43.103"),
        Requirement("botocore>=1.43.103,<2"),
        Requirement("bedrock-agentcore>=1.24.2,<2"),
        Requirement("strands-agents-evals>=1.6.0,<2"),
        Requirement("requests>=2"),
    ]

    problems = _pin_problems(drifted, _AGENTCORE_PINS)

    assert "boto3 has no major cap" in problems
    assert "unexpected requirement requests>=2" in problems


def _strands_evals_requirements() -> dict[str, Requirement]:
    data = tomllib.loads(_PYPROJECT.read_text())
    sources = {
        "[synth]": data["project"]["optional-dependencies"]["synth"],
        "dev group": data["dependency-groups"]["dev"],
    }
    found: dict[str, Requirement] = {}
    for where, raws in sources.items():
        for raw in raws:
            if isinstance(raw, str) and canonicalize_name(Requirement(raw).name) == (
                "strands-agents-evals"
            ):
                found[where] = Requirement(raw)
    return found


def test_synth_and_dev_require_a_floor_and_cap_not_an_exact_pin() -> None:
    found = _strands_evals_requirements()

    assert set(found) == {"[synth]", "dev group"}
    for where, req in found.items():
        assert _pin_problems([req], {"strands-agents-evals": _STRANDS_EVALS_PIN}) == [], where


def test_an_exact_pin_is_reported() -> None:
    problems = _pin_problems(
        [Requirement("strands-agents-evals==1.6.0")], {"strands-agents-evals": _STRANDS_EVALS_PIN}
    )

    assert "strands-agents-evals is ==1.6.0, expected >=1.6.0,<2" in problems
    assert "strands-agents-evals has no major cap" in problems
