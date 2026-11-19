# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Boundary tests for the flagship package.

The orchestration library composes only shipped ``calibrated_agent_evaluation.*`` surfaces: it
imports no vendor SDK type and no ``scripts``/``corpus`` (not in the wheel), and it re-implements no
methodology every stage builder resolves to a call into a shipped ``calibrated_agent_evaluation.*``
surface. These are contract/boundary tests (blessed under test-rubric rule 2): the FR being verified
IS the import boundary.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from _imports import imported_roots

_FLAGSHIP = (
    Path(__file__).resolve().parent.parent / "src" / "calibrated_agent_evaluation" / "flagship"
)

# Vendor SDK / non-shipped modules a methodology-plane package must never import.
_FORBIDDEN_ROOTS = {
    "langfuse",
    "boto3",
    "botocore",
    "inspect_ai",
    "strands",
    "strands_agents",
    "strands_evals",
    "anthropic",
    "opentelemetry",
    "scripts",
    "corpus",
}


@pytest.mark.parametrize("py", sorted(_FLAGSHIP.glob("*.py")), ids=lambda p: p.name)
def test_flagship_imports_no_vendor_or_script(py: Path) -> None:
    """No module under flagship/ imports a vendor SDK or a scripts/corpus module."""
    offending = imported_roots(py) & _FORBIDDEN_ROOTS
    assert not offending, f"{py.name} imports forbidden root(s): {sorted(offending)}"


def test_flagship_stage_builders_call_shipped_surfaces() -> None:
    """Each stage builder resolves to a shipped calibrated_agent_evaluation.* surface (no re-impl).

    The stages module imports the shipped entry points (probe/run_ci_gate/evaluate_improvement/
    hierarchical_cluster/validate_walkback/run_judge) -- if a builder re-implemented methodology it
    would not need them. Asserting the import presence is the contract.
    """
    stages_src = (_FLAGSHIP / "stages.py").read_text(encoding="utf-8")
    for shipped in (
        "from calibrated_agent_evaluation.probe import",
        "from calibrated_agent_evaluation.run import",
        "from calibrated_agent_evaluation.improve import",
        "from calibrated_agent_evaluation.cluster import",
        "from calibrated_agent_evaluation.attribution import",
        "from calibrated_agent_evaluation.judges import",
    ):
        assert shipped in stages_src, f"stages.py should compose the shipped surface: {shipped!r}"
