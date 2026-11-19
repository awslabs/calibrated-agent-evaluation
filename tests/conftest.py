# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shared pytest fixtures for the data-plane suite.

Builders and corpus loaders live in the importable ``_helpers`` module (test files import
them directly); this file wraps the corpus loaders as fixtures. The loaders read the
committed shared corpus (``fixtures/normalized`` + ``fixtures/labels/gold.json``) — the same
files the manual acceptance round-trips — so the automated tests exercise the real
OrderBot/ClaimsCrew traces. Reading them is filesystem-only (no network).
"""

from __future__ import annotations

import sys

import pytest

from _helpers import load_corpus_labels, load_corpus_trace
from calibrated_agent_evaluation.attribution import (
    CrewDefinition,
    ExpectedProcess,
    MandatedRule,
    build_crew_definition,
)
from calibrated_agent_evaluation.dumpload.registry import _reset_registry
from calibrated_agent_evaluation.schema.types import Label, Trace


@pytest.fixture(autouse=True)
def _clean_registry() -> None:
    """Reset the module-global codec registry before each test.

    The dump/load registry is process-global; tests that register a toy artifact type
    would otherwise leak into later tests. Resetting per test keeps each one isolated; the
    core Trace/Span/Label codecs are re-registered lazily by dump/load (register_core_codecs).
    """
    _reset_registry()


@pytest.fixture(autouse=True)
def _cli_explains(monkeypatch: pytest.MonkeyPatch) -> None:
    """Run the CLI with ``--explain`` on, so tests read the statistical detail they assert on.

    The plain default output has its own tests (``test_cli_plain_output.py``), which clear this.
    """
    monkeypatch.setenv("CALIBRATED_AGENT_EVALUATION_EXPLAIN", "1")


@pytest.fixture
def corpus_clm() -> Trace:
    """The ClaimsCrew multi-agent (5-span) trace — the both-worlds, attribution oracle."""
    return load_corpus_trace("CLM-4471")


@pytest.fixture
def corpus_ord() -> Trace:
    """The OrderBot single-agent (one-span tree) trace — the oracle."""
    return load_corpus_trace("ORD-1001")


@pytest.fixture
def corpus_labels() -> list[Label]:
    """The human gold labels, incl. crew-attribution fields."""
    return load_corpus_labels()


@pytest.fixture
def seeded_local_root(tmp_path, monkeypatch):
    """A LocalBackend root seeded with the full corpus, wired as the CLI's default backend.

    Seeds every normalized trace + gold label into a temp ``LocalBackend`` and points the CLI
    config resolver at that root, so ``calibrated-agent-evaluation <stage>`` (default
    ``--backend local``) runs against the
    seeded store. Returns the root path. Used by the CLI integration tests.
    """
    from pathlib import Path

    from calibrated_agent_evaluation.backends.local import LocalBackend

    root = tmp_path / "calibrated-agent-evaluation-store"
    backend = LocalBackend(root=root)
    for trace_id in ("ORD-1001", "ORD-1002", "CLM-4471"):
        backend.put_trace(load_corpus_trace(trace_id))
    for label in load_corpus_labels():
        backend.put_score(label)

    # Point the CLI's local-backend resolution at this seeded root. Import cli.backends first: if
    # its `from cli.config import resolve_backend` ran after the patch below, the patched function
    # would be what monkeypatch records as the original, and it would outlive this test.
    import calibrated_agent_evaluation.cli.backends  # noqa: F401  (see above)
    import calibrated_agent_evaluation.cli.config as cfg

    real = cfg.resolve_backend

    def _resolve(configured=None, *, local_root: str | Path | None = None):
        return real(configured, local_root=str(root))

    monkeypatch.setattr(cfg, "resolve_backend", _resolve)
    monkeypatch.setattr("calibrated_agent_evaluation.cli.backends.resolve_backend", _resolve)
    # And the local root itself, so the records the stages write land beside the seeded store.
    monkeypatch.setenv("CALIBRATED_AGENT_EVALUATION_LOCAL_ROOT", str(root))
    return root


@pytest.fixture
def claimscrew() -> CrewDefinition:
    """The ClaimsCrew Define artifact — the worked example for attribution + judge tests.

    S2 (vacancy exclusion): the pipeline agents run in order and correctly deny; escalation is
    conditional on the R2 fraud-flag antecedent (non-deterministic), which does not fire on the
    low-fraud CLM-4471 — so the process judge passes CLM-4471 structurally (the intake failure is a
    span-level, not a structural, defect).
    """
    s2 = ExpectedProcess(
        scenario="S2_vacancy_exclusion",
        expected_agents=("supervisor", "intake", "policy-checker", "fraud-scorer", "payout"),
        terminal_decision="deny",
        payout_allowed=False,
        escalation_required=False,
    )
    rules = (
        MandatedRule(rule_id="R1", antecedent="deny", consequent_agent="supervisor"),
        MandatedRule(
            rule_id="R2",
            antecedent="fraud flag raised",
            consequent_agent="escalation",
            deterministic=False,
        ),
        MandatedRule(
            rule_id="R3", antecedent="occupancy narrative present", consequent_agent="intake"
        ),
    )
    return build_crew_definition(
        agents=("supervisor", "intake", "policy-checker", "fraud-scorer", "payout", "escalation"),
        handoffs=(
            "intake->policy-checker",
            "policy-checker->fraud-scorer",
            "fraud-scorer->payout",
            "fraud-scorer->escalation",
        ),
        scenarios={"S2_vacancy_exclusion": s2},
        rules=rules,
    )


@pytest.fixture(autouse=True)
def _close_stubbed_aws_clients():
    """Close the AWS clients the dataset tests' stubbed account built, after each test."""
    yield
    stubs = sys.modules.get("agentcore_dataset_stubs")
    if stubs is not None:
        stubs.close_all()
