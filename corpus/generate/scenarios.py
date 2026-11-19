# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Worked-example scenarios — the single source of truth for the corpus, derived from the oracle.

The frozen `fixtures/raw/`
corpus must normalize **field-for-field** to the hand-authored `fixtures/normalized/*.json`
(the golden-file contract test). To guarantee that, and to avoid a second hand-authored
copy of the same data drifting from the oracle, a `Scenario` reads its deterministic leaf data
(tool args+results, the pinned final decision, the taxonomy cell) **directly out of the oracle**.

This is the data side of the requester's locked decision (corpus-companion §"Capture determinism"):
the *agents and their emitted wire shapes are genuine framework output* (real Strands ``Swarm`` +
tracer, real OTel-SDK spans, a real Inspect ``.eval``); only *leaf data values* are pinned so the
frozen trace reproduces the oracle deterministically. Pinning data values is orthogonal to schema
conformance — no agent imports a ``calibrated_agent_evaluation`` type; they conform to their
frameworks, not to us.

The independent-gold property the contract test relies on is preserved: the **normalizer** (built
later, in a separate session) is the code under test, and it must recover these values from genuine,
messy wire shapes (OTLP events-vs-attributes, the ``.eval`` event tree, Strands span events) — not
from this module. This module only ensures the *raw* corpus carries the oracle's data.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from functools import cache
from pathlib import Path
from typing import Any

# Repo-root-relative; this file is corpus/generate/scenarios.py -> parents[2] is the repo root.
_REPO_ROOT = Path(__file__).resolve().parents[2]
_NORMALIZED = _REPO_ROOT / "fixtures" / "normalized"


@dataclass(frozen=True)
class ToolStep:
    """One deterministic tool call: the agent invokes ``name(args)`` and gets exactly ``result``.

    Lifted verbatim from an oracle span's ``tool_calls`` entry, so a deterministic stub that returns
    ``result`` for ``(name, args)`` makes the captured tool I/O match the oracle by construction.
    """

    name: str
    args: dict[str, Any]
    result: dict[str, Any]


@dataclass(frozen=True)
class AgentStep:
    """One agent in a run: its received input, ordered tool steps, and emitted output."""

    agent: str
    role: str
    parent: str | None
    seq: int
    input: dict[str, Any]
    tool_steps: tuple[ToolStep, ...]
    output: dict[str, Any]


@dataclass(frozen=True)
class Scenario:
    """A worked example, projected from its oracle ``Trace`` into a framework-agnostic plan.

    The agents (`corpus/agents/*.py`) read this plan to drive deterministic tool stubs and to pin
    the final decision (incl. the canonical *defect*), while the framework itself produces the real
    span tree / event log. ``trace_input``/``trace_output`` are the terminal request/decision; the
    per-agent ``steps`` carry the handoff + tool detail a crew needs.
    """

    trace_id: str
    frame: str
    taxonomy_cell: dict[str, str]
    trace_input: dict[str, Any]
    trace_output: dict[str, Any]
    steps: tuple[AgentStep, ...]

    @property
    def is_single_agent(self) -> bool:
        return len(self.steps) == 1

    @property
    def root(self) -> AgentStep:
        # The root step (parent is None) — the single agent for OrderBot, the supervisor for a crew.
        return next(s for s in self.steps if s.parent is None)


@cache
def _load_oracle(trace_id: str) -> dict[str, Any]:
    path = _NORMALIZED / f"{trace_id}.json"
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        # No silent failure (conventions): name the missing oracle and where it should live.
        raise FileNotFoundError(
            f"oracle fixture {path} not found; the corpus scenario is derived from the "
            f"hand-authored fixtures/normalized/{trace_id}.json; it must exist first"
        ) from exc


def load_scenario(trace_id: str) -> Scenario:
    """Project the committed oracle ``Trace`` for ``trace_id`` into a ``Scenario`` plan.

    Reads ``fixtures/normalized/{trace_id}.json`` (stdlib ``json``, zero extra deps) and lifts the
    span tree into ordered :class:`AgentStep` / :class:`ToolStep` rows. Because every value comes
    from the oracle, an agent that faithfully replays this plan emits a trace that normalizes back
    to the oracle field-for-field (modulo capture timestamps).
    """
    oracle = _load_oracle(trace_id)
    steps: list[AgentStep] = []
    for span in sorted(oracle["spans"], key=lambda s: s["seq"]):
        tool_steps = tuple(
            ToolStep(name=tc["name"], args=tc["args"], result=tc["result"])
            for tc in span.get("tool_calls", [])
        )
        steps.append(
            AgentStep(
                agent=span["agent"],
                role=span["role"],
                parent=_strip_trace_prefix(span.get("parent"), trace_id),
                seq=span["seq"],
                input=span["input"],
                tool_steps=tool_steps,
                output=span["output"],
            )
        )
    return Scenario(
        trace_id=trace_id,
        frame=oracle["frame"],
        taxonomy_cell=oracle["taxonomy_cell"],
        trace_input=oracle["input"],
        trace_output=oracle["output"],
        steps=tuple(steps),
    )


def _strip_trace_prefix(parent_span_id: str | None, trace_id: str) -> str | None:
    """Oracle parents are span ids like ``CLM-4471/supervisor`` — return the bare agent name.

    The agents key handoffs on agent *names* (``supervisor``); the normalizer rebuilds the
    ``trace_id/agent`` span-id form. Keeping agents free of the library's span-id convention is the
    point — they speak their framework's vocabulary, not ours.
    """
    if parent_span_id is None:
        return None
    prefix = f"{trace_id}/"
    return parent_span_id[len(prefix) :] if parent_span_id.startswith(prefix) else parent_span_id


# The worked examples the corpus freezes (corpus-companion §3 "Frozen corpus"); [attr] marks the
# crew-attribution contrast traces:
#   ORD-1001 — OrderBot out-of-window over-refund (the headline single-agent defect)
#   ORD-1002 — OrderBot in-window refund, clean PASS (the control, proves a non-failing trace too)
#   CLM-4471 — ClaimsCrew manifestation != origination (intake drops the vacancy signal)
#   CLM-4473: ClaimsCrew same symptom, DIFFERENT culprit (policy-checker ignores W-7) [attr]
#   CLM-4475: ClaimsCrew outcome-pass / span-fail (intake misreads amount, still denied) [attr]
#   CLM-4477: ClaimsCrew DISPUTED culprit (intake emits occupancy un-normalized) [attr]
ORDERBOT_FAIL = "ORD-1001"
ORDERBOT_CLEAN = "ORD-1002"  # the in-window PASS control
CLAIMSCREW_FAIL = "CLM-4471"
# The multi-agent attribution contrast/trap crew traces (real-emitted via the Strands harness):
CLAIMSCREW_DIFFERENT_CULPRIT = "CLM-4473"
CLAIMSCREW_OUTCOME_PASS_SPAN_FAIL = "CLM-4475"  # noqa: S105 - a trace id, not a secret
CLAIMSCREW_DISPUTED = "CLM-4477"
CLAIMSCREW_ATTRIBUTION_TRACES = (
    CLAIMSCREW_DIFFERENT_CULPRIT,
    CLAIMSCREW_OUTCOME_PASS_SPAN_FAIL,
    CLAIMSCREW_DISPUTED,
)
ALL_TRACE_IDS = (
    ORDERBOT_FAIL,
    ORDERBOT_CLEAN,
    CLAIMSCREW_FAIL,
    *CLAIMSCREW_ATTRIBUTION_TRACES,
)
