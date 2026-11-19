# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Run the reference flagship proof, write its record and render it for a terminal.

:func:`run_reference_proof` is the producer of the ``flagship`` record kind that
``calibrated-agent-evaluation dump`` and ``calibrated-agent-evaluation load`` carry.
``scripts/flagship/run.py`` is a thin command line over it.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from pathlib import Path

from calibrated_agent_evaluation.backends import TraceStoreBackend
from calibrated_agent_evaluation.bias import BiasBatteryRecord
from calibrated_agent_evaluation.calibrate.record import CalibrationRecord
from calibrated_agent_evaluation.flagship.errors import FlagshipError
from calibrated_agent_evaluation.flagship.orchestrator import AgentPlan, prove
from calibrated_agent_evaluation.flagship.reference import ReferenceCorpus
from calibrated_agent_evaluation.flagship.report import (
    Agent,
    FlagshipConfig,
    FlagshipReport,
    ModelPlane,
    World,
)
from calibrated_agent_evaluation.flagship.store import dump_flagship_reports

__all__ = [
    "REFERENCE_AGENTS",
    "format_report",
    "live_bedrock_model_id",
    "require_live_credentials",
    "run_reference_proof",
]

REFERENCE_AGENTS: tuple[Agent, ...] = ("orderbot", "claimscrew")
"""The reference agents the committed corpus carries plans for."""


def require_live_credentials(environ: Mapping[str, str]) -> None:
    """Refuse a live model-plane run when no AWS credentials are configured.

    A live run with no credentials would fall back to nothing but the cassettes, and a cassette
    replay reported as a live pass is exactly the evidence the live leg exists to rule out.

    Args:
        environ: The environment to read, ``os.environ`` in a real run.

    Raises:
        FlagshipError: Neither ``AWS_PROFILE`` nor ``AWS_ACCESS_KEY_ID`` is set.
    """
    if not (environ.get("AWS_PROFILE") or environ.get("AWS_ACCESS_KEY_ID")):
        raise FlagshipError(
            "a live run needs AWS credentials (set AWS_PROFILE); refusing to report a live pass "
            "from a cassette replay"
        )


DEFAULT_LIVE_MODEL_ID = "global.anthropic.claude-sonnet-4-6"


def live_bedrock_model_id(environ: Mapping[str, str]) -> str:
    """The bare Bedrock model id a live run targets, read from the environment.

    The variable is ``CALIBRATED_AGENT_EVALUATION_BEDROCK_MODEL_ID``.

    Inspect callers prefix it with ``bedrock/``; Strands and the Bedrock runtime take it as is.

    Args:
        environ: The environment to read, ``os.environ`` in a real run.

    Raises:
        FlagshipError: The retired ``CALIBRATED_AGENT_EVALUATION_BEDROCK_MODEL`` is set (it would
            otherwise be silently ignored), or the id is empty or already carries the ``bedrock/``
            prefix.
    """
    if "CALIBRATED_AGENT_EVALUATION_BEDROCK_MODEL" in environ:
        raise FlagshipError(
            "CALIBRATED_AGENT_EVALUATION_BEDROCK_MODEL is no longer read; set "
            "CALIBRATED_AGENT_EVALUATION_BEDROCK_MODEL_ID to a bare Bedrock "
            "model id (e.g. global.anthropic.claude-sonnet-4-6) and unset the old name"
        )
    model_id = environ.get(
        "CALIBRATED_AGENT_EVALUATION_BEDROCK_MODEL_ID", DEFAULT_LIVE_MODEL_ID
    ).strip()
    if not model_id or model_id.startswith("bedrock/"):
        raise FlagshipError(
            f"CALIBRATED_AGENT_EVALUATION_BEDROCK_MODEL_ID={model_id!r} must be a bare Bedrock "
            "model id, without the "
            "bedrock/ prefix (e.g. global.anthropic.claude-sonnet-4-6)"
        )
    return model_id


def run_reference_proof(
    corpus: ReferenceCorpus,
    *,
    agents: Sequence[Agent],
    worlds: Sequence[World],
    model_plane: ModelPlane,
    backend_factory: Callable[[World], TraceStoreBackend],
    calibration: CalibrationRecord,
    battery: BiasBatteryRecord,
    out_dir: str | Path | None = None,
    flagship_id: str = "flagship-m1",
) -> FlagshipReport:
    """Prove the reference agents and, when ``out_dir`` is given, write the flagship record.

    Args:
        corpus: The reference corpus to prove against.
        agents: Which of :data:`REFERENCE_AGENTS` to prove.
        worlds: The worlds to run each agent in.
        model_plane: ``"cassette"`` for the hermetic replay, ``"live"`` for a live model plane.
        backend_factory: The trace-store backend for each world.
        calibration: The record the OrderBot judge is admitted on.
        battery: The bias battery the OrderBot judge is admitted on.
        out_dir: Where to write ``flagship_report.{csv,json}``; ``None`` writes nothing.
        flagship_id: The id stamped on the report.

    Returns:
        The report, whatever its verdict.

    Raises:
        FlagshipError: An unknown agent, or a proof-integrity failure ``prove`` refuses.
        FlagshipStoreError: The record could not be written.
    """
    unknown = [a for a in agents if a not in REFERENCE_AGENTS]
    if unknown:
        raise FlagshipError(
            f"no reference plan for agent(s) {unknown}; the corpus carries {list(REFERENCE_AGENTS)}"
        )
    plans: list[AgentPlan] = []
    for agent in agents:
        if agent == "orderbot":
            plans.append(corpus.orderbot_plan(calibration=calibration, battery=battery))
        else:
            plans.append(corpus.claimscrew_plan())
    config = FlagshipConfig(
        agents=tuple(agents),
        worlds=tuple(worlds),
        model_plane=model_plane,
        fixtures_root=str(corpus.root),
    )
    report = prove(config, plans, backend_factory=backend_factory, flagship_id=flagship_id)
    if out_dir is not None:
        dump_flagship_reports([report], out_dir, fmt="json")
    return report


def format_report(report: FlagshipReport) -> str:
    """The per-agent, per-world, per-stage table with parity, gates and the verdict."""
    lines = [
        f"Calibrated Agent Evaluation flagship proof: agents={list(report.agents)} "
        f"worlds={list(report.worlds)} "
        f"model_plane={report.model_plane} fixture_v={report.fixture_version}",
        "",
    ]
    for outcome in report.stages:
        mark = ("✓" if outcome.oracle.pinned else "?") if outcome.ok else "✗"
        cell = f"{outcome.agent}/{outcome.world}/{outcome.stage}"
        lines.append(f"  {mark}  {cell:34} {outcome.note}")
    lines.append("")
    for parity in report.parity:
        mark = "✓" if parity.equal else "✗"
        detail = "field-for-field equal" if parity.equal else f"DIVERGED at {parity.divergence}"
        lines.append(
            f"  {mark}  parity[{parity.agent}] "
            f"{parity.compared[0]} vs {parity.compared[1]}: {detail}"
        )
    for gate in report.gates:
        mark = "✓" if gate.admitted else "✗"
        detail = (
            "admitted (certify+bias+score)" if gate.admitted else f"REFUSED: {gate.refusal_reason}"
        )
        lines.append(f"  {mark}  judge[{gate.judge_id}] {detail}")
    lines += ["", f"  VERDICT: {report.verdict.upper()}"]
    if report.reasons:
        lines.append("  reasons:")
        lines += [f"    - {reason}" for reason in report.reasons]
    return "\n".join(lines)
