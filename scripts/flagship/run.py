#!/usr/bin/env python3
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Flagship proof runner: the reference agents through all eight stages, in both worlds.

Runs OrderBot (single agent) and ClaimsCrew (crew) end to end with bias-gated judges, prints a
per-agent, per-world, per-stage table and the verdict, and with --out writes the FlagshipReport.
Exits 0 on proven, 1 on failed or unpinned, 2 on a refusal. The proof itself is
``calibrated_agent_evaluation.flagship.run_reference_proof``; this file only parses arguments and
picks backends.

The OrderBot judge is admitted on illustrative records (a CERTIFIED calibration and a CLEAR bias
battery pinned to the committed judge file), so the run exercises the admission chain without a
calibration study.

Usage:
    just flagship --local-only                 # hermetic: cassette replay + local backend
    just langfuse-up && just flagship          # both worlds: local + self-hosted Langfuse
    AWS_PROFILE=<your-profile> just flagship --live   # plus the live-Bedrock model-plane leg
    just flagship --agent claimscrew --local-only
"""

from __future__ import annotations

import argparse
import os
import sys
import tempfile
from pathlib import Path

from calibrated_agent_evaluation._provenance import capture_provenance
from calibrated_agent_evaluation.backends import TraceStoreBackend, get_backend
from calibrated_agent_evaluation.bias import BiasBatteryRecord
from calibrated_agent_evaluation.calibrate.record import Bound, CalibrationRecord
from calibrated_agent_evaluation.flagship import (
    REFERENCE_AGENTS,
    Agent,
    FlagshipError,
    ModelPlane,
    ReferenceCorpus,
    World,
    format_report,
    run_reference_proof,
)
from calibrated_agent_evaluation.flagship.runner import (
    live_bedrock_model_id,
    require_live_credentials,
)
from calibrated_agent_evaluation.judges import ExecutableJudge

_REPO = Path(__file__).resolve().parents[2]
_FIXTURES = _REPO / "fixtures"


def _backend_factory(local_root: Path):
    def factory(world: World) -> TraceStoreBackend:
        if world == "local":
            return get_backend(None, local_root=local_root)
        from calibrated_agent_evaluation.backends.langfuse import LangfuseBackend

        # The defaults are the local dev stack's (`just langfuse-up`), not a real deployment's.
        host = os.environ.get("LANGFUSE_HOST", "http://localhost:3000")
        public = os.environ.get("LANGFUSE_PUBLIC_KEY", "pk-lf-calibrated-agent-evaluation-dev")
        secret = os.environ.get("LANGFUSE_SECRET_KEY", "sk-lf-calibrated-agent-evaluation-dev")
        return LangfuseBackend.from_credentials(public, secret, host, timeout_seconds=15)

    return factory


def _illustrative_admission(judge: ExecutableJudge) -> tuple[CalibrationRecord, BiasBatteryRecord]:
    calibration = CalibrationRecord(
        provenance=capture_provenance(),
        calib_id=f"calib-{judge.judge_id}",
        judge_id=judge.judge_id,
        model=judge.model,
        prompt_version=judge.prompt_version,
        judge_fingerprint=judge.fingerprint,
        hunted_set_id=f"hunted-{judge.judge_id}",
        n_pos=80,
        n_neg=80,
        tpr=Bound(point=0.95, lb95=0.90, n=80),
        tnr=Bound(point=0.95, lb95=0.90, n=80),
        youden_j=0.90,
        result="CERTIFIED",
    )
    battery = BiasBatteryRecord(
        provenance=capture_provenance(),
        battery_id=f"bias-{judge.judge_id}-clear",
        judge_id=judge.judge_id,
        model=judge.model,
        prompt_version=judge.prompt_version,
        judge_fingerprint=judge.fingerprint,
        is_pairwise=False,
        results=(),
        verdict="CLEAR",
        failed_biases=(),
    )
    return calibration, battery


def _live_leg(agents: tuple[Agent, ...]) -> None:
    """Re-execute the OrderBot headline trace against a live model; raise if it cannot."""
    require_live_credentials(os.environ)
    # The corpus agents live at the repo root, which is not on sys.path for a script run by path.
    if str(_REPO) not in sys.path:
        sys.path.insert(0, str(_REPO))
    try:
        from corpus.agents.orderbot_inspect import orderbot_fail

        from calibrated_agent_evaluation.execute import ExecutionError, InspectExecutor
    except ImportError as exc:
        raise FlagshipError(
            f"--live needs the corpus agents and the [inspect] and [bedrock] extras: {exc}"
        ) from exc
    model = f"bedrock/{live_bedrock_model_id(os.environ)}"
    try:
        with (
            tempfile.TemporaryDirectory() as tmp,
            InspectExecutor(model=model, log_dir=Path(tmp)) as executor,
        ):
            traces = executor.run(orderbot_fail)
    except ExecutionError as exc:
        raise FlagshipError(f"--live model unreachable through the executor: {exc}") from exc
    if not traces:
        raise FlagshipError("--live re-execution produced no trace")
    print(
        f"  live-Bedrock leg: re-executed {len(traces)} headline trace(s) against {model} "
        f"(agents={list(agents)})"
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--agent",
        choices=[*REFERENCE_AGENTS, "both"],
        default="both",
        help="which reference agent(s) to prove (default: both)",
    )
    parser.add_argument(
        "--local-only",
        action="store_true",
        help="skip the Langfuse world (offline iteration; the default runs both worlds)",
    )
    parser.add_argument(
        "--live",
        action="store_true",
        help="re-execute the headline trace against a live Bedrock model first",
    )
    parser.add_argument("--out", default=None, help="directory to write the FlagshipReport into")
    args = parser.parse_args(argv)

    agents: tuple[Agent, ...] = REFERENCE_AGENTS if args.agent == "both" else (args.agent,)
    worlds: tuple[World, ...] = ("local",) if args.local_only else ("local", "langfuse")
    model_plane: ModelPlane = "live" if args.live else "cassette"

    if args.live:
        try:
            _live_leg(agents)
        except FlagshipError as exc:
            print(f"\nFLAGSHIP REFUSED (--live): {exc}", file=sys.stderr)
            return 2

    with tempfile.TemporaryDirectory() as tmp:
        try:
            corpus = ReferenceCorpus(_FIXTURES)
            calibration, battery = _illustrative_admission(corpus.orderbot_judge())
            report = run_reference_proof(
                corpus,
                agents=agents,
                worlds=worlds,
                model_plane=model_plane,
                backend_factory=_backend_factory(Path(tmp) / "backend"),
                calibration=calibration,
                battery=battery,
                out_dir=args.out,
            )
        except FlagshipError as exc:
            print(f"\nFLAGSHIP REFUSED: {exc}", file=sys.stderr)
            return 2

    print("\n" + format_report(report))
    if args.out:
        print(f"\nFlagshipReport written to {args.out}/flagship_report.json")
    return 0 if report.verdict == "proven" else 1


if __name__ == "__main__":
    raise SystemExit(main())
