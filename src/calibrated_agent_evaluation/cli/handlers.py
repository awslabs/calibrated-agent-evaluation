# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Stage-handler registry.

Maps each ``calibrated-agent-evaluation`` stage to the callable that runs it. Handlers reuse the
feature module; they never re-implement the methodology; and import only
``calibrated_agent_evaluation.*`` (never ``scripts/``, which is not in the wheel). Handlers with
real logic live in ``cli.stages.*`` and are imported lazily (inside :func:`for_stage`) so importing
``calibrated_agent_evaluation.cli`` never pulls a stage's optional deps at module load.

Every command has its own ``cli.stages`` module. ``probe`` is the one stage whose input is an
external endpoint rather than the trace store, so it runs the real probing engine against
``--endpoint`` and persists the probed traces back into the backend for the downstream stages.
"""

from __future__ import annotations

from calibrated_agent_evaluation.cli.dispatch import Handler
from calibrated_agent_evaluation.cli.errors import CliError


def for_stage(stage: str) -> Handler:
    """Return the handler for ``stage``. Lazily imports the stage module."""
    if stage == "quickstart":
        from calibrated_agent_evaluation.cli.stages.quickstart import quickstart_handler

        return quickstart_handler
    if stage == "doctor":
        from calibrated_agent_evaluation.cli.stages.doctor import doctor_handler

        return doctor_handler
    if stage == "ingest":
        from calibrated_agent_evaluation.cli.stages.ingest import ingest_handler

        return ingest_handler
    if stage == "capture":
        from calibrated_agent_evaluation.cli.stages.capture import capture_handler

        return capture_handler
    if stage == "generate":
        from calibrated_agent_evaluation.cli.stages.generate import generate_handler

        return generate_handler
    if stage == "run":
        from calibrated_agent_evaluation.cli.stages.run import run_handler

        return run_handler
    if stage == "label":
        from calibrated_agent_evaluation.cli.stages.label import label_handler

        return label_handler
    if stage == "dump":
        from calibrated_agent_evaluation.cli.stages.dumpload import dump_handler

        return dump_handler
    if stage == "load":
        from calibrated_agent_evaluation.cli.stages.dumpload import load_handler

        return load_handler
    if stage == "calibrate":
        from calibrated_agent_evaluation.cli.stages.calibrate import calibrate_handler

        return calibrate_handler
    if stage == "bias":
        from calibrated_agent_evaluation.cli.stages.bias import bias_handler

        return bias_handler
    if stage == "judge":
        from calibrated_agent_evaluation.cli.stages.judge import judge_handler

        return judge_handler
    if stage == "prefer":
        from calibrated_agent_evaluation.cli.stages.prefer import prefer_handler

        return prefer_handler
    if stage == "improve":
        from calibrated_agent_evaluation.cli.stages.improve import improve_handler

        return improve_handler
    if stage == "cluster":
        from calibrated_agent_evaluation.cli.stages.cluster import cluster_handler

        return cluster_handler
    if stage == "define":
        from calibrated_agent_evaluation.cli.stages.define import define_handler

        return define_handler
    if stage == "dataset":
        from calibrated_agent_evaluation.cli.stages.dataset import dataset_handler

        return dataset_handler
    if stage == "audit":
        from calibrated_agent_evaluation.cli.stages.audit import audit_handler

        return audit_handler
    if stage == "online":
        from calibrated_agent_evaluation.cli.stages.online import online_handler

        return online_handler
    if stage == "score":
        from calibrated_agent_evaluation.cli.stages.score import score_handler

        return score_handler
    if stage == "results":
        from calibrated_agent_evaluation.cli.stages.results import results_handler

        return results_handler
    if stage == "plan":
        from calibrated_agent_evaluation.cli.stages.plan import plan_handler

        return plan_handler
    if stage == "probe":
        from calibrated_agent_evaluation.cli.stages.probe import probe_handler

        return probe_handler
    # Unreachable from the parser: every command it accepts is mapped above, which
    # test_cli_dispatch.py checks command by command.
    raise CliError(f"no handler for the {stage!r} command")
