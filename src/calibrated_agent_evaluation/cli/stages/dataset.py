# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""``dataset``: build test scenarios from labeled failures; move them to and from AgentCore.

``answers`` writes a starter answer key listing the failure modes on the human FAIL labels in
scope. ``build`` turns each trace every rater labeled FAIL into a single-turn scenario, taking the
expected answer from the labels and the expected tool order and assertions from the answer key.
Both read only the trace store and run on a bare install. ``publish`` and ``pull`` talk to
AgentCore and need the ``[agentcore]`` extra; ``publish`` writes nothing without ``--yes``.
``run`` holds one simulated conversation per scenario of a simulated slice and stores each as a
trace (see :mod:`calibrated_agent_evaluation.cli.stages.dataset_run`).
"""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path

from calibrated_agent_evaluation._model import replace
from calibrated_agent_evaluation.cli.dispatch import EXIT_OK, StageOutcome
from calibrated_agent_evaluation.cli.errors import CliError, UsageError
from calibrated_agent_evaluation.scenarios.errors import ScenarioError
from calibrated_agent_evaluation.schema.types import Label, Trace

_SHOW = 5

# Which flags each action reads; any other flag given is refused rather than silently ignored.
_FLAGS: dict[str, frozenset[str]] = {
    "answers": frozenset({"out", "force", "failure_mode", "trace_id"}),
    "build": frozenset({"out", "force", "failure_mode", "trace_id", "answers"}),
    "publish": frozenset({"slice", "name", "yes", "profile", "region"}),
    "pull": frozenset({"out", "force", "dataset_id", "dataset_version", "profile", "region"}),
    "run": frozenset(
        {
            "slice",
            "yes",
            "profile",
            "region",
            "executor",
            "agent",
            "agent_model",
            "cassette",
            "judge",
            "simulator",
            "simulator_model",
            "runtime_arn",
            "payload_format",
            "qualifier",
            "config_bundle_arn",
            "config_bundle_version",
            "read_timeout",
            "log_dir",
        }
    ),
}
_REQUIRED: dict[str, tuple[str, ...]] = {
    "answers": ("out",),
    "build": ("out",),
    "publish": ("slice", "name"),
    "pull": ("dataset_id", "dataset_version", "out"),
    "run": ("slice",),
}
_FLAG_NAMES = {
    "out": "--out",
    "force": "--force",
    "failure_mode": "--failure-mode",
    "trace_id": "--trace",
    "answers": "--answers",
    "slice": "SLICE",
    "name": "--name",
    "yes": "--yes",
    "dataset_id": "--dataset-id",
    "dataset_version": "--version",
    "profile": "--profile",
    "region": "--region",
    "executor": "--executor",
    "agent": "--agent",
    "agent_model": "--agent-model",
    "cassette": "--cassette",
    "judge": "--judge",
    "simulator": "--simulator",
    "simulator_model": "--simulator-model",
    "runtime_arn": "--runtime-arn",
    "payload_format": "--payload-format",
    "qualifier": "--qualifier",
    "config_bundle_arn": "--config-bundle-arn",
    "config_bundle_version": "--config-bundle-version",
    "read_timeout": "--read-timeout",
    "log_dir": "--log-dir",
}


def dataset_handler(args: argparse.Namespace) -> StageOutcome:
    """Run the ``calibrated-agent-evaluation dataset`` action ``args.action`` names."""
    action = str(args.action)
    _check_flags(args, action)
    if action == "answers":
        return _answers(args)
    if action == "build":
        return _build(args)
    if action == "publish":
        return _publish(args)
    if action == "run":
        from calibrated_agent_evaluation.cli.stages.dataset_run import run_slice

        return run_slice(args)
    return _pull(args)


def _check_flags(args: argparse.Namespace, action: str) -> None:
    given = {name for name in _FLAG_NAMES if getattr(args, name, None) not in (None, False)}
    stray = sorted(_FLAG_NAMES[n] for n in given - _FLAGS[action])
    if stray:
        raise UsageError(
            f"`calibrated-agent-evaluation dataset {action}` does not take {', '.join(stray)}"
        )
    missing = [_FLAG_NAMES[n] for n in _REQUIRED[action] if n not in given]
    if missing:
        raise UsageError(
            f"`calibrated-agent-evaluation dataset {action}` needs {', '.join(missing)}"
        )


def _refuse_existing_out(args: argparse.Namespace) -> None:
    """Refuse an existing ``--out`` before any store read, unless ``--force``."""
    out = Path(args.out)
    if out.exists() and not args.force:
        raise ScenarioError(f"{out} already exists; pass --force to replace it")


def _store(args: argparse.Namespace) -> tuple[list[Trace], dict[str, list[Label]], list[str]]:
    """The traces in scope, their labels, and one line per trace the scan left out."""
    from calibrated_agent_evaluation.cli.backends import open_backend
    from calibrated_agent_evaluation.cli.scope import read_scope, skip_lines

    with open_backend(args) as backend:
        scan = read_scope(backend, args)
        labels = {t.trace_id: list(backend.get_scores(t.trace_id)) for t in scan.traces}
    return list(scan.traces), labels, skip_lines(scan)


def _answers(args: argparse.Namespace) -> StageOutcome:
    from calibrated_agent_evaluation.scenarios.build import starter_key
    from calibrated_agent_evaluation.scenarios.slicefile import write_json_model

    _refuse_existing_out(args)
    traces, labels, skipped = _store(args)
    key = starter_key(traces, labels, args.failure_mode)
    if not key.by_failure_mode:
        raise CliError(_nothing_in_scope(args, len(traces)))
    out = Path(args.out)
    write_json_model(out, key, force=bool(args.force))
    modes = list(key.by_failure_mode)
    summary = [
        f"Wrote an answer key for {len(modes)} failure mode(s) to {out}: {_shown(modes)}.",
        "Next: fill in each mode's expected_trajectory (the tool names, in order) and assertions "
        "(plain-language checks), then run "
        f"`calibrated-agent-evaluation dataset build --answers {out} --out slice.json`.",
    ]
    return StageOutcome(
        stage="dataset", exit_code=EXIT_OK, summary="\n".join(summary), detail="\n".join(skipped)
    )


def _build(args: argparse.Namespace) -> StageOutcome:
    from calibrated_agent_evaluation.scenarios.build import build_slice
    from calibrated_agent_evaluation.scenarios.slicefile import (
        read_answer_key,
        read_slice,
        write_slice,
    )
    from calibrated_agent_evaluation.scenarios.types import DatasetSlice, PredefinedScenario

    _refuse_existing_out(args)
    key = read_answer_key(args.answers) if args.answers else None
    out = Path(args.out)
    previous = None
    if out.exists() and args.force:
        try:
            previous = read_slice(out)
        except ScenarioError as exc:
            raise ScenarioError(
                f"{out} exists but is not a slice file, so it was left unchanged: {exc}"
            ) from exc
    traces, labels, skipped = _store(args)
    result = build_slice(traces, labels, key, args.failure_mode)
    if result.slice is None:
        raise CliError(_nothing_in_scope(args, len(traces)) + _left_out_text(result.left_out))
    built: DatasetSlice = result.slice
    if previous is not None and previous.published is not None:
        built = DatasetSlice(
            form=built.form, scenarios=built.scenarios, published=previous.published
        )
    write_slice(out, built, force=bool(args.force))

    n = result.n_scenarios
    with_assertions = sum(1 for s in built.scenarios if s.assertions)
    with_tools = sum(
        1 for s in built.scenarios if isinstance(s, PredefinedScenario) and s.expected_trajectory
    )
    summary = [
        f"Built {n} scenario(s) from the human FAIL labels on {len(traces)} stored trace(s) and "
        f"wrote them to {out}: {with_assertions} with assertions, {with_tools} with an expected "
        "tool order.",
    ]
    n_left = sum(len(ids) for ids in result.left_out.values())
    if n_left:
        summary.append(f"Left out {n_left} trace(s):{_left_out_text(result.left_out)}")
    if result.unused_modes:
        summary.append(
            f"The answer key's entries for {_shown(result.unused_modes)} matched no scenario."
        )
    if with_assertions < n:
        how = (
            f"fill in their failure modes in {args.answers}"
            if args.answers
            else "`calibrated-agent-evaluation dataset answers` writes a key to fill in"
        )
        summary.append(
            f"{n - with_assertions} scenario(s) have no assertions, so AgentCore's goal check will "
            f"judge them without ground truth; {how}."
        )
    if built.published is not None:
        summary.append(
            f"Kept the published version {built.published.version} of "
            f"{built.published.dataset_name}; publishing this slice creates the next version."
        )
    summary.append(
        f"Next: `calibrated-agent-evaluation dataset publish {out} --name NAME` shows the publish "
        "plan."
    )
    detail = [
        f"scenarios: {_shown([s.scenario_id for s in built.scenarios])}",
        *(f"  left out ({reason}): {_shown(ids)}" for reason, ids in result.left_out.items()),
        *skipped,
    ]
    return StageOutcome(
        stage="dataset", exit_code=EXIT_OK, summary="\n".join(summary), detail="\n".join(detail)
    )


def _publish(args: argparse.Namespace) -> StageOutcome:
    from calibrated_agent_evaluation import agentcore
    from calibrated_agent_evaluation.scenarios.slicefile import read_slice, write_slice

    slice_path = Path(args.slice)
    slice_ = read_slice(slice_path)
    name = str(args.name)
    with agentcore.AwsClients.from_profile(
        region=args.region, profile=args.profile, handlers=[agentcore.dataset_guard()]
    ) as clients:
        plan = agentcore.plan_publish(clients, slice_, name)
        what = (
            f"creates a new dataset {name!r}"
            if plan.creates_dataset
            else f"creates version {plan.next_version} of dataset {name!r} ({plan.existing_id})"
        )
        plan_lines = [
            f"  dataset: {name} in {plan.region}",
            f"  schema: {plan.schema_type}",
            f"  scenarios: {plan.n_scenarios} ({plan.n_with_assertions} with assertions, "
            f"{plan.n_with_trajectory} with an expected tool order)",
        ]
        if plan.unchanged and slice_.published is not None:
            summary = [
                f"{slice_path} is unchanged since version {slice_.published.version} of "
                f"{slice_.published.dataset_name}, so no new version was created.",
            ]
            return _outcome(summary, plan_lines)
        if not args.yes:
            summary = [
                f"Publishing {slice_path} {what}:",
                *plan_lines,
                "Nothing was written. Run the same command with --yes to publish.",
            ]
            return _outcome(summary, [])
        published = agentcore.publish(clients, slice_, plan)
    try:
        write_slice(slice_path, replace(slice_, published=published), force=True)
    except ScenarioError as exc:
        raise ScenarioError(
            f"published version {published.version} of {published.dataset_name} "
            f"({published.dataset_arn}), but could not record it in {slice_path}: {exc}; pull "
            f"that version to recover the slice file"
        ) from exc
    summary = [
        f"Published {slice_path} as version {published.version} of dataset "
        f"{published.dataset_name} ({published.dataset_id}) in {published.region}.",
        f"The slice file now records {published.dataset_arn} version {published.version}.",
        "Next: run AgentCore's dataset evaluations on this version, or pull it back with "
        f"`calibrated-agent-evaluation dataset pull --dataset-id {published.dataset_id} --version "
        f"{published.version} "
        "--out FILE`.",
    ]
    return _outcome(summary, plan_lines)


def _pull(args: argparse.Namespace) -> StageOutcome:
    from calibrated_agent_evaluation import agentcore
    from calibrated_agent_evaluation.agentcore.datasets_codec import validate_version
    from calibrated_agent_evaluation.scenarios.slicefile import write_slice

    version = str(args.dataset_version)
    validate_version(version)
    _refuse_existing_out(args)
    out = Path(args.out)
    with agentcore.AwsClients.from_profile(
        region=args.region, profile=args.profile, handlers=[agentcore.dataset_guard()]
    ) as clients:
        slice_ = agentcore.pull(clients, str(args.dataset_id), version)
    pointer = slice_.published
    if pointer is None:  # pull always records where the slice came from
        raise CliError("the pulled slice has no dataset version recorded")
    write_slice(out, slice_, force=bool(args.force))
    summary = [
        f"Pulled version {version} of dataset {pointer.dataset_name} ({pointer.dataset_id}): "
        f"{len(slice_.scenarios)} {slice_.form} scenario(s), written to {out}.",
    ]
    return _outcome(summary, [f"  {pointer.dataset_arn} version {version}"])


def _outcome(summary: list[str], detail: list[str]) -> StageOutcome:
    return StageOutcome(
        stage="dataset", exit_code=EXIT_OK, summary="\n".join(summary), detail="\n".join(detail)
    )


def _nothing_in_scope(args: argparse.Namespace, n_traces: int) -> str:
    if args.failure_mode:
        return (
            f"no trace-level human FAIL label with failure mode {args.failure_mode!r} on "
            f"{n_traces} stored trace(s) is agreed by every rater, so there is nothing to build; "
            "check the mode name (`calibrated-agent-evaluation dataset answers` lists the modes "
            "in scope)"
        )
    return (
        f"no trace-level human FAIL label on {n_traces} stored trace(s) is agreed by every rater, "
        "so there is nothing to build; label failures with `calibrated-agent-evaluation label "
        "--serve` first"
    )


def _left_out_text(left_out: dict[str, list[str]]) -> str:
    return "".join(
        f" {len(ids)} because {reason} ({_shown(ids)})." for reason, ids in left_out.items()
    )


def _shown(items: Sequence[str]) -> str:
    shown = ", ".join(items[:_SHOW])
    return shown + (f" and {len(items) - _SHOW} more" if len(items) > _SHOW else "")
