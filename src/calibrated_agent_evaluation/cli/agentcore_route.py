# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""The checks and messages the AgentCore Runtime CLI routes share.

``calibrated-agent-evaluation generate --run --executor agentcore`` and
``calibrated-agent-evaluation probe --executor agentcore`` both send inputs to an agent on Amazon
Bedrock AgentCore Runtime. Each pays for every invocation and stores the traces only after the last
one, so whatever would refuse the command, or make that store fail, is checked first, while nothing
has been sent and nothing would be lost. Both commands run the checks here, so they refuse with the
same messages:

* :func:`refuse_agentcore_flags_without_executor`: a flag only ``--executor agentcore`` reads,
  given with another executor or none, or, for ``generate``, without ``--run``;
* :func:`flag_given`: whether a parsed flag was given at all, the test every flag refusal uses;
* :func:`require_agentcore_options`: ``--runtime-arn`` and ``--payload-format``, both named at
  once when both are missing;
* :func:`cassette_mode_from_env`: ``CALIBRATED_AGENT_EVALUATION_CASSETTE_MODE``, refused naming the
  variable;
* :func:`require_agentcore_extra`: the ``[agentcore]`` extra, for the names a command uses;
* :func:`target_fields`: the :class:`~calibrated_agent_evaluation.agentcore.RuntimeTarget` fields
  the flags give, an empty configuration bundle flag refused by name;
* :func:`manifest_dir`: where the run manifest goes;
* :func:`check_trace_store`: the trace store can take the traces;
* :func:`print_plan`: what a run that can call AWS will call, printed before ``--yes`` is honored;
* :func:`billed_call_budget`: ``CALIBRATED_AGENT_EVALUATION_AGENTCORE_MAX_BILLED_CALLS``, refused
  naming the variable;
* :func:`manifest_report`: the line naming the run manifest once a run has started.

Importing this module loads no AgentCore runtime module and no AWS SDK: each helper imports what it
needs when it runs, so ``import calibrated_agent_evaluation.cli`` stays light.
"""

from __future__ import annotations

import argparse
import os
import sys
from collections.abc import Callable, Iterable, Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any, Protocol, TextIO

from calibrated_agent_evaluation.cli.errors import CliError, UsageError

if TYPE_CHECKING:
    from calibrated_agent_evaluation.agentcore import (
        AgentCoreExtraMissing,
        BilledCallBudget,
        RuntimeTarget,
    )
    from calibrated_agent_evaluation.backends import TraceStoreBackend
    from calibrated_agent_evaluation.execute.cassette import Mode

__all__ = [
    "RecordsRuns",
    "billed_call_budget",
    "cassette_mode_from_env",
    "check_trace_store",
    "extra_missing_error",
    "flag_given",
    "manifest_dir",
    "manifest_report",
    "print_plan",
    "refuse_agentcore_flags_without_executor",
    "require_agentcore_extra",
    "require_agentcore_options",
    "target_fields",
]

# The configuration bundle flags, by parsed name; target_fields refuses an empty value for each.
_BUNDLE_FLAGS = {
    "config_bundle_arn": "--config-bundle-arn",
    "config_bundle_version": "--config-bundle-version",
}


class RecordsRuns(Protocol):
    """What :func:`manifest_report` reads: the runs started, and the records the manifest lacks.

    The runtime executor and the probe adapter both have these.
    """

    @property
    def run_records(self) -> Sequence[object]:
        """The record of every run started, whatever its outcome, in order."""
        ...

    @property
    def manifest_errors(self) -> Sequence[Exception]:
        """Why each record the run manifest could not take was not added, in order."""
        ...


def refuse_agentcore_flags_without_executor(
    args: argparse.Namespace, *, remedy: str, sent_by: str | None = None
) -> None:
    """Refuse every flag only ``--executor agentcore`` reads when the command won't send with it.

    Each such flag would otherwise be ignored, so a command runs this before it reads anything
    else.

    Args:
        args: The parsed command line.
        remedy: How this command sends its inputs to an AgentCore Runtime agent, as a clause that
            starts with "pass", e.g. "pass --executor agentcore to send the prompts to an
            AgentCore Runtime agent". The message then offers dropping the flags instead.
        sent_by: The switch without which the command sends nothing even with
            ``--executor agentcore``, such as ``generate``'s ``--run``. With the executor named
            and this switch left out, the flags are refused too, naming it. ``None`` when the
            executor alone sends, as for ``probe``.

    Raises:
        CliError: Naming every such flag given, and ``--executor agentcore`` and ``remedy``, or,
            when only ``sent_by`` is missing, ``sent_by``.
    """
    from calibrated_agent_evaluation.cli.args import AGENTCORE_ONLY_FLAGS

    agentcore = getattr(args, "executor", None) == "agentcore"
    sends = sent_by is None or bool(getattr(args, sent_by.lstrip("-").replace("-", "_"), False))
    if agentcore and sends:
        return
    given = [
        flag for dest, flag in AGENTCORE_ONLY_FLAGS.items() if flag_given(getattr(args, dest, None))
    ]
    if not given:
        return
    one = len(given) == 1
    if agentcore:
        # The executor is named, but without sent_by nothing is sent, so the flags would be
        # ignored.
        raise UsageError(
            f"{', '.join(given)} {'is' if one else 'are'} read only when {sent_by} sends the "
            f"inputs to the AgentCore Runtime agent, and without {sent_by} nothing is sent, so "
            f"{'it' if one else 'they'} would be ignored. Pass {sent_by} to send the inputs, or "
            f"drop {'it' if one else 'them'}"
        )
    raise UsageError(
        f"{', '.join(given)} only {'applies' if one else 'apply'} to --executor agentcore; "
        f"{remedy}, or drop {'it' if one else 'them'}"
    )


def flag_given(value: object) -> bool:
    """Whether a parsed flag's ``value`` says the flag was given.

    A flag left out parses to ``None``, and a switch such as ``--yes`` left out to ``False``.
    Anything else was given: a list flag such as ``--battery`` with no names parses to an empty
    list, which counts.
    """
    return value is not None and value is not False


def require_agentcore_options(args: argparse.Namespace) -> None:
    """Refuse ``--executor agentcore`` without ``--runtime-arn`` or ``--payload-format``.

    Raises:
        CliError: Naming each missing option in one message, with the four format names when
            ``--payload-format`` is missing.
    """
    from calibrated_agent_evaluation.cli.args import AGENTCORE_PAYLOAD_FORMATS

    missing: list[str] = []
    if not getattr(args, "runtime_arn", None):
        missing.append("--runtime-arn ARN, the AgentCore Runtime of the agent under test")
    if not getattr(args, "payload_format", None):
        missing.append(
            "--payload-format NAME, the request body its agent reads: one of "
            f"{', '.join(AGENTCORE_PAYLOAD_FORMATS)}"
        )
    if missing:
        raise UsageError(f"--executor agentcore needs {'; and '.join(missing)}")


def cassette_mode_from_env() -> Mode:
    """The cassette mode ``CALIBRATED_AGENT_EVALUATION_CASSETTE_MODE`` sets, ``auto`` when unset.

    Raises:
        CliError: The variable holds another value; the message names the variable and the values
            it takes. Chained to the parser's ``ValueError``.
    """
    from calibrated_agent_evaluation.execute.cassette import MODE_ENV_VAR, Mode, mode_from_env

    try:
        return mode_from_env(default=Mode.AUTO)
    except ValueError as exc:
        raise CliError(
            f"{exc}; nothing was run. Set {MODE_ENV_VAR} to one of those, or unset it"
        ) from exc


def require_agentcore_extra(names: Iterable[str]) -> None:
    """Refuse the command unless the ``[agentcore]`` extra is installed for every name it uses.

    The extra's own SDK must be importable: the AWS SDK alone may come from another extra, at a
    version older than the one these calls were checked against. Then each name is read from
    :mod:`calibrated_agent_evaluation.agentcore`, since its first access imports the adapter module
    behind it, and those need the extra's AWS SDK.

    Args:
        names: The :mod:`calibrated_agent_evaluation.agentcore` names the command uses, e.g.
            ``"RuntimeTarget"``.

    Raises:
        CliError: The extra is missing; the message holds the command that installs it.
    """
    import calibrated_agent_evaluation.agentcore as agentcore
    from calibrated_agent_evaluation.agentcore import AgentCoreExtraMissing
    from calibrated_agent_evaluation.cli.config import probe_extra

    if not probe_extra("bedrock_agentcore").available:
        raise extra_missing_error(AgentCoreExtraMissing("--executor agentcore"))
    try:
        for name in names:
            getattr(agentcore, name)
    except AgentCoreExtraMissing as exc:
        raise extra_missing_error(exc) from exc


def extra_missing_error(exc: AgentCoreExtraMissing) -> CliError:
    """The refusal for a missing ``[agentcore]`` extra, holding the command that installs it.

    For a name a command reads from :mod:`calibrated_agent_evaluation.agentcore` after
    :func:`require_agentcore_extra`; raise it ``from exc``.
    """
    return CliError(
        f"--executor agentcore needs the [agentcore] extra, so nothing was run; install it with: "
        f"{exc.hint}"
    )


def target_fields(args: argparse.Namespace) -> dict[str, Any]:
    """The :class:`~calibrated_agent_evaluation.agentcore.RuntimeTarget` arguments the flags give.

    ``qualifier`` is left out when ``--qualifier`` is not given, so the target keeps its own
    default. Call it after :func:`require_agentcore_options`.

    Raises:
        CliError: ``--config-bundle-arn`` or ``--config-bundle-version`` was given an empty
            value, naming each such flag. The target takes an empty value as no bundle, so every
            invocation would run without the bundle the command line asked for.
    """
    empty = [flag for dest, flag in _BUNDLE_FLAGS.items() if getattr(args, dest, None) == ""]
    if empty:
        one = len(empty) == 1
        raise CliError(
            f"{' and '.join(empty)} {'is' if one else 'are'} empty, so nothing was run: an empty "
            "value names no configuration bundle, and the agent would run without one. Pass the "
            "bundle's ARN and version, or drop both flags to run without a bundle; a flag given "
            '"$VAR" is empty when VAR is unset'
        )
    fields: dict[str, Any] = {
        "runtime_arn": args.runtime_arn,
        "payload_format": args.payload_format,
        "config_bundle_arn": getattr(args, "config_bundle_arn", None),
        "config_bundle_version": getattr(args, "config_bundle_version", None),
    }
    qualifier = getattr(args, "qualifier", None)
    if qualifier is not None:
        fields["qualifier"] = qualifier
    return fields


def manifest_dir(args: argparse.Namespace) -> Path:
    """The run manifest's directory: ``--log-dir``, else ``<root>/records/agentcore``.

    Both AgentCore commands share the default. It is built here rather than through the records
    helpers: those take only the record kinds that dump and load carry, and the run manifest is
    deliberately not one of them.
    """
    from calibrated_agent_evaluation.cli.backends import local_root

    log_dir = getattr(args, "log_dir", None)
    return Path(log_dir) if log_dir is not None else local_root(args) / "records" / "agentcore"


def check_trace_store(backend: TraceStoreBackend) -> None:
    """Check that ``backend`` can take the traces a run will store, before the run starts.

    A backend with a reachability check (the Langfuse backend) runs it, so an unreachable host or
    refused keys stop the command before the first billed call. The local backend has none:
    building it created its store. A backend wrapped for consistent reads is checked through to
    the store it wraps.

    Args:
        backend: The trace store the command built, as it will store the traces.

    Raises:
        BackendError: The store can't be reached, or it refuses the credentials.
    """
    from calibrated_agent_evaluation.cli.consistency import ConsistentBackend

    store = backend.inner if isinstance(backend, ConsistentBackend) else backend
    check = getattr(store, "check_reachable", None)
    if callable(check):
        check()


def print_plan(
    command: str,
    target: RuntimeTarget,
    manifest: Path,
    *,
    count: int,
    noun: str,
    reads_spans: bool,
    cassette: str | None = None,
) -> None:
    """Print to stderr what a run that can call AWS will call; call it before any AWS call.

    The plan names the runtime ARN and qualifier, the configuration bundle when the target names
    one, and the most invocations the run sends, each billed and retried when the session is busy
    or the runtime throttles. It says what is read
    back, and what is kept when an invocation fails: the run stops there and stores no trace.

    Args:
        command: The command's name, as the line's prefix, e.g. ``"generate"``.
        target: The runtime every invocation goes to.
        manifest: The run manifest's JSON file, which keeps a record of every run started.
        count: The most invocations the run sends, one per ``noun``.
        noun: What one invocation sends, in the singular, e.g. ``"input"``; adding an ``s``
            makes its plural.
        reads_spans: True when a span reader reads each run's spans. False when none does, so
            the replies are stored as ``probed``, outcome-only traces.
        cassette: The cassette the runs are recorded in, which keeps them when one fails.
    """
    sends = f"{count} {noun}" if count == 1 else f"{count} {noun}s"
    bundle = (
        f", configuration bundle {target.config_bundle_arn} at version "
        f"{target.config_bundle_version}"
        if target.config_bundle_arn is not None
        else ""
    )
    reads = (
        "the span reader reads each run's spans"
        if reads_spans
        else "no spans are read, so the replies are stored as probed, outcome-only traces"
    )
    lost = (
        f"the traces of the {noun}s before it are then not stored"
        if reads_spans
        else "none of the replies are then stored"
    )
    kept = (
        f"they stay in the cassette {cassette}"
        if cassette
        else f"only their run records in the run manifest {manifest} are kept"
    )
    print(
        f"calibrated-agent-evaluation {command}: this run can call AWS. It sends up to {sends} to "
        "the AgentCore "
        f"Runtime agent {target.runtime_arn} (qualifier {target.qualifier}{bundle}), each a billed "
        f"invocation (retried when the session is busy or the runtime throttles), and {reads}. "
        f"It stops at the first {noun} that fails, and {lost}: {kept}.",
        file=sys.stderr,
        flush=True,
    )


def billed_call_budget() -> BilledCallBudget:
    """The billed-call budget the environment sets; no limit when unset.

    The variable is ``CALIBRATED_AGENT_EVALUATION_AGENTCORE_MAX_BILLED_CALLS``.

    Raises:
        CliError: The variable is not blank and not a non-negative whole number; the message
            names the variable and its value. Chained to the budget's own error.
    """
    import calibrated_agent_evaluation.agentcore as agentcore
    from calibrated_agent_evaluation.agentcore.guards import BUDGET_ENV_VAR

    try:
        return agentcore.BilledCallBudget.from_env(os.environ)
    except agentcore.AgentCoreError as exc:
        raise CliError(
            f"{BUDGET_ENV_VAR}={os.environ.get(BUDGET_ENV_VAR, '').strip()!r} is not a "
            "non-negative whole number, so nothing was sent. Set it to the most billed calls to "
            "allow, or unset it to remove the limit"
        ) from exc


def evaluate_concurrency() -> int:
    """How many ``Evaluate`` calls to send at once: the environment's value, or 10.

    The variable is ``CALIBRATED_AGENT_EVALUATION_AGENTCORE_MAX_CONCURRENCY``.

    Read before any client is built, so a bad value sends nothing.

    Raises:
        CliError: The variable is set to anything but a whole number from 1 to 10; the message
            names the variable, its value and the range. Chained to the runner's own error.
    """
    from calibrated_agent_evaluation.agentcore.errors import AgentCoreError
    from calibrated_agent_evaluation.agentcore.evaluate import concurrency_from_env

    try:
        return concurrency_from_env(os.environ)
    except AgentCoreError as exc:
        raise CliError(f"{exc}; nothing was sent") from exc


def scoring_progress(stream: TextIO | None = None) -> Callable[[int, int], None]:
    """A ``progress(scored, total)`` callback that prints ``scored N of M trace(s)`` to stderr.

    It prints each time ``floor(10 * scored / total)`` goes up, so about ten lines a run, one per
    call when a run has fewer than ten traces, and always the last call.
    """
    out = stream if stream is not None else sys.stderr
    printed = [0]

    def progress(scored: int, total: int) -> None:
        tenth = (10 * scored) // total
        if tenth > printed[0]:
            printed[0] = tenth
            print(f"scored {scored} of {total} trace(s)", file=out, flush=True)

    return progress


def manifest_report(runs: RecordsRuns, manifest: Path) -> str:
    """What the run manifest holds once a run has started: one record per run, or which it lacks.

    Args:
        runs: What started the runs, such as the runtime executor or the probe adapter.
        manifest: The run manifest's JSON file.
    """
    started = len(runs.run_records)
    failed = runs.manifest_errors
    if not failed:
        return f"the run manifest, one record per run started, is {manifest}"
    missing = (
        "the run that started" if started == 1 else f"{len(failed)} of the {started} runs started"
    )
    # Each error names the manifest's path, the step that failed and what to do next.
    reasons = "; ".join(str(error) for error in failed)
    return f"the run manifest is {manifest}, but it holds no record of {missing}: {reasons}"
