# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""``dump`` / ``load``: the no-lock-in round-trip.

Reuses the dump/load registry (``calibrated_agent_evaluation.dumpload``, reuse, not re-implement).
``dump`` writes every registered artifact in the backend to a CSV+JSON directory, plus the stage
records under the local root into ``<dir>/records/``, and seals both with one manifest; ``load``
checks the manifest and reads both back. A corrupt directory
raises the typed ``CorruptDumpError`` (a ``CalibratedAgentEvaluationDataError``), which the CLI
boundary turns into exit 1.

``dump --judges-only --judge FILE`` writes only what a fresh store needs to admit and score with
those judges (:mod:`calibrated_agent_evaluation.cli.judgekit`), and ``load`` restores such a dump by
its manifest.
"""

from __future__ import annotations

import argparse
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Protocol, runtime_checkable

from calibrated_agent_evaluation.cli.dispatch import EXIT_OK, StageOutcome
from calibrated_agent_evaluation.cli.errors import CliError, UsageError
from calibrated_agent_evaluation.schema.types import Label, Trace


@runtime_checkable
class _RoundTripBackend(Protocol):
    """The backend surface dump/load needs, a superset of ``TraceStoreBackend``.

    ``LocalBackend`` provides ``all_traces``/``all_scores`` (the full enumeration dump/load
    requires). ``LangfuseBackend`` does not, and neither does the narrower ``TraceStoreBackend``
    protocol the CLI resolves, so the handler checks for the capability and gives a typed refusal
    for a backend that can't round-trip (rather than an ``AttributeError``).
    """

    def all_traces(self) -> list[Trace]: ...
    def all_scores(self) -> list[Label]: ...
    def put_trace(self, trace: Trace) -> None: ...
    def put_score(self, label: Label) -> None: ...


def dump_handler(args: argparse.Namespace) -> StageOutcome:
    """Dump the backend's artifacts and the local root's records to ``args.directory``."""
    from calibrated_agent_evaluation.cli.backends import local_root
    from calibrated_agent_evaluation.cli.records import export_records
    from calibrated_agent_evaluation.cli.stages.calibrate_agentcore import (
        refuse_flags_without_agentcore,
    )
    from calibrated_agent_evaluation.dumpload import dump

    judges = getattr(args, "judge_files", None) or []
    refuse_flags_without_agentcore(args, judges)
    if getattr(args, "judges_only", False):
        return _dump_judges(args, judges)
    if judges:
        raise UsageError(
            "--judge selects judges for a judges-only dump; add --judges-only, or leave --judge "
            "out to dump the whole store"
        )
    counts: dict[str, int] = {}
    with _resolve(args) as backend:
        # The records go in before the manifest, so it seals them with the artifacts.
        dump(
            backend,
            args.directory,
            extra=lambda out: counts.update(export_records(local_root(args), out)),
        )
    return StageOutcome(
        stage="dump",
        exit_code=EXIT_OK,
        summary=f"dumped the artifact set to {args.directory} (CSV+JSON); {_record_note(counts)}",
    )


def load_handler(args: argparse.Namespace) -> StageOutcome:
    """Load a CSV+JSON directory back into the backend, and its records into the local root.

    A corrupt/version-mismatched directory raises the ``CorruptDumpError`` /
    ``SchemaVersionError`` (both ``CalibratedAgentEvaluationDataError``), caught at the CLI boundary
    as exit 1, never a bare parse traceback. ``load`` version-checks everything before any
    write, so a bad directory persists nothing.
    """
    from calibrated_agent_evaluation.cli.backends import local_root
    from calibrated_agent_evaluation.cli.judgekit import is_judge_kit
    from calibrated_agent_evaluation.cli.records import prepare_import
    from calibrated_agent_evaluation.dumpload import load, verify_dump

    if is_judge_kit(Path(args.directory)):
        return _load_judges(args)
    with _resolve(args) as backend:
        # The manifest is checked, then the records read and checked, before the backend loads, so
        # any refusal persists nothing.
        verify_dump(args.directory)
        write_records = prepare_import(Path(args.directory), local_root(args))
        load(backend, args.directory, fmt=getattr(args, "fmt", "json"))
    counts = write_records()
    return StageOutcome(
        stage="load",
        exit_code=EXIT_OK,
        summary=(
            f"loaded the artifact set from {args.directory} into the backend; "
            f"{_record_note(counts, verb='new')}"
        ),
    )


def _dump_judges(args: argparse.Namespace, judges: list[str]) -> StageOutcome:
    import contextlib

    from calibrated_agent_evaluation.cli.backends import local_root
    from calibrated_agent_evaluation.cli.judgekit import dump_judges
    from calibrated_agent_evaluation.cli.stages.calibrate_agentcore import judge_file_for_identity

    dest = Path(args.directory)
    # Holds the AWS clients an AgentCore judge file opens to read its evaluator.
    with contextlib.ExitStack() as held:
        manifest = dump_judges(
            local_root(args),
            dest,
            judges,
            open_judge=lambda path: judge_file_for_identity(args, path, held),
        )
    names = ", ".join(repr(e.judge_id) for e in manifest.judges)
    return StageOutcome(
        stage="dump",
        exit_code=EXIT_OK,
        summary=(
            f"dumped {len(manifest.judges)} judge(s) ({names}) to {dest}: each judge file with "
            "the calibration record and bias battery that admit it, and no traces or labels. "
            f"Next: `calibrated-agent-evaluation load {dest}` into the store that will score with "
            "them."
        ),
        detail="\n".join(
            f"  {e.judge_id}: judges/{e.file}, calibration {e.calib_id!r}, bias {e.battery_id!r}"
            for e in manifest.judges
        ),
    )


def _load_judges(args: argparse.Namespace) -> StageOutcome:
    from calibrated_agent_evaluation.cli.backends import local_root
    from calibrated_agent_evaluation.cli.judgekit import load_judges

    src = Path(args.directory)
    manifest, counts, files = load_judges(src, local_root(args))
    shown = ", ".join(str(f) for f in files)
    first = files[0] if files else "FILE"
    return StageOutcome(
        stage="load",
        exit_code=EXIT_OK,
        summary=(
            f"loaded {len(manifest.judges)} judge(s) from {src}: {shown}; "
            f"{_record_note(counts, verb='new')}. Next: `calibrated-agent-evaluation judge "
            f"--judge {first}` scores "
            "the traces in this store."
        ),
    )


def _record_note(counts: dict[str, int], *, verb: str = "") -> str:
    """``records: 2 calibration, 1 run`` or ``no records``."""
    shown = [f"{n} {verb + ' ' if verb else ''}{kind}" for kind, n in counts.items() if n]
    return f"records: {', '.join(shown)}" if shown else "no records"


@contextmanager
def _resolve(args: argparse.Namespace) -> Iterator[_RoundTripBackend]:
    """Open a round-trip-capable backend from the CLI flag (offline local default) for a block.

    Refuses (typed ``CliError``) a backend that can't enumerate its artifacts, rather than failing
    with an ``AttributeError`` mid-dump. The backend is closed when the block exits.
    """
    from calibrated_agent_evaluation.cli.backends import open_backend

    with open_backend(args) as backend:
        if not isinstance(backend, _RoundTripBackend):
            raise CliError(
                f"the {getattr(args, 'backend', 'local')!r} backend does not support dump/load "
                "(no all_traces/all_scores); use the local backend for artifact round-trip"
            )
        yield backend
