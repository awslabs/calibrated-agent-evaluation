# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""``calibrated-agent-evaluation ingest``; normalize trace dumps into the store, or refuse loudly.

Takes one or more trace files or directories, resolves each to its normalizer adapter, and persists
the normalized Trace to the ``--backend`` store; unless ``--dry-run``, which reports the recovery
summary and writes nothing. A source that can't be parsed is skipped with a reason; a source that
would silently lose tool calls (or has duplicate span ids) is refused. The process exits non-zero if
any source was skipped or refused, so a dropped trace is never a silent success.

Reuses the ``calibrated_agent_evaluation.ingest`` feature module (imported lazily inside the
handler, so importing ``calibrated_agent_evaluation.cli`` never pulls it); computes nothing itself.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from calibrated_agent_evaluation.backends.base import TraceStoreBackend
from calibrated_agent_evaluation.cli.dispatch import StageOutcome
from calibrated_agent_evaluation.ingest import IngestReport, IngestResult

# Globs a directory arg expands to: single-trace sources (OTLP-JSON, binary OTLP, Inspect) and the
# dataset formats.
_TRACE_GLOBS: tuple[str, ...] = (
    "*.otlp.json",
    "*.json",
    "*.otlp",
    "*.eval",
    "*.jsonl",
    "*.parquet",
)


_GLOB_MAGIC = frozenset("*?[")


def _expand(raw_paths: list[str]) -> tuple[list[Path], list[Path]]:
    """Expand the path arguments to ``(files to ingest, files ignored)``, order-preserving.

    A directory is walked recursively: ``calibrated-agent-evaluation ingest exports/`` used to read
    only the top level and silently miss ``exports/run-1/trace.json``. An argument that does not
    exist and carries glob characters is expanded as a pattern from its literal prefix, so a quoted
    ``'exports/**/*.json'`` works the way it reads; one that matches nothing stays as given and is
    reported as a missing file.

    A bare ``*.json`` found by walking a directory is ignored when it parses as JSON and is plainly
    not OTLP (no ``resourceSpans``), such as a dataset's ``MANIFEST.json``. It used to be ingested,
    skipped, and fail the whole command. Ignored files are returned so the summary names them; a
    JSON file that does not parse is still attempted, so a truncated trace fails as it should. A
    file named directly is always attempted.
    """
    found: list[Path] = []
    ignored: list[Path] = []
    for raw in raw_paths:
        path = Path(raw)
        if path.is_dir():
            _walk(path, found, ignored)
        elif not path.exists() and _GLOB_MAGIC & set(raw):
            matches = _glob(path)
            if not matches:
                found.append(path)
            for match in matches:
                if match.is_dir():
                    _walk(match, found, ignored)
                else:
                    found.append(match)
        else:
            found.append(path)
    return _unique(found), _unique(ignored)


def _walk(directory: Path, found: list[Path], ignored: list[Path]) -> None:
    for pattern in _TRACE_GLOBS:
        for match in sorted(directory.rglob(pattern)):
            if not match.is_file():
                continue
            if pattern == "*.json" and _plainly_not_otlp(match):
                ignored.append(match)
            else:
                found.append(match)


def _glob(path: Path) -> list[Path]:
    parts = path.parts
    first = next(i for i, part in enumerate(parts) if _GLOB_MAGIC & set(part))
    base = Path(*parts[:first]) if first else Path()
    return sorted(base.glob(str(Path(*parts[first:]))))


def _plainly_not_otlp(path: Path) -> bool:
    """True for readable JSON with no ``resourceSpans`` key; False if it does not parse."""
    import json

    try:
        loaded = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return False
    return not (isinstance(loaded, dict) and "resourceSpans" in loaded)


def _unique(paths: list[Path]) -> list[Path]:
    seen: set[Path] = set()
    unique: list[Path] = []
    for path in paths:
        if path not in seen:
            seen.add(path)
            unique.append(path)
    return unique


def ingest_handler(args: argparse.Namespace) -> StageOutcome:
    """Ingest files, or pull a Langfuse project, into the configured backend.

    ``--from-langfuse`` reads source credentials from ``LANGFUSE_*`` env; the destination is the
    ``--backend`` selection (default local); the two are decoupled. Exit code
    follows the ingest report.
    """
    from calibrated_agent_evaluation.cli.backends import open_backend

    sample_size, seed, cell = _sample_args(args)
    with open_backend(args) as dest:
        return _ingest_into(args, dest, sample_size, seed, cell)


def _ingest_into(
    args: argparse.Namespace,
    dest: TraceStoreBackend,
    sample_size: int | None,
    seed: int | None,
    cell: dict[str, str] | None,
) -> StageOutcome:
    from calibrated_agent_evaluation.cli.backends import langfuse_from_args
    from calibrated_agent_evaluation.cli.errors import UsageError
    from calibrated_agent_evaluation.ingest import ingest_langfuse, run_ingest

    ignored: list[Path] = []
    dry_run = getattr(args, "dry_run", False)
    require_owned = getattr(args, "require_owned", False)
    overwrite = getattr(args, "overwrite", False)
    allow_partial_tools = getattr(args, "allow_partial_tools", False)
    if getattr(args, "source", None) == "agentcore":
        return _ingest_agentcore(args, dest, dry_run=dry_run, require_owned=require_owned)
    project = getattr(args, "from_langfuse", None)
    if project:
        source = langfuse_from_args(args)  # LANGFUSE_* env; CliError names the gap
        _require_project(source, project)
        report = ingest_langfuse(
            source,
            backend=dest,
            dry_run=dry_run,
            require_owned=require_owned,
            overwrite=overwrite,
            allow_partial_tools=allow_partial_tools,
            sample_size=sample_size,
            sample_seed=seed,
            sample_id=_sample_id(),
            cell=cell,
            source_label=f"langfuse:{project}",
        )
    else:
        if cell:
            raise UsageError(
                "--cell narrows a sample by taxonomy cell, and a file's cell is only known once "
                "it is normalized; sample a Langfuse project with --from-langfuse, or ingest the "
                "files and sample the store"
            )
        raw_paths = list(getattr(args, "paths", []) or [])
        fmt = getattr(args, "fmt", "auto")
        # HuggingFace repo ids are not paths, so there is nothing to expand.
        if fmt == "hf":
            files, ignored = [Path(p) for p in raw_paths], []
        else:
            files, ignored = _expand(raw_paths)
        convention = getattr(args, "convention", "auto")
        if convention != "auto" and fmt in ("strands", "inspect"):
            raise UsageError(
                f"--convention applies to OTLP and dataset sources, not --format {fmt}; drop one"
            )
        if not files:
            hint = f" (ignored {len(ignored)} non-trace JSON file(s))" if ignored else ""
            raise UsageError(
                "calibrated-agent-evaluation ingest needs one or more paths, or --from-langfuse "
                f"<project>{hint}"
            )
        report = run_ingest(
            files,
            backend=dest,
            fmt=fmt,
            dry_run=dry_run,
            require_owned=require_owned,
            convention=convention,
            overwrite=overwrite,
            allow_partial_tools=allow_partial_tools,
            frame=getattr(args, "frame", "discovery"),
            sample_size=sample_size,
            sample_seed=seed,
            sample_id=_sample_id(),
        )
    if report.sample is not None and not dry_run:
        _persist_sample(args, report.sample)
    return report_outcome("ingest", report, ignored=ignored)


def _sample_args(args: argparse.Namespace) -> tuple[int | None, int | None, dict[str, str] | None]:
    """The ``--sample`` / ``--seed`` / ``--cell`` arguments, refusing an incomplete draw."""
    from calibrated_agent_evaluation.cli.errors import UsageError
    from calibrated_agent_evaluation.cli.scope import parse_cells

    sample_size = getattr(args, "sample", None)
    seed = getattr(args, "seed", None)
    cell = parse_cells(getattr(args, "cell", None))
    if sample_size is None and seed is not None:
        raise UsageError("--seed only applies to --sample; pass the sample size too")
    if sample_size is None and cell:
        raise UsageError("--cell narrows a --sample draw; pass the sample size too")
    if sample_size is not None and seed is None:
        raise UsageError(
            "--sample needs --seed so the draw can be repeated and audited later; pick any "
            "integer and record it, e.g. --sample 100 --seed 7"
        )
    if sample_size is not None and sample_size < 1:
        raise UsageError(f"--sample needs at least one trace, got {sample_size}")
    return sample_size, seed, cell


def _sample_id() -> str:
    import secrets
    from datetime import UTC, datetime

    stamp = datetime.now(tz=UTC).strftime("%Y%m%dT%H%M%SZ")
    return f"sample-{stamp}-{secrets.token_hex(3)}"


def _persist_sample(args: argparse.Namespace, record: object) -> None:
    """Append the sample record beside the other stage records under the local root."""
    from calibrated_agent_evaluation.cli.backends import local_root
    from calibrated_agent_evaluation.cli.records import append_records

    append_records(local_root(args), "sample", [record])


def _sample_note(report: IngestReport) -> str:
    """The plain-language sampling line: what was drawn, from how many, with which seed."""
    sample = report.sample
    if sample is None:
        return ""
    stratum = ", ".join(f"{k}={v}" for k, v in sorted(sample.cell.items()))
    scope = f" in cell {stratum}" if stratum else ""
    excluded = (
        f"; {sample.excluded_unknown_cell} candidate(s) left out of the population, "
        "their taxonomy cell is unknown"
        if sample.excluded_unknown_cell
        else ""
    )
    return (
        f"\n  sampled {sample.sample_size} of {sample.population_size} trace(s){scope} "
        f"with seed {sample.provenance.seed}, tagged {sample.frame} "
        f"(record {sample.sample_id}){excluded}"
    )


def _refuse_unsupported_agentcore_flags(args: argparse.Namespace) -> None:
    """Reject flags the AgentCore source does not honor, loudly, before any AWS call.

    The AgentCore source pulls the whole window read-only into the ``--backend`` store. The file
    and Langfuse paths accept flags this path silently ignored (and mainline's own refusal
    messages tell users to pass ``--overwrite``, which would no-op here), so each is rejected
    rather than dropped.
    """
    from calibrated_agent_evaluation.cli.errors import UsageError

    unsupported: list[str] = []
    if getattr(args, "paths", None):
        unsupported.append("positional path(s)")
    if getattr(args, "from_langfuse", None):
        unsupported.append("--from-langfuse")
    if getattr(args, "overwrite", False):
        unsupported.append("--overwrite")
    if getattr(args, "allow_partial_tools", False):
        unsupported.append("--allow-partial-tools")
    if getattr(args, "sample", None) is not None:
        unsupported.append("--sample")
    if getattr(args, "seed", None) is not None:
        unsupported.append("--seed")
    if getattr(args, "frame", "discovery") != "discovery":
        unsupported.append("--frame")
    if unsupported:
        raise UsageError(
            "calibrated-agent-evaluation ingest --source agentcore does not support "
            + ", ".join(unsupported)
            + "; it pulls the whole window read-only into the --backend store. Remove the "
            "unsupported flag(s) and re-run"
        )


def _ingest_agentcore(
    args: argparse.Namespace, dest: TraceStoreBackend, *, dry_run: bool, require_owned: bool
) -> StageOutcome:
    """Pull AgentCore telemetry into the ``--backend`` store (local or Langfuse).

    The AgentCore account is a read-only source; the destination stays the configured local or
    Langfuse store. Needs the ``[agentcore]`` extra; an absent extra surfaces as the typed
    missing-extra error when the AWS clients are built.
    """
    import dataclasses
    import os

    import calibrated_agent_evaluation.agentcore as agentcore
    from calibrated_agent_evaluation.cli.errors import UsageError

    runtime = getattr(args, "runtime", None)
    since = getattr(args, "since", None)
    until = getattr(args, "until", None)
    if not runtime:
        raise UsageError(
            "calibrated-agent-evaluation ingest --source agentcore needs --runtime: the runtime's "
            "OpenTelemetry "
            "service.name, which for the default endpoint is <agent>.DEFAULT, not the runtime id"
        )
    if not (since and until):
        raise UsageError(
            "calibrated-agent-evaluation ingest --source agentcore needs --since and --until "
            "(ISO-8601 window bounds)"
        )
    _refuse_unsupported_agentcore_flags(args)
    config = agentcore.AgentCoreSourceConfig(
        runtime=runtime,
        window=agentcore.TimeWindow(start=since, end=until),
        omni=getattr(args, "omni", False),
        rekey_sessions=getattr(args, "rekey_sessions", False),
        with_scores=getattr(args, "with_scores", False),
    )
    clients = agentcore.AwsClients.from_profile(
        region=args.region,
        profile=args.profile,
        handlers=[agentcore.BilledCallBudget.from_env(os.environ)],
    )
    with clients:
        source = agentcore.agentcore_source(clients, config)
        report = agentcore.ingest_agentcore(
            source,
            backend=dest,
            dry_run=dry_run,
            require_owned=require_owned,
            with_scores=config.with_scores,
        )
    outcome = report_outcome("ingest", report)
    if source.rekeyed:
        lines = [
            f"  [re-keyed] {old} → runtime session {new} (mixed-session trace)"
            for old, new in source.rekeyed
        ]
        outcome = dataclasses.replace(outcome, summary=outcome.summary + "\n" + "\n".join(lines))
    return outcome


def _require_project(source: object, project: str) -> None:
    """Refuse unless the ``LANGFUSE_*`` credentials address the project ``--from-langfuse`` names.

    Langfuse API keys are project-scoped and its public API has no project selector, so the key pair
    alone decides which project is read. The value used to be a truthiness check, and a wrong name
    silently pulled whatever project the environment's keys belonged to. It is now checked against
    the key's project, by id or by name.
    """
    from calibrated_agent_evaluation.cli.errors import CliError

    identify = getattr(source, "project", None)
    if identify is None:
        return
    project_id, name = identify()
    if project not in (project_id, name):
        raise CliError(
            f"--from-langfuse names project {project!r}, but the LANGFUSE_* keys belong to project "
            f"{name!r} (id {project_id!r}); Langfuse keys are project-scoped, so point "
            "LANGFUSE_PUBLIC_KEY / LANGFUSE_SECRET_KEY at that project's keys"
        )


def report_outcome(
    stage: str, report: IngestReport, *, ignored: list[Path] | None = None
) -> StageOutcome:
    """Render an ingest report to a StageOutcome (for `ingest` and `capture`)."""
    verb = "would ingest" if report.dry_run else "ingested"
    # The probed count rides on the ingested figure rather than beside it, because that is what it
    # qualifies: those traces DID ingest. Shown only when non-zero, for the same reason the
    # contentless count is (see `_render`); a field that is always there stops being read. Without
    # it, a batch that degraded end to end reported "N ingested, 0 refused, 0 skipped" and exit 0,
    # and the only trace of the degradation was a logger.warning nobody reading a report sees.
    probed = f" ({report.probed} probed)" if report.probed else ""
    header = (
        f"calibrated-agent-evaluation {stage}: {report.ingested} {verb}{probed}, "
        f"{report.refused} refused, {report.skipped} skipped"
        f"{_sample_note(report)}"
    )
    lines = [_render(result) for result in report.results]
    # Ignored files do not change the exit code, but each is named: a file left out of a batch is
    # only acceptable when the reader can see it was left out.
    lines.extend(
        f"  [ignored] {path}: JSON with no resourceSpans, not a trace" for path in ignored or []
    )
    body = "\n".join(lines)
    return StageOutcome(
        stage=stage, exit_code=report.exit_code, summary=f"{header}\n{body}" if body else header
    )


def _render(r: IngestResult) -> str:
    """One report line per source: a check for ingested, the reason for skipped/refused."""
    if r.status == "ingested":
        # The contentless-node count is appended only when non-zero. Printing "0
        # contentless" on every healthy line would train the reader to skip the field, which is
        # how a reported number becomes an unread one; a figure that shows up only when something
        # is missing keeps its signal. The count is recorded unconditionally on the IngestResult,
        # so a programmatic caller never depends on this rendering choice.
        empty = (
            f", {r.contentless_node_count} of {r.span_count} node(s) with no content"
            if r.contentless_node_count
            else ""
        )
        unrecovered = (
            f", {r.unrecovered_tool_spans} source tool span(s) not recovered as tool calls"
            if r.unrecovered_tool_spans
            else ""
        )
        return (
            f"  [ok] {r.source} → {r.trace_id} "
            f"({r.adapter}, {r.span_count} span(s), "
            f"{r.tool_call_count} tool call(s), {r.ownership}{empty}{unrecovered})"
        )
    tag = "REFUSED" if r.status == "refused" else "skipped"
    return f"  [{tag}] {r.source}: {r.reason}"
