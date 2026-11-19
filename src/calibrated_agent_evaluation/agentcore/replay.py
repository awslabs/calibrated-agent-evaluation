# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""The paired re-test's checks: one held-out slice, replayed under both bundle versions.

Vendor-free. Before the improve evaluation runs, :func:`check_replay` proves the two replay
stores are one input set run under the two arms' bundles, held out from the change's development
traces and unused by any earlier re-test, and, for judge verdicts, scored by judges
``calibrated-agent-evaluation judge`` would admit.
Inputs are compared by their prompt text, so a trace the ingest normalized from a user message
and one an executor sent match when they ask the same question.
"""

from __future__ import annotations

import json
import tempfile
from collections.abc import Callable, Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import TYPE_CHECKING, Protocol

from calibrated_agent_evaluation.agentcore.audit_types import (
    BundleVersionRef,
    RecommendationClaimRecord,
    RunProvenance,
    canonical_json,
    sha256_text,
)
from calibrated_agent_evaluation.agentcore.errors import (
    AgentCoreDevelopmentInputMissing,
    AgentCoreDevelopmentSourceChanged,
    AgentCoreDevelopmentSourceUnread,
    AgentCoreError,
    AgentCoreNotReplayable,
    AgentCoreReplayProvenanceError,
    AgentCoreSliceReused,
    AgentCoreUncertifiedJudge,
)

if TYPE_CHECKING:
    from calibrated_agent_evaluation.agentcore.audit_types import AuditRecord
    from calibrated_agent_evaluation.backends.base import SourceTrace
    from calibrated_agent_evaluation.bias.record import BiasBatteryRecord
    from calibrated_agent_evaluation.calibrate.record import CalibrationRecord
    from calibrated_agent_evaluation.improve.report import ImproveRecord
    from calibrated_agent_evaluation.schema.types import Label, Trace

__all__ = [
    "PROMPT_KEY",
    "DevelopmentSet",
    "OnlineEvalConfig",
    "ProvenanceStore",
    "ReplayCheck",
    "RunProvenanceSource",
    "RuntimeRunManifest",
    "WindowRead",
    "check_replay",
    "confirmation_slice_id",
    "prompt_digest",
    "prompt_text",
    "provenance_digest",
    "recommendation_development",
    "replay_provenance",
    "root_development",
    "unguarded_modes",
]

PROMPT_KEY = "single-string-input/1"
_DEFAULT_SESSION_TIMEOUT_MINUTES = 15
_MARGIN_MINUTES = 60
_JUDGE_KIND = "llm-judge"


def prompt_text(trace: Trace) -> str | None:
    """The trace's prompt: its input's only value when the input has one entry, a string."""
    values = list(trace.input.values()) if isinstance(trace.input, Mapping) else []
    if len(values) == 1 and isinstance(values[0], str):
        return values[0]
    return None


def prompt_digest(text: str) -> str:
    """The SHA-256 of the prompt text's UTF-8 bytes, with no normalization."""
    return sha256_text(text)


def confirmation_slice_id(trace_ids: Sequence[str]) -> str:
    """The improve evaluation's id for a confirmation slice: a digest of its sorted trace ids."""
    from calibrated_agent_evaluation.improve.confirm import slice_id

    return slice_id("confirm", trace_ids)


@dataclass(frozen=True)
class DevelopmentSet:
    """The traces a change was developed on, keyed by trace id with their prompt digests."""

    source: str
    traces: Mapping[str, Trace]
    window_start: str | None = None
    window_end: str | None = None
    margin_minutes: int | None = None
    log_groups: tuple[str, ...] = ()
    groups_read: tuple[str, ...] = ()
    updated_at: str | None = None

    @property
    def trace_ids(self) -> tuple[str, ...]:
        """The set's trace ids, sorted."""
        return tuple(sorted(self.traces))


@dataclass(frozen=True)
class OnlineEvalConfig:
    """What a development window depends on in an online evaluation configuration."""

    config_arn: str
    service_name: str
    log_group_names: tuple[str, ...]
    log_group_name_prefixes: tuple[str, ...]
    session_timeout_minutes: int | None
    updated_at: str

    def __post_init__(self) -> None:
        # The development window is checked against this time, so a configuration whose update
        # time is missing or has no offset is refused here rather than failing as a bare ValueError.
        try:
            parsed = datetime.fromisoformat(self.updated_at)
        except (TypeError, ValueError):
            parsed = None
        if parsed is None or parsed.tzinfo is None:
            raise AgentCoreError(
                f"online evaluation {self.config_arn} gives its updatedAt as "
                f"{self.updated_at!r}, not a timezone-aware ISO-8601 time, so the development "
                "window can't be checked against it; read the configuration again"
            )


class WindowRead(Protocol):
    """A trace source over one window that says, after its pull, which log groups it covered."""

    def pull(self) -> Iterator[SourceTrace]:
        """Yield every trace in the window."""
        ...

    def groups_read(self) -> tuple[str, ...]:
        """The log groups the pull covered, once it has ended."""
        ...


class RunProvenanceSource(Protocol):
    """Where a replay root's per-trace bundle versions come from."""

    label: str

    def provenance(self) -> dict[str, RunProvenance]:
        """Each replayed trace's provenance, by trace id."""
        ...


def _traces_of(root: Path) -> dict[str, Trace]:
    from calibrated_agent_evaluation.backends.local import LocalBackend

    if not (root / "store.sqlite3").exists():
        return {}
    with LocalBackend(root=root) as backend:
        return {t.trace_id: t for t in backend.query()}


def _labels_of(root: Path) -> dict[str, list[Label]]:
    from calibrated_agent_evaluation.backends.local import LocalBackend
    from calibrated_agent_evaluation.schema.serde import latest_verdicts

    with LocalBackend(root=root) as backend:
        return {
            t.trace_id: latest_verdicts(backend.get_scores(t.trace_id)) for t in backend.query()
        }


def root_development(root: str | Path) -> DevelopmentSet:
    """Every trace of the store at ``root``, as a change's development set.

    Raises:
        AgentCoreDevelopmentInputMissing: The store holds no trace.
    """
    path = Path(root)
    traces = _traces_of(path)
    if not traces:
        raise AgentCoreDevelopmentInputMissing(
            f"the development store {path} holds no trace, so nothing can show the confirmation "
            "slice is held out from it"
        )
    return DevelopmentSet(source=f"root:{path}", traces=traces)


class ProvenanceStore:
    """The run provenance a replay root holds under ``records/agentcore-run-provenance``."""

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)
        self.label = str(self.root)

    def provenance(self) -> dict[str, RunProvenance]:
        """The root's stored run provenance, by trace id; empty when it holds none."""
        from calibrated_agent_evaluation.agentcore.audit_types import load_run_provenance

        directory = self.root / "records" / "agentcore-run-provenance"
        if not directory.exists():
            return {}
        return _by_trace(load_run_provenance(directory, fmt="json"), self.label)


def replay_provenance(root: str | Path) -> RunProvenanceSource:
    """A replay root's run provenance: its stored records, else the executor's run manifest in it.

    The AgentCore Runtime executor, run through
    ``calibrated-agent-evaluation generate --run --executor agentcore`` without ``--log-dir``,
    writes its run manifest to ``<root>/records/agentcore``. A root that holds provenance records
    uses those; one with neither gets the empty store, which the bundle check refuses naming the
    root.
    """
    from calibrated_agent_evaluation.agentcore.run_manifest import manifest_path

    stored = ProvenanceStore(root)
    log_dir = Path(root) / "records" / "agentcore"
    if (
        not (stored.root / "records" / "agentcore-run-provenance").exists()
        and manifest_path(log_dir).exists()
    ):
        return RuntimeRunManifest(log_dir)
    return stored


class RuntimeRunManifest:
    """The run provenance the AgentCore Runtime executor's run manifest records."""

    def __init__(self, log_dir: str | Path) -> None:
        self.log_dir = Path(log_dir)
        self.label = str(self.log_dir)

    def provenance(self) -> dict[str, RunProvenance]:
        """One provenance record per run, from the bundle each run named.

        Raises:
            AgentCoreReplayProvenanceError: A run named no bundle, so its arm is unknown, or two
                runs of one trace id named different bundle versions.
        """
        from calibrated_agent_evaluation.agentcore.run_manifest import RUN_RECORD_STORE

        records: list[RunProvenance] = []
        for run in RUN_RECORD_STORE.load(self.log_dir, "json"):
            if not run.config_bundle_arn or not run.config_bundle_version:
                raise AgentCoreReplayProvenanceError(
                    f"run {run.trace_id!r} in {self.log_dir} named no configuration bundle, so "
                    "which arm it replayed is unknown"
                )
            records.append(
                RunProvenance(
                    trace_id=run.trace_id,
                    bundle_arn=run.config_bundle_arn,
                    bundle_version=run.config_bundle_version,
                    runtime_arn=run.runtime_arn,
                    qualifier=run.qualifier,
                )
            )
        return _by_trace(records, self.label)


def _by_trace(records: Sequence[RunProvenance], where: str) -> dict[str, RunProvenance]:
    """The provenance by trace id; a trace id run again under the same bundle keeps the last run.

    Raises:
        AgentCoreReplayProvenanceError: One trace id names two bundle versions, so which arm its
            stored trace ran is unknown.
    """
    out: dict[str, RunProvenance] = {}
    for record in records:
        earlier = out.get(record.trace_id)
        if earlier is not None and (earlier.bundle_arn, earlier.bundle_version) != (
            record.bundle_arn,
            record.bundle_version,
        ):
            raise AgentCoreReplayProvenanceError(
                f"trace {record.trace_id!r} in {where} was run under bundle version "
                f"{earlier.bundle_version!r} and again under {record.bundle_version!r}, so which "
                "arm its stored trace ran is unknown; replay each arm into a root of its own"
            )
        out[record.trace_id] = record
    return out


def provenance_digest(records: Mapping[str, RunProvenance]) -> str:
    """The digest of a root's run provenance, sorted by trace id."""
    docs = [records[t].model_dump(mode="json") for t in sorted(records)]
    return sha256_text(canonical_json(docs))


@dataclass(frozen=True)
class ReplayCheck:
    """A replay that passed every check: what the audit record keeps of it."""

    trace_ids: tuple[str, ...]
    confirmation_digests: tuple[str, ...]
    slice_id: str
    before_provenance_sha256: str
    after_provenance_sha256: str
    raters: dict[str, str] = field(default_factory=dict)


def _check_provenance(
    traces: Mapping[str, Trace], source: RunProvenanceSource, arm: BundleVersionRef, store: Path
) -> dict[str, RunProvenance]:
    records = source.provenance()
    for trace_id in sorted(traces):
        record = records.get(trace_id)
        if record is None:
            raise AgentCoreReplayProvenanceError(
                f"trace {trace_id!r} in {store} has no run provenance, so it can't be shown to "
                f"have run under bundle version {arm.version_id}"
            )
        if record.bundle_arn != arm.bundle_arn or record.bundle_version != arm.version_id:
            raise AgentCoreReplayProvenanceError(
                f"trace {trace_id!r} in {store} ran under bundle version {record.bundle_version}, "
                f"not this arm's {arm.version_id}"
            )
    return records


def _digests(traces: Mapping[str, Trace], where: str) -> dict[str, str]:
    missing = {t: tr for t, tr in traces.items() if prompt_text(tr) is None}
    if missing:
        detail = "; ".join(
            f"{t} (input keys: {', '.join(sorted(tr.input)) or 'none'})"
            for t, tr in sorted(missing.items())
        )
        raise AgentCoreReplayProvenanceError(
            f"{len(missing)} trace(s) in {where} have no single prompt text to compare: {detail}"
        )
    return {t: prompt_digest(str(prompt_text(tr))) for t, tr in traces.items()}


def _newest(
    records: Sequence[CalibrationRecord] | Sequence[BiasBatteryRecord],
    judge_id: str,
    fingerprint: str,
) -> CalibrationRecord | BiasBatteryRecord | None:
    mine = [
        (i, r)
        for i, r in enumerate(records)
        if r.judge_id == judge_id and r.judge_fingerprint == fingerprint
    ]
    if not mine:
        return None
    return max(mine, key=lambda ir: (ir[1].provenance.run_at, ir[0]))[1]


def _check_judges(
    stores: Sequence[tuple[Path, Mapping[str, Sequence[Label]]]],
    calibration: Sequence[CalibrationRecord],
    bias: Sequence[BiasBatteryRecord],
) -> dict[str, str]:
    """The one certified rater of every judged failure mode, or a typed refusal."""
    raters: dict[str, dict[str, list[Path]]] = {}
    for store, labels in stores:
        for trace_id in sorted(labels):
            for label in labels[trace_id]:
                if label.span_id is not None or label.kind != _JUDGE_KIND or not label.failure_mode:
                    continue
                rater = label.rater_id
                judge, _, fingerprint = (rater or "").rpartition("@")
                if not judge or not fingerprint:
                    raise AgentCoreUncertifiedJudge(
                        f"trace {trace_id!r} in {store} has a {label.failure_mode!r} judge verdict "
                        f"whose rater {rater or 'none'!r} names no judge@fingerprint, so the judge "
                        "behind it can't be checked"
                    )
                stores_of = raters.setdefault(label.failure_mode, {}).setdefault(str(rater), [])
                if store not in stores_of:
                    stores_of.append(store)
    for mode in sorted(raters):
        if len(raters[mode]) > 1:
            listed = "; ".join(
                f"{r} in {', '.join(str(s) for s in stores_of)}"
                for r, stores_of in sorted(raters[mode].items())
            )
            raise AgentCoreUncertifiedJudge(
                f"failure mode {mode!r} is judged by {len(raters[mode])} raters across the stores "
                f"({listed}); one judge must score both stores, so the comparison measures the "
                "change and not a change of judge"
            )
    out: dict[str, str] = {}
    for mode in sorted(raters):
        [(rater, stores_of)] = raters[mode].items()
        judge, _, fingerprint = rater.rpartition("@")
        where = f"failure mode {mode!r}, rater {rater}, in {stores_of[0]}"
        for kind, records, wanted, field_name in (
            ("calibration", calibration, "CERTIFIED", "result"),
            ("bias", bias, "CLEAR", "verdict"),
        ):
            newest = _newest(records, judge, fingerprint)
            if newest is None:
                carried = sorted({str(r.judge_fingerprint) for r in records if r.judge_id == judge})
                raise AgentCoreUncertifiedJudge(
                    f"{where}: no {kind} record for this judge and fingerprint"
                    + (f" (its records carry {', '.join(carried)})" if carried else "")
                    + "; run `calibrated-agent-evaluation "
                    + f"{'calibrate' if kind == 'calibration' else 'bias'}` first"
                )
            value = getattr(newest, field_name)
            if value != wanted:
                raise AgentCoreUncertifiedJudge(
                    f"{where}: its newest {kind} record is {value}, not {wanted}"
                )
        out[mode] = rater
    return out


def check_replay(
    before_root: str | Path,
    after_root: str | Path,
    *,
    before_arm: BundleVersionRef,
    after_arm: BundleVersionRef,
    before_provenance: RunProvenanceSource,
    after_provenance: RunProvenanceSource,
    development: DevelopmentSet,
    prior_audits: Sequence[AuditRecord],
    prior_improves: Sequence[ImproveRecord],
    verdict_source: str,
    judge_records: tuple[Sequence[CalibrationRecord], Sequence[BiasBatteryRecord]] = ((), ()),
) -> ReplayCheck:
    """Check, in order, that two replay stores can decide a change, before it is evaluated.

    1. Every trace in both stores has run provenance naming its arm's bundle version.
    2. Every development trace is stored (checked as the development set is built).
    3. Every development and confirmation trace has one prompt text.
    4. One trace id names one prompt across the two stores.
    5. No confirmation prompt is a development prompt.
    6. No confirmation prompt was in an earlier re-test, and no earlier improve evaluation used
       this slice.
    7. With judge verdicts, each judged mode has one rater across both stores, with a current
       ``CERTIFIED`` calibration record and ``CLEAR`` bias record.

    Raises:
        AgentCoreReplayProvenanceError: Checks 1, 3 or 4.
        ConfirmationContaminationError: Check 5 (the improve evaluation's error).
        AgentCoreSliceReused: Check 6.
        AgentCoreUncertifiedJudge: Check 7.
    """
    from calibrated_agent_evaluation.improve import ConfirmationContaminationError

    before_path, after_path = Path(before_root), Path(after_root)
    before, after = _traces_of(before_path), _traces_of(after_path)
    before_records = _check_provenance(before, before_provenance, before_arm, before_path)
    after_records = _check_provenance(after, after_provenance, after_arm, after_path)
    dev_digests = _digests(development.traces, f"the development set ({development.source})")
    before_digests = _digests(before, str(before_path))
    after_digests = _digests(after, str(after_path))
    for trace_id in sorted(set(before_digests) & set(after_digests)):
        if before_digests[trace_id] != after_digests[trace_id]:
            raise AgentCoreReplayProvenanceError(
                f"trace {trace_id!r} asks a different prompt in the two replays (digests "
                f"{before_digests[trace_id][:12]} and {after_digests[trace_id][:12]}), so its two "
                "runs are not one input"
            )
    confirmation = {**after_digests, **before_digests}
    by_digest = {d: t for t, d in dev_digests.items()}
    shared = sorted((t, by_digest[d]) for t, d in confirmation.items() if d in by_digest)
    if shared:
        pairs = ", ".join(f"{c} (development {d})" for c, d in shared)
        raise ConfirmationContaminationError(
            f"{len(shared)} confirmation trace(s) ask a question the change was developed on: "
            f"{pairs}; the slice is not held out"
        )
    digests = set(confirmation.values())
    trace_ids = tuple(sorted(confirmation))
    slice_id = confirmation_slice_id(trace_ids)
    reused = [
        f"{a.audit_id} (by prompt digest)"
        for a in prior_audits
        if digests & set(a.confirmation_prompt_digests)
    ] + [
        f"{r.improve_id} (by slice id {slice_id})"
        for r in prior_improves
        if r.confirmation_slice_id == slice_id
    ]
    if reused:
        raise AgentCoreSliceReused(
            f"this confirmation slice already decided another change: {', '.join(reused)}; a "
            "held-out slice decides one change, so replay a fresh one"
        )
    raters: dict[str, str] = {}
    if verdict_source == "judge":
        calibration, bias = judge_records
        raters = _check_judges(
            [(before_path, _labels_of(before_path)), (after_path, _labels_of(after_path))],
            calibration,
            bias,
        )
    return ReplayCheck(
        trace_ids=trace_ids,
        confirmation_digests=tuple(sorted(digests)),
        slice_id=slice_id,
        before_provenance_sha256=provenance_digest({t: before_records[t] for t in before}),
        after_provenance_sha256=provenance_digest({t: after_records[t] for t in after}),
        raters=raters,
    )


def unguarded_modes(
    before: str | Path,
    after: str | Path,
    development: str | Path | None,
    *,
    target: str,
    panel_modes: Sequence[str],
) -> tuple[str, ...]:
    """The failure modes a label in the three stores names that no slice guards, sorted.

    The target and every guard slice are left out.
    """
    named: set[str] = set()
    for root in (before, after, development):
        if root is None:
            continue
        for labels in _labels_of(Path(root)).values():
            named.update(lb.failure_mode for lb in labels if lb.failure_mode)
    return tuple(sorted(named - {target, *panel_modes}))


# --- a recommendation's development set ---------------------------------------------------


def _log_group_name(arn: str) -> str | None:
    if ":log-group:" not in arn:
        return None
    name = arn.split(":log-group:", 1)[1]
    return name[:-2] if name.endswith(":*") else name


def _normalize(source_trace: SourceTrace) -> Trace:
    from calibrated_agent_evaluation.ingest.detect import detect_source
    from calibrated_agent_evaluation.normalize import normalize

    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "trace.otlp.json"
        path.write_text(json.dumps(source_trace.otlp), encoding="utf-8")
        return normalize(
            detect_source(path, trace_id=source_trace.source_id, frame="production-representative")
        )


def _window_traces(read: WindowRead, *, window: str, service: str) -> dict[str, Trace]:
    traces: dict[str, Trace] = {}
    errors: list[str] = []
    for delivered in read.pull():
        if delivered.error is not None:
            errors.append(delivered.source_id)
            continue
        if delivered.trace is not None:
            traces[delivered.source_id] = delivered.trace
        elif delivered.otlp is not None:
            traces[delivered.source_id] = _normalize(delivered)
    if errors:
        raise AgentCoreDevelopmentInputMissing(
            f"development trace(s) {', '.join(sorted(errors))} in {window} came back without their "
            "input, so the confirmation slice can't be shown to be held out from them"
        )
    if not traces:
        raise AgentCoreDevelopmentInputMissing(
            f"the development window {window} of service {service!r} holds no trace"
        )
    return traces


def _check_groups(
    named: Sequence[str], prefixes: Sequence[str], read: WindowRead
) -> tuple[str, ...]:
    covered = set(read.groups_read())
    names, bad = [], []
    for entry in named:
        name = entry if entry.startswith("/") or ":" not in entry else _log_group_name(entry)
        if name is None:
            bad.append(entry)
        else:
            names.append(name)
    outside = sorted(n for n in names if n not in covered)
    if outside or prefixes or bad:
        parts = [f"group {g}" for g in outside] + [f"name prefix {p}" for p in prefixes]
        parts += [f"ARN {a} (no :log-group: part)" for a in bad]
        raise AgentCoreDevelopmentSourceUnread(
            f"the development read did not cover {', '.join(parts)}; it covered "
            f"{', '.join(sorted(covered)) or 'no group'}, so a trace the recommendation read may "
            "be missing from the development set"
        )
    return tuple(sorted(names))


def recommendation_development(
    record: RecommendationClaimRecord,
    *,
    online_config: OnlineEvalConfig | None,
    source_factory: Callable[[str, tuple[str, str]], WindowRead],
    development_root: str | Path | None,
) -> DevelopmentSet:
    """The traces a recommendation was developed on, read from its own agent-traces source.

    Session spans are looked up by trace id in ``development_root``; a batch evaluation's set is
    that root; a CloudWatch Logs source's window, and an online evaluation's window widened by
    its session timeout and an hour, are read through ``source_factory``.

    Raises:
        AgentCoreNotReplayable: A batch-evaluation source with no development root.
        AgentCoreDevelopmentSourceChanged: The online evaluation changed after the window began.
        AgentCoreDevelopmentSourceUnread: The read missed a log group the source names.
        AgentCoreDevelopmentInputMissing: A trace has no input, or the set holds none.
    """
    source = record.traces_source
    if source == "sessionSpans":
        root = Path(str(development_root))
        stored = _traces_of(root)
        wanted = record.traces_inline_trace_ids
        missing = [t for t in wanted if t not in stored]
        if missing:
            raise AgentCoreDevelopmentInputMissing(
                f"{len(missing)} of the recommendation's sessionSpans trace id(s) are not in "
                f"{root}: {', '.join(missing)}"
            )
        if not wanted:
            raise AgentCoreDevelopmentInputMissing(
                f"the recommendation's sessionSpans name no trace, so {root} gives no "
                "development set"
            )
        return DevelopmentSet(
            source=f"sessionSpans:{len(wanted)} ids in root:{root}",
            traces={t: stored[t] for t in wanted},
        )
    if source == "batchEvaluation":
        if development_root is None:
            raise AgentCoreNotReplayable(
                f"the recommendation read batch evaluation {record.traces_batch_evaluation_arn}, "
                "whose traces Calibrated Agent Evaluation can't list; pass --development-root "
                "with them"
            )
        return root_development(development_root)
    start, end = record.traces_window_start, record.traces_window_end
    if start is None or end is None:
        raise AgentCoreError(f"recommendation {record.recommendation_id!r} names no trace window")
    if source == "cloudwatchLogs":
        service = record.traces_service_name or ""
        read = source_factory(service, (start, end))
        window = f"window {start} to {end}"
        traces = _window_traces(read, window=window, service=service)
        groups = _check_groups(record.traces_log_group_arns, (), read)
        return DevelopmentSet(
            source=f"cloudwatchLogs:{service}@{start}/{end}",
            traces=traces,
            window_start=start,
            window_end=end,
            log_groups=groups,
            groups_read=tuple(sorted(read.groups_read())),
        )
    if online_config is None:
        raise AgentCoreError(
            f"recommendation {record.recommendation_id!r} reads an online evaluation whose "
            "configuration was not given"
        )
    timeout = online_config.session_timeout_minutes or _DEFAULT_SESSION_TIMEOUT_MINUTES
    margin = timeout + _MARGIN_MINUTES
    widened = (datetime.fromisoformat(start) - timedelta(minutes=margin)).isoformat()
    if datetime.fromisoformat(online_config.updated_at) > datetime.fromisoformat(widened):
        raise AgentCoreDevelopmentSourceChanged(
            f"online evaluation {online_config.config_arn} was updated at "
            f"{online_config.updated_at}, after the development window's start {widened} "
            f"(its startTime {start} less {margin} minutes), so the traces it chose may have "
            "followed another configuration"
        )
    service = online_config.service_name
    read = source_factory(service, (widened, end))
    window = f"window {widened} to {end}"
    traces = _window_traces(read, window=window, service=service)
    named = [*online_config.log_group_names]
    groups = _check_groups(named, online_config.log_group_name_prefixes, read)
    return DevelopmentSet(
        source=f"onlineEvaluation:{service}@{widened}/{end}",
        traces=traces,
        window_start=widened,
        window_end=end,
        margin_minutes=margin,
        log_groups=groups,
        groups_read=tuple(sorted(read.groups_read())),
        updated_at=online_config.updated_at,
    )
