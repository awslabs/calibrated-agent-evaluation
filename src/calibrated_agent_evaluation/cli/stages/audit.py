# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""``audit``: AgentCore's A/B and recommendation claims, each trace's arm, and a re-test.

Every AWS call goes through clients whose read-only guard allows only the audit's reads and the
trace reads the AgentCore ingest's readers make. AgentCore's figures are stored and printed as a
vendor claim; the re-test's verdict is `calibrated-agent-evaluation improve`'s, run on a held-out
replay of both bundle versions.
"""

from __future__ import annotations

import argparse
import os
from collections.abc import Sequence
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

from calibrated_agent_evaluation.cli.dispatch import EXIT_OK, StageOutcome
from calibrated_agent_evaluation.cli.errors import CliError, UsageError

if TYPE_CHECKING:
    from calibrated_agent_evaluation.agentcore.arms import ArmRead
    from calibrated_agent_evaluation.agentcore.audit_types import (
        AbTestClaimRecord,
        ArmAssignment,
        AuditRecord,
        BundleVersionRef,
        RecommendationClaimRecord,
    )
    from calibrated_agent_evaluation.agentcore.replay import DevelopmentSet

__all__ = [
    "audit_handler",
    "require_agentcore",
    "resolve_arms_read",
    "resolve_audit_record",
    "resolve_claim_record",
    "resolve_recommendation_record",
]

_SUBJECT = {"ab_test_id", "recommendation_id"}
_FLAGS: dict[str, set[str]] = {
    "abtest": {"ab_test_id", "region", "profile"},
    "recommendation": {"recommendation_id", "region", "profile"},
    "arms": {
        "ab_test_id",
        "claim_record",
        "runtime",
        "since",
        "until",
        "omni",
        "rekey_sessions",
        "applied_bundle_attribute",
        "applied_bundle_require",
        "region",
        "profile",
    },
    "retest": {
        "metric",
        *_SUBJECT,
        "arms_read",
        "claim_record",
        "before_root",
        "after_root",
        "development_root",
        "verdicts",
        "region",
        "profile",
    },
    "report": {*_SUBJECT, "audit_id", "claim_record", "arms_read"},
}
_REQUIRED: dict[str, tuple[str, ...]] = {
    "abtest": ("ab_test_id",),
    "recommendation": ("recommendation_id",),
    "arms": ("ab_test_id", "runtime", "since", "until"),
    "retest": ("metric", "before_root", "after_root"),
    "report": (),
}
_FLAG_NAMES = {
    "metric": "METRIC",
    "ab_test_id": "--ab-test-id",
    "recommendation_id": "--recommendation-id",
    "claim_record": "--claim-record",
    "arms_read": "--arms-read",
    "audit_id": "--audit-id",
    "runtime": "--runtime",
    "since": "--since",
    "until": "--until",
    "omni": "--omni",
    "rekey_sessions": "--rekey-sessions",
    "applied_bundle_attribute": "--applied-bundle-attribute",
    "applied_bundle_require": "--applied-bundle-require",
    "before_root": "--before-root",
    "after_root": "--after-root",
    "development_root": "--development-root",
    "verdicts": "--verdicts",
    "region": "--region",
    "profile": "--profile",
}


def audit_handler(args: argparse.Namespace) -> StageOutcome:
    """Run the ``calibrated-agent-evaluation audit`` action ``args.action`` names."""
    action = str(args.action)
    _check_flags(args, action)
    if action in _AWS_ACTIONS:
        require_agentcore(f"audit {action}")
    if action == "abtest":
        return _abtest(args)
    if action == "recommendation":
        return _recommendation(args)
    if action == "arms":
        return _arms(args)
    if action == "retest":
        return _retest(args)
    return _report(args)


#: The actions that read from AWS, through the ``[agentcore]`` extra's clients.
_AWS_ACTIONS = frozenset({"abtest", "recommendation", "arms"})


def require_agentcore(command: str) -> None:
    """Refuse ``command`` unless the ``[agentcore]`` extra is installed, naming its install command.

    Checked on the extra's own SDK, as :func:`~calibrated_agent_evaluation.cli.agentcore_route.
    require_agentcore_extra` does: the AWS SDK alone may come from another extra.

    Raises:
        AgentCoreExtraMissing: The extra is not installed.
    """
    from calibrated_agent_evaluation.agentcore.errors import AgentCoreExtraMissing
    from calibrated_agent_evaluation.cli.config import probe_extra

    if not probe_extra("bedrock_agentcore").available:
        raise AgentCoreExtraMissing(f"`calibrated-agent-evaluation {command}`")


def _given(args: argparse.Namespace, name: str) -> bool:
    return getattr(args, name, None) not in (None, False, [])


def _check_flags(args: argparse.Namespace, action: str) -> None:
    stray = sorted(
        _FLAG_NAMES[n] for n in _FLAG_NAMES if _given(args, n) and n not in _FLAGS[action]
    )
    if stray:
        raise UsageError(
            f"`calibrated-agent-evaluation audit {action}` does not take {', '.join(stray)}"
        )
    missing = [_FLAG_NAMES[n] for n in _REQUIRED[action] if getattr(args, n, None) in (None, "")]
    if missing:
        raise UsageError(f"`calibrated-agent-evaluation audit {action}` needs {', '.join(missing)}")
    if action in ("retest", "report"):
        subjects = [n for n in _SUBJECT if _given(args, n)]
        if len(subjects) != 1:
            raise UsageError(
                f"`calibrated-agent-evaluation audit {action}` needs exactly one of --ab-test-id "
                "and --recommendation-id"
            )
        if _given(args, "arms_read") and not _given(args, "ab_test_id"):
            raise UsageError(
                "--arms-read goes with --ab-test-id: a recommendation has no arms read"
            )


# --- choosing stored records ----------------------------------------------------------------


def _several(kind: str, subject: str, listed: str, flag: str) -> CliError:
    return CliError(f"several {kind} are stored for {subject}: {listed}. Name one with {flag}")


def resolve_claim_record(
    records: Sequence[AbTestClaimRecord], ab_test_id: str, claim_record: str | None
) -> AbTestClaimRecord:
    """The stored claim record of an A/B test an audit step works from.

    The one ``--claim-record`` names, else the only one stored for the test. Every
    ``calibrated-agent-evaluation audit abtest`` read stores a new one, so with several stored and
    none named the step refuses rather than pick one.

    Raises:
        CliError: None is stored for the test, the named id is not one of its records, or several
            are stored and none is named.
    """
    mine = [r for r in records if r.ab_test_id == ab_test_id]
    run_first = f"run `calibrated-agent-evaluation audit abtest --ab-test-id {ab_test_id}` first"
    if claim_record is not None:
        named = [r for r in mine if r.record_id == claim_record]
        if not named:
            raise CliError(
                f"no claim record {claim_record!r} is stored for A/B test {ab_test_id!r}; "
                f"{run_first}"
            )
        return named[0]
    if not mine:
        raise CliError(f"no claim record is stored for A/B test {ab_test_id!r}; {run_first}")
    if len(mine) > 1:
        listed = "; ".join(
            f"{r.record_id} (read {r.read_at}, {r.execution_status})"
            for r in sorted(mine, key=lambda r: r.read_at)
        )
        raise _several("claim records", f"A/B test {ab_test_id!r}", listed, "--claim-record")
    return mine[0]


def resolve_recommendation_record(
    records: Sequence[RecommendationClaimRecord], recommendation_id: str, claim_record: str | None
) -> RecommendationClaimRecord:
    """The stored claim record of a recommendation, chosen as :func:`resolve_claim_record` does.

    Raises:
        CliError: None is stored, the named id is not one of them, or several and none named.
    """
    mine = [r for r in records if r.recommendation_id == recommendation_id]
    run_first = (
        "run `calibrated-agent-evaluation audit recommendation "
        f"--recommendation-id {recommendation_id}` first"
    )
    if claim_record is not None:
        named = [r for r in mine if r.record_id == claim_record]
        if not named:
            raise CliError(
                f"no claim record {claim_record!r} is stored for recommendation "
                f"{recommendation_id!r}; {run_first}"
            )
        return named[0]
    if not mine:
        raise CliError(
            f"no claim record is stored for recommendation {recommendation_id!r}; {run_first}"
        )
    if len(mine) > 1:
        listed = "; ".join(
            f"{r.record_id} (read {r.read_at}, {r.status})"
            for r in sorted(mine, key=lambda r: r.read_at)
        )
        raise _several(
            "claim records", f"recommendation {recommendation_id!r}", listed, "--claim-record"
        )
    return mine[0]


def resolve_arms_read(
    records: Sequence[ArmAssignment], ab_test_arn: str, arms_read: str | None
) -> list[ArmAssignment] | None:
    """The stored arms read a re-test or report uses: the named one, the only one, or none.

    Raises:
        CliError: The named read is not stored for the test, or several are and none is named.
    """
    reads: dict[str, list[ArmAssignment]] = {}
    for a in records:
        if a.ab_test_arn == ab_test_arn:
            reads.setdefault(a.read_id, []).append(a)
    if arms_read is not None:
        if arms_read not in reads:
            raise CliError(f"no arms read {arms_read!r} is stored for {ab_test_arn}")
        return reads[arms_read]
    if not reads:
        return None
    if len(reads) > 1:
        listed = "; ".join(
            f"{rid} ({group[0].window_start} to {group[0].window_end})"
            for rid, group in sorted(reads.items())
        )
        raise _several("arms reads", ab_test_arn, listed, "--arms-read")
    return next(iter(reads.values()))


def resolve_audit_record(
    records: Sequence[AuditRecord], subject_arn: str, audit_id: str | None
) -> AuditRecord | None:
    """The stored audit record a report prints: the named one, the only one, or none.

    Raises:
        CliError: The named id is no audit record of the subject, or several are stored and none
            is named.
    """
    mine = [r for r in records if r.subject_arn == subject_arn]
    if audit_id is not None:
        named = [r for r in mine if r.audit_id == audit_id]
        if not named:
            raise CliError(f"no audit record {audit_id!r} is stored for {subject_arn}")
        return named[0]
    if len(mine) > 1:
        listed = "; ".join(f"{r.audit_id} (improve {r.improve_id})" for r in mine)
        raise _several("audit records", subject_arn, listed, "--audit-id")
    return mine[0] if mine else None


# --- reads -------------------------------------------------------------------------------------


def _root(args: argparse.Namespace) -> Path:
    from calibrated_agent_evaluation.cli.backends import local_root

    return local_root(args)


def _abtest(args: argparse.Namespace) -> StageOutcome:
    from calibrated_agent_evaluation.agentcore import optimization
    from calibrated_agent_evaluation.agentcore.audit_report import claim_lines
    from calibrated_agent_evaluation.cli.records import append_records

    with optimization.audit_clients(args.region, args.profile) as clients:
        record = optimization.read_ab_test(clients, str(args.ab_test_id))
    append_records(_root(args), "agentcore-abtest", [record])
    lines = claim_lines(record)
    return StageOutcome(
        stage="audit", exit_code=EXIT_OK, summary="\n".join(lines[:3]), detail="\n".join(lines)
    )


def _recommendation(args: argparse.Namespace) -> StageOutcome:
    from calibrated_agent_evaluation.agentcore import optimization
    from calibrated_agent_evaluation.agentcore.audit_report import recommendation_lines
    from calibrated_agent_evaluation.cli.records import append_records

    with optimization.audit_clients(args.region, args.profile) as clients:
        record = optimization.read_recommendation(clients, str(args.recommendation_id))
    append_records(_root(args), "agentcore-recommendation", [record])
    lines = recommendation_lines(record)
    return StageOutcome(
        stage="audit", exit_code=EXIT_OK, summary="\n".join(lines[:3]), detail="\n".join(lines)
    )


def _require_pairs(values: Sequence[str] | None) -> list[tuple[str, str]]:
    pairs: list[tuple[str, str]] = []
    for item in values or ():
        key, sep, value = item.partition("=")
        if not sep or not key or not value:
            raise UsageError(f"--applied-bundle-require takes KEY=VALUE, got {item!r}")
        pairs.append((key, value))
    keys = [k for k, _ in pairs]
    if len(keys) != len(set(keys)):
        raise UsageError("--applied-bundle-require names one key twice")
    return pairs


def _store_ids(root: Path) -> set[str] | None:
    """The trace ids the root's local trace store holds, or ``None`` when it holds none."""
    if not (root / "store.sqlite3").exists():
        return None
    from calibrated_agent_evaluation.backends.local import LocalBackend

    with LocalBackend(root) as backend:
        ids = {t.trace_id for t in backend.all_traces()}
    return ids or None


def _arms(args: argparse.Namespace) -> StageOutcome:
    from calibrated_agent_evaluation.agentcore import optimization
    from calibrated_agent_evaluation.agentcore.arms import RawTraceLog, merge_arms_read, read_arms
    from calibrated_agent_evaluation.agentcore.audit_report import arms_lines
    from calibrated_agent_evaluation.agentcore.readers import LogsInsightsReader, OmniReader
    from calibrated_agent_evaluation.agentcore.sources import (
        AgentCoreSourceConfig,
        AgentCoreTraceSource,
        TimeWindow,
    )
    from calibrated_agent_evaluation.cli.records import (
        append_records,
        load_records_of,
        records_lock,
    )

    root = _root(args)
    claim = resolve_claim_record(
        load_records_of(root, "agentcore-abtest"), str(args.ab_test_id), args.claim_record
    )
    require = _require_pairs(args.applied_bundle_require)
    config = AgentCoreSourceConfig(
        runtime=str(args.runtime),
        window=TimeWindow(start=str(args.since), end=str(args.until)),
        omni=bool(args.omni),
        rekey_sessions=bool(args.rekey_sessions),
    )
    versions = {v.name: v.bundle.version_id for v in claim.variants if v.bundle is not None}
    with optimization.audit_clients(args.region, args.profile) as clients:
        reader = (
            OmniReader(clients, config) if config.uses_omni else LogsInsightsReader(clients, config)
        )
        log = RawTraceLog(reader)
        source = AgentCoreTraceSource(log, config)
        read: ArmRead = read_arms(
            source,
            ab_test_arn=claim.ab_test_arn,
            runtime=config.runtime,
            window=config.window,
            omni=config.omni,
            rekey_sessions=config.rekey_sessions,
            rekey_collisions=(
                (lambda: log.collisions(source.rekeyed)) if config.rekey_sessions else None
            ),
            contentless=lambda: log.contentless(source.rekeyed),
            applied_bundle_attribute=args.applied_bundle_attribute,
            applied_bundle_require=require,
            bundle_versions=versions,
        )
    with records_lock(root, "agentcore-arms"):
        new = merge_arms_read(load_records_of(root, "agentcore-arms"), read)
        stored = append_records(root, "agentcore-arms", new, locked=True) if new else 0
    held = _store_ids(root)
    missing = (
        None
        if held is None
        else sorted(
            a.trace_id
            for a in read.assignments
            if a.status == "assigned" and a.trace_id not in held
        )
    )
    lines = arms_lines(
        read_id=read.read_id,
        counts=read.counts,
        stored=stored,
        omni=config.omni,
        rekey_sessions=config.rekey_sessions,
        missing_from_store=missing,
        arns_seen=[a for a in read.arns_seen if a != claim.ab_test_arn],
    )
    return StageOutcome(
        stage="audit", exit_code=EXIT_OK, summary="\n".join(lines[:1]), detail="\n".join(lines)
    )


# --- the re-test ------------------------------------------------------------------------------


def _order(frozen_at: str, label: str, when: str | None) -> dict[str, str]:
    if when is None:
        return {label: "not recorded"}
    before = datetime.fromisoformat(frozen_at) < datetime.fromisoformat(when)
    return {label: "before" if before else "after"}


def _prereg_entry(root: Path, metric: str) -> Any:
    from calibrated_agent_evaluation.cli.records import load_ledger

    entries = [e for e in load_ledger(root).entries if e.target_metric == metric]
    return entries[-1] if entries else None


def _ab_subject(
    args: argparse.Namespace, root: Path
) -> tuple[Any, BundleVersionRef, BundleVersionRef, list[Any] | None, DevelopmentSet]:
    from calibrated_agent_evaluation.agentcore.errors import AgentCoreNotReplayable
    from calibrated_agent_evaluation.agentcore.replay import root_development
    from calibrated_agent_evaluation.cli.records import load_records_of

    test_id = str(args.ab_test_id)
    if not args.development_root:
        raise UsageError(
            f"`calibrated-agent-evaluation audit retest --ab-test-id {test_id}` needs "
            "--development-root, the store of "
            "the traces the change was developed on, so the replay can be shown to be held out"
        )
    claim = resolve_claim_record(
        load_records_of(root, "agentcore-abtest"), test_id, args.claim_record
    )
    arms = {v.name: v for v in claim.variants}
    targets = [v for v in claim.variants if v.bundle is None]
    if targets:
        raise AgentCoreNotReplayable(
            f"A/B test {test_id!r} routes arm {targets[0].name} to gateway target "
            f"{targets[0].target_name!r}, not a bundle version, so there is no version to replay"
        )
    read = resolve_arms_read(
        load_records_of(root, "agentcore-arms"), claim.ab_test_arn, args.arms_read
    )
    development = root_development(args.development_root)
    before, after = arms["C"].bundle, arms["T1"].bundle
    if before is None or after is None:  # both checked above; keeps the types exact
        raise AgentCoreNotReplayable(f"A/B test {test_id!r} has an arm with no bundle version")
    return claim, before, after, read, development


def _recommendation_subject(
    args: argparse.Namespace, root: Path, metric: str
) -> tuple[Any, BundleVersionRef, BundleVersionRef, DevelopmentSet]:
    from calibrated_agent_evaluation.agentcore.errors import AgentCoreNotReplayable
    from calibrated_agent_evaluation.cli.records import load_records_of
    from calibrated_agent_evaluation.improve import ImproveError

    rec_id = str(args.recommendation_id)
    claim = resolve_recommendation_record(
        load_records_of(root, "agentcore-recommendation"), rec_id, args.claim_record
    )
    source = claim.traces_source
    if source == "sessionSpans" and not args.development_root:
        raise UsageError(
            f"`calibrated-agent-evaluation audit retest --recommendation-id {rec_id}` needs "
            "--development-root, the "
            "store holding the traces its sessionSpans name"
        )
    if source in ("cloudwatchLogs", "onlineEvaluation") and args.development_root:
        raise UsageError(
            f"--development-root is not taken for recommendation {rec_id!r}: its {source} source "
            "names its own window, and the development set is read from it"
        )
    if claim.status != "COMPLETED" or claim.error_code:
        reason = f"error {claim.error_code}" if claim.error_code else f"status {claim.status}"
        raise AgentCoreNotReplayable(
            f"recommendation {rec_id!r} has {reason}, so it produced nothing to replay"
        )
    if claim.input_bundle is None or claim.result_bundle is None:
        missing = "an inline-text input" if claim.input_bundle is None else "no result bundle"
        raise AgentCoreNotReplayable(
            f"recommendation {rec_id!r} has {missing}, so there is no bundle version to pin for "
            "both replays"
        )
    entry = _prereg_entry(root, metric)
    if entry is not None and datetime.fromisoformat(entry.frozen_at) >= datetime.fromisoformat(
        claim.created_at
    ):
        raise ImproveError(
            f"the pre-registration of {metric!r} was frozen at {entry.frozen_at}, not before the "
            f"recommendation was created at {claim.created_at}; register the target before "
            "starting the recommendation"
        )
    development = _recommendation_development(args, claim)
    return claim, claim.input_bundle, claim.result_bundle, development


def _recommendation_development(args: argparse.Namespace, claim: Any) -> DevelopmentSet:
    from calibrated_agent_evaluation.agentcore.replay import recommendation_development

    if claim.traces_source not in ("cloudwatchLogs", "onlineEvaluation"):
        return recommendation_development(
            claim,
            online_config=None,
            source_factory=_no_window,
            development_root=args.development_root,
        )
    from calibrated_agent_evaluation.agentcore import optimization

    with optimization.audit_clients(args.region, args.profile) as clients:
        online = (
            optimization.read_online_eval_config(
                clients, str(claim.traces_online_evaluation_config_arn)
            )
            if claim.traces_source == "onlineEvaluation"
            else None
        )
        return recommendation_development(
            claim,
            online_config=online,
            source_factory=lambda service, window: _window_read(clients, service, window),
            development_root=None,
        )


def _no_window(service: str, window: tuple[str, str]) -> Any:
    raise CliError(f"no window is read for this source (asked for {service} {window})")


class _WindowSource:
    """The AgentCore ingest's Logs Insights read of one window, with the log groups it covered."""

    def __init__(self, clients: Any, service: str, window: tuple[str, str]) -> None:
        from calibrated_agent_evaluation.agentcore.arms import RawTraceLog
        from calibrated_agent_evaluation.agentcore.readers import LogsInsightsReader
        from calibrated_agent_evaluation.agentcore.sources import (
            AgentCoreSourceConfig,
            AgentCoreTraceSource,
            TimeWindow,
        )

        config = AgentCoreSourceConfig(
            runtime=service,
            window=TimeWindow(start=window[0], end=window[1]),
            omni=False,
            rekey_sessions=True,
        )
        self._log = RawTraceLog(LogsInsightsReader(clients, config))
        self._source = AgentCoreTraceSource(self._log, config)

    def pull(self) -> Any:
        return self._source.pull()

    def groups_read(self) -> tuple[str, ...]:
        return self._log.groups_read()


def _window_read(clients: Any, service: str, window: tuple[str, str]) -> Any:
    return _WindowSource(clients, service, window)


def _retest(args: argparse.Namespace) -> StageOutcome:
    import secrets

    from calibrated_agent_evaluation.agentcore.replay import check_replay, replay_provenance
    from calibrated_agent_evaluation.cli.config import require_extras
    from calibrated_agent_evaluation.cli.records import load_records_of, records_dir
    from calibrated_agent_evaluation.cli.stages.improve import evaluate_store

    require_extras("scipy", "statsmodels")
    metric = str(args.metric)
    # One verdict source for every step, resolved the way
    # `calibrated-agent-evaluation improve --evaluate` does.
    verdict_source = str(getattr(args, "verdicts", None) or "judge")
    args.verdicts = verdict_source
    root = _root(args)
    read: list[Any] | None = None
    if getattr(args, "ab_test_id", None):
        claim, before_arm, after_arm, read, development = _ab_subject(args, root)
        subject, subject_arn = "ab-test", claim.ab_test_arn
    else:
        claim, before_arm, after_arm, development = _recommendation_subject(args, root, metric)
        subject, subject_arn = "recommendation", claim.recommendation_arn
    before_root, after_root = Path(str(args.before_root)), Path(str(args.after_root))
    check = check_replay(
        before_root,
        after_root,
        before_arm=before_arm,
        after_arm=after_arm,
        before_provenance=replay_provenance(before_root),
        after_provenance=replay_provenance(after_root),
        development=development,
        prior_audits=load_records_of(root, "agentcore-audit"),
        prior_improves=load_records_of(root, "improve"),
        verdict_source=verdict_source,
        judge_records=(
            (load_records_of(root, "calibration"), load_records_of(root, "bias"))
            if verdict_source == "judge"
            else ((), ())
        ),
    )
    audit_id = f"audit-{secrets.token_hex(6)}"
    improve_id = f"improve-{audit_id[6:]}"
    dev_dir = records_dir(root, "agentcore-audit") / "development"
    dev_dir.mkdir(parents=True, exist_ok=True)
    dev_file = dev_dir / f"{audit_id}.txt"
    dev_file.write_text("\n".join(development.trace_ids) + "\n", encoding="utf-8")
    if not os.access(records_dir(root, "agentcore-audit"), os.W_OK):
        dev_file.unlink(missing_ok=True)
        raise CliError(
            f"the audit records directory {records_dir(root, 'agentcore-audit')} is not writable, "
            "so the re-test was not run (its improve record would be stored without an audit "
            "record); make it writable and run the re-test again"
        )
    args.development = str(dev_file)
    args.improve_id = improve_id
    try:
        outcome = evaluate_store(args, metric)
    except BaseException:
        # The improve evaluation refused, so no improve record names this file; remove it.
        dev_file.unlink(missing_ok=True)
        raise
    try:
        return _record_audit(
            args,
            root,
            metric,
            outcome,
            improve_id,
            audit_id,
            dev_file,
            subject,
            subject_arn,
            claim,
            before_arm,
            after_arm,
            before_root,
            after_root,
            check,
            verdict_source,
            development,
            read,
        )
    except BaseException as exc:
        # The improve record is stored and its held-out slice now counts as decided; say so
        # rather than leave an improve record with no audit record and no word about it.
        raise CliError(
            f"the re-test's improve record {improve_id} is stored, but its audit record could not "
            f"be written ({type(exc).__name__}: {exc}); the held-out slice now counts as decided, "
            f"and the development ids it was checked against are in {dev_file}"
        ) from exc


def _record_audit(
    args: argparse.Namespace,
    root: Path,
    metric: str,
    outcome: StageOutcome,
    improve_id: str,
    audit_id: str,
    dev_file: Path,
    subject: Literal["ab-test", "recommendation"],
    subject_arn: str,
    claim: Any,
    before_arm: Any,
    after_arm: Any,
    before_root: Path,
    after_root: Path,
    check: Any,
    verdict_source: str,
    development: Any,
    read: Any,
) -> StageOutcome:
    """Build and store the audit record for a re-test whose improve record is stored."""
    from calibrated_agent_evaluation._provenance import capture_provenance
    from calibrated_agent_evaluation.agentcore.audit_report import read_summary
    from calibrated_agent_evaluation.agentcore.audit_types import AuditRecord
    from calibrated_agent_evaluation.agentcore.replay import unguarded_modes
    from calibrated_agent_evaluation.cli.records import append_records, load_records_of

    [improve] = [r for r in load_records_of(root, "improve") if r.improve_id == improve_id]
    panel_modes = [s.name for s in improve.panel.slices]
    raters = check.raters
    guard_modes = {m: raters.get(m, verdict_source) for m in panel_modes}
    summary = read_summary(read or [])
    frozen = improve.prereg.frozen_at
    if subject == "ab-test":
        order = {
            **_order(frozen, "ab_test_started_at", claim.started_at),
            **_order(frozen, "analysis_timestamp", claim.analysis_timestamp),
        }
    else:
        order = _order(frozen, "recommendation_created_at", claim.created_at)
    record = AuditRecord(
        audit_id=audit_id,
        subject=subject,
        subject_arn=subject_arn,
        claim_record_id=claim.record_id,
        improve_id=improve_id,
        target_metric=metric,
        before_arm=before_arm,
        after_arm=after_arm,
        before_root=str(before_root),
        after_root=str(after_root),
        before_provenance_sha256=check.before_provenance_sha256,
        after_provenance_sha256=check.after_provenance_sha256,
        confirmation_trace_ids=check.trace_ids,
        confirmation_prompt_digests=check.confirmation_digests,
        confirmation_slice_id=improve.confirmation_slice_id,
        development_source=development.source,
        development_window_start=development.window_start,
        development_window_end=development.window_end,
        development_margin_minutes=development.margin_minutes,
        development_source_updated_at=development.updated_at,
        development_digest_count=len(development.traces),
        development_log_groups=development.log_groups,
        development_groups_read=development.groups_read,
        arms_read_id=summary.read_id,
        prereg_frozen_at=frozen,
        prereg_order=order,
        arm_counts=summary.counts,
        applied_bundle_checked=summary.applied_bundle_checked,
        target_success_threshold=improve.prereg.success_threshold,
        verdict_source=verdict_source,  # type: ignore[arg-type]
        target_rater=raters.get(metric, verdict_source),
        guard_modes=guard_modes,
        unguarded_modes=unguarded_modes(
            before_root,
            after_root,
            args.development_root,
            target=metric,
            panel_modes=panel_modes,
        ),
        inputs={"improve_id": improve_id, "development_file": str(dev_file)},
        provenance=capture_provenance(),
    )
    append_records(root, "agentcore-audit", [record])
    return StageOutcome(
        stage="audit",
        exit_code=outcome.exit_code,
        summary=outcome.summary,
        detail=f"{outcome.detail}\nrecorded audit {audit_id} for {subject_arn}",
    )


# --- the report -------------------------------------------------------------------------------


def _report(args: argparse.Namespace) -> StageOutcome:
    from calibrated_agent_evaluation.agentcore.audit_report import (
        ReadSummary,
        read_summary,
        render_report,
    )
    from calibrated_agent_evaluation.cli.records import load_records_of

    root = _root(args)
    if getattr(args, "ab_test_id", None):
        test_id = str(args.ab_test_id)
        claims = load_records_of(root, "agentcore-abtest")
        mine = [c for c in claims if c.ab_test_id == test_id]
        if not mine:
            raise CliError(
                f"no claim record is stored for A/B test {test_id!r}; run "
                f"`calibrated-agent-evaluation audit abtest --ab-test-id {test_id}` first"
            )
        subject_arn, by_id = mine[0].ab_test_arn, {c.record_id: c for c in claims}
        hint = f"calibrated-agent-evaluation audit retest METRIC --ab-test-id {test_id} ..."
    else:
        rec_id = str(args.recommendation_id)
        claims = load_records_of(root, "agentcore-recommendation")
        mine = [c for c in claims if c.recommendation_id == rec_id]
        if not mine:
            raise CliError(
                f"no claim record is stored for recommendation {rec_id!r}; run "
                f"`calibrated-agent-evaluation audit recommendation --recommendation-id {rec_id}` "
                "first"
            )
        subject_arn, by_id = mine[0].recommendation_arn, {c.record_id: c for c in claims}
        hint = f"calibrated-agent-evaluation audit retest METRIC --recommendation-id {rec_id} ..."
    audit = resolve_audit_record(
        load_records_of(root, "agentcore-audit"), subject_arn, args.audit_id
    )
    if audit is None:
        if getattr(args, "ab_test_id", None):
            claim: Any = resolve_claim_record(claims, str(args.ab_test_id), args.claim_record)
            read = resolve_arms_read(
                load_records_of(root, "agentcore-arms"), subject_arn, args.arms_read
            )
            summary = read_summary(read or [])
        else:
            claim = resolve_recommendation_record(
                claims, str(args.recommendation_id), args.claim_record
            )
            summary = ReadSummary(read_id=None)
        lines = render_report(claim, summary, retest_hint=hint)
        return StageOutcome(stage="audit", exit_code=EXIT_OK, summary="\n".join(lines), detail="")
    for flag, named, own in (
        ("--claim-record", args.claim_record, audit.claim_record_id),
        ("--arms-read", getattr(args, "arms_read", None), audit.arms_read_id),
    ):
        if named is not None and named != own:
            raise CliError(
                f"audit {audit.audit_id} rests on {flag[2:]} {own or 'none'}, not {named!r}; "
                f"drop {flag} or name {own or 'no read'}"
            )
    claim = by_id.get(audit.claim_record_id)
    improves = {r.improve_id: r for r in load_records_of(root, "improve")}
    improve = improves.get(audit.improve_id)
    stored_reads = {a.read_id for a in load_records_of(root, "agentcore-arms")}
    for what, found, wanted in (
        ("claim record", claim is not None, audit.claim_record_id),
        ("improve record", improve is not None, audit.improve_id),
        (
            "arms read",
            audit.arms_read_id is None or audit.arms_read_id in stored_reads,
            audit.arms_read_id,
        ),
    ):
        if not found:
            raise CliError(f"audit {audit.audit_id} names {what} {wanted!r}, which is not stored")
    summary = ReadSummary(
        read_id=audit.arms_read_id,
        counts=dict(audit.arm_counts),
        applied_bundle_checked=audit.applied_bundle_checked,
    )
    lines = render_report(claim, summary, audit=audit, improve=improve, retest_hint=hint)
    return StageOutcome(stage="audit", exit_code=EXIT_OK, summary="\n".join(lines), detail="")
