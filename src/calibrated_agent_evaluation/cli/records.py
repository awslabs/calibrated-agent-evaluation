# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Where the CLI keeps the records its stages produce: ``<local_root>/records/<kind>/``.

A stage's record (a calibration, run, improve or bias record, and every other kind in
``RECORD_KINDS``) is not a trace-store object, so it does not live in the backend. Each kind is
written through that kind's own store module (the same codec the library ships), read back from
the lossless JSON file, and kept beside the local store whichever backend is selected: with the
Langfuse backend the records are still local files.

The improve stage's pre-registration ledger lives in ``records/improve/`` next to the improve
records, since a ledger entry is what an improve record gates on.

``calibrated-agent-evaluation dump`` copies every kind into ``<dir>/records/<kind>/`` and
``calibrated-agent-evaluation load`` merges them back;
:func:`export_records` / :func:`import_records` are that copy. A record id present on both sides
with different content is refused rather than overwritten.

Every read-modify-write of a kind's records, and of the ledger, holds an exclusive lock on
``<local_root>/records/<kind>/.lock``, so two stages appending at once (two
``calibrated-agent-evaluation calibrate`` runs, say) cannot lose a record.
"""

from __future__ import annotations

import contextlib
import hashlib
from collections.abc import Callable, Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from calibrated_agent_evaluation._recordstore import directory_lock
from calibrated_agent_evaluation.cli.errors import CliError, GoldSetReusedError

__all__ = [
    "NO_LEDGER_HEAD",
    "RECORD_KINDS",
    "append_records",
    "export_records",
    "import_records",
    "latest_for_judge",
    "ledger_head",
    "load_ledger",
    "load_records_of",
    "prepare_import",
    "records_dir",
    "records_lock",
    "save_ledger",
]

RECORD_KINDS: tuple[str, ...] = (
    "calibration",
    "run",
    "improve",
    "bias",
    "flagship",
    "preference",
    "judge",
    "agreement",
    "codebook",
    "judge-check",
    "sample",
    "probe",
    "proposed-mode",
    "agentcore-abtest",
    "agentcore-arms",
    "agentcore-recommendation",
    "agentcore-audit",
    "agentcore-run-provenance",
    "agentcore-online",
    "agentcore-session",
    "agentcore-read",
)

# Records are read from the JSON file, the lossless one of the pair every store writes.
_FMT = "json"


@dataclass(frozen=True)
class _Codec:
    load: Callable[[Path], list[Any]]
    dump: Callable[[Sequence[Any], Path], None]
    id_of: Callable[[Any], str]


def _codec(kind: str) -> _Codec:
    """The store functions for ``kind``, imported on use so the CLI imports none of them early.

    Most record packages import their engine (numpy, scipy) on import, so on an install without it
    reading those records is a typed install hint, never a raw ``ModuleNotFoundError``.
    """
    try:
        return _import_codec(kind)
    except ModuleNotFoundError as exc:
        from calibrated_agent_evaluation.cli.config import probe_extra

        hint = probe_extra(exc.name or "").install_hint
        raise CliError(
            f"reading the stored {kind} records needs an engine that isn't installed: {hint}"
        ) from exc


def _import_codec(kind: str) -> _Codec:
    if kind == "calibration":
        from calibrated_agent_evaluation.calibrate import dump_records, load_records

        return _Codec(
            load=lambda p: load_records(p, fmt=_FMT),
            dump=lambda rs, p: dump_records(rs, p, fmt=_FMT),
            id_of=lambda r: r.calib_id,
        )
    if kind == "run":
        from calibrated_agent_evaluation.run.store import dump_run_records, load_run_records

        return _Codec(
            load=lambda p: load_run_records(p, fmt=_FMT),
            dump=lambda rs, p: dump_run_records(rs, p, fmt=_FMT),
            id_of=lambda r: r.run_id,
        )
    if kind == "improve":
        from calibrated_agent_evaluation.improve.store import (
            dump_improve_records,
            load_improve_records,
        )

        return _Codec(
            load=lambda p: load_improve_records(p, fmt=_FMT),
            dump=lambda rs, p: dump_improve_records(rs, p, fmt=_FMT),
            id_of=lambda r: r.improve_id,
        )
    if kind == "bias":
        from calibrated_agent_evaluation.bias.store import dump_battery, load_battery

        return _Codec(
            load=lambda p: load_battery(p, fmt=_FMT),
            dump=lambda rs, p: dump_battery(rs, p, fmt=_FMT),
            id_of=lambda r: r.battery_id,
        )
    if kind == "flagship":
        from calibrated_agent_evaluation.flagship.store import (
            dump_flagship_reports,
            load_flagship_reports,
        )

        return _Codec(
            load=lambda p: load_flagship_reports(p, fmt=_FMT),
            dump=lambda rs, p: dump_flagship_reports(rs, p, fmt=_FMT),
            id_of=lambda r: r.flagship_id,
        )
    if kind == "preference":
        from calibrated_agent_evaluation.subjective.store import (
            dump_preference_records,
            load_preference_records,
        )

        return _Codec(
            load=lambda p: load_preference_records(p, fmt=_FMT),
            dump=lambda rs, p: dump_preference_records(rs, p, fmt=_FMT),
            id_of=lambda r: r.preference_id,
        )
    if kind == "judge":
        from calibrated_agent_evaluation.judges.run_record import (
            dump_run_records as dump_judge_runs,
        )
        from calibrated_agent_evaluation.judges.run_record import (
            load_run_records as load_judge_runs,
        )

        return _Codec(
            load=lambda p: load_judge_runs(p, fmt=_FMT),
            dump=lambda rs, p: dump_judge_runs(rs, p, fmt=_FMT),
            id_of=lambda r: r.run_id,
        )
    if kind == "agreement":
        from calibrated_agent_evaluation.review.store import (
            dump_agreement_records,
            load_agreement_records,
        )

        return _Codec(
            load=lambda p: load_agreement_records(p, fmt=_FMT),
            dump=lambda rs, p: dump_agreement_records(rs, p, fmt=_FMT),
            id_of=lambda r: r.agreement_id,
        )
    if kind == "codebook":
        from calibrated_agent_evaluation.review.store import (
            dump_codebook_records,
            load_codebook_records,
        )

        return _Codec(
            load=lambda p: load_codebook_records(p, fmt=_FMT),
            dump=lambda rs, p: dump_codebook_records(rs, p, fmt=_FMT),
            id_of=lambda r: r.codebook_id,
        )
    if kind == "judge-check":
        from calibrated_agent_evaluation.run.judge_check import dump_judge_checks, load_judge_checks

        return _Codec(
            load=lambda p: load_judge_checks(p, fmt=_FMT),
            dump=lambda rs, p: dump_judge_checks(rs, p, fmt=_FMT),
            id_of=lambda r: r.check_id,
        )
    if kind == "probe":
        from calibrated_agent_evaluation.probe.store import dump_probes, load_probes

        return _Codec(
            load=lambda p: load_probes(p, fmt=_FMT),
            dump=lambda rs, p: dump_probes(rs, p, fmt=_FMT),
            id_of=lambda r: r.probe_id,
        )
    if kind == "sample":
        from calibrated_agent_evaluation.ingest.sample import dump_samples, load_samples

        return _Codec(
            load=lambda p: load_samples(p, fmt=_FMT),
            dump=lambda rs, p: dump_samples(rs, p, fmt=_FMT),
            id_of=lambda r: r.sample_id,
        )
    if kind == "proposed-mode":
        from calibrated_agent_evaluation.review.store import (
            dump_proposed_modes,
            load_proposed_modes,
        )

        return _Codec(
            load=lambda p: load_proposed_modes(p, fmt=_FMT),
            dump=lambda rs, p: dump_proposed_modes(rs, p, fmt=_FMT),
            id_of=lambda r: r.proposal_id,
        )
    if kind == "agentcore-abtest":
        from calibrated_agent_evaluation.agentcore.audit_types import (
            dump_abtest_claims,
            load_abtest_claims,
        )

        return _Codec(
            load=lambda p: load_abtest_claims(p, fmt=_FMT),
            dump=lambda rs, p: dump_abtest_claims(rs, p, fmt=_FMT),
            id_of=lambda r: r.record_id,
        )
    if kind == "agentcore-arms":
        from calibrated_agent_evaluation.agentcore.audit_types import (
            dump_arm_assignments,
            load_arm_assignments,
        )

        return _Codec(
            load=lambda p: load_arm_assignments(p, fmt=_FMT),
            dump=lambda rs, p: dump_arm_assignments(rs, p, fmt=_FMT),
            id_of=lambda r: f"{r.ab_test_arn}|{r.read_id}|{r.trace_id}",
        )
    if kind == "agentcore-recommendation":
        from calibrated_agent_evaluation.agentcore.audit_types import (
            dump_recommendation_claims,
            load_recommendation_claims,
        )

        return _Codec(
            load=lambda p: load_recommendation_claims(p, fmt=_FMT),
            dump=lambda rs, p: dump_recommendation_claims(rs, p, fmt=_FMT),
            id_of=lambda r: r.record_id,
        )
    if kind == "agentcore-audit":
        from calibrated_agent_evaluation.agentcore.audit_types import (
            dump_audit_records,
            load_audit_records,
        )

        return _Codec(
            load=lambda p: load_audit_records(p, fmt=_FMT),
            dump=lambda rs, p: dump_audit_records(rs, p, fmt=_FMT),
            id_of=lambda r: r.audit_id,
        )
    if kind == "agentcore-run-provenance":
        from calibrated_agent_evaluation.agentcore.audit_types import (
            dump_run_provenance,
            load_run_provenance,
        )

        return _Codec(
            load=lambda p: load_run_provenance(p, fmt=_FMT),
            dump=lambda rs, p: dump_run_provenance(rs, p, fmt=_FMT),
            id_of=lambda r: r.trace_id,
        )
    if kind == "agentcore-online":
        from calibrated_agent_evaluation.agentcore.online_record import (
            dump_online_records,
            load_online_records,
        )

        return _Codec(
            load=lambda p: load_online_records(p, fmt=_FMT),
            dump=lambda rs, p: dump_online_records(rs, p, fmt=_FMT),
            id_of=lambda r: r.config_id,
        )
    if kind == "agentcore-session":
        from calibrated_agent_evaluation.agentcore.online_record import (
            dump_session_logs,
            load_session_logs,
            session_log_id,
        )

        return _Codec(
            load=lambda p: load_session_logs(p, fmt=_FMT),
            dump=lambda rs, p: dump_session_logs(rs, p, fmt=_FMT),
            id_of=session_log_id,
        )
    if kind == "agentcore-read":
        from calibrated_agent_evaluation.agentcore.online_record import (
            dump_read_records,
            load_read_records,
        )

        return _Codec(
            load=lambda p: load_read_records(p, fmt=_FMT),
            dump=lambda rs, p: dump_read_records(rs, p, fmt=_FMT),
            id_of=lambda r: r.read_id,
        )
    raise CliError(f"unknown record kind {kind!r} (expected one of {list(RECORD_KINDS)})")


def records_dir(root: Path, kind: str) -> Path:
    """``<root>/records/<kind>``."""
    if kind not in RECORD_KINDS:
        raise CliError(f"unknown record kind {kind!r} (expected one of {list(RECORD_KINDS)})")
    return root / "records" / kind


@contextlib.contextmanager
def records_lock(root: Path, kind: str) -> Iterator[None]:
    """Hold an exclusive lock on the ``kind`` records under ``root`` for the ``with`` block.

    A stage that reads the stored records, decides on them and appends (calibrate seeds its
    single-use ledger from them) holds this across all three steps. It is the shared
    :func:`~calibrated_agent_evaluation._recordstore.directory_lock` on the kind's directory,
    created first if missing: advisory and process-wide, and not reentrant, so a holder calls
    :func:`append_records` with ``locked=True``.

    Raises:
        CliError: ``kind`` is not a record kind, or its directory can't be created or locked.
    """
    directory = records_dir(root, kind)
    with contextlib.ExitStack() as held:
        try:
            directory.mkdir(parents=True, exist_ok=True)
            held.enter_context(directory_lock(directory))
        except OSError as exc:
            raise CliError(f"cannot lock the {kind} records in {directory}: {exc}") from exc
        # Outside the try, so an OSError raised in the caller's block is not called a lock failure.
        yield


def load_records_of(root: Path, kind: str) -> list[Any]:
    """Every ``kind`` record under ``root``; an empty list when none were written yet."""
    directory = records_dir(root, kind)
    if not directory.exists():
        return []
    return _codec(kind).load(directory)


def append_records(root: Path, kind: str, records: Sequence[Any], *, locked: bool = False) -> int:
    """Add ``records`` to the ``kind`` records under ``root``; returns how many were new.

    A record whose id is already stored with the same content is skipped. One whose id is stored
    with different content is refused: two different records under one id would leave a reader
    unable to tell which one a later stage used. A new calibration record on a gold set an
    existing record already consumed is refused too, so a locked test stays single-use even when
    two runs certify on it at once.

    The read and the write happen under :func:`records_lock`; pass ``locked=True`` when the caller
    already holds it.
    """
    if not locked:
        with records_lock(root, kind):
            return append_records(root, kind, records, locked=True)
    codec = _codec(kind)
    target = records_dir(root, kind)
    existing = codec.load(target)
    added = _new_records(kind, codec, existing, records, target)
    if kind == "calibration":
        _refuse_consumed_gold(existing, added)
    if added:
        codec.dump([*existing, *added], target)
    return len(added)


def _refuse_consumed_gold(existing: Sequence[Any], added: Sequence[Any]) -> None:
    """Refuse a new calibration record whose locked gold a stored record already used.

    The same rule ``calibrated-agent-evaluation calibrate`` checks before any model call
    (:func:`refuse_used_gold`), applied again at write time so two runs racing on one set cannot
    both store a record.
    """
    seen = list(existing)
    for record in added:
        refuse_used_gold(
            set_id=record.hunted_set_id,
            trace_ids=set(record.trace_ids),
            judge_id=record.judge_id,
            fingerprint=record.judge_fingerprint,
            priors=seen,
        )
        seen.append(record)


def gold_mode_tag(failure_mode: str, agent: str | None) -> str:
    """A short tag for the failure mode (and agent) a locked test tests."""
    return hashlib.sha256(f"{failure_mode}|{agent or ''}".encode()).hexdigest()[:6]


def _mode_of(set_id: str) -> str | None:
    """The mode tag of a ``gold-<mode>-<content>`` set id; ``None`` for an id without one."""
    parts = set_id.split("-")
    return parts[1] if len(parts) == 3 and parts[0] == "gold" else None


def _topped_up(prior: Any, fingerprint: str | None, trace_ids: set[str]) -> bool:
    """Whether a new set grows ``prior``: the same judge, ``prior`` short of plan, all its traces.

    ``calibrated-agent-evaluation calibrate`` tells a judge refused for too few labels to label more
    and run again. That retest grows the set with the judge unchanged, so nothing was tuned on it;
    the record counts the looks. Any other reuse (a changed judge, a set that certified or was
    refused at its planned size, a new set that drops some of the refused one's traces) is a retest
    on traces the judge, or an earlier version of it, was already scored on. The same rule as
    :class:`calibrated_agent_evaluation.calibrate.InMemoryLockedTestLedger`.
    """
    from calibrated_agent_evaluation.calibrate import open_for_top_up

    return (
        open_for_top_up(prior)
        and fingerprint is not None
        and prior.judge_fingerprint == fingerprint
        and set(prior.trace_ids) < trace_ids
    )


def same_mode_priors(set_id: str, judge_id: str, priors: Sequence[Any]) -> list[Any]:
    """The stored calibration records that tested the same failure mode as ``set_id``."""
    mode = _mode_of(set_id)
    return [
        p
        for p in priors
        if (
            _mode_of(p.hunted_set_id) == mode
            if _mode_of(p.hunted_set_id) is not None
            else p.judge_id == judge_id
        )
    ]


def refuse_used_gold(
    *,
    set_id: str,
    trace_ids: set[str],
    judge_id: str,
    fingerprint: str | None,
    priors: Sequence[Any],
) -> None:
    """Refuse gold that already tested a judge for its failure mode, in whole or in part.

    The single-use rule for locked tests, in one place. A locked test is used up by being
    evaluated, whatever the result, so a stored record with the same ``set_id`` refuses it,
    whatever judge made the record. A set that shares traces with an earlier test of the same
    failure mode is refused too, since a changed judge scored again on traces it was tuned against
    would certify on a retest; the one exception is :func:`_topped_up`. The mode is read from the
    set id's tag; a set id without one (made before the tag) is read as the same mode when it
    was made for the same ``judge_id``.

    Raises:
        GoldSetReusedError: The gold was already used.
    """
    exact = next((r for r in priors if r.hunted_set_id == set_id), None)
    if exact is not None:
        raise GoldSetReusedError(
            f"this gold set was already used to test judge {exact.judge_id!r} "
            f"({exact.calib_id!r}, {exact.model}@{exact.prompt_version}, {exact.result}); a "
            "locked test is single-use, whatever its result, so a judge is tested on gold it has "
            "not been scored on. Label fresh traces for a new locked test.",
            judge_id=judge_id,
            prior_result=exact.result,
        )
    mode = _mode_of(set_id)
    for prior in priors:
        prior_mode = _mode_of(prior.hunted_set_id)
        same_mode = prior_mode == mode if prior_mode is not None else prior.judge_id == judge_id
        if not same_mode or _topped_up(prior, fingerprint, trace_ids):
            continue
        shared = trace_ids & set(prior.trace_ids)
        if shared:
            raise GoldSetReusedError(
                f"{len(shared)} of this gold set's {len(trace_ids)} trace(s) were already used to "
                f"test judge {prior.judge_id!r} for this mode ({prior.calib_id!r}, "
                f"{prior.model}@{prior.prompt_version}, {prior.result}); a locked test is "
                "single-use, so a judge is tested only on traces no tested judge for this mode "
                "has been scored on. Leave those traces out, or label fresh ones.",
                judge_id=judge_id,
                prior_result=prior.result,
                n_overlap=len(shared),
            )


def _new_records(
    kind: str, codec: _Codec, existing: Sequence[Any], records: Sequence[Any], target: Path
) -> list[Any]:
    """The members of ``records`` not already in ``existing``; refuses an id clash."""
    by_id = {codec.id_of(r): r for r in existing}
    added: list[Any] = []
    for record in records:
        rid = codec.id_of(record)
        prior = by_id.get(rid)
        if prior is None:
            by_id[rid] = record
            added.append(record)
        elif prior != record:
            raise CliError(
                f"{kind} record {rid!r} already exists in {target} with different content; "
                "refusing to overwrite it"
            )
    return added


def load_ledger(root: Path) -> Any:
    """The pre-registration ledger under ``root`` (empty if none was written)."""
    from calibrated_agent_evaluation.improve.store import load_prereg_ledger

    return load_prereg_ledger(records_dir(root, "improve"))


def save_ledger(root: Path, ledger: Any, *, locked: bool = False) -> None:
    """Write the pre-registration ledger under ``root``, under the improve records' lock.

    A caller that read the ledger, extended it and is writing it back holds
    ``records_lock(root, "improve")`` across all of it and passes ``locked=True``.
    """
    from calibrated_agent_evaluation.improve.store import dump_prereg_ledger

    if not locked:
        with records_lock(root, "improve"):
            save_ledger(root, ledger, locked=True)
        return
    dump_prereg_ledger(ledger, records_dir(root, "improve"))


def export_records(root: Path, dest: Path) -> dict[str, int]:
    """Copy every record kind (and the ledger) from ``root`` into ``dest/records/``.

    Returns the count written per kind; a kind with no records is not written.
    """
    counts: dict[str, int] = {}
    for kind in RECORD_KINDS:
        records = load_records_of(root, kind)
        if records:
            _codec(kind).dump(records, records_dir(dest, kind))
        counts[kind] = len(records)
    ledger = _stored_ledger(root)
    if ledger is not None and ledger.entries:
        save_ledger(dest, ledger)
    counts["prereg"] = len(ledger.entries) if ledger is not None else 0
    return counts


# Same value as calibrated_agent_evaluation.improve.EMPTY_LEDGER_HEAD, kept here so reading it never
# imports improve.
NO_LEDGER_HEAD = "empty"


def ledger_head(root: Path) -> tuple[str, int]:
    """The pre-registration ledger's head under ``root``: its last entry's hash and its length.

    Run and judge records stamp it, so an evaluation can later tell whether the ledger they were
    recorded against still holds. An absent or empty ledger is ``(NO_LEDGER_HEAD, 0)``.
    """
    ledger = _stored_ledger(root)
    if ledger is None or not ledger.entries:
        return NO_LEDGER_HEAD, 0
    return ledger.entries[-1].content_hash, len(ledger.entries)


def _stored_ledger(root: Path) -> Any:
    """The ledger under ``root``, or ``None`` without importing the improve package when absent."""
    if not records_dir(root, "improve").exists():
        return None
    _codec("improve")
    return load_ledger(root)


def import_records(src: Path, root: Path) -> dict[str, int]:
    """Merge every record kind (and the ledger) from ``src/records/`` into ``root``."""
    return prepare_import(src, root)()


def prepare_import(src: Path, root: Path) -> Callable[[], dict[str, int]]:
    """Read and check ``src/records/`` against ``root``; return the step that writes it.

    Every kind is read (and so version-checked) and every id clash refused before anything is
    written, so a caller can do its other checks between the two steps and a refusal anywhere
    persists nothing. The returned step merges and returns the count of new records per kind. A
    ledger is taken whole: an empty local ledger adopts the dumped one, an identical one is kept,
    and two different ledgers are refused, because an append-only ledger cannot be merged entry by
    entry without breaking its chain.
    """
    incoming = {kind: load_records_of(src, kind) for kind in RECORD_KINDS}
    ledger = _stored_ledger(src)
    local_ledger = _stored_ledger(root)
    has = ledger is not None and bool(ledger.entries)
    has_local = local_ledger is not None and bool(local_ledger.entries)
    if has and has_local and ledger != local_ledger:
        raise CliError(
            f"the pre-registration ledger in {src} differs from the one in {root}; a ledger is "
            "append-only and is not merged, so load into a fresh --root"
        )
    for kind, recs in incoming.items():
        if recs:
            codec = _codec(kind)
            target = records_dir(root, kind)
            existing = codec.load(target)
            added = _new_records(kind, codec, existing, recs, target)
            if kind == "calibration":
                _refuse_consumed_gold(existing, added)

    def _apply() -> dict[str, int]:
        counts = {
            kind: append_records(root, kind, recs) if recs else 0 for kind, recs in incoming.items()
        }
        if ledger is None or not has or has_local:
            counts["prereg"] = 0
            return counts
        save_ledger(root, ledger)
        counts["prereg"] = len(ledger.entries)
        return counts

    return _apply


def latest_for_judge(records: Sequence[Any], judge: Any) -> Any:
    """The record a gate should see for ``judge``, or ``None`` if there is none for its id.

    The newest record pinned to the judge as it is now (model, prompt version and content
    fingerprint, see ``pin_mismatch``); failing that the
    newest for its id, which the gate then refuses as stale. Never an older record picked because
    it passes: a newer REFUSED calibration or BIASED battery for the same version is the evidence
    that counts. Records are ordered by their provenance ``run_at``, when their numbers were made,
    and ``run_at`` has one-second resolution, so two records stamped in the same second are ordered
    by their position in ``records`` (the order they were appended): the later one counts.
    """
    mine = [(i, r) for i, r in enumerate(records) if r.judge_id == judge.judge_id]
    if not mine:
        return None
    from calibrated_agent_evaluation.calibrate.engine import pin_mismatch

    current = [(i, r) for i, r in mine if pin_mismatch(r, judge) is None]
    return max(current or mine, key=lambda ir: (ir[1].provenance.run_at, ir[0]))[1]
