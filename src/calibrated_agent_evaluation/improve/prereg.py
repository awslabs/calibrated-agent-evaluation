# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""The pre-registration ledger -- append-only, immutable, and anchored in time.

The target metric + success threshold must be *pre-registered* BEFORE a fix. The
ledger is an append-only, frozen record: :func:`freeze` stamps a content hash over an entry's
frozen fields, :func:`register_prereg` returns a new ledger with the frozen entry appended (no
update/delete path), and :func:`require_prereg` re-verifies the hash on read -- so a metric or
threshold edited *after* the entry was frozen fails the check and is refused. This is the enforcing
reading: the guard cannot compute a headline for a metric chosen after seeing the result (never
pick-metric -> score).

A content hash alone is regenerable: edit an entry, re-freeze it, and the new hash is internally
consistent. So each frozen entry also carries the UTC time it was frozen (``frozen_at``), the hash
of the entry before it (``prev_hash``) and the commit of the agent under test (``code_ref``), all
inside the hash. :func:`verify_ledger` walks the chain; re-freezing an edited entry changes its
hash and breaks the link from the next one.

The chain cannot see an edit to its last entry, or a ledger deleted and started again, because
nothing follows the entry that changed. Three checks outside the chain cover that:

  * :func:`~calibrated_agent_evaluation.improve.dump_prereg_ledger` writes only a ledger that
    extends the one on disk, and :func:`freeze` refuses an entry that is already frozen;
  * :func:`verify_anchors` compares the ledger with the entries earlier evaluations recorded in
    their :class:`~calibrated_agent_evaluation.improve.ImproveRecord`, a copy held outside the
    ledger file;
  * :func:`require_frozen_before` refuses an entry frozen after the earliest time recorded in the
    after run, which is where a re-frozen or re-created entry lands;
  * :func:`verify_heads` compares the ledger with the heads (:func:`ledger_head`) that run and
    judge records were stamped with when they were made, so a ledger replaced after a run that
    used it is refused even when no evaluation has recorded an entry yet.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from datetime import UTC, datetime

from calibrated_agent_evaluation.improve.errors import ImproveError, LedgerIntegrityError
from calibrated_agent_evaluation.improve.report import PreRegEntry, PreRegLedger

# The attribute a stored run or judge record carries the ledger head under.
LEDGER_HEAD_FIELD = "ledger_head"
# The head of a ledger with no entries. Every ledger extends the empty one, so it never refuses.
EMPTY_LEDGER_HEAD = "empty"


def ledger_head(ledger: PreRegLedger) -> str:
    """The ledger's head: its last entry's ``content_hash``, or ``"empty"`` for no entries.

    The hash chain makes the head name the whole ledger up to it, so a record stamped with the
    head when it was made can later tell whether the ledger it was made against still exists.
    """
    return ledger.entries[-1].content_hash if ledger.entries else EMPTY_LEDGER_HEAD


def verify_heads(ledger: PreRegLedger, stamped: Iterable[tuple[str, str]]) -> None:
    """Refuse a ledger that no longer holds a head a stored record was stamped with.

    ``stamped`` pairs a description of each record (``"run record 'run-1'"``) with the head it
    carries. A head is held when it is ``"empty"`` or the ``content_hash`` of an entry still in
    the ledger. Raises :class:`LedgerIntegrityError` naming the first record whose head is gone.
    """
    held = {entry.content_hash for entry in ledger.entries} | {EMPTY_LEDGER_HEAD}
    for what, head in stamped:
        if head in held:
            continue
        raise LedgerIntegrityError(
            f"the {what} was made when the pre-registration ledger ended at entry {head[:12]}, "
            "and the ledger no longer holds that entry, so it was edited, replaced or deleted "
            "after that record was made. Restore the ledger the record was made against (from "
            "version control or a `calibrated-agent-evaluation dump`) before registering or "
            "evaluating."
        )


def _stamp(now: datetime | None) -> str:
    when = now or datetime.now(tz=UTC)
    if when.utcoffset() is None:
        raise ImproveError("freeze: now must be timezone-aware")
    return when.astimezone(UTC).isoformat(timespec="microseconds")


def freeze(
    entry: PreRegEntry, *, prev_hash: str | None = None, now: datetime | None = None
) -> PreRegEntry:
    """Return ``entry`` stamped with ``frozen_at``, ``prev_hash`` and its ``content_hash``.

    ``frozen_at`` is the current UTC time (``now`` exists for tests and must be timezone-aware);
    ``prev_hash`` is the hash of the ledger entry this one follows, which :func:`register_prereg`
    supplies. ``code_ref`` is kept as given. The stamped entry is what :func:`register_prereg`
    stores; :func:`require_prereg` recomputes the hash on read and refuses a mismatch. Refuses a
    ``success_threshold`` outside (0, 1): it is a minimum absolute reduction in fail rate, so 0
    would accept any significant drop and 1 or more can never be met.
    """
    if entry.content_hash or entry.frozen_at:
        raise LedgerIntegrityError(
            f"freeze: the pre-registration for {entry.target_metric!r} is already frozen (at "
            f"{entry.frozen_at or 'an unrecorded time'}); a frozen entry is never re-frozen, since "
            "that would give an edited threshold a fresh, consistent hash. Register a new metric."
        )
    if not 0.0 < entry.success_threshold < 1.0:
        raise ImproveError(
            f"freeze: success_threshold {entry.success_threshold!r} for {entry.target_metric!r} is "
            "not in (0, 1); it is the minimum absolute reduction in fail rate the fix must show "
            ""
        )
    stamped = PreRegEntry(
        prereg_id=entry.prereg_id,
        target_metric=entry.target_metric,
        success_threshold=entry.success_threshold,
        # nosemgrep: is-function-without-parentheses  -- bool dataclass field, not a method
        is_crew=entry.is_crew,
        frozen_at=_stamp(now),
        prev_hash=prev_hash,
        code_ref=entry.code_ref,
    )
    return PreRegEntry(
        prereg_id=stamped.prereg_id,
        target_metric=stamped.target_metric,
        success_threshold=stamped.success_threshold,
        # nosemgrep: is-function-without-parentheses  -- bool dataclass field, not a method
        is_crew=stamped.is_crew,
        content_hash=stamped.compute_hash(),
        frozen_at=stamped.frozen_at,
        prev_hash=stamped.prev_hash,
        code_ref=stamped.code_ref,
    )


def register_prereg(
    ledger: PreRegLedger, entry: PreRegEntry, *, now: datetime | None = None
) -> PreRegLedger:
    """Append ``entry`` (freezing it, chained to the last entry) to ``ledger``; return a NEW ledger.

    Refuses a duplicate ``target_metric`` -- a second pre-registration for the same metric would let
    a later, result-fitted threshold shadow the one committed before the fix (the exact evasion the
    ledger exists to stop). The returned ledger is a new object; the input is unchanged.
    """
    prev = ledger.entries[-1].content_hash if ledger.entries else None
    frozen = freeze(entry, prev_hash=prev, now=now)
    if any(e.target_metric == frozen.target_metric for e in ledger.entries):
        raise ImproveError(
            f"register_prereg: metric {frozen.target_metric!r} is already pre-registered; the "
            "ledger is append-only and a metric is registered exactly once, before its fix "
            "(re-registering would let a result-fitted threshold shadow the committed one)"
        )
    return PreRegLedger(entries=(*ledger.entries, frozen))


def verify_ledger(ledger: PreRegLedger) -> None:
    """Refuse a ledger whose hashes, chain, or freeze times do not hold.

    Every entry must hash to its ``content_hash``; the first must have no ``prev_hash`` and each
    later one must name its predecessor's hash; ``frozen_at`` must be a UTC timestamp that never
    goes backwards. Raises :class:`ImproveError` naming the first broken entry.
    """
    prev_hash: str | None = None
    prev_time: datetime | None = None
    for i, entry in enumerate(ledger.entries):
        if entry.content_hash != entry.compute_hash():
            raise ImproveError(
                f"verify_ledger: entry {i} ({entry.target_metric!r}) does not hash to its "
                "content_hash; it was edited after it was frozen"
            )
        if entry.prev_hash != prev_hash:
            raise ImproveError(
                f"verify_ledger: entry {i} ({entry.target_metric!r}) does not chain to the entry "
                "before it; an earlier entry was edited and re-frozen, or entries were reordered"
            )
        try:
            when = datetime.fromisoformat(entry.frozen_at)
        except ValueError as exc:
            raise ImproveError(
                f"verify_ledger: entry {i} has no valid frozen_at ({entry.frozen_at!r})"
            ) from exc
        if when.utcoffset() is None:
            raise ImproveError(f"verify_ledger: entry {i} frozen_at {entry.frozen_at!r} is naive")
        if prev_time is not None and when < prev_time:
            raise ImproveError(
                f"verify_ledger: entry {i} was frozen at {entry.frozen_at}, before the entry it "
                "follows"
            )
        prev_hash, prev_time = entry.content_hash, when


def require_prereg(ledger: PreRegLedger, metric: str) -> PreRegEntry:
    """Return the frozen pre-registration for ``metric``; refuse if absent, mutated, or unchained.

    Verifies the whole ledger first (:func:`verify_ledger`), so an entry whose metric, threshold,
    or crew flag was edited after :func:`freeze`, or a ledger whose chain was broken by a
    re-freeze, is refused before any headline is measured. An unregistered metric is likewise
    refused rather than silently measured.
    """
    for entry in ledger.entries:
        if entry.target_metric == metric and entry.content_hash != entry.compute_hash():
            raise ImproveError(
                f"require_prereg: pre-registration for {metric!r} was mutated after it was "
                "frozen (content hash mismatch); refusing to measure a headline against a "
                "pre-registration edited after the fix"
            )
    verify_ledger(ledger)
    for entry in ledger.entries:
        if entry.target_metric == metric:
            return entry
    raise ImproveError(
        f"require_prereg: metric {metric!r} was never pre-registered; refusing to compute a "
        "before/after headline for a metric chosen after the fix"
    )


def verify_anchors(ledger: PreRegLedger, anchors: Sequence[PreRegEntry]) -> None:
    """Refuse a ledger that no longer holds, unchanged, every entry in ``anchors``.

    ``anchors`` are the pre-registrations earlier evaluations recorded (each improve record keeps
    the entry it gated on). An anchor missing from the ledger, or present with a different hash,
    means the ledger was edited, replaced or deleted after that evaluation, so a threshold it now
    holds may have been chosen after a result was seen. Raises :class:`LedgerIntegrityError`
    naming the first such entry.
    """
    held = {entry.content_hash: entry for entry in ledger.entries}
    by_id = {entry.prereg_id: entry for entry in ledger.entries}
    for anchor in anchors:
        if held.get(anchor.content_hash) == anchor:
            continue
        now = by_id.get(anchor.prereg_id)
        state = (
            "is missing from the ledger"
            if now is None
            else f"now reads threshold {now.success_threshold} frozen at {now.frozen_at}"
        )
        raise LedgerIntegrityError(
            f"verify_anchors: an earlier evaluation recorded pre-registration {anchor.prereg_id!r} "
            f"({anchor.target_metric!r}, threshold {anchor.success_threshold}, frozen at "
            f"{anchor.frozen_at}, hash {anchor.content_hash[:12]}), and it {state}; the ledger was "
            "edited, replaced or deleted after that evaluation. Restore the ledger it was recorded "
            "against (from version control or a `calibrated-agent-evaluation dump`) before "
            "evaluating."
        )


def require_frozen_before(entry: PreRegEntry, after_run_started_at: datetime | None) -> None:
    """Refuse ``entry`` if it was frozen at or after ``after_run_started_at``.

    A pre-registration exists to fix the bar before the result is known, so it must predate the
    after run it gates. ``None`` means the after run recorded no time, and nothing is checked; the
    caller says so in its output. Raises :class:`LedgerIntegrityError` when the order is wrong and
    :class:`ImproveError` for a naive ``after_run_started_at``.
    """
    if after_run_started_at is None:
        return
    if after_run_started_at.utcoffset() is None:
        raise ImproveError("require_frozen_before: after_run_started_at must be timezone-aware")
    frozen = datetime.fromisoformat(entry.frozen_at)
    if frozen >= after_run_started_at:
        raise LedgerIntegrityError(
            f"require_frozen_before: the pre-registration for {entry.target_metric!r} was frozen "
            f"at {entry.frozen_at}, after the after run began "
            f"({after_run_started_at.astimezone(UTC).isoformat()}); a threshold set once the "
            "result could be seen does not gate it. Register before the fix and measure the after "
            "run again."
        )
