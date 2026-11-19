# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""The single-use ledger for locked test sets.

A locked test may be evaluated once: certifying on it, changing the prompt, and certifying again
on the same gold is tuning on the test. :func:`~calibrated_agent_evaluation.calibrate.certify`
consumes the set from the ledger it is given. The ledger keys on what the set holds, not only on its
name: a set whose content was already consumed is refused under any id, and so is a set that shares
a trace with one already consumed, since re-scoring a judge on traces it was tuned against is the
same retest.

One exception: a set refused for having fewer labels than its plan may be grown. ``certify``
opens such a set for a top-up by the same judge (same content fingerprint), and a later set that
holds every trace of it plus fresh ones is accepted. Nothing was tuned on the refused set, but
the judge was looked at on part of the new set, so the ledger counts the looks and ``certify``
records them on the calibration record for disclosure. A certified set, a set refused at its
planned size, a set tested by another judge, and any partial overlap stay refused.

The caller owns the ledger, so its reach is explicit: one ledger per failure mode for a process
session, or one seeded from stored calibration records (:meth:`InMemoryLockedTestLedger.
from_records`) so the refusal survives across runs (what ``calibrated-agent-evaluation calibrate``
does).
"""

from __future__ import annotations

import hashlib
from collections.abc import Iterable, Sequence
from typing import Protocol, runtime_checkable

from calibrated_agent_evaluation.calibrate.errors import LockedTestReuseError
from calibrated_agent_evaluation.calibrate.record import CalibrationRecord, CalibrationSet

__all__ = ["InMemoryLockedTestLedger", "LockedTestLedger", "locked_set_digest", "open_for_top_up"]

_SHOWN_TRACES = 5


def locked_set_digest(locked: CalibrationSet) -> str:
    """A hash of the set's targets and their gold outcomes, independent of order and ``set_id``."""
    members = sorted(
        f"{p.gold.trace_id}\x1f{p.gold.span_id or ''}\x1f{p.gold.outcome}" for p in locked.pairs
    )
    return hashlib.sha256("\x1e".join(members).encode()).hexdigest()


def open_for_top_up(record: CalibrationRecord) -> bool:
    """Whether ``record``'s set may be grown: REFUSED short of its plan, by a fingerprinted judge.

    A judge refused at its planned size, or certified, was measured on as much gold as was asked
    for, so more gold on top is a second test, not the first one completed.
    """
    return (
        record.result == "REFUSED"
        and record.judge_fingerprint is not None
        and record.planned_per_class is not None
        and min(record.n_pos, record.n_neg) < record.planned_per_class
    )


@runtime_checkable
class LockedTestLedger(Protocol):
    """Records which locked test sets have been evaluated."""

    def consume(self, locked: CalibrationSet, *, judge_fingerprint: str | None = None) -> int:
        """Mark ``locked`` consumed and return how many looks the judge has had at its gold.

        1 for a fresh set; one more for each refused set it tops up. Raises
        :class:`LockedTestReuseError` if the set, its content or a trace was already used.
        """
        ...

    def allow_top_up(self, set_id: str, *, judge_fingerprint: str) -> None:
        """Let a superset of the consumed set ``set_id`` be consumed by the same judge."""
        ...


class InMemoryLockedTestLedger:
    """A :class:`LockedTestLedger` held in memory, optionally seeded with consumed sets.

    Refuses a set whose id, content digest, or any trace was already consumed, except a strict
    superset of sets opened with :meth:`allow_top_up` for the same judge fingerprint. Not safe
    for concurrent ``consume`` calls on one instance.

    Args:
        consumed: Locked-set ids already used, for example from stored calibration records.
        consumed_trace_ids: Trace ids already used in a locked test, for example the
            ``trace_ids`` of stored calibration records for the same failure mode.
    """

    def __init__(
        self, consumed: Iterable[str] = (), *, consumed_trace_ids: Iterable[str] = ()
    ) -> None:
        self._ids: set[str] = set(consumed)
        self._digests: dict[str, str] = {}
        self._traces: dict[str, str] = dict.fromkeys(consumed_trace_ids, "an earlier locked test")
        self._members: dict[str, frozenset[str]] = {}
        self._looks: dict[str, int] = {}
        self._open: dict[str, str] = {}

    @classmethod
    def from_records(cls, records: Sequence[CalibrationRecord]) -> InMemoryLockedTestLedger:
        """A ledger holding every set ``records`` consumed, with refused-short sets left open.

        Pass the records for one failure mode. A refused set already topped up by a later
        record (one holding all its traces) is closed, so it cannot be grown twice.
        """
        ledger = cls()
        for record in records:
            members = frozenset(record.trace_ids)
            ledger._ids.add(record.hunted_set_id)
            ledger._members[record.hunted_set_id] = members
            ledger._looks[record.hunted_set_id] = record.looks
            ledger._traces.update(dict.fromkeys(record.trace_ids, record.hunted_set_id))
            if open_for_top_up(record) and record.judge_fingerprint is not None:
                ledger._open[record.hunted_set_id] = record.judge_fingerprint
        for record in records:
            members = ledger._members[record.hunted_set_id]
            for other in list(ledger._open):
                if other != record.hunted_set_id and ledger._members[other] < members:
                    del ledger._open[other]
        return ledger

    def is_consumed(self, set_id: str) -> bool:
        """Whether a set with id ``set_id`` was already consumed."""
        return set_id in self._ids

    def allow_top_up(self, set_id: str, *, judge_fingerprint: str) -> None:
        """Open the consumed set ``set_id`` to a top-up by the judge with ``judge_fingerprint``.

        Raises:
            LockedTestReuseError: ``set_id`` was never consumed here.
        """
        if set_id not in self._members:
            raise LockedTestReuseError(
                f"locked test {set_id!r} was not consumed by this ledger, so there is nothing to "
                "top up"
            )
        self._open[set_id] = judge_fingerprint

    def consume(self, locked: CalibrationSet, *, judge_fingerprint: str | None = None) -> int:
        """Mark ``locked`` consumed, refusing a reused id, reused content or a shared trace.

        A set holding every trace of open sets for the same ``judge_fingerprint``, plus at least
        one fresh trace, is their top-up: it is accepted and the looks are counted.

        Returns:
            How many looks the judge has had at this gold, 1 for a fresh set.

        Raises:
            LockedTestReuseError: The set, its content or one of its traces was already used.
        """
        if locked.set_id in self._ids:
            raise LockedTestReuseError(
                f"locked test {locked.set_id!r} was already consumed once; it is single-use, so "
                "re-iterating after touching the locked test is refused"
            )
        digest = locked_set_digest(locked)
        if digest in self._digests:
            first = self._digests[digest]
            raise LockedTestReuseError(
                f"locked test {locked.set_id!r} holds exactly the gold of {first!r}, "
                "which was already consumed; renaming a locked test does not make it fresh. Label "
                "new traces for a new locked test"
            )
        traces = list(dict.fromkeys(p.gold.trace_id for p in locked.pairs))
        members = frozenset(traces)
        topped = self._topped_up(members, judge_fingerprint)
        covered = {t for set_id in topped for t in self._members[set_id]}
        shared = [t for t in traces if t in self._traces and t not in covered]
        if shared:
            shown = ", ".join(shared[:_SHOWN_TRACES])
            more = len(shared) - _SHOWN_TRACES
            raise LockedTestReuseError(
                f"{len(shared)} of locked test {locked.set_id!r}'s {len(traces)} trace(s) were "
                f"already used in {self._traces[shared[0]]!r}: {shown}"
                + (f" and {more} more" if more > 0 else "")
                + ". A judge re-scored on traces it was tested on is a retest, so the set is "
                "refused. Leave those traces out, or label fresh ones"
            )
        looks = 1 + max((self._looks.get(s, 1) for s in topped), default=0)
        for set_id in topped:
            del self._open[set_id]
        self._ids.add(locked.set_id)
        self._digests[digest] = locked.set_id
        self._members[locked.set_id] = members
        self._looks[locked.set_id] = looks
        self._traces.update(dict.fromkeys(traces, locked.set_id))
        return looks

    def _topped_up(self, members: frozenset[str], fingerprint: str | None) -> list[str]:
        """The open sets ``members`` strictly contains, for the same judge."""
        if fingerprint is None:
            return []
        return [
            set_id
            for set_id, opened_for in self._open.items()
            if opened_for == fingerprint and self._members[set_id] < members
        ]
