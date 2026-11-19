# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the pre-registration ledger.

Behavioural: registering freezes an entry and the ledger is append-only; require_prereg returns a
matching frozen entry, and refuses an unregistered metric or one whose frozen fields were mutated
after freezing (the "metric chosen after the fix" guard).
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from calibrated_agent_evaluation._model import replace
from calibrated_agent_evaluation.improve import (
    ImproveError,
    LedgerIntegrityError,
    PreRegEntry,
    PreRegLedger,
    freeze,
    register_prereg,
    require_prereg,
)
from calibrated_agent_evaluation.improve.prereg import verify_ledger

T0 = datetime(2026, 10, 14, 9, 0, tzinfo=UTC)


def test_register_freezes_and_require_returns_entry() -> None:
    """Registering freezes the entry (stamps a hash); require_prereg returns it."""
    led = register_prereg(
        PreRegLedger(),
        PreRegEntry(
            prereg_id="p1", target_metric="trusted_broken_policy_tool", success_threshold=0.10
        ),
    )
    entry = require_prereg(led, "trusted_broken_policy_tool")
    assert entry.content_hash  # stamped by freeze
    assert entry.content_hash == entry.compute_hash()
    assert entry.success_threshold == 0.10


def test_ledger_is_append_only() -> None:
    """A second register yields a 2-entry ledger; the input ledger is unchanged."""
    led1 = register_prereg(
        PreRegLedger(), PreRegEntry(prereg_id="p1", target_metric="mode_a", success_threshold=0.10)
    )
    led2 = register_prereg(
        led1, PreRegEntry(prereg_id="p2", target_metric="mode_b", success_threshold=0.05)
    )
    assert len(led1.entries) == 1  # original unchanged (append-only, new object)
    assert len(led2.entries) == 2
    assert {e.target_metric for e in led2.entries} == {"mode_a", "mode_b"}


def test_duplicate_metric_refused() -> None:
    """Re-registering the same metric is refused (a fitted threshold can't shadow)."""
    led = register_prereg(
        PreRegLedger(), PreRegEntry(prereg_id="p1", target_metric="mode_a", success_threshold=0.10)
    )
    with pytest.raises(ImproveError):
        register_prereg(
            led, PreRegEntry(prereg_id="p2", target_metric="mode_a", success_threshold=0.20)
        )


def test_unregistered_metric_refused() -> None:
    """require_prereg for a metric never registered raises (no headline for a late-chosen
    metric)."""
    led = register_prereg(
        PreRegLedger(), PreRegEntry(prereg_id="p1", target_metric="mode_a", success_threshold=0.10)
    )
    with pytest.raises(ImproveError):
        require_prereg(led, "mode_never_registered")


def test_mutated_frozen_entry_refused() -> None:
    """An entry whose frozen field was edited after freeze fails the hash check."""
    frozen = freeze(PreRegEntry(prereg_id="p1", target_metric="mode_a", success_threshold=0.10))
    # Tamper: raise the threshold after freezing but keep the stale content_hash.
    tampered = replace(frozen, success_threshold=0.20)
    led = PreRegLedger(entries=(tampered,))
    with pytest.raises(ImproveError):
        require_prereg(led, "mode_a")


def test_crew_flag_is_part_of_the_frozen_identity() -> None:
    """Flipping is_crew after freezing also breaks the hash (the crew metric is fixed)."""
    frozen = freeze(
        PreRegEntry(
            prereg_id="p1",
            target_metric="workflow_outcome_rate",
            success_threshold=0.10,
            is_crew=True,
        )
    )
    tampered = replace(frozen, is_crew=False)
    with pytest.raises(ImproveError):
        require_prereg(PreRegLedger(entries=(tampered,)), "workflow_outcome_rate")


# --- a temporal anchor an operator cannot regenerate ---------------


def _two_entry_ledger() -> PreRegLedger:
    led = register_prereg(
        PreRegLedger(),
        PreRegEntry(
            prereg_id="p1", target_metric="mode_a", success_threshold=0.10, code_ref="abc123"
        ),
        now=T0,
    )
    return register_prereg(
        led,
        PreRegEntry(prereg_id="p2", target_metric="mode_b", success_threshold=0.05),
        now=T0 + timedelta(hours=1),
    )


def test_freeze_stamps_time_chain_and_code_ref_inside_the_hash() -> None:
    led = _two_entry_ledger()
    first, second = led.entries
    assert first.frozen_at == "2026-10-14T09:00:00.000000+00:00"
    assert first.prev_hash is None
    assert first.code_ref == "abc123"
    assert second.prev_hash == first.content_hash
    verify_ledger(led)
    # Each anchor field is hashed: changing any one of them breaks the entry's own hash.
    for field, value in (("frozen_at", "2026-10-13T09:00:00+00:00"), ("code_ref", "def456")):
        with pytest.raises(ImproveError, match="does not hash"):
            verify_ledger(PreRegLedger(entries=(replace(first, **{field: value}),)))


def test_edit_and_refreeze_breaks_the_chain() -> None:
    """Editing a persisted threshold and re-freezing gives a consistent hash but a broken chain."""
    led = _two_entry_ledger()
    first, second = led.entries
    with pytest.raises(LedgerIntegrityError, match="already frozen"):
        freeze(replace(first, success_threshold=0.20), now=T0 + timedelta(days=30))
    edited = freeze(
        PreRegEntry(prereg_id=first.prereg_id, target_metric="mode_a", success_threshold=0.20),
        prev_hash=None,
        now=T0 + timedelta(days=30),
    )
    assert edited.content_hash == edited.compute_hash()  # internally consistent on its own
    forged = PreRegLedger(entries=(edited, second))
    with pytest.raises(ImproveError, match="does not chain"):
        verify_ledger(forged)
    with pytest.raises(ImproveError, match="does not chain"):
        require_prereg(forged, "mode_b")
    # The re-frozen entry also carries the late time, so it no longer predates the fix it gates.
    assert edited.frozen_at > second.frozen_at


def test_reordered_or_backdated_entries_refused() -> None:
    led = _two_entry_ledger()
    with pytest.raises(ImproveError, match="does not chain"):
        verify_ledger(PreRegLedger(entries=tuple(reversed(led.entries))))
    early = register_prereg(
        PreRegLedger(),
        PreRegEntry(prereg_id="p1", target_metric="mode_a", success_threshold=0.10),
        now=T0,
    )
    backdated = register_prereg(
        early,
        PreRegEntry(prereg_id="p2", target_metric="mode_b", success_threshold=0.05),
        now=T0 - timedelta(hours=1),
    )
    with pytest.raises(ImproveError, match="before the entry it follows"):
        verify_ledger(backdated)


def test_freeze_refuses_a_naive_time() -> None:
    with pytest.raises(ImproveError, match="timezone-aware"):
        freeze(
            PreRegEntry(prereg_id="p1", target_metric="mode_a", success_threshold=0.10),
            now=datetime(2026, 10, 14),  # noqa: DTZ001
        )


@pytest.mark.parametrize("threshold", [0.0, 1.0, -0.1, 1.5])
def test_freeze_refuses_a_threshold_outside_the_unit_interval(threshold: float) -> None:
    """success_threshold is a minimum reduction in fail rate, so it must be in (0, 1)."""
    with pytest.raises(ImproveError, match=r"not in \(0, 1\)"):
        register_prereg(
            PreRegLedger(),
            PreRegEntry(prereg_id="p1", target_metric="mode_a", success_threshold=threshold),
        )
