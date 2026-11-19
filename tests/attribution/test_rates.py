# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Per-agent Rogan-Gladen-corrected rates.

Each agent is corrected by its OWN judge (rows don't share a fate); a weak per-agent judge gets a
wider/refused interval; a foreign calibration record is refused. Every test embeds its FR and
asserts behavior (the returned intervals, the denominator label, the typed refusal), never mocks.
"""

from __future__ import annotations

from dataclasses import dataclass

import pytest

from _helpers import make_label, rate_ids
from calibrated_agent_evaluation.attribution import AgentRecordMismatchError, per_agent_rates
from calibrated_agent_evaluation.calibrate import InMemoryLockedTestLedger
from calibrated_agent_evaluation.calibrate.engine import certify
from calibrated_agent_evaluation.calibrate.errors import UncalibratedJudgeError
from calibrated_agent_evaluation.calibrate.record import (
    CalibrationRecord,
    CalibrationSet,
    LabelPair,
)


@dataclass(frozen=True)
class _Judge:
    judge_id: str
    model: str = "bedrock/claude-sonnet-4"
    prompt_version: str = "sha1"


def _pairs(n_pos: int, tp: int, n_neg: int, tn: int, tag: str) -> tuple[LabelPair, ...]:
    pairs: list[LabelPair] = []
    for i in range(n_pos):
        v = "FAIL" if i < tp else "PASS"
        pairs.append(
            LabelPair(
                gold=make_label(trace_id=f"{tag}-p{i}", kind="human", outcome="FAIL"),
                verdict=make_label(trace_id=f"{tag}-p{i}", kind="llm-judge", outcome=v),
            )
        )
    for i in range(n_neg):
        v = "PASS" if i < tn else "FAIL"
        pairs.append(
            LabelPair(
                gold=make_label(trace_id=f"{tag}-n{i}", kind="human", outcome="PASS"),
                verdict=make_label(trace_id=f"{tag}-n{i}", kind="llm-judge", outcome=v),
            )
        )
    return tuple(pairs)


def _certified(agent: str, *, tp: int, tn: int) -> CalibrationRecord:
    """A certified record for ``agent``'s judge at 80/class with the given TP/TN counts."""
    judge = _Judge(judge_id=f"{agent}-judge")
    locked = CalibrationSet(
        pairs=_pairs(80, tp, 80, tn, agent), split="locked_test", set_id=f"locked-{agent}"
    )
    return certify(judge, locked, ledger=InMemoryLockedTestLedger())


# --- two agents → two independent intervals; weak judge → wider/refused point ---


def test_per_agent_intervals_are_independent() -> None:
    intake = _certified("intake", tp=76, tn=76)  # strong (95%): tighter bound
    policy = _certified("policy-checker", tp=72, tn=72)  # weaker-but-certified (90%): wider bound
    assert intake.result == "CERTIFIED"
    assert policy.result == "CERTIFIED"

    rates = per_agent_rates(
        {"intake": intake, "policy-checker": policy},
        {
            "intake": ([1] * 24 + [0] * 176, "prod-intake", rate_ids(range(200))),
            "policy-checker": ([1] * 40 + [0] * 160, "prod-policy", rate_ids(range(200))),
        },
        agent_judges={"intake": "intake-judge", "policy-checker": "policy-checker-judge"},
        denominator="rate-over-traffic",
    )
    by_agent = {r.agent: r for r in rates}
    assert set(by_agent) == {"intake", "policy-checker"}
    # Independent intervals: the two agents' rates are computed separately, not shared/averaged —
    # different measured rates (12% vs 20% failures) yield different corrected points + upper bound.
    assert by_agent["intake"].interval is not by_agent["policy-checker"].interval
    assert by_agent["intake"].interval.point != by_agent["policy-checker"].interval.point
    assert by_agent["intake"].interval.hi != by_agent["policy-checker"].interval.hi


def test_weaker_agent_judge_gets_a_wider_interval() -> None:
    # Rows don't share a fate: the lower-accuracy (but still certified) judge honestly gets a WIDER
    # corrected interval — the width IS the signal (a weak judge = less certainty), not hidden.
    strong = _certified("intake", tp=76, tn=76)  # 95%
    weaker = _certified("policy-checker", tp=72, tn=72)  # 90%
    rates = per_agent_rates(
        {"intake": strong, "policy-checker": weaker},
        {
            "intake": ([1] * 40 + [0] * 160, "prod-intake", rate_ids(range(200))),
            "policy-checker": ([1] * 40 + [0] * 160, "prod-policy", rate_ids(range(200))),
        },
        agent_judges={"intake": "intake-judge", "policy-checker": "policy-checker-judge"},
        denominator="rate-over-traffic",
    )
    by_agent = {r.agent: r for r in rates}
    strong_width = by_agent["intake"].interval.hi - by_agent["intake"].interval.lo
    weaker_width = by_agent["policy-checker"].interval.hi - by_agent["policy-checker"].interval.lo
    assert weaker_width > strong_width


# --- reported denominator == the one requested, per call (not swapped/defaulted) ---


def test_denominator_label_recorded() -> None:
    # requires each reported per-agent rate to be LABELLED share-of-failures vs
    # rate-over-traffic. The label is a caller-supplied dimension (it does not change the RG
    # computation), so the honest check is that the returned AgentRate carries exactly the
    # denominator requested for that call — never silently swapped or defaulted to the other value.
    rec = _certified("intake", tp=76, tn=76)
    share = per_agent_rates(
        {"intake": rec},
        {"intake": ([1] * 24 + [0] * 176, "prod-intake", rate_ids(range(200)))},
        agent_judges={"intake": "intake-judge"},
        denominator="share-of-failures",
    )
    assert share[0].denominator == "share-of-failures"

    rec2 = _certified("intake", tp=76, tn=76)
    traffic = per_agent_rates(
        {"intake": rec2},
        {"intake": ([1] * 24 + [0] * 176, "prod-intake-2", rate_ids(range(200)))},
        agent_judges={"intake": "intake-judge"},
        denominator="rate-over-traffic",
    )
    assert traffic[0].denominator == "rate-over-traffic"


# --- (edge) zero-failure rate sample → corrected point ~0, label retained ---


def test_zero_failure_sample_keeps_label_and_no_error() -> None:
    rec = _certified("intake", tp=76, tn=76)
    rates = per_agent_rates(
        {"intake": rec},
        {"intake": ([0] * 200, "prod-intake", rate_ids(range(200)))},  # zero measured failures
        agent_judges={"intake": "intake-judge"},
        denominator="share-of-failures",
    )
    interval = rates[0].interval
    assert rates[0].denominator == "share-of-failures"  # label retained
    # A zero measured rate corrects to ~0 (never negative, never a silent swap to another number).
    if interval.point is not None:
        assert 0.0 <= interval.point <= 0.05
    assert interval.lo >= 0.0


# --- (negative) foreign judge record → AgentRecordMismatchError ---


def test_foreign_judge_record_refused() -> None:
    # intake is handed policy-checker's record: rows must not share a fate.
    policy_record = _certified("policy-checker", tp=72, tn=72)
    with pytest.raises(AgentRecordMismatchError, match="its declared judge"):
        per_agent_rates(
            {"intake": policy_record},
            {"intake": ([1] * 24 + [0] * 176, "prod-intake", rate_ids(range(200)))},
            agent_judges={"intake": "intake-judge"},
            denominator="rate-over-traffic",
        )


def test_agent_without_declared_judge_refused() -> None:
    # An agent absent from agent_judges must NOT bypass the own-judge guard: an unverifiable
    # binding is refused, never silently corrected by whatever judge the record happens to carry
    # (the guard is mandatory, not opt-in).
    policy_record = _certified("policy-checker", tp=72, tn=72)
    with pytest.raises(AgentRecordMismatchError, match="no declared judge"):
        per_agent_rates(
            {"intake": policy_record},
            {"intake": ([1] * 24 + [0] * 176, "prod-intake", rate_ids(range(200)))},
            agent_judges={},  # intake undeclared → cannot verify → refuse
            denominator="rate-over-traffic",
        )


# --- (negative) uncertified agent judge → UncalibratedJudgeError propagates ---


def test_missing_rate_sample_refused() -> None:
    # An agent with a record but no rate sample must not raise a bare KeyError across the surface.
    rec = _certified("intake", tp=76, tn=76)
    with pytest.raises(AgentRecordMismatchError, match="no .*rate_set_id.* entry"):
        per_agent_rates(
            {"intake": rec},
            {},  # no verdicts for intake
            agent_judges={"intake": "intake-judge"},
            denominator="rate-over-traffic",
        )


def test_uncertified_agent_judge_refused() -> None:
    # 60% accuracy → lower bounds miss 0.80 → REFUSED; corrected_rate then refuses.
    refused = _certified("intake", tp=48, tn=48)
    assert refused.result == "REFUSED"
    with pytest.raises(UncalibratedJudgeError):
        per_agent_rates(
            {"intake": refused},
            {"intake": ([1] * 24 + [0] * 176, "prod-intake", rate_ids(range(200)))},
            agent_judges={"intake": "intake-judge"},
            denominator="rate-over-traffic",
        )
