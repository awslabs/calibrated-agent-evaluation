# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""A judge's calibration and bias records pin its content, not only its version label.

Editing the prompt in place under an unchanged ``prompt_version`` used to keep the old
certification. These tests pin that the content fingerprint voids it, that a record made before
fingerprints existed proves nothing, and that judges without a fingerprint keep the old pinning.
"""

from __future__ import annotations

from dataclasses import dataclass

import pytest

from _judge_helpers import certified_record, judge
from calibrated_agent_evaluation._model import replace
from calibrated_agent_evaluation._provenance import capture_provenance
from calibrated_agent_evaluation.bias import (
    BiasBatteryRecord,
    StaleBiasBatteryError,
    enforce_bias_gate,
)
from calibrated_agent_evaluation.calibrate import certifies, pin_mismatch
from calibrated_agent_evaluation.enforce import UncalibratedScoreError, enforce_score_gate
from calibrated_agent_evaluation.judges import VerdictCache
from calibrated_agent_evaluation.schema.types import Label


def test_the_fingerprint_changes_with_the_prompt_and_not_with_the_id() -> None:
    base = judge()
    assert base.fingerprint == judge().fingerprint
    assert replace(base, prompt_template=base.prompt_template + " ").fingerprint != base.fingerprint
    assert replace(base, temperature=0.0).fingerprint != base.fingerprint
    assert replace(base, judge_id="another-id").fingerprint == base.fingerprint


def test_an_in_place_prompt_edit_voids_the_certification() -> None:
    original = judge()
    record = certified_record(original)
    edited = replace(original, prompt_template=original.prompt_template + " Be lenient.")
    assert certifies(record, original)
    assert not certifies(record, edited)
    reason = pin_mismatch(record, edited)
    assert reason is not None
    assert "prompt changed since it was certified" in reason
    assert "re-certify on a fresh locked set" in reason
    with pytest.raises(UncalibratedScoreError, match="prompt changed since it was certified"):
        enforce_score_gate(record, edited)


def test_a_record_made_before_fingerprints_does_not_certify_a_fingerprinted_judge() -> None:
    current = judge()
    legacy = certified_record(current, judge_fingerprint=None)
    assert not certifies(legacy, current)
    reason = pin_mismatch(legacy, current)
    assert reason is not None
    assert "before judge prompts were fingerprinted" in reason


@dataclass(frozen=True)
class _PlainJudge:
    judge_id: str
    model: str
    prompt_version: str


def test_a_judge_without_a_fingerprint_is_pinned_on_model_and_version() -> None:
    fingerprinted = judge()
    record = certified_record(fingerprinted, judge_fingerprint=None)
    plain = _PlainJudge(fingerprinted.judge_id, fingerprinted.model, fingerprinted.prompt_version)
    assert certifies(record, plain)
    assert not certifies(record, replace(plain, prompt_version="v2"))


def test_a_battery_for_the_old_prompt_is_stale_for_the_edited_one() -> None:
    original = judge()
    edited = replace(original, prompt_template=original.prompt_template + " Be lenient.")
    battery = BiasBatteryRecord(
        provenance=capture_provenance(),
        battery_id="bias-1",
        judge_id=original.judge_id,
        model=original.model,
        prompt_version=original.prompt_version,
        judge_fingerprint=original.fingerprint,
        is_pairwise=False,
        results=(),
        verdict="CLEAR",
    )
    linked = replace(certified_record(edited), bias_battery_ref="bias-1")
    enforce_bias_gate(replace(linked, judge_fingerprint=original.fingerprint), original, battery)
    with pytest.raises(StaleBiasBatteryError, match="prompt changed since it was certified"):
        enforce_bias_gate(linked, edited, battery)


def test_the_verdict_cache_misses_after_a_prompt_edit() -> None:
    original = judge()
    edited = replace(original, prompt_template=original.prompt_template + " Be lenient.")
    cache = VerdictCache()
    cache.put(
        original,
        Label(trace_id="T1", span_id=None, kind="llm-judge", outcome="PASS", critique="fine"),
    )
    assert cache.get(original, "T1", None) is not None
    assert cache.get(edited, "T1", None) is None


def test_pointing_a_judge_at_another_mode_or_agent_changes_its_fingerprint() -> None:
    base = replace(judge(), failure_mode="wrong_refund", agent=None)
    assert replace(base, failure_mode="late_refund").fingerprint != base.fingerprint
    assert replace(base, agent="payout").fingerprint != base.fingerprint
    record = certified_record(base)
    assert certifies(record, base)
    assert not certifies(record, replace(base, failure_mode="late_refund"))
    assert not certifies(record, replace(base, agent="payout"))


def test_a_judge_with_no_target_keeps_the_fingerprint_it_always_had() -> None:
    assert replace(judge(), failure_mode=None, agent=None).fingerprint == judge().fingerprint


def test_a_judge_file_carries_its_mode_and_agent_into_the_fingerprint(tmp_path) -> None:
    import json

    from calibrated_agent_evaluation.cli.judgefile import load_judge_file

    doc = {
        "judge_id": "j",
        "model": "bedrock/m",
        "prompt_version": "v1",
        "level": "span",
        "prompt_template": "Grade it. Critique first, then a pass/fail verdict.",
        "failure_mode": "wrong_refund",
    }
    loaded = []
    for agent in ("intake", "payout"):
        path = tmp_path / f"{agent}.json"
        path.write_text(json.dumps({**doc, "agent": agent}), encoding="utf-8")
        loaded.append(load_judge_file(path).judge)
    intake, payout = loaded
    assert (intake.agent, intake.failure_mode) == ("intake", "wrong_refund")
    assert intake.fingerprint != payout.fingerprint


def test_a_record_for_another_judge_id_never_certifies() -> None:
    current = judge()
    record = certified_record(current)
    other = replace(current, judge_id="a-copy")
    assert not certifies(record, other)
    reason = pin_mismatch(record, other)
    assert reason is not None
    assert "made for judge" in reason


def test_a_fingerprinted_record_does_not_certify_a_judge_without_one() -> None:
    fingerprinted = judge()
    record = certified_record(fingerprinted)
    plain = _PlainJudge(fingerprinted.judge_id, fingerprinted.model, fingerprinted.prompt_version)
    assert not certifies(record, plain)
    reason = pin_mismatch(record, plain)
    assert reason is not None
    assert "carries no fingerprint" in reason


def test_a_verdict_names_the_judge_version_that_made_it() -> None:
    from calibrated_agent_evaluation.execute import StructuredReply
    from calibrated_agent_evaluation.judges import reply_to_label, verdict_rater_id

    current = judge()
    reply = StructuredReply(fields={"critique": "fine", "verdict": "pass"}, via="response_schema")
    label = reply_to_label(reply, current, trace_id="T1", span_id=None)
    assert label.rater_id == f"{current.judge_id}@{current.fingerprint}"
    edited = replace(current, prompt_template=current.prompt_template + " Be lenient.")
    assert verdict_rater_id(edited) != label.rater_id


def test_the_verdict_cache_keys_a_repeat_judgment_separately() -> None:
    current = judge()
    cache = VerdictCache()
    first = Label(trace_id="T1", span_id=None, kind="llm-judge", outcome="PASS", critique="ok")
    repeat = Label(trace_id="T1", span_id=None, kind="llm-judge", outcome="FAIL", critique="no")
    cache.put(current, first)
    assert cache.get(current, "T1", None, sample=1) is None
    cache.put(current, repeat, sample=1)
    assert cache.get(current, "T1", None) == first
    assert cache.get(current, "T1", None, sample=1) == repeat


@pytest.mark.parametrize(
    "payload",
    [b"[1, 2]", b"\xff\xfe not utf-8", b'{"entries": [{"key": "k", "label": {"trace_id": 1}}]}'],
)
def test_a_malformed_verdict_cache_is_a_typed_refusal(tmp_path, payload: bytes) -> None:
    from calibrated_agent_evaluation.judges import JudgeRuntimeError

    (tmp_path / "verdict_cache.json").write_bytes(payload)
    with pytest.raises(JudgeRuntimeError):
        VerdictCache.load(tmp_path)
