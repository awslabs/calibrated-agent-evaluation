# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Map a model's structured reply to a validated verdict ``Label``.

The seam (``execute/judge_model.py``) guarantees a *structured* payload exists; this module turns
that payload into a normalized ``Label`` and enforces the verdict contract: a non-empty ``critique``
and a binary ``verdict``. A payload that violates the contract raises
:class:`JudgeRuntimeError`; never a fabricated or half-formed ``Label``. The verdict is
built from the structured fields only; there is no free-text parsing here.
"""

from __future__ import annotations

from typing import Protocol

from calibrated_agent_evaluation.execute import StructuredReply
from calibrated_agent_evaluation.judges.errors import JudgeRuntimeError
from calibrated_agent_evaluation.judges.executable import ExecutableJudge
from calibrated_agent_evaluation.schema.types import Label

# Map the schema's lowercase verdict enum to the normalized binary Outcome.
_VERDICT_TO_OUTCOME = {"pass": "PASS", "fail": "FAIL"}


def reply_to_label(
    reply: StructuredReply,
    judge: ExecutableJudge,
    *,
    trace_id: str,
    span_id: str | None,
) -> Label:
    """Build a validated ``kind="llm-judge"`` ``Label`` from ``reply``.

    Raises :class:`JudgeRuntimeError` if the structured payload is missing ``critique``/``verdict``,
    has an empty/whitespace critique, or a non-binary verdict; the runtime refuses to
    persist a malformed verdict. (The ``Label`` dataclass + ``schema.validate`` also reject an empty
    critique, but we raise the typed judge error here so the boundary is the judge family.)
    """
    fields = reply.fields
    critique = fields.get("critique")
    verdict = fields.get("verdict")

    if not isinstance(critique, str) or not critique.strip():
        raise JudgeRuntimeError(
            f"judge {judge.judge_id!r} on trace {trace_id!r}: structured reply had an empty or "
            f"non-string 'critique' (via={reply.via}); refusing to persist a verdict"
        )
    if not isinstance(verdict, str) or verdict.lower() not in _VERDICT_TO_OUTCOME:
        raise JudgeRuntimeError(
            f"judge {judge.judge_id!r} on trace {trace_id!r}: structured reply 'verdict' was "
            f"{verdict!r}, not one of {sorted(_VERDICT_TO_OUTCOME)} (via={reply.via}); "
            "refusing to persist a non-binary verdict"
        )

    return Label(
        trace_id=trace_id,
        span_id=span_id,
        kind="llm-judge",
        outcome=_VERDICT_TO_OUTCOME[verdict.lower()],  # type: ignore[arg-type]
        critique=critique,
        rater_id=verdict_rater_id(judge),
    )


class _Fingerprinted(Protocol):
    """A judge as its verdicts name it: its id and the fingerprint of its current version."""

    @property
    def judge_id(self) -> str: ...

    @property
    def fingerprint(self) -> str: ...


def verdict_rater_id(judge: _Fingerprinted) -> str:
    """The ``rater_id`` a verdict from ``judge`` carries: ``"<judge_id>@<fingerprint>"``.

    A stored verdict says which judge, as it is now, produced it, so a rate is corrected only with
    the calibration of the judge that made the verdicts. The fingerprint is 16 hex digits, so the
    id is read back by splitting on the last ``@``.
    """
    return f"{judge.judge_id}@{judge.fingerprint}"
