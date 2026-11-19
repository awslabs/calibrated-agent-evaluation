# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Typed error family for the enforcement / lint-policy layer.

Every hard refusal the layer makes is a typed, explained error -- never a silent no-op (the refusal
message is a feature, not an error to swallow; "no silent failure"). All errors
subclass :class:`EnforcementError`, which subclasses the
:class:`~calibrated_agent_evaluation.schema.errors.CalibratedAgentEvaluationDataError` base so a
caller can catch the whole methodology-plane family at the plane boundary.

Each subclass maps to exactly one refusal (one refusal -> one typed
error -> one negative test):

  * :class:`UncalibratedScoreError` -- a score/rate was requested for a judge whose
    calibration does not certify it.
  * :class:`MultiAxisLikertError`   -- a judge scores >1 non-binary axis in one
    call.
  * :class:`CritiqueOrderError`     -- the judge output schema omits
    ``critique`` or orders ``verdict`` before it.
  * :class:`WeightedCollapseError`  -- a checklist declares a weighted-threshold
    collapse rule.
  * :class:`LikertNotAllowedError`  -- a Likert axis appears while ``allow_likert``
    is disabled.
  * :class:`GenericMetricInCIError` -- a generic generation-quality metric is used
    while the context is in CI mode.
  * :class:`EnforcementStoreError`  -- a decision-store read/write failed (wraps
    ``OSError`` / ``json.JSONDecodeError``); no raw stdlib exception escapes the store surface.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from calibrated_agent_evaluation.schema.errors import CalibratedAgentEvaluationDataError

if TYPE_CHECKING:
    from calibrated_agent_evaluation.enforce.decision import EnforcementDecision


class EnforcementError(CalibratedAgentEvaluationDataError):
    """Base class for every enforcement-layer refusal.

    Carries the optional :class:`~calibrated_agent_evaluation.enforce.decision.EnforcementDecision`
    that recorded the refusal, so a hard refusal both *raises* (control flow) and *records* (the
    audit trail) -- a caller catching the error reads ``err.decision`` for the audit record.
    """

    def __init__(self, message: str, *, decision: EnforcementDecision | None = None) -> None:
        super().__init__(message)
        self.decision = decision


class UncalibratedScoreError(EnforcementError):
    """A score or corrected rate was requested for a judge with no certifying calibration.

    The calibration verdict is read from the calibration record (`certifies`); a non-CERTIFIED
    record, or a record whose pinned ``(model, prompt_version)`` no longer matches the judge,
    refuses the score.
    """


class MultiAxisLikertError(EnforcementError):
    """A judge definition scores more than one non-binary (Likert) axis in a single call.

    The specifically-attested-harmful pattern -- a hard lint-time refusal.
    """


class CritiqueOrderError(EnforcementError):
    """A judge output schema omits ``critique`` or orders ``verdict`` before it.

    The binary/critique-first discipline is structural; critique precedes verdict.
    """


class WeightedCollapseError(EnforcementError):
    """A checklist judge declares a weighted-threshold collapse rule.

    Only critical/non-critical tagging is offered; tunable weights reintroduce the
    vibe-scoring the binary rule exists to kill.
    """


class LikertNotAllowedError(EnforcementError):
    """A Likert-scaled axis appears while the context's ``allow_likert`` capability is disabled.

    Likert is available only behind the explicit opt-in, which is disabled by default.
    """


class GenericMetricInCIError(EnforcementError):
    """A generic generation-quality metric was used as a CI gate.

    ROUGE/BERTScore/generic-"helpfulness" are exploration-only; refused while the context is
    in CI mode.
    """


class EnforcementStoreError(EnforcementError):
    """A decision-store file read or write failed.

    Wraps the underlying ``OSError`` / ``json.JSONDecodeError`` so a filesystem/parse failure
    surfaces inside the typed family rather than as a raw stdlib exception escaping the store's
    public surface ("wrap external errors at the boundary").
    """
