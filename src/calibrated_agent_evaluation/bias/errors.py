# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Typed error family for the bias-check promotion gate.

Every refusal the gate makes is a typed, explained error -- never a silent no-op (conventions: "no
silent failure"). All errors subclass :class:`BiasError`, which subclasses the
:class:`~calibrated_agent_evaluation.schema.errors.CalibratedAgentEvaluationDataError` base so a
caller can catch the whole methodology-plane family at the plane boundary -- exactly as
``CalibrationError`` and ``EnforcementError`` do.

Each subclass maps to one refusal (one refusal -> one typed error -> one negative test):

  * :class:`BiasedJudgeError`         -- a must-pass bias's upper bound exceeds threshold.
  * :class:`EmptyProbeError`          -- a must-pass probe scored zero trials, so its bias is
    unmeasured (fail-closed).
  * :class:`IncompleteBatteryError`   -- the battery has no probes, or no must-pass probe for a
    must-pass bias (fail-closed).
  * :class:`MissingBiasThresholdError`-- no model-relative threshold for a must-pass bias
    (fail-closed).
  * :class:`MissingBiasBatteryError`  -- promotion checked with no bias battery run.
  * :class:`StaleBiasBatteryError`    -- the battery's (model, prompt_version) does
    not match the judge, or the referenced battery id does not match.
  * :class:`ProbeExecutionError`      -- a probe run failed (wraps the executor
    ``ExecutionError``); no CLEAR verdict from partial results.
  * :class:`BiasStoreError`           -- a record read/write failed (wraps ``OSError`` /
    ``json.JSONDecodeError``).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from calibrated_agent_evaluation.schema.errors import CalibratedAgentEvaluationDataError

if TYPE_CHECKING:
    from calibrated_agent_evaluation.enforce.decision import EnforcementDecision


class BiasError(CalibratedAgentEvaluationDataError):
    """Base class for every bias-gate refusal.

    Carries the optional :class:`~calibrated_agent_evaluation.enforce.decision.EnforcementDecision`
    that recorded the refusal, so a hard refusal both *raises* (control flow) and *records* (the
    audit trail) -- a caller catching the error reads ``err.decision`` for the audit record. Reuses
    the enforcement decision type rather than inventing a parallel one.
    """

    def __init__(self, message: str, *, decision: EnforcementDecision | None = None) -> None:
        super().__init__(message)
        self.decision = decision


class BiasedJudgeError(BiasError):
    """A must-pass bias's one-sided 95% **upper** bound exceeds its model-relative threshold.

    The judge may have passed calibration; the bias gate is orthogonal and refuses promotion.

    Upper, not lower: bias magnitude must stay under a ceiling, so the conservative bound is the
    upper edge. See ``ProbeResult`` for why the lower bound was wrong here.
    """


class EmptyProbeError(BiasError):
    """A must-pass probe scored zero trials.

    Fail-closed, and the counterpart to :class:`MissingBiasThresholdError`: a configured threshold
    with nothing measured against it certifies nothing. Zero trials used to yield
    ``magnitude_lb95=0.0`` and therefore ``passed=True`` unconditionally, so an unrun must-pass
    probe silently cleared the gate -- absence of evidence read as evidence of absence.
    """


class IncompleteBatteryError(BiasError):
    """The battery has no probes, or lacks a must-pass probe for a must-pass bias.

    Fail-closed: a CLEAR verdict claims every must-pass bias was measured and stayed under its
    threshold. An empty battery, or one missing the sycophancy probe, used to come back CLEAR with
    nothing measured. ``missing`` names the must-pass biases with no must-pass probe.
    """

    def __init__(
        self,
        message: str,
        *,
        missing: tuple[str, ...] = (),
        decision: EnforcementDecision | None = None,
    ) -> None:
        super().__init__(message, decision=decision)
        self.missing = missing


class MissingBiasThresholdError(BiasError):
    """No model-relative threshold is configured for a must-pass bias.

    Fail-closed: an unconfigured model must not pass on a borrowed/permissive default -- bias
    magnitude varies ~10x across judge models ("enforcement, not enablement").
    """


class MissingBiasBatteryError(BiasError):
    """Promotion was checked for a judge whose ``bias_battery_ref`` is null.

    No battery has been run, so bias is unproven -- refuse rather than promote on absence.
    """


class StaleBiasBatteryError(BiasError):
    """The referenced battery does not match the judge under promotion.

    Either its pinned ``(model, prompt_version)`` differs (a changed judge invalidates the
    battery), or the calibration record's ``bias_battery_ref`` does not name this battery. Treated
    as absent -- a stale battery cannot certify a changed judge.
    """


class ProbeExecutionError(BiasError):
    """A judge probe run failed.

    Wraps the executor ``ExecutionError`` (chained ``from exc``) so an execution failure surfaces
    inside the typed family; ``run_battery`` lets it propagate rather than recording a CLEAR
    verdict from partial results (no silent partial pass).
    """


class BiasStoreError(BiasError):
    """A bias-battery record file read or write failed.

    Wraps the underlying ``OSError`` / ``json.JSONDecodeError`` so a filesystem/parse failure
    surfaces inside the typed family rather than as a raw stdlib exception escaping the store.
    """
