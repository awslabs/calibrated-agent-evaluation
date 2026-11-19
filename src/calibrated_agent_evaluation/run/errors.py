# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Typed error family for the run / CI-gate stage.

Every refusal the run stage makes is a typed, explained error -- never a silent no-op or a vacuous
pass (conventions: "no silent failure"; the refusal message is a feature). All errors subclass
:class:`RunError`, which subclasses the
:class:`~calibrated_agent_evaluation.schema.errors.CalibratedAgentEvaluationDataError` base so a
caller can catch the whole methodology-plane family at the plane boundary -- exactly as
``CalibrationError`` / ``EnforcementError`` do.

The run stage also *reuses* (never redefines) the existing typed errors raised by the surfaces it
builds on:

  * :class:`~calibrated_agent_evaluation.enforce.UncalibratedScoreError` -- a non-CERTIFIED /
    version- drifted judge presented to the CI gate.
  * :class:`~calibrated_agent_evaluation.sampling.FrameContaminationError` -- a discovery-frame
    trace fed into production-frame rate estimation.
  * :class:`~calibrated_agent_evaluation.schema.errors.BackendError` -- the dashboard adapter's
    Langfuse backend is unreachable.
  * :class:`~calibrated_agent_evaluation.calibrate.errors.StatisticsInputError` -- an
    invalid multiplicity method or an empty/invalid drift input.

So a caller catching ``CalibratedAgentEvaluationDataError`` catches every run-stage refusal,
whichever module raised it.
"""

from __future__ import annotations

from calibrated_agent_evaluation.schema.errors import CalibratedAgentEvaluationDataError


class RunError(CalibratedAgentEvaluationDataError):
    """Base class for every run-stage refusal."""


class RunGateError(RunError):
    """The CI gate refuses to produce a pass/fail decision.

    Raised when an exploration-lane scorer is presented as a gate, or when the gate is
    asked to run over an empty input (no traces / no verdicts) rather than returning a vacuous
    ``passed=True``.
    """


class RunStoreError(RunError):
    """A run-record / dashboard-sink file read or write failed.

    Wraps the underlying ``OSError`` / ``json.JSONDecodeError`` / ``csv`` failure so a
    filesystem/parse error surfaces inside the typed family rather than as a raw stdlib exception
    escaping the store's public surface ("wrap external errors at the boundary"). A stored
    ``schema_version`` != the running ``RUN_SCHEMA_VERSION`` also refuses here (no coercion).
    """
