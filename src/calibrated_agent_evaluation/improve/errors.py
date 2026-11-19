# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Typed error family for the causal improve-guard stage.

Every refusal the improve stage makes is a typed, explained error -- never a silent no-op, a
vacuous "no regression", or a certified judge-vs-judge result (conventions: "no silent failure";
the refusal message is the feature). All errors subclass :class:`ImproveError`, which subclasses the
:class:`~calibrated_agent_evaluation.schema.errors.CalibratedAgentEvaluationDataError` base so a
caller can catch the whole methodology-plane family at the plane boundary -- exactly as ``RunError``
/ ``CalibrationError`` / ``AttributionError`` do.

The improve stage also *reuses* (never redefines) the existing typed errors raised by the surfaces
it builds on:

  * :class:`~calibrated_agent_evaluation.calibrate.errors.StatisticsInputError` -- an invalid
    McNemar table, a forbidden multiplicity method (``fdr_bh``), or an empty statistical input.
  * :class:`~calibrated_agent_evaluation.sampling.FrameContaminationError` -- a discovery-frame
    trace fed into a production-frame rate estimate (the shared sampling guard).
  * :class:`~calibrated_agent_evaluation.review.errors.RatificationError` -- the ratification
    family; :class:`AdjudicationRequiredError` is the improve-stage refusal that no discordant pair
    reaches McNemar without a ``kind="human"`` adjudicating label.

So a caller catching ``CalibratedAgentEvaluationDataError`` catches every improve-stage refusal,
whichever module raised it.
"""

from __future__ import annotations

from calibrated_agent_evaluation.schema.errors import CalibratedAgentEvaluationDataError


class ImproveError(CalibratedAgentEvaluationDataError):
    """Base class for every improve-stage refusal."""


class ImproveStoreError(ImproveError):
    """An ``ImproveRecord`` file read or write failed.

    Wraps the underlying ``OSError`` / ``json.JSONDecodeError`` / ``csv`` failure so a
    filesystem/parse error surfaces inside the typed family rather than as a raw stdlib exception
    escaping the store's public surface ("wrap external errors at the boundary"). A stored
    ``schema_version`` != the running ``IMPROVE_SCHEMA_VERSION`` also refuses here (no coercion).
    """


class ConfirmationContaminationError(ImproveError):
    """The confirmation slice overlaps the development / tuning set.

    Measuring a headline before/after on traces the fix was tuned against is regression-to-the-mean
    contamination -- the dev set scores better by construction. Raised (naming the overlapping
    trace_id) rather than measuring a contaminated headline.
    """


class AdjudicationRequiredError(ImproveError):
    """A judge-labeled discordant pair lacks a human-ratified adjudication.

    McNemar assumes ground-truth labels; a judge's own run-to-run flip-flops land in exactly the
    discordant cells it counts. So when an LLM judge produced the before/after labels, the improve
    guard refuses to run the test until every discordant pair carries a ``kind="human"``
    adjudicating label -- the human-gold boundary (the "refuse to calibrate from gold"
    analogue). The improve stage writes no label itself; it requires the human-provenance one the
    review surface produces.
    """


class LedgerIntegrityError(ImproveError):
    """The pre-registration ledger was edited, replaced or frozen after the run it gates.

    Raised when a write would change or drop an entry already on disk, when the ledger no longer
    holds an entry an earlier evaluation recorded, and when a pre-registration was frozen after the
    earliest time recorded in the after run. Each is how a threshold chosen after seeing the result
    would get in, so the gate refuses rather than measure against it.
    """
