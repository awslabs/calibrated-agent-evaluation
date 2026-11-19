# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Typed error family for the review surface.

Subclasses :class:`CalibratedAgentEvaluationDataError` so a caller can catch the whole Calibrated
Agent Evaluation data/methodology family at the plane boundary. Every review refusal is a typed,
explained error; never swallowed, never a silent success ("no silent failure"; "the
refusal message is a feature, not an error to swallow").
"""

from __future__ import annotations

from calibrated_agent_evaluation.schema import CalibratedAgentEvaluationDataError


class ReviewError(CalibratedAgentEvaluationDataError):
    """Base class for every review-surface error."""


class RatificationError(ReviewError):
    """An LLM-suggested label was submitted without an explicit human accept/override.

    The human-gold boundary: an LLM ``Suggestion`` may pre-fill the form, but a
    ``kind="human"`` label is refused unless a human ratified it. No LLM-origin label reaches
    the gold path unratified.
    """


class AgreementDataError(ReviewError):
    """The labels handed to the inter-rater agreement path cannot form an agreement set.

    Fewer than two distinct raters, a rater with two labels on one unit, a non-human label under a
    named rater, or no label in the requested codebook version. Each would otherwise mean a silent
    choice about which labels the number measures.
    """


class LabelSubmitError(ReviewError):
    """A label write through the backend failed and could not be reconciled.

    Wraps the underlying ``SchemaError``/``BackendError`` (chained with ``raise … from``)
    so the review session surfaces the failure and re-queues the trace as unlabeled rather than
    presenting it as successfully labeled.
    """
