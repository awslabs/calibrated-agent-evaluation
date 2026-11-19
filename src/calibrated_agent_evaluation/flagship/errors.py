# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Typed error family for the flagship proof harness.

The harness makes exactly one kind of failure of its own: it refuses to declare the proof complete
when the proof cannot be trusted -- a stale corpus, a missing oracle, an unreachable ``--live``
model, a cross-world divergence, or a stage that raised. Every one is a typed, explained refusal,
never a silent no-op or a partial pass (no silent failure; the refusal message is a
feature). :class:`FlagshipError` subclasses the
:class:`~calibrated_agent_evaluation.schema.errors.CalibratedAgentEvaluationDataError` base so a
caller can catch the whole methodology-plane family at the plane boundary -- exactly as ``RunError``
/ ``ImproveError`` do.

The harness *reuses* (never redefines) every stage's own typed error -- it surfaces the real
error type on the offending ``StageOutcome`` and lets the family propagate, so the failure is
diagnosable rather than flattened. Reused (non-exhaustive):

  * :class:`~calibrated_agent_evaluation.run.RunGateError` -- the CI gate refused / ran over an
    empty input.
  * :class:`~calibrated_agent_evaluation.improve.ImproveError` -- an improve-guard refusal.
  * :class:`~calibrated_agent_evaluation.bias.BiasedJudgeError` -- a biased judge refused promotion.
  * :class:`~calibrated_agent_evaluation.enforce.UncalibratedScoreError` -- an uncertified judge
    refused a score.
  * :class:`~calibrated_agent_evaluation.execute.ExecutionError` /
    :class:`~calibrated_agent_evaluation.execute.CassetteMiss` -- an executor failure (the
    ``--live`` unreachable path re-raises this as ``FlagshipError``).

So a caller catching ``CalibratedAgentEvaluationDataError`` catches every flagship refusal,
whichever module raised it.
"""

from __future__ import annotations

from calibrated_agent_evaluation.schema.errors import CalibratedAgentEvaluationDataError


class FlagshipError(CalibratedAgentEvaluationDataError):
    """Base class for every flagship-harness refusal.

    Raised when the proof cannot be trusted: a seeded fixture version that does not match the
    running ``SCHEMA_VERSION``, a missing/corrupt oracle for a stage that claims one,
    a ``--live`` model that is unreachable/unconfigured (re-raised from the
    executor's ``ExecutionError`` so the harness never silently replays the cassette and reports a
    live pass), or a stage's shipped feature raising a typed domain error mid-proof. It is
    a refusal to declare the proof complete, not a bug in a composed stage.
    """


class FlagshipStoreError(FlagshipError):
    """A ``FlagshipReport`` dump/load file read or write failed.

    Wraps the underlying ``OSError`` / ``json.JSONDecodeError`` / ``csv`` failure so a
    filesystem/parse error surfaces inside the typed family rather than as a raw stdlib exception
    escaping the store's public surface (wrap external errors at the boundary). A stored
    ``schema_version`` != the running ``FLAGSHIP_SCHEMA_VERSION`` also refuses here (no coercion).
    """
