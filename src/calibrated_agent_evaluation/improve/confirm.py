# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""The confirmation-slice guard -- held-out and fresh-and-paired.

The headline before/after is measured on a *held-out / freshly-drawn* confirmation slice
(one the fix was NOT tuned against), run through *both* old and new agent on the *same* inputs. This
module guards those two properties before any measurement:

  * :func:`assert_held_out` -- refuses (naming the overlap) if the confirmation slice shares any
    ``trace_id`` with the development / tuning set; measuring on the traces you tuned against is
    regression-to-the-mean contamination -- the dev set scores better by construction.
  * :func:`pair_observations` -- pairs the before/after labels on the SAME inputs (by ``trace_id``);
    a mismatched input set (an input present on only one side) is refused rather than silently
    dropped, because McNemar's validity rests on a genuine pairing.

The improve stage does NOT *draw* the slice (that is the sampling stage) -- it only guards the
disjointness + pairing of a caller-supplied one.
"""

from __future__ import annotations

import hashlib
from collections.abc import Mapping, Sequence

from calibrated_agent_evaluation.improve.errors import ConfirmationContaminationError, ImproveError
from calibrated_agent_evaluation.improve.report import PairedObs


def slice_id(prefix: str, trace_ids: Sequence[str]) -> str:
    """The id of a set of traces: ``prefix``, then a digest of its sorted trace ids.

    An improve record names its confirmation slice and development set this way, so another
    stage can tell whether a slice it holds is one a record already decided.
    """
    digest = hashlib.sha256("\n".join(sorted(trace_ids)).encode()).hexdigest()[:12]
    return f"{prefix}-{digest}"


def assert_held_out(confirmation_ids: Sequence[str], development_ids: Sequence[str]) -> None:
    r"""Refuse if the confirmation slice overlaps the development set.

    Raises :class:`ConfirmationContaminationError` naming the shared ``trace_id``\ s. An empty
    confirmation slice is refused too -- there is nothing to measure a headline on (a vacuous
    "held out" is not held out).
    """
    if not confirmation_ids:
        raise ImproveError(
            "assert_held_out: the confirmation slice is empty; there are no held-out traces to "
            "measure a before/after headline on"
        )
    overlap = sorted(set(confirmation_ids) & set(development_ids))
    if overlap:
        raise ConfirmationContaminationError(
            f"assert_held_out: the confirmation slice shares {len(overlap)} trace(s) with the "
            f"development set ({overlap[:5]}{'...' if len(overlap) > 5 else ''}); a headline "
            "measured on traces the fix was tuned against is regression-to-the-mean, not a "
            "fresh measurement"
        )


def pair_observations(
    before: Mapping[str, bool], after: Mapping[str, bool], *, judge_produced: bool = False
) -> list[PairedObs]:
    """Pair before/after FAIL verdicts on the same inputs into :class:`PairedObs`.

    ``before`` / ``after`` map ``trace_id -> before/after FAIL`` (True = the agent failed on that
    input). The two must cover exactly the same inputs -- an input present on only one side means
    the slice was not run through both agents on the same inputs, breaking the paired assumption, so
    it is refused (naming the offending ids) rather than silently dropped. ``judge_produced`` marks
    every observation as judge-labeled (drives the adjudication gate downstream).
    """
    if not before or not after:
        raise ImproveError(
            "pair_observations: both the before and after label sets must be non-empty to form a "
            "paired confirmation slice"
        )
    before_ids, after_ids = set(before), set(after)
    if before_ids != after_ids:
        only_before = sorted(before_ids - after_ids)
        only_after = sorted(after_ids - before_ids)
        raise ImproveError(
            "pair_observations: the before and after label sets cover different inputs "
            f"(only-before={only_before[:5]}, only-after={only_after[:5]}); a paired before/after "
            "requires the SAME inputs run through both agents"
        )
    return [
        PairedObs(
            trace_id=tid,
            before_fail=bool(before[tid]),
            after_fail=bool(after[tid]),
            labels_are_judge_produced=judge_produced,
        )
        for tid in sorted(before_ids)
    ]
