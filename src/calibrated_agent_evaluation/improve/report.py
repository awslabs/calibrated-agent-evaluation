# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Improve-stage data types -- the frozen artifacts every guard reads and the store persists.

Pure, frozen data holders (the "typed functional core" house style): the numeric work lives
in :mod:`calibrated_agent_evaluation.improve.prereg` / ``confirm`` / ``paired`` / ``panel`` /
``guard`` and the shared :mod:`calibrated_agent_evaluation.stats.mcnemar`, never in ``__init__`` or
here. No vendor SDK type appears in this plane. Every reported effect / rate is the shared
:class:`~calibrated_agent_evaluation.calibrate.Interval` (reused, never re-modelled) so a refused
correction carries ``point=None`` + ``refusal_reason`` rather than a fabricated float.

  * :class:`PreRegEntry` / :class:`PreRegLedger` -- the append-only, frozen pre-registration ledger;
    a mutated frozen entry fails its ``content_hash`` on read.
  * :class:`PairedObs` -- one paired before/after observation (the McNemar cell source).
  * :class:`ImproveRequest` -- the full input to
    :func:`calibrated_agent_evaluation.improve.evaluate_improvement`.
  * :class:`McNemarResult` -- the paired significance (exact / midp / asymptotic / undefined),
    effect as an :class:`Interval`.
  * :class:`GuardSlice` / :class:`GuardPanelReport` -- the Holm-corrected guard panel + its ``m``;
    ``relocation`` carries the :class:`RelocationResult` for a crew.
  * :class:`ImproveVerdict` -- the composed accept/reject decision.
  * :class:`ImproveRecord` -- the persisted, version-pinned audit artifact.

``CrewDefinition`` / ``CrewOutcomes`` / ``RelocationResult`` are deliberately NOT defined here:
the improve stage imports the real :mod:`calibrated_agent_evaluation.attribution` types so
:func:`calibrated_agent_evaluation.attribution.relocation_check`
consumes them directly (single source of truth). ``Label`` is the normalized label.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from datetime import datetime
from typing import ClassVar, Literal, Self

from pydantic import ConfigDict, model_validator

from calibrated_agent_evaluation._model import CalibratedAgentEvaluationModel
from calibrated_agent_evaluation._provenance import ProvenanceStamp
from calibrated_agent_evaluation.attribution import CrewDefinition, CrewOutcomes, RelocationResult
from calibrated_agent_evaluation.calibrate.record import Interval
from calibrated_agent_evaluation.improve.errors import ImproveError
from calibrated_agent_evaluation.schema import Label

# Stamped on every persisted ImproveRecord; load refuses on mismatch (no coercion). Homed here (not
# __init__) so the store + types share one constant without a circular import (the run/report.py
# precedent).
# "2" added `provenance`; "3" added Interval kind fields; "4" the pre-registration's frozen_at,
# prev_hash and code_ref; "5" the guard panel is Holm at FWER 0.05, not BY-FDR; "6" the
# provenance stamp names calibrated_agent_evaluation_version (records from before the rename are
# refused)
IMPROVE_SCHEMA_VERSION = "6"

Verdict = Literal["accepted", "rejected"]
# How the paired McNemar was computed, recorded so the record is auditable at any n.
PairedMethod = Literal["asymptotic", "exact", "midp", "undefined"]
# The provenance of a guard slice: the pre-registered target, another crew agent, a crew handoff
# seam, or a non-crew non-target failure mode.
SliceKind = Literal["target", "agent", "handoff", "other"]


class PreRegEntry(CalibratedAgentEvaluationModel):
    """One frozen pre-registration: a target metric + success threshold committed BEFORE a fix.

    ``freeze`` (in :mod:`calibrated_agent_evaluation.improve.prereg`) stamps ``content_hash`` over
    :meth:`frozen_fields`; :func:`calibrated_agent_evaluation.improve.require_prereg` re-verifies it
    on read, so a metric or threshold edited after the entry was frozen fails the check and is
    refused. ``is_crew`` marks ``target_metric`` as the crew's workflow outcome rate.

    ``success_threshold`` is the minimum absolute reduction in the target's fail rate the fix must
    show, in (0, 1): 0.10 means the fail rate must fall by at least ten percentage points. The
    improve gate accepts only when the upper bound of the paired effect interval (after minus
    before) is at most ``-success_threshold``, so the whole interval shows that reduction, not just
    its point. :func:`~calibrated_agent_evaluation.improve.freeze` refuses a value
    outside (0, 1).

    The temporal anchor: ``frozen_at`` is the UTC time
    :func:`~calibrated_agent_evaluation.improve.freeze` stamped, ``prev_hash`` the ``content_hash``
    of the ledger entry before this one (``None`` for the first), and ``code_ref`` the commit of the
    agent under test the entry was frozen against, supplied by the caller. All three are inside the
    hash, so an entry edited and re-frozen gets a new ``frozen_at`` and breaks the chain to every
    later entry.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")
    invalid_error: ClassVar[type[Exception]] = ImproveError

    prereg_id: str
    target_metric: str
    success_threshold: float
    is_crew: bool = False
    content_hash: str = ""
    frozen_at: str = ""
    prev_hash: str | None = None
    code_ref: str | None = None

    def frozen_fields(self) -> tuple[object, ...]:
        """The tuple the ``content_hash`` is computed over (everything except the hash itself)."""
        return (
            self.prereg_id,
            self.target_metric,
            self.success_threshold,
            # nosemgrep: is-function-without-parentheses  -- bool field, not a method
            self.is_crew,
            self.frozen_at,
            self.prev_hash,
            self.code_ref,
        )

    def compute_hash(self) -> str:
        """Deterministic content hash over :meth:`frozen_fields` (JSON-canonical, sha256)."""
        payload = json.dumps(self.frozen_fields(), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()


class PreRegLedger(CalibratedAgentEvaluationModel):
    """The append-only ledger of frozen pre-registrations.

    :func:`calibrated_agent_evaluation.improve.register_prereg` returns a NEW ledger with the frozen
    entry appended; there is no update or delete path -- immutability is what makes "the metric was
    chosen before the fix" a guarantee rather than a convention.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")
    invalid_error: ClassVar[type[Exception]] = ImproveError

    entries: tuple[PreRegEntry, ...] = ()


@dataclass(frozen=True)
class PairedObs:
    """One paired before/after observation on a single confirmation-slice input.

    ``before_fail`` / ``after_fail`` are the old/new agent's binary FAIL verdict on the SAME input
    (fresh-and-paired); together across the slice they populate the 2x2 McNemar table.
    ``labels_are_judge_produced`` drives the adjudication gate.
    """

    trace_id: str
    before_fail: bool
    after_fail: bool
    labels_are_judge_produced: bool = False


class McNemarResult(CalibratedAgentEvaluationModel):
    """The paired before/after significance.

    ``method`` records how it was computed -- ``asymptotic`` (>= 25 discordant), ``exact`` /
    ``midp`` (< 25 discordant), or ``undefined`` (zero discordant pairs: the test has no info, so
    ``p_value`` is ``None`` and ``undefined_reason`` is populated -- never a fabricated p-value).
    ``effect`` is the change in fail-rate (after - before) as an :class:`Interval`.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")
    invalid_error: ClassVar[type[Exception]] = ImproveError

    method: PairedMethod
    n_discordant: int
    p_value: float | None
    effect: Interval
    undefined_reason: str | None = None

    @model_validator(mode="after")
    def _check_invariants(self) -> Self:
        self._post_init()
        return self

    def _post_init(self) -> None:
        # An undefined test carries no p-value, and a defined test always carries one -- the
        # two must never disagree (a numeric p on an undefined table would be a fabricated result).
        if self.method == "undefined" and self.p_value is not None:
            raise ImproveError(
                "McNemarResult(method='undefined') must have p_value=None "
                "(a zero-discordant table has no p-value to report --)"
            )
        if self.method != "undefined" and self.p_value is None:
            raise ImproveError(f"McNemarResult(method={self.method!r}) requires a p_value")


class GuardSlice(CalibratedAgentEvaluationModel):
    r"""One non-targeted slice watched for regression, with its confidence-treated rates.

    ``regression_p`` is the p-value that this slice regressed (worsened) between before and after;
    the panel corrects these across all slices with Holm. ``rate_before`` /
    ``rate_after`` are :class:`Interval`\ s.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")
    invalid_error: ClassVar[type[Exception]] = ImproveError

    name: str
    kind: SliceKind
    regression_p: float
    rate_before: Interval
    rate_after: Interval


class GuardPanelReport(CalibratedAgentEvaluationModel):
    """The Holm-corrected guard-slice panel.

    ``method`` is the literal ``"holm"`` only: one surviving slice rejects the fix, so the panel
    controls the family-wise error rate, which Holm does under any dependence between the
    overlapping slices (``fdr_bh`` and ``fdr_by`` are not accepted).
    ``m`` is the comparison count, stated per run. ``regressed`` names the slices whose regression
    survived the correction. ``relocation`` carries the crew verdict for a crew improvement,
    ``None`` for a single-agent one.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")
    invalid_error: ClassVar[type[Exception]] = ImproveError

    slices: tuple[GuardSlice, ...]
    method: Literal["holm"]
    m: int
    regressed: tuple[str, ...]
    relocation: RelocationResult | None = None


class ImproveVerdict(CalibratedAgentEvaluationModel):
    """The composed improve decision.

    ``verdict`` is ``rejected`` if the paired test did not show an improvement, OR any guard slice
    regressed, OR a crew improvement relocated its failure; ``accepted`` only when all guards pass.
    ``reasons`` records why a rejection happened (named slices / relocation / not-significant).
    """

    model_config = ConfigDict(frozen=True, extra="forbid")
    invalid_error: ClassVar[type[Exception]] = ImproveError

    verdict: Verdict
    prereg: PreRegEntry
    paired: McNemarResult
    panel: GuardPanelReport
    reasons: tuple[str, ...] = ()


@dataclass(frozen=True)
class ImproveRequest:
    """The full input to :func:`calibrated_agent_evaluation.improve.evaluate_improvement`.

    ``ledger`` + ``target_metric`` select the frozen pre-registration to gate on;
    ``pairs`` are the paired before/after observations on the confirmation slice;
    ``confirmation_ids`` / ``development_ids`` prove disjointness; ``slices`` are the
    non-targeted guard panel. ``adjudications`` maps a discordant ``trace_id`` to the
    ``kind="human"`` :class:`Label` that adjudicates it -- the KEY presence + human provenance IS
    the ratification record (there is no separate ``Adjudication`` type; the review surface records
    it via ``Label.kind="human"``, not a label field). The ``crew`` / ``crew_before`` /
    ``crew_after`` / ``touched_agent`` fields are populated only for a crew improvement.
    ``success_threshold`` is optional and never used as the threshold: the gate reads the
    pre-registered one, and a caller who states a threshold that differs from it is refused.
    ``anchors`` are the pre-registration entries earlier evaluations recorded (each
    :class:`ImproveRecord` carries its own); the ledger must still hold every one unchanged.
    ``after_run_started_at`` is the earliest time recorded in the after run, when known; the
    pre-registration must have been frozen before it.
    """

    ledger: PreRegLedger
    target_metric: str
    pairs: tuple[PairedObs, ...]
    confirmation_ids: tuple[str, ...]
    development_ids: tuple[str, ...]
    slices: tuple[GuardSlice, ...] = ()
    adjudications: dict[str, Label] = field(default_factory=dict)
    crew: CrewDefinition | None = None
    crew_before: CrewOutcomes | None = None
    crew_after: CrewOutcomes | None = None
    touched_agent: str | None = None
    success_threshold: float | None = None
    anchors: tuple[PreRegEntry, ...] = ()
    after_run_started_at: datetime | None = None


class ImproveRecord(CalibratedAgentEvaluationModel):
    """The persisted, version-pinned audit artifact for one improvement evaluation.

    Carries the referenced pre-registration, the paired result, the guard panel (with its
    multiplicity ``m`` and any crew relocation), the confirmation/development slice ids it was
    proven disjoint against, the final verdict, and the ``inputs`` that produced it.
    Round-trips losslessly through :mod:`calibrated_agent_evaluation.improve.store`.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")
    invalid_error: ClassVar[type[Exception]] = ImproveError

    improve_id: str
    verdict: Verdict
    prereg: PreRegEntry
    paired: McNemarResult
    panel: GuardPanelReport
    confirmation_slice_id: str
    development_set_id: str
    inputs: dict[str, str]
    reasons: tuple[str, ...] = ()
    schema_version: str = IMPROVE_SCHEMA_VERSION
    # The significance levels and pre-registered threshold applied.
    provenance: ProvenanceStamp

    @model_validator(mode="after")
    def _check_invariants(self) -> Self:
        self._post_init()
        return self

    def _post_init(self) -> None:
        if not self.inputs:
            raise ImproveError(
                f"ImproveRecord(improve_id={self.improve_id!r}) requires non-empty inputs "
                "(a record with no inputs is not reproducible --)"
            )
