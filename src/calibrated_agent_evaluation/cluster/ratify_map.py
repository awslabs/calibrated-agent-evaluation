# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Ratify failure modes from a map file and plan which stored FAIL labels they stamp.

A ratification map is the human's decision, written down: which failure mode each FAIL label
shows, under which codebook version, ratified by whom::

    {
      "codebook_version": "1",
      "ratified_by": "alice",
      "modes": [{"name": "refund_outside_window", "description": "refund past the window"}],
      "assign": [
        {"trace_id": "ORD-1001", "mode": "refund_outside_window"},
        {"trace_id": "CLM-4471", "span_id": "policy-checker", "rater_id": "bob", "mode": "..."}
      ]
    }

``modes`` is optional; without it the vocabulary is the set of modes the assignments name.

Two optional keys settle the modes labelers proposed from the review form under this version
(:class:`~calibrated_agent_evaluation.review.record.ProposedModeRecord`):
``"adopt": ["promised callback"]`` makes a proposal a ratified mode (its labels already carry it),
and ``"merge": {"refund too late": "refund_outside_window"}`` folds a proposal into another mode and
retags its labels. A map may hold only these, with no ``assign``. An
assignment names a unit (``trace_id`` and, for a span-level label, ``span_id``) and, when more than
one rater labeled that unit FAIL, the ``rater_id``: stamping one rater's mode onto another rater's
label would manufacture agreement on the failure-mode axis. Each rater is tagged on their newest
label for the unit, the same relabel rule the agreement gate reads.

:func:`plan_ratification` checks the whole map against the stored labels before anything is
written and returns the ratified :class:`~calibrated_agent_evaluation.cluster.types.Codebook` (built
by :func:`~calibrated_agent_evaluation.cluster.codebook.ratify`) and the label replacements (built
by :func:`~calibrated_agent_evaluation.cluster.codebook.tag_label`). Pure: no I/O.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path
from typing import ClassVar, Self

from pydantic import ConfigDict, Field, model_validator

from calibrated_agent_evaluation._model import CalibratedAgentEvaluationModel
from calibrated_agent_evaluation.cluster.codebook import ratify, tag_label
from calibrated_agent_evaluation.cluster.errors import ClusteringError
from calibrated_agent_evaluation.cluster.types import Codebook, DraftCluster, DraftCodebook
from calibrated_agent_evaluation.review.record import CodebookMode, ProposedModeRecord, mode_key
from calibrated_agent_evaluation.review.relabel import newest_per_rater_unit
from calibrated_agent_evaluation.schema.types import Label

__all__ = [
    "Assignment",
    "RatificationMap",
    "RatificationPlan",
    "load_ratification_map",
    "plan_ratification",
]


class Assignment(CalibratedAgentEvaluationModel):
    """One FAIL label's failure mode: the unit, optionally the rater, and the mode."""

    model_config = ConfigDict(frozen=True, extra="forbid")
    invalid_error: ClassVar[type[Exception]] = ClusteringError

    trace_id: str = Field(min_length=1)
    span_id: str | None = None
    rater_id: str | None = None
    mode: str = Field(min_length=1)


class RatificationMap(CalibratedAgentEvaluationModel):
    """The ratification map file (see the module docstring for its shape)."""

    model_config = ConfigDict(frozen=True, extra="forbid")
    invalid_error: ClassVar[type[Exception]] = ClusteringError

    codebook_version: str = Field(min_length=1)
    ratified_by: str = Field(min_length=1)
    parent_version: str | None = None
    modes: tuple[CodebookMode, ...] = ()
    assign: tuple[Assignment, ...] = ()
    adopt: tuple[str, ...] = ()
    merge: dict[str, str] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _check(self) -> Self:
        if not (self.assign or self.adopt or self.merge):
            raise ClusteringError("ratification map: needs at least one of assign, adopt or merge")
        names = [m.name for m in self.modes]
        if len(set(names)) != len(names):
            raise ClusteringError("ratification map modes: each failure-mode name may appear once")
        both = sorted({mode_key(n) for n in self.adopt} & {mode_key(n) for n in self.merge})
        if both:
            raise ClusteringError(
                f"ratification map: {both} is both adopted and merged; settle each proposal once"
            )
        if names:
            listed = set(names) | set(self.adopt)
            unknown = sorted({a.mode for a in self.assign} - listed)
            if unknown:
                raise ClusteringError(
                    f"ratification map: assign names mode(s) not listed in modes: {unknown}"
                )
        return self

    def vocabulary(self) -> tuple[CodebookMode, ...]:
        """The listed modes, or the assigned ones in first-assigned order."""
        if self.modes:
            return self.modes
        return tuple(CodebookMode(name=m) for m in dict.fromkeys(a.mode for a in self.assign))


class RatificationPlan(CalibratedAgentEvaluationModel):
    """What a ratification writes: the codebook, the label replacements and what was skipped.

    ``replacements`` pairs each stored label with its tagged copy. ``unchanged`` counts labels that
    already carry the assigned mode under this version.
    """

    model_config = ConfigDict(frozen=True, extra="forbid", arbitrary_types_allowed=True)
    invalid_error: ClassVar[type[Exception]] = ClusteringError

    codebook: Codebook
    modes: tuple[CodebookMode, ...]
    replacements: tuple[tuple[Label, Label], ...]
    unchanged: int
    merged: dict[str, str] = Field(default_factory=dict)
    adopted: tuple[str, ...] = ()


def load_ratification_map(path: str | Path) -> RatificationMap:
    """Read a :class:`RatificationMap` from JSON; every failure is a :class:`ClusteringError`."""
    src = Path(path)
    try:
        doc = json.loads(src.read_text(encoding="utf-8"))
    except OSError as exc:
        raise ClusteringError(
            f"cannot read the ratification map {src}: {exc.strerror or exc}"
        ) from exc
    except json.JSONDecodeError as exc:
        raise ClusteringError(f"the ratification map {src} is not valid JSON: {exc}") from exc
    if not isinstance(doc, dict):
        raise ClusteringError(f"the ratification map {src} must be a JSON object")
    return RatificationMap.model_validate(doc)


def _unit_name(a: Assignment) -> str:
    where = f"trace {a.trace_id!r}" + (f" span {a.span_id!r}" if a.span_id else "")
    return where + (f" by rater {a.rater_id!r}" if a.rater_id else "")


def _target(a: Assignment, fails: Sequence[Label]) -> Label:
    on_unit = [lb for lb in fails if lb.trace_id == a.trace_id and lb.span_id == a.span_id]
    if a.rater_id is not None:
        on_unit = [lb for lb in on_unit if lb.rater_id == a.rater_id]
    if not on_unit:
        raise ClusteringError(
            f"no human FAIL label on {_unit_name(a)} to ratify a mode onto; check the ids against "
            "the store (a PASS carries no failure mode)"
        )
    raters = sorted({lb.rater_id or "(unattributed)" for lb in on_unit})
    if len(raters) > 1:
        raise ClusteringError(
            f"{_unit_name(a)} has FAIL labels from {len(raters)} raters ({', '.join(raters)}); "
            "name "
            "the rater_id the mode is for, so one rater's mode is never stamped on another's label"
        )
    return on_unit[-1]


def _settled_proposals(
    ratification: RatificationMap,
    proposals: Sequence[ProposedModeRecord],
    ratified: Sequence[CodebookMode],
    vocabulary: tuple[CodebookMode, ...],
) -> tuple[tuple[CodebookMode, ...], dict[str, str]]:
    """The adopted proposals as modes, and each merged proposal's stored name to its target."""
    pending = {mode_key(p.name): p for p in proposals}

    def find(name: str) -> ProposedModeRecord:
        found = pending.get(mode_key(name))
        if found is None:
            waiting = sorted(p.name for p in proposals) or ["none"]
            raise ClusteringError(
                f"{name!r} is not a pending proposed mode under codebook version "
                f"{ratification.codebook_version!r} (pending: {', '.join(waiting)})"
            )
        return found

    adopted = tuple(
        CodebookMode(name=p.name, description=p.note) for p in map(find, ratification.adopt)
    )
    targets = {m.name for m in (*vocabulary, *adopted, *ratified)}
    merged: dict[str, str] = {}
    for source, target in ratification.merge.items():
        proposal = find(source)
        if target not in targets:
            raise ClusteringError(
                f"merge target {target!r} for {proposal.name!r} is not a mode of this map, an "
                "adopted proposal or a ratified mode of the version; name an existing mode"
            )
        merged[proposal.name] = target
    return adopted, merged


def plan_ratification(
    ratification: RatificationMap,
    labels: Sequence[Label],
    *,
    parent_version: str | None = None,
    proposals: Sequence[ProposedModeRecord] = (),
    ratified: Sequence[CodebookMode] = (),
) -> RatificationPlan:
    """Check ``ratification`` against ``labels`` (oldest first) and plan the tagging.

    ``parent_version`` is the version this one follows when the map does not name one.
    ``proposals`` are the pending proposed modes under the map's version and ``ratified`` the
    modes already ratified under it, for the map's ``adopt`` and ``merge``. A merged proposal's
    labels (each rater's newest FAIL carrying it under the version) are retagged to its target.

    Raises:
        ClusteringError: An assignment matches no human FAIL label, matches several raters'
            labels without a ``rater_id``, two assignments give one label different modes, a
            label already carries another mode under this version, the version does not
            advance past its parent, or an adopt or merge names no pending proposal or a merge
            target no mode.
    """
    human_fails = [lb for lb in labels if lb.kind == "human" and lb.outcome == "FAIL"]
    newest, _ = newest_per_rater_unit(human_fails)
    chosen: dict[int, tuple[Label, str]] = {}
    for a in ratification.assign:
        target = _target(a, newest)
        prior = chosen.get(id(target))
        if prior is not None and prior[1] != a.mode:
            raise ClusteringError(
                f"{_unit_name(a)} is assigned both {prior[1]!r} and {a.mode!r}; give it one mode"
            )
        chosen[id(target)] = (target, a.mode)
    version = ratification.codebook_version
    for target, mode in chosen.values():
        if target.codebook_version == version and target.failure_mode not in (None, mode):
            raise ClusteringError(
                f"the FAIL label on trace {target.trace_id!r} already has mode "
                f"{target.failure_mode!r} under codebook version {version!r}; a version's modes "
                "are fixed once ratified, so ratify the change under a new codebook_version"
            )
    vocabulary = ratification.vocabulary()
    adopted, merged = _settled_proposals(ratification, proposals, ratified, vocabulary)
    known = {m.name for m in vocabulary}
    by_name = {m.name: m for m in ratified}
    vocabulary = (
        *vocabulary,
        *(m for m in adopted if m.name not in known),
        *(
            by_name.get(t, CodebookMode(name=t))
            for t in dict.fromkeys(merged.values())
            if t not in known and t not in {m.name for m in adopted}
        ),
    )
    for label in newest:
        if label.codebook_version == version and label.failure_mode in merged:
            if id(label) in chosen:
                raise ClusteringError(
                    f"the FAIL label on trace {label.trace_id!r} is both assigned and carries the "
                    f"merged proposal {label.failure_mode!r}; leave it out of assign"
                )
            chosen[id(label)] = (label, merged[label.failure_mode])
    members: dict[str, list[str]] = {m.name: [] for m in vocabulary}
    for target, mode in chosen.values():
        members[mode].append(target.critique)
    draft = DraftCodebook(
        clusters=tuple(
            DraftCluster(name=name, member_indices=(), member_critiques=tuple(critiques))
            for name, critiques in members.items()
        )
    )
    codebook = ratify(
        draft,
        ratified_by=ratification.ratified_by,
        version=version,
        parent_version=ratification.parent_version or parent_version,
    )
    replacements: list[tuple[Label, Label]] = []
    unchanged = 0
    for target, mode in chosen.values():
        tagged = tag_label(target, mode, codebook)
        if tagged == target:
            unchanged += 1
        else:
            replacements.append((target, tagged))
    return RatificationPlan(
        codebook=codebook,
        modes=tuple(vocabulary),
        replacements=tuple(replacements),
        unchanged=unchanged,
        merged=merged,
        adopted=tuple(m.name for m in adopted),
    )
