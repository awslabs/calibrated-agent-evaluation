# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""The detector's result types: the failure vocabulary, detections, cluster names, provenance.

Vendor-free, so they import on a bare install and the methodology plane can hold them. Each is a
frozen pydantic model; invalid input raises the plug-in's typed error.

A result's ``provenance.stamp`` records the environment of the run that produced it (the run
time, the checkout's commit, the Calibrated Agent Evaluation and Python versions and the numeric
libraries' versions), so a replayed result never equals a committed one in full. ``oracle_view()``
is the part a replay must reproduce: the full dump with exactly those values removed.
"""

from __future__ import annotations

from typing import Any, ClassVar, Literal

from pydantic import ConfigDict

from calibrated_agent_evaluation._model import CalibratedAgentEvaluationModel
from calibrated_agent_evaluation._provenance import STATS_DISTRIBUTIONS, ProvenanceStamp
from calibrated_agent_evaluation.agentcore.errors import DetectorError

__all__ = [
    "Confidence",
    "DetectedFailure",
    "Detection",
    "DetectorProvenance",
    "FailureVocabulary",
    "NamingResult",
    "VocabularyEntry",
]

Confidence = Literal["low", "medium", "high"]
"""The detector's confidence in one failure, and the threshold a caller filters at."""

# The stamp fields that describe the run's environment rather than what the detector was given.
_ENVIRONMENT_STAMP_FIELDS: tuple[str, ...] = (
    "run_at",
    "git_sha",
    "calibrated_agent_evaluation_version",
    "python_version",
)
# The stamp's numeric-library versions, from the one list capture_provenance fills; they vary
# with the interpreter and decide nothing here.
_ENVIRONMENT_DISTRIBUTIONS: tuple[str, ...] = STATS_DISTRIBUTIONS


class VocabularyEntry(CalibratedAgentEvaluationModel):
    """One failure category the detector may name.

    ``parent`` is the category's group with its bold markup removed, ``None`` only for
    ``other``. ``definition`` and ``examples`` are the table's cells with their markup (such as
    ``<br>``) kept as written; ``examples`` is empty when the table gives none, and for
    ``other``, whose definition is the template's guideline for using it.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")
    invalid_error: ClassVar[type[Exception]] = DetectorError

    name: str
    parent: str | None
    definition: str
    examples: str


class FailureVocabulary(CalibratedAgentEvaluationModel):
    """The detector's categories in table order, then ``other``, and the template they came from.

    ``detector_version`` is the installed ``strands-agents-evals`` version, ``prompt_template``
    the template version and ``prompt_sha256`` the SHA-256 of the prompt the detector sends for
    an empty session.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")
    invalid_error: ClassVar[type[Exception]] = DetectorError

    entries: tuple[VocabularyEntry, ...]
    detector_version: str
    prompt_template: str
    prompt_sha256: str

    def names(self) -> tuple[str, ...]:
        """The category names in table order, ``other`` last.

        Returns:
            One name per entry.
        """
        return tuple(entry.name for entry in self.entries)

    def descriptions(self) -> dict[str, str]:
        """Each category's definition by name, as ``cluster.ratify(descriptions=...)`` takes them.

        Returns:
            A mapping from category name to definition.
        """
        return {entry.name: entry.definition for entry in self.entries}


class DetectorProvenance(CalibratedAgentEvaluationModel):
    """What produced a detection or a set of cluster names.

    ``model_config_sent`` is the model's config without its ``model_id``, as sent.
    ``confidence_threshold`` is the threshold the detection was filtered at, ``None`` for cluster
    names. ``stamp.dep_versions`` records the ``strands-agents-evals`` version beside the
    numeric libraries'.
    """

    # pydantic before 2.10 reserves the whole "model_" prefix and warns on model_id and
    # model_config_sent; neither collides with a BaseModel attribute.
    model_config = ConfigDict(frozen=True, extra="forbid", protected_namespaces=())
    invalid_error: ClassVar[type[Exception]] = DetectorError

    detector_package: Literal["strands-agents-evals"]
    detector_version: str
    prompt_template: str
    prompt_sha256: str
    model_id: str
    model_config_sent: dict[str, Any]
    confidence_threshold: Confidence | None
    stamp: ProvenanceStamp


class DetectedFailure(CalibratedAgentEvaluationModel):
    """One category the detector named at one location, with its evidence.

    ``span_id`` is the Calibrated Agent Evaluation span the failure is on. ``tool_call_index`` and
    ``tool_name`` are set when the detector pointed at one of that span's tool calls. A category the
    detector named twice at one location appears twice, each with its own evidence.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")
    invalid_error: ClassVar[type[Exception]] = DetectorError

    span_id: str
    tool_call_index: int | None
    tool_name: str | None
    category: str
    parent: str | None
    confidence: Confidence
    evidence: str


class Detection(CalibratedAgentEvaluationModel):
    """The detector's findings on one trace.

    ``trace_sha256`` is the SHA-256 of the canonical JSON of the trace that was detected.
    ``times`` says whether the spans' recorded times were sent or placeholders stood in for them.
    ``failures`` holds the failures at or above the confidence threshold, in span order and then
    response order; ``below_threshold`` counts the ones the threshold dropped. No failures with
    ``below_threshold`` 0 means the detector found none.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")
    invalid_error: ClassVar[type[Exception]] = DetectorError

    trace_id: str
    trace_sha256: str
    times: Literal["recorded", "placeholder"]
    failures: tuple[DetectedFailure, ...]
    below_threshold: int
    provenance: DetectorProvenance

    def oracle_view(self) -> dict[str, Any]:
        """The detection as JSON, without the values that describe the run's environment.

        Returns:
            ``model_dump(mode="json")`` without the stamp's ``run_at``, ``git_sha``,
            ``calibrated_agent_evaluation_version`` and ``python_version``, and without the
            numeric-library versions ``capture_provenance`` records (``STATS_DISTRIBUTIONS``). Every
            other field is kept.
        """
        return _without_run_environment(self)


class NamingResult(CalibratedAgentEvaluationModel):
    """The names one draft-naming call gave a set of clusters.

    ``names`` holds one name per cluster, a repeated name suffixed `` #2``, `` #3``. ``shown``
    holds, per cluster, how many critiques the prompt showed and how many the cluster has, at
    most ``max_critiques_per_cluster`` shown. ``namer_prompt_sha256`` hashes the part of the
    namer prompt that holds no cluster data.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")
    invalid_error: ClassVar[type[Exception]] = DetectorError

    names: tuple[str, ...]
    shown: tuple[tuple[int, int], ...]
    namer_prompt_version: Literal["namer-v1"]
    namer_prompt_sha256: str
    max_critiques_per_cluster: int
    provenance: DetectorProvenance

    def oracle_view(self) -> dict[str, Any]:
        """The naming result as JSON, without the values that describe the run's environment.

        Returns:
            ``model_dump(mode="json")`` without the stamp's ``run_at``, ``git_sha``,
            ``calibrated_agent_evaluation_version`` and ``python_version``, and without the
            numeric-library versions ``capture_provenance`` records (``STATS_DISTRIBUTIONS``). Every
            other field is kept.
        """
        return _without_run_environment(self)


def _without_run_environment(result: CalibratedAgentEvaluationModel) -> dict[str, Any]:
    view = result.model_dump(mode="json")
    stamp = view["provenance"]["stamp"]
    for field in _ENVIRONMENT_STAMP_FIELDS:
        del stamp[field]
    for distribution in _ENVIRONMENT_DISTRIBUTIONS:
        # A stamp built by capture_provenance always holds these; one built by hand may not.
        stamp["dep_versions"].pop(distribution, None)
    return view
