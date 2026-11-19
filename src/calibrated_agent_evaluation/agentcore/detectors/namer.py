# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Name draft failure-mode clusters from the failure detector's category vocabulary.

:class:`DetectorDraftNamer` is a ``DraftNamer`` for
:func:`calibrated_agent_evaluation.cluster.build_draft`. Each
:meth:`~DetectorDraftNamer.name_clusters` call makes one model call carrying a Calibrated Agent
Evaluation-authored prompt (version ``namer-v1``): every vocabulary name with its definition, then
each cluster's first critiques in member order with a note of how many are shown, and a request for
one JSON array holding one vocabulary name per cluster. The answer is read with the detector's three
tolerance rules (the bare array, one fenced block, or the only such array in surrounding text).
Anything else is refused with :class:`~calibrated_agent_evaluation.agentcore.DraftNamingError`; the
namer never falls back to ``mode_0`` names. A name the model gives more than one cluster stays on
the first and is suffixed `` #2``, `` #3`` on the later ones, so every cluster stays its own mode.

The model call goes through the same guarded stack as the detector's: a live call past
``max_live_calls`` is refused before it is sent, a cassette records and replays it, and whatever
it raises reaches the caller as a typed error with the original chained.
"""

from __future__ import annotations

import hashlib
import json
import threading
from collections import Counter
from collections.abc import Sequence
from datetime import datetime
from pathlib import Path
from typing import Any, Literal

from strands.models.model import Model
from strands.types.content import Messages

from calibrated_agent_evaluation._provenance import capture_provenance
from calibrated_agent_evaluation.agentcore.detectors._payload import PayloadPickError, pick_one_json
from calibrated_agent_evaluation.agentcore.detectors.models import (
    call_error,
    prepare_model,
    run_stream,
)
from calibrated_agent_evaluation.agentcore.detectors.types import (
    DetectorProvenance,
    FailureVocabulary,
    NamingResult,
)
from calibrated_agent_evaluation.agentcore.detectors.vocabulary import template_provenance
from calibrated_agent_evaluation.agentcore.errors import DetectorConfigError, DraftNamingError
from calibrated_agent_evaluation.execute.cassette import Mode

__all__ = ["NAMER_PROMPT_SHA256", "NAMER_PROMPT_VERSION", "DetectorDraftNamer"]

NAMER_PROMPT_VERSION: Literal["namer-v1"] = "namer-v1"
"""The version of the prompt the namer sends, recorded on every naming result."""

_DETECTOR_DISTRIBUTION = "strands-agents-evals"
_DEFAULT_MAX_CRITIQUES = 10
# A response fragment quoted in an error is cut to this many characters, and the message says so.
_QUOTE_LIMIT = 200

# The prompt, with the data left out. A recorded naming cassette keys on the full prompt text, so
# any change here needs a new prompt version and a new recording.
_PROMPT_TEMPLATE = """\
You are helping a human reviewer start a codebook of failure modes for an AI agent.

Reviewers wrote critiques of agent traces that failed, and the critiques were grouped into \
clusters of similar critiques. Name each cluster with the one failure category from the \
vocabulary below whose definition best describes what the cluster's critiques say went wrong.

Rules:
- Use only category names from the vocabulary, spelled exactly as they are listed.
- Give every cluster exactly one name. Two clusters may get the same name when the same category \
fits both.
- A cluster's heading says how many of its critiques are shown. Name the cluster from the \
critiques shown.
- Answer with one JSON array of strings and nothing else: one category name per cluster, in \
cluster order, so the first name is for Cluster 1, the second for Cluster 2, and so on.

Vocabulary, one category per line as "- name: definition":
{vocabulary}

Clusters, {count} in all, each critique written as a JSON string:
{clusters}

Answer with one JSON array of exactly {count} category names, in cluster order."""
_ENTRY_LINE = "- {name}: {definition}"
_CLUSTER_HEADING = "Cluster {number} ({shown} of {total} critiques shown):"
_CRITIQUE_LINE = "- {critique}"

NAMER_PROMPT_SHA256: str = hashlib.sha256(
    "\n\n".join((_PROMPT_TEMPLATE, _ENTRY_LINE, _CLUSTER_HEADING, _CRITIQUE_LINE)).encode("utf-8")
).hexdigest()
"""The SHA-256 of the namer prompt's data-free part, recorded on every naming result.

It hashes the prompt template and the line templates the vocabulary and the critiques are filled
into, joined by blank lines, so it changes with the prompt's wording or layout and never with the
vocabulary or the clusters.
"""


class DetectorDraftNamer:
    """Names draft clusters from the detector vocabulary, one model call per naming.

    A ``DraftNamer`` for :func:`calibrated_agent_evaluation.cluster.build_draft`, which it satisfies
    structurally. The configuration is checked when the namer is built, before any model call. The
    model is always the one passed: ``None`` and a model id string are refused, because the detector
    library would build a Bedrock model of its own from them.

    Args:
        model: The strands ``Model`` that names the clusters, for example
            ``detector_bedrock_model(model_id)``.
        vocabulary: The names the clusters are named from, usually ``failure_vocabulary()``.
        model_id: The model id, when the model's config has none; if both are given they must be
            equal. Every naming result and cassette entry records it.
        max_critiques_per_cluster: How many of a cluster's critiques the prompt shows, its first
            ones in member order; the prompt states how many were shown out of how many.
        cassette_path: The cassette file that records and replays the naming calls, or ``None``
            to make every call live.
        cassette_mode: The cassette mode; ``None`` reads
            ``CALIBRATED_AGENT_EVALUATION_CASSETTE_MODE`` (default auto).
        max_live_calls: How many live model calls the namer may make over its lifetime. The
            default 0 allows none, so only cassette replays run.
        now: The time every naming result's provenance stamp records, for tests; ``None`` reads
            the clock at each call. It must be timezone-aware.

    Attributes:
        results: One :class:`~calibrated_agent_evaluation.agentcore.detectors.types.NamingResult`
            per naming call that returned names, in call order.

    Raises:
        DetectorConfigError: The model is missing, a string or not a strands ``Model``; no model
            id can be resolved, or the passed one differs from the config's; the cassette cannot
            be loaded, or is missing in replay mode; ``CALIBRATED_AGENT_EVALUATION_CASSETTE_MODE``
            names no mode; ``max_live_calls`` is negative; ``max_critiques_per_cluster`` is below 1;
            the vocabulary is empty or not a ``FailureVocabulary``; or ``now`` is naive.
        AgentCoreError: The installed detector package reports no version.
    """

    def __init__(
        self,
        model: Model,
        vocabulary: FailureVocabulary,
        *,
        model_id: str | None = None,
        max_critiques_per_cluster: int = _DEFAULT_MAX_CRITIQUES,
        cassette_path: str | Path | None = None,
        cassette_mode: Mode | None = None,
        max_live_calls: int = 0,
        now: datetime | None = None,
    ) -> None:
        _check_cap(max_critiques_per_cluster)
        _check_vocabulary(vocabulary)
        _check_now(now)
        self._prepared = prepare_model(
            model,
            model_id=model_id,
            cassette_path=cassette_path,
            cassette_mode=cassette_mode,
            max_live_calls=max_live_calls,
        )
        # Read once here, so a broken install is refused before any billed call.
        self._template = template_provenance()
        self._vocabulary = vocabulary
        self._known = frozenset(vocabulary.names())
        self._max_critiques = max_critiques_per_cluster
        self._now = now
        # The model stack allows one call per arm, so one naming runs at a time.
        self._lock = threading.Lock()
        self.results: list[NamingResult] = []

    def name_clusters(self, clusters: list[list[str]]) -> list[str]:
        """Name each cluster from the vocabulary, with one model call.

        Args:
            clusters: The member critiques of each cluster, in cluster order.

        Returns:
            One name per cluster, in cluster order: a vocabulary name, with `` #2``, `` #3``
            appended to the second and later clusters given the same name. The call's
            :class:`~calibrated_agent_evaluation.agentcore.detectors.types.NamingResult` is appended
            to :attr:`results`.

        Raises:
            DraftNamingError: ``clusters`` is empty or holds an empty cluster (refused before
                any call), or the response holds no JSON array of exactly one vocabulary name
                per cluster.
            DetectorCallLimitExceeded: The call would be a live call past ``max_live_calls``;
                it was not sent.
            DetectorCallError: The model or the cassette failed; the original error is chained.
            DetectorConfigError: The model's config now names another model id than the one the
                namer was built for, or cannot be read; nothing was sent.
        """
        _check_clusters(clusters)
        shown = tuple(
            (min(len(cluster), self._max_critiques), len(cluster)) for cluster in clusters
        )
        prompt = _render_prompt(self._vocabulary, clusters, self._max_critiques)
        messages: Messages = [{"role": "user", "content": [{"text": prompt}]}]
        with self._lock:
            # Read before the call, so the result records the configuration it was made with.
            config_sent = self._prepared.config_sent()
            stack = self._prepared.model
            stack.arm(prompt)
            try:
                text = run_stream(stack, messages)
            except Exception as exc:
                error = call_error(exc)
                if error is exc:
                    raise
                raise error from error.__cause__
            names = _suffix_repeats(_parse_names(text, known=self._known, count=len(clusters)))
            self.results.append(
                NamingResult(
                    names=names,
                    shown=shown,
                    namer_prompt_version=NAMER_PROMPT_VERSION,
                    namer_prompt_sha256=NAMER_PROMPT_SHA256,
                    max_critiques_per_cluster=self._max_critiques,
                    provenance=self._provenance(config_sent),
                )
            )
        return list(names)

    def _provenance(self, config_sent: dict[str, Any]) -> DetectorProvenance:
        return DetectorProvenance(
            detector_package=_DETECTOR_DISTRIBUTION,
            detector_version=self._template.detector_version,
            prompt_template=self._template.prompt_template,
            prompt_sha256=self._template.prompt_sha256,
            model_id=self._prepared.model_id,
            model_config_sent=config_sent,
            confidence_threshold=None,
            stamp=capture_provenance(now=self._now, extra_distributions=(_DETECTOR_DISTRIBUTION,)),
        )


def _render_prompt(
    vocabulary: FailureVocabulary, clusters: Sequence[Sequence[str]], max_critiques: int
) -> str:
    """The full prompt: the template filled with the vocabulary and each cluster's critiques.

    Built only from sequences in their given order, so the same input gives the same text in
    every process and a recorded call replays.
    """
    entries = "\n".join(
        _ENTRY_LINE.format(name=entry.name, definition=entry.definition)
        for entry in vocabulary.entries
    )
    blocks = []
    for number, cluster in enumerate(clusters, start=1):
        shown = cluster[:max_critiques]
        heading = _CLUSTER_HEADING.format(number=number, shown=len(shown), total=len(cluster))
        # Each critique as a JSON string, so a line break inside one cannot read as a new line.
        lines = [
            _CRITIQUE_LINE.format(critique=json.dumps(critique, ensure_ascii=False))
            for critique in shown
        ]
        blocks.append("\n".join((heading, *lines)))
    return _PROMPT_TEMPLATE.format(
        vocabulary=entries, count=len(clusters), clusters="\n\n".join(blocks)
    )


def _is_name_array(value: Any) -> bool:
    """Whether a decoded JSON value is a non-empty array of strings."""
    return isinstance(value, list) and bool(value) and all(isinstance(v, str) for v in value)


def _parse_names(text: str, *, known: frozenset[str], count: int) -> list[str]:
    """The one array of exactly one vocabulary name per cluster that the response holds.

    Only such an array is a candidate answer, so an example the model quotes beside its answer,
    such as ``["other"]``, is not counted against it. When there is none, the one array of
    strings the response does hold, if any, says what was wrong with it.
    """

    def is_answer(value: Any) -> bool:
        return _is_name_array(value) and len(value) == count and all(v in known for v in value)

    try:
        return pick_one_json(text, opener="[", accept=is_answer)
    except PayloadPickError as exc:
        refused = exc
    problem = None if refused.candidates else _answer_problem(text, known=known, count=count)
    if problem is None:
        raise DraftNamingError(
            f"the namer's response cannot be read as cluster names ({refused}); it must hold one "
            f"JSON array of exactly {_count(count, 'vocabulary name')}, one per cluster, bare, "
            "in one fenced block, or as the only such array in the text. No draft names were "
            "made: name the clusters again"
        ) from refused
    raise DraftNamingError(problem)


def _answer_problem(text: str, *, known: frozenset[str], count: int) -> str | None:
    """What is wrong with the one array of strings ``text`` holds; ``None`` without just one."""
    try:
        names: list[str] = pick_one_json(text, opener="[", accept=_is_name_array)
    except PayloadPickError:
        return None  # the caller's refusal says the response holds no answer
    if len(names) != count:
        return (
            f"the namer's response gives {_count(len(names), 'name')} for "
            f"{_count(count, 'cluster')}; every cluster needs exactly one name, so no draft "
            "names were made: name the clusters again"
        )
    unknown = list(dict.fromkeys(name for name in names if name not in known))
    if unknown:
        return (
            f"the namer's response gives {_count(len(unknown), 'name')} that the vocabulary "
            f"does not hold: {', '.join(_quote(name) for name in unknown)}; every cluster must be "
            "named from the vocabulary, so no draft names were made: name the clusters again"
        )
    return None


def _suffix_repeats(names: Sequence[str]) -> tuple[str, ...]:
    """Keep each name on its first cluster and number its later uses `` #2``, `` #3``."""
    uses: Counter[str] = Counter()
    suffixed: list[str] = []
    for name in names:
        uses[name] += 1
        suffixed.append(name if uses[name] == 1 else f"{name} #{uses[name]}")
    return tuple(suffixed)


def _check_cap(max_critiques: object) -> None:
    if isinstance(max_critiques, bool) or not isinstance(max_critiques, int) or max_critiques < 1:
        raise DetectorConfigError(
            f"max_critiques_per_cluster={max_critiques!r} is not a number of critiques to show: "
            f"pass a whole number, 1 or more (the default is {_DEFAULT_MAX_CRITIQUES})"
        )


def _check_vocabulary(vocabulary: object) -> None:
    if not isinstance(vocabulary, FailureVocabulary):
        raise DetectorConfigError(
            f"the vocabulary must be a FailureVocabulary, not {type(vocabulary).__name__}; "
            "failure_vocabulary() reads the installed detector's"
        )
    if not vocabulary.entries:
        raise DetectorConfigError(
            "the vocabulary holds no names, so no cluster could be named: pass failure_vocabulary()"
        )


def _check_now(now: object) -> None:
    if now is not None and (not isinstance(now, datetime) or now.utcoffset() is None):
        raise DetectorConfigError(
            f"now={now!r} is not a timezone-aware datetime: pass one with a timezone, such as "
            "datetime.now(tz=UTC), or leave now out to read the clock"
        )


def _check_clusters(clusters: Sequence[Sequence[str]]) -> None:
    if not clusters:
        raise DraftNamingError(
            "there are no clusters to name, so no model call was made: pass at least one cluster"
        )
    empty = [number for number, cluster in enumerate(clusters, start=1) if not cluster]
    if empty:
        raise DraftNamingError(
            f"cluster {empty[0]} holds no critiques, so nothing could name it and no model call "
            "was made: pass only clusters with at least one critique"
        )


def _count(number: int, noun: str) -> str:
    return f"{number} {noun}" if number == 1 else f"{number} {noun}s"


def _quote(name: str) -> str:
    if len(name) <= _QUOTE_LIMIT:
        return repr(name)
    return f"{name[:_QUOTE_LIMIT]!r} (cut to its first {_QUOTE_LIMIT} characters)"
