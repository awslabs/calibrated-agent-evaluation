# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""The draft namer names clusters from the detector vocabulary in one model call, or refuses.

Every naming call sends one prompt listing every vocabulary name with its definition and each
cluster's first critiques, and reads back one JSON array of vocabulary names, one per cluster. A
response it cannot read that way is refused, never replaced by ``mode_0`` names. A name given to
more than one cluster is suffixed, so each cluster stays its own failure mode, and every call
records which detector, template, prompt and model produced the names.

Unless a test says otherwise, the scripted model stands in for the real one, so each of its calls
is a live call and the namer is built with ``max_live_calls=1``.
"""

from __future__ import annotations

import json
from collections import Counter
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest

pytest.importorskip("strands_evals")

from strands.types.exceptions import ContextWindowOverflowException

from _detector_fakes import ScriptedModel, spy_connections, unset_aws_env
from _helpers import load_corpus_labels, make_label
from calibrated_agent_evaluation.agentcore.detectors._payload import PayloadPickError
from calibrated_agent_evaluation.agentcore.detectors.models import detector_bedrock_model
from calibrated_agent_evaluation.agentcore.detectors.namer import DetectorDraftNamer
from calibrated_agent_evaluation.agentcore.detectors.types import FailureVocabulary
from calibrated_agent_evaluation.agentcore.detectors.vocabulary import failure_vocabulary
from calibrated_agent_evaluation.agentcore.errors import (
    DetectorCallError,
    DetectorCallLimitExceeded,
    DetectorConfigError,
    DraftNamingError,
)
from calibrated_agent_evaluation.cluster import (
    build_draft,
    hierarchical_cluster,
    ratify,
    saturation_check,
    tag_label,
)
from calibrated_agent_evaluation.cluster.embed import DraftNamer
from calibrated_agent_evaluation.cluster.types import DraftCodebook
from calibrated_agent_evaluation.execute import CassetteMiss, CassetteModelMismatch, Mode
from calibrated_agent_evaluation.schema.types import Label
from test_cluster_helpers import FakeEmbedder

NOW = datetime(2026, 10, 27, 9, 30, tzinfo=UTC)
VERIFIED_TEMPLATE_SHA256 = "30d6baa2559a514b87e9477f58afefb977371cc413f5c841eaa3ad9798c809a9"
# The hash of the namer prompt's data-free part. A change to the prompt's wording or layout changes
# it, and a recorded naming cassette would no longer replay, so the value is pinned here.
NAMER_PROMPT_SHA256 = "14d34f523d690189bb47a4cc98a8342492d4aec6586e7e64276d963d15d37fda"


@pytest.fixture(scope="module")
def vocabulary() -> FailureVocabulary:
    return failure_vocabulary()


def _namer(model: Any, vocabulary: FailureVocabulary, **kwargs: Any) -> DetectorDraftNamer:
    kwargs.setdefault("max_live_calls", 1)
    kwargs.setdefault("now", NOW)
    return DetectorDraftNamer(model, vocabulary, **kwargs)


def _answer(*names: str) -> str:
    return json.dumps(list(names))


def _critiques(prefix: str, count: int) -> list[str]:
    return [f"{prefix} critique {number:02d}" for number in range(1, count + 1)]


def _sent_prompt(model: ScriptedModel) -> str:
    """The text of the one user message the model was sent."""
    (messages,) = model.calls
    (message,) = messages
    assert message["role"] == "user"
    (block,) = message["content"]
    assert "text" in block
    return block["text"]


def _line_holding(lines: list[str], *parts: str) -> int:
    matches = [index for index, line in enumerate(lines) if all(part in line for part in parts)]
    assert len(matches) == 1, f"expected one prompt line holding {parts!r}, found {len(matches)}"
    return matches[0]


def _cassette_entries(path: Path) -> list[dict[str, Any]]:
    return json.loads(path.read_text(encoding="utf-8"))["entries"]


def _mode_counts(
    labels: list[Label], draft: DraftCodebook, *, descriptions: dict[str, str] | None = None
) -> list[int]:
    """Ratify the draft, tag each label with its cluster's mode, count traces per mode.

    The counts are taken the way ``calibrated-agent-evaluation cluster`` takes them: a mode's count
    is the number of distinct traces with a label tagged with it. They are returned in draft order.
    """
    codebook = ratify(draft, ratified_by="namer-test", descriptions=descriptions)
    mode_of = {
        index: cluster.name for cluster in draft.clusters for index in cluster.member_indices
    }
    tagged = [tag_label(label, mode_of[index], codebook) for index, label in enumerate(labels)]
    per_mode = Counter(mode for _, mode in {(lb.trace_id, lb.failure_mode) for lb in tagged})
    return [per_mode[cluster.name] for cluster in draft.clusters]


def test_the_namer_returns_one_vocabulary_name_per_cluster(vocabulary: FailureVocabulary) -> None:
    names = vocabulary.names()
    chosen = (names[0], names[5], names[-1])
    scripted = ScriptedModel([_answer(*chosen)])
    big, single, pair = _critiques("big", 12), _critiques("single", 1), _critiques("pair", 2)
    namer = _namer(scripted, vocabulary)

    assert namer.name_clusters([big, single, pair]) == list(chosen)

    prompt = _sent_prompt(scripted)
    lines = prompt.splitlines()
    entry_lines = [_line_holding(lines, e.name, e.definition) for e in vocabulary.entries]
    assert entry_lines == sorted(entry_lines)
    assert "10 of 12" in prompt
    shown_lines = [_line_holding(lines, critique) for critique in big[:10]]
    assert shown_lines == sorted(shown_lines)
    assert [critique for critique in big[10:] if critique in prompt] == []
    assert all(critique in prompt for critique in single + pair)
    (result,) = namer.results
    assert result.names == chosen
    assert result.shown == ((10, 12), (1, 1), (2, 2))
    assert result.max_critiques_per_cluster == 10


def test_every_naming_result_records_its_provenance(vocabulary: FailureVocabulary) -> None:
    names = vocabulary.names()
    config = {"model_id": "scripted-namer", "temperature": 0.0}
    scripted = ScriptedModel([_answer(names[1]), _answer(names[2], names[3])], config=config)
    namer = _namer(scripted, vocabulary, max_live_calls=2)

    namer.name_clusters([["the agent refunded twice"]])
    namer.name_clusters([["the agent invented an order"], ["the agent looped on one tool"]])

    assert len(namer.results) == 2
    for result in namer.results:
        provenance = result.provenance
        assert provenance.detector_package == "strands-agents-evals"
        assert provenance.detector_version == "1.6.0"
        assert provenance.prompt_template == "v0"
        assert provenance.prompt_sha256 == VERIFIED_TEMPLATE_SHA256
        assert provenance.model_id == "scripted-namer"
        assert provenance.model_config_sent == {"temperature": 0.0}
        assert provenance.confidence_threshold is None
        assert provenance.stamp.dep_versions["strands-agents-evals"] == "1.6.0"
        assert datetime.fromisoformat(provenance.stamp.run_at) == NOW
        assert result.namer_prompt_version == "namer-v1"
    # The same hash for different clusters: it covers the prompt's wording, not its data.
    assert [result.namer_prompt_sha256 for result in namer.results] == [NAMER_PROMPT_SHA256] * 2


def test_the_critique_cap_is_stated_and_must_be_positive(vocabulary: FailureVocabulary) -> None:
    exactly_ten = _critiques("full", 10)
    scripted = ScriptedModel([_answer(vocabulary.names()[0])])
    namer = _namer(scripted, vocabulary)

    namer.name_clusters([exactly_ten])

    prompt = _sent_prompt(scripted)
    assert "10 of 10" in prompt
    assert all(critique in prompt for critique in exactly_ten)
    assert namer.results[0].shown == ((10, 10),)

    with pytest.raises(DetectorConfigError, match="max_critiques_per_cluster"):
        _namer(ScriptedModel([]), vocabulary, max_critiques_per_cluster=0)


def test_repeated_names_are_suffixed_in_cluster_order(vocabulary: FailureVocabulary) -> None:
    a, b, c, d = vocabulary.names()[:4]
    clusters = [[f"critique of cluster {number}"] for number in range(4)]
    repeating = _namer(ScriptedModel([_answer(a, b, a, a)]), vocabulary)
    distinct = _namer(ScriptedModel([_answer(a, b, c, d)]), vocabulary)

    assert repeating.name_clusters(clusters) == [a, b, f"{a} #2", f"{a} #3"]
    assert repeating.results[0].names == (a, b, f"{a} #2", f"{a} #3")
    assert distinct.name_clusters(clusters) == [a, b, c, d]


@pytest.mark.parametrize(
    ("response", "problem", "cause_type"),
    [
        pytest.param(
            lambda names: _answer(*names[:6]), "6 names for 7 clusters", None, id="six-names"
        ),
        pytest.param(
            lambda names: _answer(*names[:6], "made-up-category"),
            "'made-up-category'",
            None,
            id="unknown-name",
        ),
        pytest.param(
            lambda names: (
                f"First guess: {_answer(*names[:7])}\nSecond guess: {_answer(*names[1:8])}"
            ),
            "2 candidates",
            PayloadPickError,
            id="two-arrays-in-prose",
        ),
        pytest.param(
            lambda names: (
                f"```json\n{_answer(*names[:7])}\n```\nOr, read again: {_answer(*names[1:8])}"
            ),
            "2 candidates",
            PayloadPickError,
            id="a-fenced-array-and-another-in-prose",
        ),
    ],
)
def test_a_bad_namer_response_is_refused_and_no_draft_is_built(
    vocabulary: FailureVocabulary,
    response: Callable[[tuple[str, ...]], str],
    problem: str,
    cause_type: type[Exception] | None,
) -> None:
    critiques = [f"failing trace {number} went wrong its own way" for number in range(7)]
    embeddings = FakeEmbedder().embed(critiques)
    namer = _namer(ScriptedModel([response(vocabulary.names())]), vocabulary)

    with pytest.raises(DraftNamingError, match=problem) as refused:
        build_draft(critiques, embeddings, list(range(7)), namer=namer)

    cause = refused.value.__cause__
    assert (None if cause is None else type(cause)) is cause_type
    assert namer.results == []


def test_a_wrapped_name_array_parses_like_a_bare_one(vocabulary: FailureVocabulary) -> None:
    names = vocabulary.names()[2:5]
    bare = _answer(*names)
    wrapped = {
        "bare": bare,
        "after-prose": f"Each name below is for cluster [1], [2] and [3], in order:\n{bare}",
        "fenced": f"Here are the names.\n```json\n{bare}\n```\n",
        # An array that is not one vocabulary name per cluster is no candidate answer.
        "after-an-example": f'An answer looks like ["other"]. Mine:\n{bare}',
        "fenced-after-an-example": f'An answer looks like ["other"].\n```json\n{bare}\n```',
        # The prose before the block ends, and the prose after it starts, with half of a full
        # answer. Joined across the cut-out block they would read as a second candidate.
        "fenced-between-fragments-of-an-answer": (
            f'A draft began ["{names[0]}", "{names[1]}",\n```json\n{bare}\n```\n'
            f'"{names[2]}"] and was cut short.'
        ),
    }
    clusters = [["a"], ["b"], ["c"]]

    named = {
        form: _namer(ScriptedModel([text]), vocabulary).name_clusters(clusters)
        for form, text in wrapped.items()
    }

    assert named == {form: list(names) for form in wrapped}


def test_vocabulary_names_leave_the_saturation_verdict_unchanged(
    vocabulary: FailureVocabulary,
) -> None:
    fails = [label for label in load_corpus_labels() if label.outcome == "FAIL"]
    critiques = [label.critique for label in fails]
    embeddings = FakeEmbedder().embed(critiques)
    assignments = hierarchical_cluster(embeddings, threshold=0.5)
    assert len(set(assignments)) == 7
    names = vocabulary.names()
    namer = _namer(ScriptedModel([_answer(*names[:6], names[0])]), vocabulary)

    named = build_draft(critiques, embeddings, assignments, namer=namer)
    default = build_draft(critiques, embeddings, assignments)

    assert [cluster.name for cluster in named.clusters] == [*names[:6], f"{names[0]} #2"]
    verdict = saturation_check(_mode_counts(fails, named, descriptions=vocabulary.descriptions()))
    assert verdict == saturation_check(_mode_counts(fails, default))
    assert verdict.saturated is False
    assert verdict.reason == "insufficient_observations: 7 < 30, the minimum failure count"


def test_vocabulary_naming_keeps_every_cluster_a_separate_mode_in_the_counts(
    vocabulary: FailureVocabulary,
) -> None:
    sizes = (10, 8, 6, 5, 3)
    assignments = [cluster for cluster, size in enumerate(sizes) for _ in range(size)]
    labels = [
        make_label(trace_id=f"trace-{index:02d}", critique=f"failure {index:02d} in cluster {c}")
        for index, c in enumerate(assignments)
    ]
    critiques = [label.critique for label in labels]
    embeddings = FakeEmbedder().embed(critiques)
    first, second, third, fourth = vocabulary.names()[:4]
    namer = _namer(ScriptedModel([_answer(first, second, first, third, fourth)]), vocabulary)

    named = build_draft(critiques, embeddings, assignments, namer=namer)
    default = build_draft(critiques, embeddings, assignments)
    named_counts = _mode_counts(labels, named, descriptions=vocabulary.descriptions())
    default_counts = _mode_counts(labels, default)

    assert [cluster.name for cluster in named.clusters] == [
        first,
        second,
        f"{first} #2",
        third,
        fourth,
    ]
    assert named_counts == [10, 8, 6, 5, 3]
    verdict = saturation_check(named_counts)
    assert verdict == saturation_check(default_counts)
    assert (verdict.total_failures, verdict.total_modes) == (32, 5)


def test_the_namer_never_uses_the_vendor_default_model(
    vocabulary: FailureVocabulary, monkeypatch: pytest.MonkeyPatch
) -> None:
    connections = spy_connections(monkeypatch)

    with pytest.raises(DetectorConfigError, match="no model was passed"):
        DetectorDraftNamer(None, vocabulary)  # type: ignore[arg-type]
    with pytest.raises(DetectorConfigError, match="model id string"):
        DetectorDraftNamer("global.anthropic.claude-sonnet-4-6", vocabulary)  # type: ignore[arg-type]
    name = vocabulary.names()[7]
    accepted = _namer(ScriptedModel([_answer(name)]), vocabulary)

    assert isinstance(accepted, DraftNamer)
    assert accepted.name_clusters([["the agent ignored the policy"]]) == [name]
    assert connections == []


def _failing_runtime_model(
    tmp_path: Path, vocabulary: FailureVocabulary, clusters: list[list[str]]
) -> DetectorDraftNamer:
    secret = RuntimeError("the endpoint refused token ASIAEXAMPLESECRET0042 for this request")
    return _namer(ScriptedModel([secret]), vocabulary)


def _context_overflow_model(
    tmp_path: Path, vocabulary: FailureVocabulary, clusters: list[list[str]]
) -> DetectorDraftNamer:
    overflow = ContextWindowOverflowException("the request is over the model's input limit")
    return _namer(ScriptedModel([overflow]), vocabulary)


def _too_long_message_model(
    tmp_path: Path, vocabulary: FailureVocabulary, clusters: list[list[str]]
) -> DetectorDraftNamer:
    too_long = RuntimeError("Input is too long for requested model; max_tokens exceeded")
    return _namer(ScriptedModel([too_long]), vocabulary)


def _replay_miss(
    tmp_path: Path, vocabulary: FailureVocabulary, clusters: list[list[str]]
) -> DetectorDraftNamer:
    path = tmp_path / "namer.cassette.json"
    recorder = _namer(
        ScriptedModel([_answer(vocabulary.names()[0])]),
        vocabulary,
        cassette_path=path,
        cassette_mode=Mode.RECORD,
    )
    recorder.name_clusters([["a cluster the replay never asks about"]])
    return _namer(ScriptedModel([]), vocabulary, cassette_path=path, cassette_mode=Mode.REPLAY)


def _replay_under_another_model(
    tmp_path: Path, vocabulary: FailureVocabulary, clusters: list[list[str]]
) -> DetectorDraftNamer:
    path = tmp_path / "namer.cassette.json"
    recorder = _namer(
        ScriptedModel(
            [_answer(vocabulary.names()[0])], config={"model_id": "m-1", "max_tokens": 64}
        ),
        vocabulary,
        cassette_path=path,
        cassette_mode=Mode.RECORD,
    )
    recorder.name_clusters(clusters)
    other = ScriptedModel([], config={"model_id": "m-2", "max_tokens": 64})
    return _namer(other, vocabulary, cassette_path=path, cassette_mode=Mode.REPLAY)


@pytest.mark.parametrize(
    ("build", "cause_type"),
    [
        pytest.param(_failing_runtime_model, RuntimeError, id="model-raises"),
        # Only a detection reads an overflow as "too large"; a naming call reports the failure.
        pytest.param(
            _context_overflow_model, ContextWindowOverflowException, id="model-context-overflow"
        ),
        pytest.param(_too_long_message_model, RuntimeError, id="model-says-too-long"),
        pytest.param(_replay_miss, CassetteMiss, id="cassette-miss"),
        pytest.param(_replay_under_another_model, CassetteModelMismatch, id="other-model"),
    ],
)
def test_namer_model_and_cassette_failures_become_detector_call_errors(
    tmp_path: Path,
    vocabulary: FailureVocabulary,
    build: Callable[[Path, FailureVocabulary, list[list[str]]], DetectorDraftNamer],
    cause_type: type[Exception],
) -> None:
    critique = "the agent quoted the customer's card number back to them"
    clusters = [[critique]]
    namer = build(tmp_path, vocabulary, clusters)

    with pytest.raises(DetectorCallError) as failed:
        namer.name_clusters(clusters)

    assert type(failed.value.__cause__) is cause_type
    message = str(failed.value)
    assert critique not in message
    assert vocabulary.entries[0].definition not in message
    assert "ASIAEXAMPLESECRET0042" not in message


_Unusable = tuple[dict[str, Any], dict[str, Any]]


def _no_model_id(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> _Unusable:
    return {}, {}


def _mismatched_model_id(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> _Unusable:
    return {"model_id": "m-1"}, {"model_id": "m-2"}


def _corrupt_cassette(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> _Unusable:
    corrupt = tmp_path / "corrupt.cassette.json"
    corrupt.write_text("this is not a cassette", encoding="utf-8")
    return {"model_id": "m-1"}, {"cassette_path": corrupt}


def _missing_replay_cassette(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> _Unusable:
    absent = tmp_path / "absent.cassette.json"
    return {"model_id": "m-1"}, {"cassette_path": absent, "cassette_mode": Mode.REPLAY}


def _bad_mode_variable(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> _Unusable:
    monkeypatch.setenv("CALIBRATED_AGENT_EVALUATION_CASSETTE_MODE", "rewind")
    return {"model_id": "m-1"}, {}


def _replay_without_cassette(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> _Unusable:
    return {"model_id": "m-1"}, {"cassette_mode": Mode.REPLAY}


def _replay_variable_without_cassette(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> _Unusable:
    monkeypatch.setenv("CALIBRATED_AGENT_EVALUATION_CASSETTE_MODE", "replay")
    return {"model_id": "m-1"}, {}


@pytest.mark.parametrize(
    ("unusable", "problem"),
    [
        pytest.param(_no_model_id, "no model_id and none was passed", id="no-model-id"),
        pytest.param(_mismatched_model_id, "'m-2' was passed", id="mismatched-model-id"),
        pytest.param(_corrupt_cassette, "corrupt.cassette.json", id="corrupt-cassette"),
        pytest.param(_missing_replay_cassette, "does not exist", id="missing-replay-cassette"),
        pytest.param(
            _bad_mode_variable, "CALIBRATED_AGENT_EVALUATION_CASSETTE_MODE", id="bad-mode-variable"
        ),
        pytest.param(_replay_without_cassette, "no cassette_path", id="replay-without-cassette"),
        pytest.param(
            _replay_variable_without_cassette,
            "no cassette_path",
            id="replay-variable-without-cassette",
        ),
    ],
)
def test_an_unusable_configuration_is_refused_when_the_namer_is_built(
    tmp_path: Path,
    vocabulary: FailureVocabulary,
    monkeypatch: pytest.MonkeyPatch,
    unusable: Callable[[Path, pytest.MonkeyPatch], _Unusable],
    problem: str,
) -> None:
    config, kwargs = unusable(tmp_path, monkeypatch)
    scripted = ScriptedModel([_answer(vocabulary.names()[0])], config=config)

    with pytest.raises(DetectorConfigError, match=problem):
        _namer(scripted, vocabulary, **kwargs)

    assert scripted.calls == []


def test_a_recorded_naming_call_replays_offline_under_its_model_id(
    tmp_path: Path, vocabulary: FailureVocabulary, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "namer.cassette.json"
    names = vocabulary.names()
    clusters = [["the agent paid an excluded claim"], ["the agent skipped the policy check"]]
    recorder = _namer(
        ScriptedModel([_answer(names[4], names[9])], config={}),
        vocabulary,
        model_id="m-1",
        cassette_path=path,
        cassette_mode=Mode.RECORD,
    )
    recorded = recorder.name_clusters(clusters)
    (entry,) = _cassette_entries(path)
    assert entry["model"] == {"id": "m-1", "config": {}}

    unset_aws_env(monkeypatch)
    connections = spy_connections(monkeypatch)
    replayer = _namer(
        detector_bedrock_model("m-1"),
        vocabulary,
        model_id="m-1",
        cassette_path=path,
        cassette_mode=Mode.REPLAY,
        max_live_calls=0,
    )
    assert replayer.name_clusters(clusters) == recorded == [names[4], names[9]]
    assert connections == []

    other = ScriptedModel([], config={})
    wrong_model = _namer(
        other,
        vocabulary,
        model_id="m-2",
        cassette_path=path,
        cassette_mode=Mode.REPLAY,
        max_live_calls=0,
    )
    with pytest.raises(DetectorCallError) as refused:
        wrong_model.name_clusters(clusters)
    assert isinstance(refused.value.__cause__, CassetteModelMismatch)
    assert other.calls == []


def test_a_model_reconfigured_after_the_namer_is_built_is_recorded_as_it_was_called(
    vocabulary: FailureVocabulary,
) -> None:
    names = vocabulary.names()
    scripted = ScriptedModel(
        [_answer(names[0]), _answer(names[1])], config={"model_id": "m-1", "temperature": 0.0}
    )
    namer = _namer(scripted, vocabulary, max_live_calls=2)

    scripted.update_config(temperature=0.9)
    namer.name_clusters([["the agent refunded twice"]])

    (result,) = namer.results
    assert result.provenance.model_id == "m-1"
    assert result.provenance.model_config_sent == {"temperature": 0.9}

    scripted.update_config(model_id="m-2")
    with pytest.raises(DetectorConfigError, match="'m-2'") as refused:
        namer.name_clusters([["the agent invented an order"]])
    assert "'m-1'" in str(refused.value)
    assert len(scripted.calls) == 1
    assert len(namer.results) == 1


def test_the_live_call_limit_reaches_the_namers_caller(vocabulary: FailureVocabulary) -> None:
    scripted = ScriptedModel([_answer(*vocabulary.names()[:3])])
    namer = _namer(scripted, vocabulary, max_live_calls=0)

    with pytest.raises(DetectorCallLimitExceeded, match="max_live_calls=0") as refused:
        namer.name_clusters([["one"], ["two"], ["three"]])

    assert not isinstance(refused.value, DetectorCallError)
    assert refused.value.limit == 0
    assert scripted.calls == []
    assert namer.results == []
