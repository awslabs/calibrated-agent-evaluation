# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""The installed detector packages still have the surface Calibrated Agent Evaluation was verified
against.

The vocabulary, the detector's prompt and every vendor name the detector adapter relies on were
read from strands-agents-evals 1.6.0, strands-agents 1.45.0 and botocore 1.43.106. A release that
changes any of them fails here, in ``just check``, before a detection or a replay meets the
changed surface. The two private vendor helpers the adapter mirrors, ``_serialize_session`` (the
preflight rebuilds the prompt's session string) and ``_is_context_exceeded`` (the adapter keeps a
copy of its substrings), are read here and nowhere else.
"""

from __future__ import annotations

import hashlib
import inspect
import itertools
import typing
from collections.abc import Callable, Mapping, Sequence
from types import ModuleType, SimpleNamespace
from typing import Any, Literal

import pytest

pytest.importorskip("strands_evals")

import botocore.exceptions
import botocore.loaders
import strands.models.model as strands_model_module
from strands.models.bedrock import BedrockModel
from strands.models.model import Model
from strands_evals.detectors import (
    ConfidenceLevel,
    FailureDetectionStructuredOutput,
    detect_failures,
)
from strands_evals.detectors.chunking import estimate_tokens, would_exceed_context
from strands_evals.detectors.prompt_templates.failure_detection import get_template
from strands_evals.detectors.utils import _is_context_exceeded, _serialize_session

from _helpers import load_corpus_trace
from calibrated_agent_evaluation.agentcore.detectors.models import CONTEXT_SUBSTRINGS
from calibrated_agent_evaluation.agentcore.detectors.session import session_json, trace_to_session
from calibrated_agent_evaluation.agentcore.detectors.vocabulary import (
    TEMPLATE_VERSION,
    failure_vocabulary,
    parse_categories,
    template_provenance,
)
from calibrated_agent_evaluation.agentcore.errors import VocabularyParseError

VERIFIED_PACKAGE = "strands-agents-evals 1.6.0"
"""The detector release the values below were copied from."""

VERIFIED_NAMES: tuple[str, ...] = (
    "execution-error-category-authentication",
    "execution-error-category-resource-not-found",
    "execution-error-category-service-errors",
    "execution-error-category-rate-limiting",
    "execution-error-category-formatting",
    "execution-error-category-timeout",
    "execution-error-category-resource-exhaustion",
    "execution-error-category-environment",
    "execution-error-category-tool-schema",
    "task-instruction-category-non-compliance",
    "task-instruction-category-problem-id",
    "incorrect-actions-category-tool-selection",
    "incorrect-actions-category-poor-information-retrieval",
    "incorrect-actions-category-clarification",
    "incorrect-actions-category-inappropriate-info-request",
    "context-handling-error-category-context-handling-failures",
    "hallucination-category-hall-capabilities",
    "hallucination-category-hall-misunderstand",
    "hallucination-category-hall-usage",
    "hallucination-category-hall-history",
    "hallucination-category-hall-params",
    "hallucination-category-fabricate-tool-outputs",
    "repetitive-behavior-category-repetition-tool",
    "repetitive-behavior-category-repetition-info",
    "repetitive-behavior-category-step-repetition",
    "orchestration-related-errors-category-reasoning-mismatch",
    "orchestration-related-errors-category-goal-deviation",
    "orchestration-related-errors-category-premature-termination",
    "orchestration-related-errors-category-unaware-termination",
    "llm-output-category-nonsensical",
    "configuration-mismatch-category-tool-definition",
    "coding-use-case-specific-failure-types-category-edge-case-oversights",
    "coding-use-case-specific-failure-types-category-dependency-issues",
)
"""The template's 33 category names in table order, as the verified release ships them."""

VERIFIED_PROMPT_SHA256 = "30d6baa2559a514b87e9477f58afefb977371cc413f5c841eaa3ad9798c809a9"
"""The verified template's hash: its system prompt, a blank line, then the empty-session prompt."""

_OTHER_NAME = "other"
_OTHER_GUIDELINE_PREFIX = '- **Use "other"**: '
_INSIGHTS_SERVICE = "bedrock-agentcore"
_INSIGHTS_SHAPE = "InsightsFailureCategory"
# AgentCore Insights stopped taking new jobs pending a redesign, so a changed or missing enum is
# the likeliest sign that the redesign has shipped.
_REDESIGN_HINT = (
    "re-read the AgentCore Insights redesign before changing the vocabulary or this check"
)

Template = ModuleType | SimpleNamespace


def surface_problems(template: Template, insights_enum: Sequence[str] | None) -> list[str]:
    """Each way the template and the Insights enum differ from the verified surface, as a message.

    Args:
        template: The failure-detection template module, or a stand-in carrying its attributes.
        insights_enum: botocore's ``InsightsFailureCategory`` values in order, ``None`` when the
            installed service model has none.

    Returns:
        One plain message per difference; empty when the surface is the verified one.
    """
    problems: list[str] = []
    names: tuple[str, ...] | None
    try:
        names = tuple(entry.name for entry in parse_categories(template.CATEGORIES))
    except VocabularyParseError as exc:
        names = None
        problems.append(f"the template's category table no longer reads as a vocabulary: {exc}")
    if names is not None and names != VERIFIED_NAMES:
        problems.append(
            f"the template's category names differ from the {len(VERIFIED_NAMES)} verified on "
            f"{VERIFIED_PACKAGE}: {_first_difference(names, VERIFIED_NAMES)}"
        )
    digest = _prompt_sha256(template)
    if digest != VERIFIED_PROMPT_SHA256:
        problems.append(
            f"the template's prompt hash is {digest}, not the {VERIFIED_PROMPT_SHA256} verified on "
            f"{VERIFIED_PACKAGE}; read the installed prompt before updating the verified values"
        )
    if insights_enum is None:
        problems.append(
            f"botocore's {_INSIGHTS_SERVICE} service model has no {_INSIGHTS_SHAPE} enum; "
            f"{_REDESIGN_HINT}"
        )
    elif names is not None and tuple(insights_enum) != (*names, _OTHER_NAME):
        problems.append(
            f"botocore's {_INSIGHTS_SHAPE} enum differs from the vocabulary names (the table's "
            f"names, then {_OTHER_NAME!r}): "
            f"{_first_difference(insights_enum, (*names, _OTHER_NAME))}; {_REDESIGN_HINT}"
        )
    return problems


def installed_service_model() -> Mapping[str, Any]:
    """botocore's bundled ``bedrock-agentcore`` service model, empty when botocore has none.

    A local file read: no client is built, no credentials are resolved, no connection is opened.
    """
    try:
        return botocore.loaders.create_loader().load_service_model(_INSIGHTS_SERVICE, "service-2")
    except botocore.exceptions.DataNotFoundError:
        return {}


def insights_enum(service_model: Mapping[str, Any]) -> list[str] | None:
    """The ``InsightsFailureCategory`` values of a service model, ``None`` when it has none."""
    shape = service_model.get("shapes", {}).get(_INSIGHTS_SHAPE)
    values = None if shape is None else shape.get("enum")
    return None if values is None else list(values)


def _prompt_sha256(template: Template) -> str:
    """The template's hash, as the vocabulary's provenance computes it for the installed one."""
    prompt = f"{template.SYSTEM_PROMPT}\n\n{template.build_prompt(session_json='[]')}"
    return hashlib.sha256(prompt.encode()).hexdigest()


def _first_difference(installed: Sequence[str], expected: Sequence[str]) -> str:
    """Where ``installed`` first departs from ``expected``, counting entries from 1."""
    position, have, want = next(
        (position, have, want)
        for position, (have, want) in enumerate(itertools.zip_longest(installed, expected), 1)
        if have != want
    )
    reordered = (
        " (the same names in another order)" if sorted(installed) == sorted(expected) else ""
    )
    return f"entry {position} is {_shown(have)} where {_shown(want)} was expected{reordered}"


def _shown(value: str | None) -> str:
    return "nothing" if value is None else repr(value)


def _parameters(function: Callable[..., object]) -> list[tuple[str, object, object]]:
    return [
        (parameter.name, parameter.kind, parameter.default)
        for parameter in inspect.signature(function).parameters.values()
    ]


def _template_with(**changes: object) -> SimpleNamespace:
    """A stand-in carrying the installed template's attributes, with ``changes`` applied."""
    template = get_template(TEMPLATE_VERSION)
    attributes = {name: value for name, value in vars(template).items() if not name.startswith("_")}
    return SimpleNamespace(**{**attributes, **changes})


def test_the_installed_detector_matches_the_verified_surface() -> None:
    template = get_template(TEMPLATE_VERSION)
    enum = insights_enum(installed_service_model())

    assert surface_problems(template, enum) == []
    assert template_provenance().prompt_sha256 == VERIFIED_PROMPT_SHA256
    assert enum == list(failure_vocabulary().names())
    assert any(
        line.startswith(_OTHER_GUIDELINE_PREFIX)
        for line in template.DEFAULT_ANNOTATION_GUIDELINES.splitlines()
    )

    empty = inspect.Parameter.empty
    positional = inspect.Parameter.POSITIONAL_OR_KEYWORD
    keyword = inspect.Parameter.KEYWORD_ONLY
    assert _parameters(detect_failures) == [
        ("session", positional, empty),
        ("confidence_threshold", keyword, ConfidenceLevel.LOW),
        ("model", keyword, None),
    ]
    assert _parameters(would_exceed_context) == [
        ("prompt_text", positional, empty),
        ("max_input_tokens", positional, 200_000),
    ]

    errors = FailureDetectionStructuredOutput.model_fields
    assert list(errors) == ["errors"]
    assert typing.get_origin(errors["errors"].annotation) is list
    (entry,) = typing.get_args(errors["errors"].annotation)
    assert {name: field.annotation for name, field in entry.model_fields.items()} == {
        "location": str,
        "category": list[str],
        "confidence": list[Literal["low", "medium", "high"]],
        "evidence": list[str],
    }

    for trace_id in ("ORD-1001", "CLM-4471"):
        converted = trace_to_session(load_corpus_trace(trace_id))
        assert session_json(converted) == _serialize_session(converted.session), trace_id
    assert [s for s in CONTEXT_SUBSTRINGS if not _is_context_exceeded(RuntimeError(s))] == []

    assert Model.__abstractmethods__ == frozenset(
        {"get_config", "stream", "structured_output", "update_config"}
    )
    assert _parameters(BedrockModel.__init__) == [
        ("self", positional, empty),
        ("boto_session", keyword, None),
        ("boto_client_config", keyword, None),
        ("region_name", keyword, None),
        ("endpoint_url", keyword, None),
        ("model_config", inspect.Parameter.VAR_KEYWORD, empty),
    ]
    # Without this helper the token estimate is characters / 3 and loads nothing; a strands
    # release that adds it would make the estimate fetch tiktoken's encoding over the network.
    assert not hasattr(strands_model_module, "_get_encoding")
    assert estimate_tokens("x" * 30) == 10


def test_a_changed_category_name_is_reported() -> None:
    template = get_template(TEMPLATE_VERSION)
    installed, renamed = VERIFIED_NAMES[2], "execution-error-category-service-faults"
    changed = _template_with(
        CATEGORIES=template.CATEGORIES.replace(f"|{installed}|", f"|{renamed}|")
    )

    names_problem, enum_problem = surface_problems(
        changed, insights_enum(installed_service_model())
    )

    assert "category names differ from the 33 verified on strands-agents-evals 1.6.0" in (
        names_problem
    )
    assert f"entry 3 is {renamed!r} where {installed!r} was expected" in names_problem
    assert f"entry 3 is {installed!r} where {renamed!r} was expected" in enum_problem


def test_a_changed_prompt_is_reported() -> None:
    template = get_template(TEMPLATE_VERSION)
    changed = _template_with(SYSTEM_PROMPT=f"{template.SYSTEM_PROMPT}\nBe brief.")

    (problem,) = surface_problems(changed, insights_enum(installed_service_model()))

    assert f"not the {VERIFIED_PROMPT_SHA256} verified on strands-agents-evals 1.6.0" in problem


class _LoaderWithoutAgentCore:
    """A botocore loader whose bundled data has no ``bedrock-agentcore`` service model."""

    def load_service_model(
        self, service_name: str, type_name: str, api_version: str | None = None
    ) -> Mapping[str, Any]:
        raise botocore.exceptions.UnknownServiceError(
            service_name=service_name, known_service_names="bedrock-runtime"
        )


def test_a_changed_or_missing_insights_enum_is_reported(monkeypatch: pytest.MonkeyPatch) -> None:
    template = get_template(TEMPLATE_VERSION)
    service_model = installed_service_model()
    enum = insights_enum(service_model)
    assert enum is not None
    without_shape = {
        **service_model,
        "shapes": {k: v for k, v in service_model["shapes"].items() if k != _INSIGHTS_SHAPE},
    }

    (longer,) = surface_problems(template, [*enum, "tool-misuse"])
    (absent,) = surface_problems(template, None)
    (shape_removed,) = surface_problems(template, insights_enum(without_shape))
    monkeypatch.setattr(botocore.loaders, "create_loader", lambda: _LoaderWithoutAgentCore())
    (service_removed,) = surface_problems(template, insights_enum(installed_service_model()))

    assert "entry 35 is 'tool-misuse' where nothing was expected" in longer
    assert "has no InsightsFailureCategory enum" in absent
    assert "re-read the AgentCore Insights redesign" in absent
    assert "re-read the AgentCore Insights redesign" in longer
    assert shape_removed == service_removed == absent
