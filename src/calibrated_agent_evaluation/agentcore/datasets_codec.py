# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Convert scenarios to and from AgentCore evaluation dataset examples, losing nothing.

AgentCore stores dataset examples as untyped JSON documents in one of two documented shapes,
``AGENTCORE_EVALUATION_PREDEFINED_V1`` and ``AGENTCORE_EVALUATION_SIMULATED_V1`` (field names
verified against ``bedrock_agentcore.evaluation.runner.dataset_types`` 1.24.2). Calibrated Agent
Evaluation-only fields ride under ``metadata["calibrated_agent_evaluation"]``, so
``decode_example(encode_example(s), schema) == s``. Decoding is strict: the SDK's own parser ignores
keys it does not know, so an unknown field here is a typed refusal, never a silent drop. The listed
form of an example also carries the service-assigned ``exampleId``, which is accepted and left out
of the scenario.

This module is pure (no AWS SDK import); it lives in ``calibrated_agent_evaluation.agentcore``
because it speaks AgentCore's wire shape.
"""

from __future__ import annotations

import hashlib
import json
import re
from typing import Any

from calibrated_agent_evaluation.agentcore.errors import (
    DatasetDecodeError,
    DatasetInputError,
    DatasetSchemaUnsupported,
)
from calibrated_agent_evaluation.scenarios.errors import ScenarioError
from calibrated_agent_evaluation.scenarios.types import (
    CALIBRATED_AGENT_EVALUATION_METADATA_KEY,
    CalibratedAgentEvaluationScenarioMeta,
    DatasetSlice,
    Form,
    PredefinedScenario,
    Scenario,
    SimulatedScenario,
)

__all__ = [
    "ACTOR_FIELDS",
    "NAME_PATTERN",
    "PREDEFINED_FIELDS",
    "PREDEFINED_SCHEMA",
    "SERVICE_FIELDS",
    "SIMULATED_FIELDS",
    "SIMULATED_SCHEMA",
    "SUPPORTED_SCHEMAS",
    "TURN_FIELDS",
    "VERSION_PATTERN",
    "content_digest",
    "decode_example",
    "encode_example",
    "form_for_schema",
    "schema_for_form",
    "validate_dataset_name",
    "validate_version",
]

PREDEFINED_SCHEMA = "AGENTCORE_EVALUATION_PREDEFINED_V1"
SIMULATED_SCHEMA = "AGENTCORE_EVALUATION_SIMULATED_V1"
SUPPORTED_SCHEMAS: dict[str, Form] = {
    PREDEFINED_SCHEMA: "predefined",
    SIMULATED_SCHEMA: "simulated",
}

PREDEFINED_FIELDS = frozenset(
    {"scenario_id", "turns", "expected_trajectory", "assertions", "metadata"}
)
SIMULATED_FIELDS = frozenset(
    {
        "scenario_id",
        "actor_profile",
        "input",
        "max_turns",
        "scenario_description",
        "assertions",
        "metadata",
    }
)
TURN_FIELDS = frozenset({"input", "expected_response"})
ACTOR_FIELDS = frozenset({"context", "goal", "traits"})
SERVICE_FIELDS = frozenset({"exampleId"})


NAME_PATTERN = re.compile(r"^[a-zA-Z][a-zA-Z0-9_]{0,47}$")
VERSION_PATTERN = re.compile(r"^[0-9]+$")


def validate_dataset_name(name: str) -> None:
    """Refuse a dataset name AgentCore would reject, before any call."""
    if not NAME_PATTERN.fullmatch(name):
        raise DatasetInputError(
            f"dataset name {name!r} is not allowed: use 1 to 48 letters, digits and underscores, "
            "starting with a letter (no hyphens)"
        )


def validate_version(version: str) -> None:
    """Refuse anything but a published version number, before any call."""
    if not VERSION_PATTERN.fullmatch(version):
        raise DatasetInputError(
            f"version {version!r} can't be pulled: only published versions (1, 2, ...) can, "
            "because the DRAFT can still change"
        )


def schema_for_form(form: Form) -> str:
    """The AgentCore schema type a slice of ``form`` is published as."""
    return PREDEFINED_SCHEMA if form == "predefined" else SIMULATED_SCHEMA


def form_for_schema(schema_type: str) -> Form:
    """The scenario form a dataset schema holds, or a typed refusal for any other schema."""
    try:
        return SUPPORTED_SCHEMAS[schema_type]
    except KeyError:
        raise DatasetSchemaUnsupported(
            f"dataset schema {schema_type!r} is not supported; Calibrated Agent Evaluation reads "
            "and writes "
            f"{PREDEFINED_SCHEMA} and {SIMULATED_SCHEMA}"
        ) from None


def encode_example(scenario: Scenario) -> dict[str, Any]:
    """The AgentCore example document for ``scenario``; empty optional fields are left out."""
    doc: dict[str, Any] = {"scenario_id": scenario.scenario_id}
    if isinstance(scenario, PredefinedScenario):
        doc["turns"] = [
            {"input": t.input}
            | ({} if t.expected_response is None else {"expected_response": t.expected_response})
            for t in scenario.turns
        ]
        if scenario.expected_trajectory:
            doc["expected_trajectory"] = list(scenario.expected_trajectory)
    else:
        actor: dict[str, Any] = {
            "context": scenario.actor_profile.context,
            "goal": scenario.actor_profile.goal,
        }
        if scenario.actor_profile.traits:
            actor["traits"] = dict(scenario.actor_profile.traits)
        doc["actor_profile"] = actor
        doc["input"] = scenario.input
        doc["max_turns"] = scenario.max_turns
        if scenario.scenario_description:
            doc["scenario_description"] = scenario.scenario_description
    if scenario.assertions:
        doc["assertions"] = list(scenario.assertions)
    metadata = dict(scenario.metadata)
    if not scenario.calibrated_agent_evaluation.is_empty():
        metadata[CALIBRATED_AGENT_EVALUATION_METADATA_KEY] = (
            scenario.calibrated_agent_evaluation.model_dump(mode="json", exclude_defaults=True)
        )
    if metadata:
        doc["metadata"] = metadata
    return doc


def decode_example(doc: object, schema_type: str) -> Scenario:
    """The scenario an AgentCore example document holds, refusing any field that does not map."""
    form = form_for_schema(schema_type)
    if not isinstance(doc, dict):
        raise DatasetDecodeError(
            f"a dataset example must be a JSON object, got {type(doc).__name__}"
        )
    scenario_id = doc.get("scenario_id")
    where = f"scenario {scenario_id!r}" if isinstance(scenario_id, str) else "a dataset example"
    allowed = PREDEFINED_FIELDS if form == "predefined" else SIMULATED_FIELDS
    _refuse_unknown(doc, allowed | SERVICE_FIELDS, where, schema_type)
    fields = {k: v for k, v in doc.items() if k not in SERVICE_FIELDS}

    metadata = fields.pop("metadata", None)
    metadata = {} if metadata is None else metadata
    if not isinstance(metadata, dict):
        raise DatasetDecodeError(f"{where}: metadata must be a JSON object")
    metadata = dict(metadata)
    meta_raw = metadata.pop(CALIBRATED_AGENT_EVALUATION_METADATA_KEY, None)
    try:
        meta = CalibratedAgentEvaluationScenarioMeta.model_validate(
            {} if meta_raw is None else meta_raw
        )
        if form == "predefined":
            for turn in fields.get("turns") or []:
                if isinstance(turn, dict):
                    _refuse_unknown(turn, TURN_FIELDS, f"{where} turn", schema_type)
            return PredefinedScenario.model_validate(
                {**fields, "metadata": metadata, "calibrated_agent_evaluation": meta}
            )
        actor = fields.get("actor_profile")
        if isinstance(actor, dict):
            _refuse_unknown(actor, ACTOR_FIELDS, f"{where} actor_profile", schema_type)
        return SimulatedScenario.model_validate(
            {**fields, "metadata": metadata, "calibrated_agent_evaluation": meta}
        )
    except ScenarioError as exc:
        raise DatasetDecodeError(f"{where} does not map to a {form} scenario: {exc}") from exc


def content_digest(slice_: DatasetSlice) -> str:
    """A SHA-256 over the slice's encoded examples, so an unchanged slice can be recognized."""
    canonical = json.dumps(
        {"form": slice_.form, "examples": [encode_example(s) for s in slice_.scenarios]},
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _refuse_unknown(doc: dict[str, Any], allowed: frozenset[str], where: str, schema: str) -> None:
    unknown = sorted(set(doc) - allowed)
    if unknown:
        raise DatasetDecodeError(
            f"{where} has field(s) {unknown} that {schema} does not define; refusing rather than "
            "dropping them"
        )
