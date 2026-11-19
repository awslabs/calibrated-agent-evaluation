# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Scenarios encode to AgentCore's documented example shapes and decode back without loss."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from calibrated_agent_evaluation.agentcore.datasets_codec import (
    PREDEFINED_SCHEMA,
    SIMULATED_SCHEMA,
    decode_example,
    encode_example,
)
from calibrated_agent_evaluation.agentcore.errors import (
    DatasetDecodeError,
    DatasetSchemaUnsupported,
)
from calibrated_agent_evaluation.scenarios.build import expected_text
from calibrated_agent_evaluation.scenarios.types import (
    ActorProfile,
    CalibratedAgentEvaluationScenarioMeta,
    PredefinedScenario,
    Scenario,
    SimulatedScenario,
    Turn,
)

FIXTURES = Path(__file__).resolve().parent.parent / "fixtures"
ORD_EXPECTED = {
    "action": "REFUND_DENIED",
    "amount_usd": 0,
    "basis": "out-of-window: days_since_delivery=51 > 30-day return window "
    "(despite the tool's within_window=true)",
}


def _ord_1001() -> PredefinedScenario:
    return PredefinedScenario(
        scenario_id="ORD-1001",
        turns=[
            Turn(
                input={
                    "message": "I want a refund for order A-7781, the blender just isn't for me."
                },
                expected_response=expected_text(ORD_EXPECTED),
            )
        ],
        expected_trajectory=["lookup_order", "check_refund_policy"],
        assertions=["Agent denies a refund outside the return window"],
        calibrated_agent_evaluation=CalibratedAgentEvaluationScenarioMeta(
            source_trace_id="ORD-1001",
            frame="production-representative",
            taxonomy_cell={"feature": "refund_handling", "persona": "P1_consumer"},
            failure_modes=["trusted_broken_policy_tool"],
            expected=ORD_EXPECTED,
        ),
    )


def _angry_customer() -> SimulatedScenario:
    return SimulatedScenario(
        scenario_id="angry-late-refund",
        actor_profile=ActorProfile(
            context="Order A-7781 arrived 51 days ago",
            goal="Get a refund to the original card",
            traits={"tone": "angry"},
        ),
        input="Where is my refund?!",
        max_turns=6,
        assertions=["Agent explains the 30-day return window"],
    )


def test_scenarios_encode_to_the_documented_example_shapes() -> None:
    assert encode_example(_ord_1001()) == {
        "scenario_id": "ORD-1001",
        "turns": [
            {
                "input": {
                    "message": "I want a refund for order A-7781, the blender just isn't for me."
                },
                "expected_response": (
                    '{"action":"REFUND_DENIED","amount_usd":0,"basis":"out-of-window: '
                    "days_since_delivery=51 > 30-day return window (despite the tool's "
                    'within_window=true)"}'
                ),
            }
        ],
        "expected_trajectory": ["lookup_order", "check_refund_policy"],
        "assertions": ["Agent denies a refund outside the return window"],
        "metadata": {
            "calibrated_agent_evaluation": {
                "source_trace_id": "ORD-1001",
                "frame": "production-representative",
                "taxonomy_cell": {"feature": "refund_handling", "persona": "P1_consumer"},
                "failure_modes": ["trusted_broken_policy_tool"],
                "expected": ORD_EXPECTED,
            }
        },
    }
    assert encode_example(_angry_customer()) == {
        "scenario_id": "angry-late-refund",
        "actor_profile": {
            "context": "Order A-7781 arrived 51 days ago",
            "goal": "Get a refund to the original card",
            "traits": {"tone": "angry"},
        },
        "input": "Where is my refund?!",
        "max_turns": 6,
        "assertions": ["Agent explains the 30-day return window"],
    }


def test_encoding_omits_empty_fields_and_is_order_stable() -> None:
    bare = PredefinedScenario(scenario_id="bare", turns=[Turn(input="hi")])
    plain_actor = SimulatedScenario(
        scenario_id="plain", actor_profile=ActorProfile(context="c", goal="g"), input="hi"
    )

    assert encode_example(bare) == {"scenario_id": "bare", "turns": [{"input": "hi"}]}
    assert encode_example(plain_actor) == {
        "scenario_id": "plain",
        "actor_profile": {"context": "c", "goal": "g"},
        "input": "hi",
        "max_turns": 10,
    }
    # The same expected output with keys in another order gives the same text.
    assert expected_text({"b": 1, "a": 2}) == expected_text({"a": 2, "b": 1}) == '{"a":2,"b":1}'
    # Non-ASCII text is kept as written, not escaped.
    assert expected_text({"note": "remboursé"}) == '{"note":"remboursé"}'


_text = st.text(
    st.characters(codec="utf-8", exclude_categories=("Cs",)), min_size=1, max_size=12
).filter(str.strip)
_json_leaf = st.one_of(st.integers(-5, 5), st.booleans(), _text)
_small_dict = st.dictionaries(_text, _json_leaf, max_size=3)
_meta = st.builds(
    CalibratedAgentEvaluationScenarioMeta,
    source_trace_id=st.none() | _text,
    frame=st.none() | st.sampled_from(["discovery", "production-representative"]),
    taxonomy_cell=st.dictionaries(_text, _text, max_size=2),
    failure_modes=st.lists(_text, max_size=2),
    expected=st.none() | _small_dict,
)
_foreign_metadata = st.dictionaries(
    _text.filter(lambda k: k != "calibrated_agent_evaluation"), _json_leaf, max_size=2
)
_input = st.one_of(_text, st.dictionaries(_text, _text, min_size=1, max_size=2))
_predefined = st.builds(
    PredefinedScenario,
    scenario_id=_text,
    turns=st.lists(
        st.builds(Turn, input=_input, expected_response=st.none() | _text), min_size=1, max_size=3
    ),
    expected_trajectory=st.lists(_text, max_size=3),
    assertions=st.lists(_text, max_size=2),
    metadata=_foreign_metadata,
    calibrated_agent_evaluation=_meta,
)
_simulated = st.builds(
    SimulatedScenario,
    scenario_id=_text,
    actor_profile=st.builds(ActorProfile, context=_text, goal=_text, traits=_small_dict),
    input=_input,
    max_turns=st.integers(1, 20),
    scenario_description=st.just("") | _text,
    assertions=st.lists(_text, max_size=2),
    metadata=_foreign_metadata,
    calibrated_agent_evaluation=_meta,
)


@settings(max_examples=150, deadline=None)
@given(st.one_of(_predefined, _simulated))
def test_decode_of_encode_returns_the_same_scenario(scenario: Scenario) -> None:
    schema = PREDEFINED_SCHEMA if scenario.form == "predefined" else SIMULATED_SCHEMA
    # Through JSON text too, as the documents travel over the wire.
    wire = json.loads(json.dumps(encode_example(scenario)))

    assert decode_example(wire, schema) == scenario


@pytest.mark.parametrize(
    ("change", "named"),
    [
        (lambda d: d | {"expected_output": "x"}, "expected_output"),
        (lambda d: d | {"exampleIdx": "x"}, "exampleIdx"),
        (
            lambda d: {
                **d,
                "metadata": {"calibrated_agent_evaluation": {"source_trace": "ORD-1001"}},
            },
            "source_trace",
        ),
        (lambda d: {**d, "turns": [{"input": "hi", "expected": "x"}]}, "expected"),
    ],
)
def test_decode_refuses_unknown_fields_and_drops_the_example_id(change: Any, named: str) -> None:
    doc = encode_example(_ord_1001())

    with pytest.raises(DatasetDecodeError, match=named) as refused:
        decode_example(change(doc), PREDEFINED_SCHEMA)
    assert "ORD-1001" in str(refused.value)

    listed = doc | {"exampleId": "00000000-0000-4000-8000-000000000003"}
    decoded = decode_example(listed, PREDEFINED_SCHEMA)
    assert decoded == _ord_1001()
    assert "exampleId" not in encode_example(decoded)


def test_unsupported_schemas_are_refused_by_name() -> None:
    captured = json.loads(
        (FIXTURES / "raw/agentcore/datasets/list_dataset_examples_third_party.json").read_text()
    )
    third_party_doc = captured["examples"][0]

    for schema in ("THIRD_PARTY_EVALUATION_V1", "SOMETHING_ELSE_V9"):
        with pytest.raises(DatasetSchemaUnsupported, match=schema) as refused:
            decode_example(third_party_doc, schema)
        assert PREDEFINED_SCHEMA in str(refused.value)
        assert SIMULATED_SCHEMA in str(refused.value)
