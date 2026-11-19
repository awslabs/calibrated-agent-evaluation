# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""A value JSON cannot represent is refused by one check: a NaN or infinite number anywhere in it,
or nesting deeper than the fixed limit, the same on every supported Python and never by running
out of stack. Every nested value here is built with a loop, never by recursion."""

from __future__ import annotations

import json
import math
from collections.abc import Callable
from types import MappingProxyType

import pytest

from calibrated_agent_evaluation._finite import (
    MAX_JSON_DEPTH,
    NestedTooDeepError,
    NonFiniteNumberError,
    refuse_json_constant,
    require_finite,
)

_WHERE = "the test value"
_SENTINEL = "SENTINEL-4c1d"


def _nested(levels: int, wrap: Callable[[object], object], leaf: object = 1.5) -> object:
    """``leaf`` inside ``levels`` containers, each made by ``wrap``."""
    value = leaf
    for _ in range(levels):
        value = wrap(value)
    return value


# A set's items are walked too, and each set counts as one level.
_WRAPPERS = [
    pytest.param(lambda value: {_SENTINEL: value}, id="dicts"),
    pytest.param(lambda value: [value], id="lists"),
    pytest.param(lambda value: (value,), id="tuples"),
    pytest.param(lambda value: frozenset({value}), id="frozensets"),
]


@pytest.mark.parametrize(
    ("number", "rule"),
    [(math.nan, "NaN"), (math.inf, "infinite"), (-math.inf, "infinite")],
    ids=["nan", "inf", "-inf"],
)
@pytest.mark.parametrize(
    "place",
    [
        pytest.param(lambda number: number, id="top-level"),
        pytest.param(lambda number: {"a": {_SENTINEL: number}}, id="nested-dict-value"),
        pytest.param(lambda number: {"a": [1, "two", number]}, id="list-item"),
        pytest.param(lambda number: [("x", number)], id="tuple-item"),
        pytest.param(lambda number: {"a": {number: _SENTINEL}}, id="mapping-key"),
        pytest.param(lambda number: {"tags": {1.0, number}}, id="set-item"),
        pytest.param(lambda number: [frozenset({number})], id="frozenset-item"),
    ],
)
def test_require_finite_refuses_nan_and_infinities_at_any_depth(
    number: float, rule: str, place: Callable[[float], object]
) -> None:
    with pytest.raises(NonFiniteNumberError) as refused:
        require_finite(place(number), where=_WHERE)

    message = str(refused.value)
    assert _WHERE in message
    assert rule in message
    assert _SENTINEL not in message
    assert "two" not in message
    # A ValueError, so every caller's existing "not JSON" handling catches it.
    assert isinstance(refused.value, ValueError)


@pytest.mark.parametrize(
    "value",
    [
        0,
        -7,
        10**30,
        1.5,
        -0.0,
        1.7976931348623157e308,
        True,
        "NaN",
        "Infinity",
        None,
        {},
        [],
        {"a": [1, 2.5, None, True, "x"], "b": {"c": (1, 2)}},
        MappingProxyType({"a": 1.0}),
        frozenset({1.0, 2}),
    ],
    ids=repr,
)
def test_require_finite_accepts_finite_numbers_and_other_values(value: object) -> None:
    assert require_finite(value, where=_WHERE) is None


@pytest.mark.parametrize("text", ["1e999", "[-1e999]", '{"score": 1e999}'])
def test_a_literal_too_large_for_a_float_is_refused_after_decoding(text: str) -> None:
    # The decoder reads it as an infinity without calling the constant hook, so the check after
    # decoding is what refuses it.
    decoded = json.loads(text, parse_constant=refuse_json_constant)

    with pytest.raises(NonFiniteNumberError) as refused:
        require_finite(decoded, where=_WHERE)

    assert "infinite" in str(refused.value)
    assert _WHERE in str(refused.value)


@pytest.mark.parametrize(
    ("text", "token"),
    [("NaN", "NaN"), ("[1, Infinity]", "Infinity"), ('{"a": -Infinity}', "-Infinity")],
)
def test_the_json_constant_hook_refuses_nan_and_infinity_tokens(text: str, token: str) -> None:
    with pytest.raises(NonFiniteNumberError) as refused:
        json.loads(text, parse_constant=refuse_json_constant)

    assert token in str(refused.value)


@pytest.mark.parametrize("wrap", _WRAPPERS)
def test_require_finite_accepts_256_levels_and_refuses_257(
    wrap: Callable[[object], object],
) -> None:
    assert MAX_JSON_DEPTH == 256
    require_finite(_nested(256, wrap), where=_WHERE)

    with pytest.raises(NestedTooDeepError) as refused:
        require_finite(_nested(257, wrap), where=_WHERE)

    message = str(refused.value)
    assert _WHERE in message
    assert "256" in message
    assert _SENTINEL not in message
    assert "1.5" not in message
    assert isinstance(refused.value, ValueError)
    with pytest.raises(NonFiniteNumberError):
        require_finite(_nested(256, wrap, leaf=math.nan), where=_WHERE)


@pytest.mark.parametrize("wrap", _WRAPPERS)
def test_require_finite_refuses_a_value_nested_100000_levels_without_recursion_error(
    wrap: Callable[[object], object],
) -> None:
    deep = _nested(100_000, wrap)

    with pytest.raises(NestedTooDeepError) as refused:
        require_finite(deep, where=_WHERE)

    assert not isinstance(refused.value, RecursionError)
    assert _WHERE in str(refused.value)


def test_require_finite_refuses_a_value_that_holds_itself() -> None:
    # A cycle never ends, so the walk stops at the depth limit instead of looping.
    looped: list[object] = []
    looped.append(looped)

    with pytest.raises(NestedTooDeepError):
        require_finite(looped, where=_WHERE)
