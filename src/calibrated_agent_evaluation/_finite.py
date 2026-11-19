# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Refuse a value that standard JSON cannot represent, by one rule on every supported Python.

Two things make a value one that JSON cannot represent here:

- **A NaN or infinite number.** Python's ``json`` writes and reads the ``NaN``, ``Infinity`` and
  ``-Infinity`` tokens unless told not to, but they are not JSON. A literal too large for a float,
  such as ``1e999``, decodes to an infinity with no token, so a decoded value is checked too.
- **Nesting deeper than** :data:`MAX_JSON_DEPTH` **levels.** How deep Python's encoder and decoder
  go before ``RecursionError`` differs between versions, so a fixed limit, checked before any
  encoder runs, gives one answer everywhere.

:func:`require_finite` checks a value for both, and :func:`refuse_json_constant` is the
``parse_constant`` hook that refuses the three tokens during a decode. Both raise ``ValueError``
subclasses, so a caller's existing "not JSON" handling catches them. Their messages name where the
value came from and the rule it broke, never a key or a value of it.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Final, NoReturn

__all__ = [
    "MAX_JSON_DEPTH",
    "NestedTooDeepError",
    "NonFiniteNumberError",
    "refuse_json_constant",
    "require_finite",
]

MAX_JSON_DEPTH: Final = 256
"""The deepest nesting a value may have.

A value's depth is the number of containers (mappings, lists, tuples, sets and frozensets) on its
deepest path, the outermost included: a scalar has depth 0, ``[]`` and ``{"a": 1}`` have depth 1,
and ``[[1]]`` has depth 2. A mapping's keys count as items of the mapping. Each caller counts from
the value it checks.
"""


class NonFiniteNumberError(ValueError):
    """A value holds a NaN or infinite number, which standard JSON has no form for."""


class NestedTooDeepError(ValueError):
    """A value nests more than :data:`MAX_JSON_DEPTH` levels deep."""


def require_finite(value: object, *, where: str) -> None:
    """Refuse ``value`` if it holds a NaN or infinite number, or nests too deep.

    Walks mapping keys and values and the items of lists, tuples, sets and frozensets, stopping at
    the first problem. It keeps its own stack and never looks inside a container past the limit,
    so it cannot raise ``RecursionError``. Any other type is left to the caller's encoder.

    Args:
        value: The value to check, its depth counted from here.
        where: What the value is, such as ``"request body"``; the message names it.

    Raises:
        NonFiniteNumberError: A ``float`` in the value is NaN or infinite.
        NestedTooDeepError: A container in the value sits deeper than :data:`MAX_JSON_DEPTH`
            levels.
    """
    # Each item with the level of the container holding it: 0 for the value itself.
    pending: list[tuple[object, int]] = [(value, 0)]
    while pending:
        item, holder_level = pending.pop()
        if isinstance(item, float):
            if not math.isfinite(item):
                raise NonFiniteNumberError(
                    f"{where} holds a number JSON cannot represent ({_kind(item)})"
                )
            continue
        children = _children(item)
        if children is None:
            continue
        level = holder_level + 1
        if level > MAX_JSON_DEPTH:
            raise NestedTooDeepError(f"{where} nests more than {MAX_JSON_DEPTH} levels deep")
        # Reversed, so items are met in the order the value holds them.
        pending.extend((child, level) for child in reversed(children))


def refuse_json_constant(token: str) -> NoReturn:
    """The ``parse_constant`` hook of a standard-JSON decode: refuse every constant it is given.

    ``json.loads`` calls it only for the ``NaN``, ``Infinity`` and ``-Infinity`` tokens.

    Args:
        token: The token the decoder met.

    Raises:
        NonFiniteNumberError: Always, naming the token.
    """
    # Python spells the three tokens the way float() reads them.
    raise NonFiniteNumberError(
        f"the JSON text holds the token {token}, a number JSON cannot represent "
        f"({_kind(float(token))})"
    )


def _kind(number: float) -> str:
    return "NaN" if math.isnan(number) else "infinite"


def _children(item: object) -> list[object] | None:
    # A container's items, keys included; None for anything that is not a container.
    if isinstance(item, Mapping):
        children: list[object] = []
        for key, child in item.items():
            children.append(key)
            children.append(child)
        return children
    if isinstance(item, list | tuple | set | frozenset):
        return list(item)
    return None
