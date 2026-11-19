# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""The nesting bound for decoded trace content.

A hostile or broken trace can nest lists or objects tens of thousands of levels deep. Whether the
JSON decoder accepts that depends on the platform's C stack, and anything that later walks the
value recursively (``str``, ``repr``, Pydantic, ``json.dumps``) can overflow it. Content deeper
than this bound is kept as text instead of being decoded.
"""

from __future__ import annotations

from typing import Any

MAX_CONTENT_DEPTH = 256
TOO_DEEP_NOTE = f"<content nested deeper than {MAX_CONTENT_DEPTH} levels, not decoded>"


def nested_deeper_than(value: Any, limit: int = MAX_CONTENT_DEPTH) -> bool:
    """Whether ``value`` holds lists, tuples or dicts nested more than ``limit`` levels deep."""
    stack: list[tuple[Any, int]] = [(value, 0)]
    while stack:
        item, depth = stack.pop()
        if isinstance(item, dict):
            children = item.values()
        elif isinstance(item, (list, tuple)):
            children = item
        else:
            continue
        if depth >= limit:
            return True
        stack.extend((child, depth + 1) for child in children)
    return False
