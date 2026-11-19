# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Pick the one JSON payload a model's text holds, and refuse any text that holds several.

Vendor-free and stdlib only. The failure detector and the draft namer read their model's text
with the same three rules, tried in order:

1. the whole stripped text decodes as JSON and the caller accepts the value;
2. else the text holds exactly one fenced block (a ```` ```json ```` or a bare ```` ``` ````
   fence), its body decodes to an accepted value, and neither the text before that block nor
   the text after it holds an accepted value of its own (each found as rule 3 finds them, the
   two scanned separately);
3. else ``json.JSONDecoder().raw_decode`` at every occurrence of the opener finds exactly one
   accepted value.

Rule 3 tries every opener, nested ones included, so an accepted value inside another counts as a
candidate of its own. When no rule yields a value, the text held no accepted value or more than
one, and :class:`PayloadPickError` says which. The picker never takes the first of several: a
fenced answer beside another answer in the prose around it is two candidates, not the fenced one.
"""

from __future__ import annotations

import json
import re
from collections.abc import Callable
from typing import Any, Literal

# A fenced block: an opening fence at the start of a line with an optional info string, the
# body, and a closing fence alone on a later line. Rule 2 counts only blocks whose info string is
# ``json`` or empty.
_FENCE = re.compile(
    r"^[ \t]*```(?P<info>[^\n`]*)\n(?P<body>.*?)^[ \t]*```[ \t]*$",
    re.MULTILINE | re.DOTALL,
)
_JSON_INFO = frozenset({"", "json"})
_OPENERS = frozenset({"{", "["})


class PayloadPickError(ValueError):
    """A model's text held no accepted JSON value, or more than one.

    The message starts ``no candidate`` or ``<N> candidates`` and quotes none of the text, and
    ``candidates`` holds that number (0 for none). Callers chain it into their own typed error,
    which names the trace or the call.
    """

    def __init__(self, message: str, *, candidates: int) -> None:
        super().__init__(message)
        self.candidates = candidates


def pick_one_json(text: str, *, opener: Literal["{", "["], accept: Callable[[Any], bool]) -> Any:
    """Return the one JSON value in ``text`` that ``accept`` holds for.

    Args:
        text: The model's response text.
        opener: The character the payload starts with: ``"{"`` for an object, ``"["`` for an
            array. Rule 3 decodes from every occurrence of it.
        accept: Whether a decoded value is the payload the caller wants, for example a dict with
            an ``errors`` key or a list of strings.

    Returns:
        The decoded value, from the first rule that yields one.

    Raises:
        PayloadPickError: No rule yields a value and ``text`` holds no accepted value, or several.
        ValueError: ``opener`` is neither ``"{"`` nor ``"["``.
    """
    if opener not in _OPENERS:
        raise ValueError(f"opener must be '{{' or '[', got {opener!r}")
    whole = _accepted_document(text.strip(), accept)
    if whole:
        return whole[0]
    fenced = [match for match in _FENCE.finditer(text) if match.group("info").strip() in _JSON_INFO]
    if len(fenced) == 1 and (body := _accepted_document(fenced[0].group("body").strip(), accept)):
        (block,) = fenced
        # Each side of the block is scanned on its own, so the block's value is not counted again
        # and a fragment ending the text before it cannot join one starting the text after it into
        # a value the model never wrote, such as '{"errors":' and '[]}'.
        before, after = text[: block.start()], text[block.end() :]
        others = [
            *_accepted_from_every_opener(before, opener, accept),
            *_accepted_from_every_opener(after, opener, accept),
        ]
        if not others:
            return body[0]
        raise _several(1 + len(others), opener)
    candidates = _accepted_from_every_opener(text, opener, accept)
    if len(candidates) == 1:
        return candidates[0]
    if not candidates:
        raise PayloadPickError(
            "no candidate: the text is not an accepted JSON value, holds no single fenced block "
            f"with one, and no {opener!r} in it starts one",
            candidates=0,
        )
    raise _several(len(candidates), opener)


def _several(count: int, opener: str) -> PayloadPickError:
    return PayloadPickError(
        f"{count} candidates: the text holds {count} accepted JSON values starting with "
        f"{opener!r}, and the picker does not choose between them",
        candidates=count,
    )


def _accepted_document(text: str, accept: Callable[[Any], bool]) -> list[Any]:
    """``[value]`` when ``text`` decodes as one JSON value ``accept`` holds for, else ``[]``.

    A list, because ``null`` decodes to ``None`` and so cannot mean "nothing here".
    """
    try:
        value = json.loads(text)
    except (ValueError, RecursionError):
        # Not JSON, or nested past the decoder's depth: not this rule's payload either way.
        return []
    return [value] if accept(value) else []


def _accepted_from_every_opener(text: str, opener: str, accept: Callable[[Any], bool]) -> list[Any]:
    """Every accepted value that ``raw_decode`` finds starting at an occurrence of ``opener``."""
    decoder = json.JSONDecoder()
    found: list[Any] = []
    for match in re.finditer(re.escape(opener), text):
        try:
            value, _ = decoder.raw_decode(text, match.start())
        except (ValueError, RecursionError):
            continue  # this occurrence starts no JSON value; a later one may
        if accept(value):
            found.append(value)
    return found
