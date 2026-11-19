# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Content-coercion helper tests: as_object never leaks a vendor type or drops content.

``as_object`` is the shared per-field decoder every adapter uses. Its contract is that the result
is always a JSON-serializable dict of plain types — no vendor SDK object crosses the normalized
boundary, and no recovered content is silently dropped. These tests pin the
two failure modes the real ``inspect_ai`` rich-content types expose, plus the plain-shape
regressions the corpus relies on.
"""

from __future__ import annotations

import json

import pytest

from calibrated_agent_evaluation.normalize.content import as_object


def test_bare_vendor_content_object_is_coerced_to_prose_not_stored_raw() -> None:
    # Inspect's ToolEvent.result may be a ContentText (a pydantic vendor object). Storing it raw
    # would leak an SDK type past the normalized boundary AND break json.dumps/serde.
    ct = pytest.importorskip("inspect_ai._util.content").ContentText(text="the tool said hello")
    result = as_object(ct)
    assert result == {"answer": "the tool said hello"}
    json.dumps(result)  # must be serializable — no vendor object inside


def test_multimodal_content_list_is_preserved_not_silently_dropped() -> None:
    # A list of rich-content objects (a multimodal tool result) must survive as a JSON-serializable
    # list, never collapse to {} (the silent-drop anti-pattern the old code hit).
    content = pytest.importorskip("inspect_ai._util.content")
    result = as_object([content.ContentImage(image="data:image/png;base64,abc")])
    assert result != {}
    assert len(result["answer"]) == 1
    json.dumps(result)  # serializable (each element coerced to a plain dict)


def test_content_text_list_is_preserved_as_prose() -> None:
    content = pytest.importorskip("inspect_ai._util.content")
    result = as_object([content.ContentText(text="a"), content.ContentText(text="b")])
    assert result == {"answer": ["a", "b"]}


# --- plain-shape regressions (the shapes the corpus actually emits) -----------------------


def test_dict_passthrough() -> None:
    assert as_object({"decision": "DENY"}) == {"decision": "DENY"}


def test_choice_list_inner_json_object_is_unwrapped() -> None:
    assert as_object('[{"text": "{\\"decision\\": \\"DENY\\"}"}]') == {"decision": "DENY"}


def test_choice_list_prose_is_wrapped() -> None:
    assert as_object('[{"text": "hello world"}]') == {"answer": "hello world"}


def test_plain_prose_and_scalars_and_empty() -> None:
    assert as_object("just text") == {"answer": "just text"}
    assert as_object(42) == {"answer": 42}
    assert as_object("") == {}
    assert as_object(None) == {}
