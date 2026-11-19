# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Cassette fingerprints stay stable across inspect-ai releases that add unset content fields."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from pydantic import TypeAdapter

pytest.importorskip("inspect_ai")

from inspect_ai._util.content import ContentText
from inspect_ai.model import ChatMessage, ChatMessageUser

from calibrated_agent_evaluation.execute.cassette import fingerprint
from calibrated_agent_evaluation.execute.cassette_provider import _stable_message


def _fp(message: ChatMessageUser) -> str:
    return fingerprint({"messages": [_stable_message(message)]})


_RECORDED = (
    Path(__file__).resolve().parents[1] / "fixtures/raw/cassettes/inspect_ORD-1001.cassette.json"
)


def test_recorded_messages_rebuild_to_their_recorded_form_under_the_installed_inspect() -> None:
    entries = json.loads(_RECORDED.read_text())["entries"]
    recorded = [m for e in entries for m in e["request"]["messages"]]
    with_blocks = [m for m in recorded if isinstance(m.get("content"), list)]
    assert with_blocks, "the fixture cassette should hold messages with content blocks"

    for message in recorded:
        rebuilt = TypeAdapter(ChatMessage).validate_python(message)
        assert _stable_message(rebuilt) == message


def test_a_set_cache_breakpoint_changes_the_fingerprint() -> None:
    plain = ChatMessageUser(content=[ContentText(text="refund ORD-1001")])
    cached = ChatMessageUser(content=[ContentText(text="refund ORD-1001", cache_breakpoint=True)])

    assert _stable_message(cached)["content"][0]["cache_breakpoint"] is True
    assert _fp(plain) != _fp(cached)


def test_string_content_is_left_untouched() -> None:
    message = ChatMessageUser(content="refund ORD-1001")

    assert _stable_message(message)["content"] == "refund ORD-1001"
