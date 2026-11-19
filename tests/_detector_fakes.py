# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Test doubles for the failure detector and the draft namer.

``ScriptedModel`` implements the four abstract methods of the installed
``strands.models.model.Model`` and streams scripted texts as ``contentBlockDelta`` text events,
the only event shape the detector library reads, or raises a scripted exception. ``spy_connections``
and ``unset_aws_env`` let a test show that a replay opens no connection and needs no AWS
configuration.
"""

from __future__ import annotations

import os
import socket
from collections.abc import AsyncGenerator, AsyncIterable, Sequence
from typing import Any, TypeVar

import pytest
from pydantic import BaseModel
from strands.models.model import Model
from strands.types.content import Messages, SystemContentBlock
from strands.types.streaming import StreamEvent
from strands.types.tools import ToolChoice, ToolSpec

T = TypeVar("T", bound=BaseModel)

DEFAULT_CONFIG: dict[str, Any] = {"model_id": "scripted-model"}


class ScriptedModel(Model):
    """A strands ``Model`` that answers each ``stream`` call with the next scripted response.

    A text response is streamed as two text deltas, so a consumer has to join them; an exception
    response is raised. Every call's messages are kept in ``calls``, in call order.
    """

    def __init__(
        self,
        responses: Sequence[str | BaseException] = (),
        *,
        config: dict[str, Any] | None = None,
    ) -> None:
        self._responses = list(responses)
        self._config = dict(DEFAULT_CONFIG if config is None else config)
        self.calls: list[Messages] = []

    def get_config(self) -> dict[str, Any]:
        return dict(self._config)

    def update_config(self, **model_config: Any) -> None:
        self._config.update(model_config)

    def structured_output(
        self,
        output_model: type[T],
        prompt: Messages,
        system_prompt: str | None = None,
        **kwargs: Any,
    ) -> AsyncGenerator[dict[str, T | Any], None]:
        raise NotImplementedError("ScriptedModel answers stream calls only")

    async def stream(
        self,
        messages: Messages,
        tool_specs: list[ToolSpec] | None = None,
        system_prompt: str | None = None,
        *,
        tool_choice: ToolChoice | None = None,
        system_prompt_content: list[SystemContentBlock] | None = None,
        invocation_state: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> AsyncIterable[StreamEvent]:
        self.calls.append(messages)
        if not self._responses:
            raise AssertionError(
                f"ScriptedModel got call {len(self.calls)} but has no scripted response left"
            )
        response = self._responses.pop(0)
        if isinstance(response, BaseException):
            raise response
        middle = len(response) // 2
        for part in (response[:middle], response[middle:]):
            yield {"contentBlockDelta": {"delta": {"text": part}}}


def spy_connections(monkeypatch: pytest.MonkeyPatch) -> list[Any]:
    """Refuse every outgoing socket connection, and return the list of addresses tried."""
    attempts: list[Any] = []

    def connect(_sock: socket.socket, address: Any) -> None:
        attempts.append(address)
        raise OSError(f"the test refused a network connection to {address!r}")

    def create_connection(address: Any, *_args: Any, **_kwargs: Any) -> socket.socket:
        attempts.append(address)
        raise OSError(f"the test refused a network connection to {address!r}")

    monkeypatch.setattr(socket.socket, "connect", connect)
    monkeypatch.setattr(socket, "create_connection", create_connection)
    return attempts


def unset_aws_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Remove every ``AWS_*`` variable from the environment for the rest of the test."""
    for name in list(os.environ):
        if name.startswith("AWS_"):
            monkeypatch.delenv(name)
