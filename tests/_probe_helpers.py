# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Fake transports for probe tests — mirror the real (status, line-iterator) surface.

Injected via ``EndpointAdapter(config, transport=...)`` / ``probe(..., transport=...)`` so unit
tests exercise the real parsing + circuit-breaker paths without opening a socket (rubric-blessed
stub: it mirrors the verified transport contract; the live path is an opt-in integration test).
"""

from __future__ import annotations

import json
import urllib.error
from collections.abc import Iterable


def json_transport(text: str = "hello, I can help with orders"):
    def _t(url: str, headers: dict, body: bytes, timeout: float) -> tuple[int, Iterable[bytes]]:
        return 200, [json.dumps({"response": text}).encode()]

    return _t


def sse_transport(text: str = "done", tools: tuple[str, ...] = ("lookup_order",)):
    def _t(url: str, headers: dict, body: bytes, timeout: float) -> tuple[int, Iterable[bytes]]:
        lines = [
            b"data: " + json.dumps({"type": "tool_use", "tool_name": t}).encode() for t in tools
        ]
        lines.append(b"data: " + json.dumps({"type": "response", "response": text}).encode())
        return 200, lines

    return _t


def empty_transport():
    """A 200 with no parseable content — drives the degrade-to-outcome-only path."""

    def _t(url: str, headers: dict, body: bytes, timeout: float) -> tuple[int, Iterable[bytes]]:
        return 200, [b""]

    return _t


def unreachable_transport():
    def _t(url: str, headers: dict, body: bytes, timeout: float) -> tuple[int, Iterable[bytes]]:
        raise urllib.error.URLError("connection refused")

    return _t
