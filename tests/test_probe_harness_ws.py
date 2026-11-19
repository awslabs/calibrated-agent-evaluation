# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Harness routing + missing-extra fail-closed.

Exercises the URL-scheme routing in ``probe.harness.probe()``: `http`/`https` → HTTP adapter,
`ws`/`wss` → WebSocket adapter; a `wss://` config without `websockets` importable raises
``ProbeMissingExtraError`` BEFORE the authorization gate would send anything (fail-closed).
"""

from __future__ import annotations

import builtins
import json
from collections.abc import Iterator
from typing import Any

import pytest

from calibrated_agent_evaluation.probe import (
    BearerAuth,
    ConnectionConfig,
    ProbeMissingExtraError,
    SubprotocolAuth,
    probe,
)


class _StubWsConn:
    def __init__(self, replies: list[str]) -> None:
        self._replies = iter(replies)
        self.closed = False

    def __enter__(self) -> _StubWsConn:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    def send(self, message: str | bytes) -> None:
        pass

    def recv(self, timeout: float | None = None) -> str:
        try:
            return next(self._replies)
        except StopIteration as stop:
            raise TimeoutError("stub idle") from stop

    def close(self) -> None:
        self.closed = True


def _ws_transport() -> Any:
    def _connect(uri: str, **_kwargs: Any) -> _StubWsConn:
        _connect.calls.append(uri)  # type: ignore[attr-defined]
        return _StubWsConn([json.dumps({"text": f"ws reply for {uri}"})])

    _connect.calls = []  # type: ignore[attr-defined]
    return _connect


def _http_transport() -> Any:
    def _fake(url: str, headers: dict[str, str], body: bytes, timeout: float) -> Any:
        _fake.calls.append(url)  # type: ignore[attr-defined]
        return 200, [json.dumps({"message": f"http reply for {url}"}).encode("utf-8")]

    _fake.calls = []  # type: ignore[attr-defined]
    return _fake


def test_wss_scheme_routes_to_ws_adapter(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("CALIBRATED_AGENT_EVALUATION_TOKEN", "t")
    ws_t = _ws_transport()
    http_t = _http_transport()
    report = probe(
        ConnectionConfig(
            endpoint="wss://taco.example/agent",
            auth=SubprotocolAuth(token_env="CALIBRATED_AGENT_EVALUATION_TOKEN"),
        ),
        authorized=True,
        seeds=("hi",),
        transport=http_t,
        ws_transport=ws_t,
    )
    assert ws_t.calls == ["wss://taco.example/agent"]  # type: ignore[attr-defined]
    assert http_t.calls == []  # type: ignore[attr-defined]
    assert len(report.traces) == 1
    assert report.traces[0].ownership == "probed"
    # A WebSocket probe negotiates no payload format, so `None` is the honest report rather than a
    # borrowed HTTP format name.
    assert report.detected_format is None


def test_https_scheme_routes_to_http_adapter(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("CALIBRATED_AGENT_EVALUATION_TOKEN", "t")
    ws_t = _ws_transport()
    http_t = _http_transport()
    probe(
        ConnectionConfig(
            endpoint="https://ord.example/agent",
            auth=BearerAuth(token_env="CALIBRATED_AGENT_EVALUATION_TOKEN"),
        ),
        authorized=True,
        seeds=("hi",),
        transport=http_t,
        ws_transport=ws_t,
    )
    # Routing intent: HTTP config -> HTTP transport (called at least once for the seed prompt,
    # possibly twice because the HTTP adapter auto-detects payload format on first send); WS
    # transport is NEVER touched.
    assert http_t.calls, "HTTP transport should have been called for an https:// config"  # type: ignore[attr-defined]
    assert all(u == "https://ord.example/agent" for u in http_t.calls)  # type: ignore[attr-defined]
    assert ws_t.calls == []  # type: ignore[attr-defined]


def test_wss_without_probe_ws_extra_raises_missing_extra(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Simulate the `[probe-ws]` extra not being installed by making `import websockets.sync.client`
    # raise ImportError. This is the exact condition the missing-extra hint targets.
    original_import = builtins.__import__

    def _fake_import(name: str, *args: Any, **kwargs: Any) -> Any:
        if name.startswith("websockets"):
            raise ImportError("mocked missing extra")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", _fake_import)
    monkeypatch.setenv("CALIBRATED_AGENT_EVALUATION_TOKEN", "t")

    http_t = _http_transport()
    with pytest.raises(ProbeMissingExtraError, match=r"\[probe-ws\]"):
        probe(
            ConnectionConfig(
                endpoint="wss://taco.example/agent",
                auth=SubprotocolAuth(token_env="CALIBRATED_AGENT_EVALUATION_TOKEN"),
            ),
            authorized=True,
            seeds=("hi",),
            transport=http_t,
            ws_transport=None,  # Force the real import path -> caught by the fake_import
        )
    # No silent fallback to HTTP -- the HTTP adapter must never be touched.
    assert http_t.calls == []  # type: ignore[attr-defined]


def test_missing_extra_fails_closed_before_authorization(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # authorized=False on an HTTP endpoint raises AuthorizationError; on a WS endpoint WITHOUT the
    # extra, the missing-extra check happens FIRST (per harness spec) — proving the fail-closed
    # order (transport availability > authorization > network).
    original_import = builtins.__import__

    def _fake_import(name: str, *args: Any, **kwargs: Any) -> Any:
        if name.startswith("websockets"):
            raise ImportError("mocked missing extra")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", _fake_import)
    monkeypatch.setenv("CALIBRATED_AGENT_EVALUATION_TOKEN", "t")

    with pytest.raises(ProbeMissingExtraError):
        probe(
            ConnectionConfig(
                endpoint="wss://taco.example/agent",
                auth=SubprotocolAuth(token_env="CALIBRATED_AGENT_EVALUATION_TOKEN"),
            ),
            authorized=False,  # would raise AuthorizationError on the HTTP path
            seeds=("hi",),
            ws_transport=None,
        )


@pytest.fixture(autouse=True)
def _no_leak_between_tests() -> Iterator[None]:
    import os

    original = os.environ.get("CALIBRATED_AGENT_EVALUATION_TOKEN")
    yield
    if original is None:
        os.environ.pop("CALIBRATED_AGENT_EVALUATION_TOKEN", None)
    else:
        os.environ["CALIBRATED_AGENT_EVALUATION_TOKEN"] = original
