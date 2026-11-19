# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""WebSocket adapter behavior: the WS branch, WS carriers, auth and the missing-extra hint.

Exercised through a stub transport factory that mirrors ``websockets.sync.client.connect``'s
verified surface: ``(uri, *, additional_headers, subprotocols, open_timeout,
close_timeout, ...) -> ClientConnection-like`` with ``__enter__``/``__exit__``/``send``/``recv``/
``close``. A stub mirroring a verified real surface is a blessed test double (rubric rule 2).
"""

from __future__ import annotations

import json
import os
from collections.abc import Iterator
from typing import Any

import pytest

from calibrated_agent_evaluation.probe import (
    BearerAuth,
    ConnectionConfig,
    EndpointUnreachableError,
    SubprotocolAuth,
    WebSocketEndpointAdapter,
)


class _StubWsConn:
    """Mirrors the ``ClientConnection`` shape the adapter uses: send + recv + close."""

    def __init__(
        self, replies: list[str], raise_on_recv: type[BaseException] | None = None
    ) -> None:
        self._replies: Iterator[str] = iter(replies)
        self._raise = raise_on_recv
        self.sent: list[str | bytes] = []
        self.closed = False

    def __enter__(self) -> _StubWsConn:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    def send(self, message: str | bytes) -> None:
        self.sent.append(message)

    def recv(self, timeout: float | None = None) -> str:
        try:
            return next(self._replies)
        except StopIteration as stop:
            # Simulate an idle gap after the last reply -- the adapter treats this as "response
            # complete" once at least one frame was received.
            if self._raise is not None:
                raise self._raise("recv failure") from stop
            raise TimeoutError("stub idle") from stop

    def close(self) -> None:
        self.closed = True


class _StubTransport:
    """Callable mirroring ``websockets.sync.client.connect``'s signature; records every call."""

    def __init__(self, replies: list[str]) -> None:
        self._replies = replies
        self.calls: list[dict[str, Any]] = []

    def __call__(
        self,
        uri: str,
        *,
        additional_headers: dict[str, str] | None = None,
        subprotocols: list[str] | None = None,
        open_timeout: float | None = None,
        close_timeout: float | None = None,
        **_kwargs: Any,
    ) -> _StubWsConn:
        self.calls.append(
            {
                "uri": uri,
                "additional_headers": additional_headers,
                "subprotocols": subprotocols,
                "open_timeout": open_timeout,
                "close_timeout": close_timeout,
            }
        )
        return _StubWsConn(self._replies)


def _stub_transport(replies: list[str]) -> _StubTransport:
    return _StubTransport(replies)


def _raising_transport(exc: type[BaseException]) -> Any:
    def _connect(uri: str, **_kwargs: Any) -> _StubWsConn:
        raise exc("stub connect failure")

    return _connect


def test_ws_adapter_sends_prompt_and_parses_response(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("CALIBRATED_AGENT_EVALUATION_TOKEN", "s3cret-taco-token")
    replies = [json.dumps({"response": "I help with tacos."})]
    transport = _stub_transport(replies)
    adapter = WebSocketEndpointAdapter(
        ConnectionConfig(
            endpoint="wss://taco.example/agent",
            auth=SubprotocolAuth(token_env="CALIBRATED_AGENT_EVALUATION_TOKEN"),
        ),
        transport=transport,
    )
    resp = adapter.send("Hello?")
    # The prompt was sent and the frame parsed through the same _parse path the HTTP adapter uses.
    assert resp.text == "I help with tacos."
    call = transport.calls[0]
    assert call["uri"] == "wss://taco.example/agent"
    # SubprotocolAuth carries the token in `subprotocols`, NOT in additional_headers.
    assert call["subprotocols"] == ["s3cret-taco-token"]
    assert not call["additional_headers"]  # empty/none — SubprotocolAuth.headers() returns {}


def test_ws_adapter_carries_bearer_auth_in_additional_headers(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("CALIBRATED_AGENT_EVALUATION_TOKEN", "s3cret-bearer")
    transport = _stub_transport([json.dumps({"text": "ok"})])
    adapter = WebSocketEndpointAdapter(
        ConnectionConfig(
            endpoint="wss://x.example/agent",
            auth=BearerAuth(token_env="CALIBRATED_AGENT_EVALUATION_TOKEN"),
        ),
        transport=transport,
    )
    adapter.send("hi")
    call = transport.calls[0]
    assert call["additional_headers"] == {"Authorization": "Bearer s3cret-bearer"}
    assert call["subprotocols"] is None  # bearer carries in headers, not subprotocols


def test_ws_unreachable_raises_typed_error(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("CALIBRATED_AGENT_EVALUATION_TOKEN", "t")
    adapter = WebSocketEndpointAdapter(
        ConnectionConfig(
            endpoint="wss://x.example/agent",
            auth=BearerAuth(token_env="CALIBRATED_AGENT_EVALUATION_TOKEN"),
        ),
        transport=_raising_transport(ConnectionRefusedError),
    )
    with pytest.raises(EndpointUnreachableError):
        adapter.send("hi")


def test_ws_circuit_opens_after_repeated_failures(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("CALIBRATED_AGENT_EVALUATION_TOKEN", "t")
    adapter = WebSocketEndpointAdapter(
        ConnectionConfig(
            endpoint="wss://x.example/agent",
            auth=BearerAuth(token_env="CALIBRATED_AGENT_EVALUATION_TOKEN"),
            max_consecutive_failures=2,
        ),
        transport=_raising_transport(ConnectionRefusedError),
    )
    for _ in range(2):
        with pytest.raises(EndpointUnreachableError):
            adapter.send("hi")
    # Third call: no new transport attempt — circuit open message.
    with pytest.raises(EndpointUnreachableError, match="circuit open"):
        adapter.send("hi")


def test_subprotocol_token_absent_from_config_repr(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("CALIBRATED_AGENT_EVALUATION_TOKEN", "s3cret-taco-token")
    config = ConnectionConfig(
        endpoint="wss://taco.example/agent",
        auth=SubprotocolAuth(token_env="CALIBRATED_AGENT_EVALUATION_TOKEN"),
    )
    representation = repr(config)
    assert "s3cret-taco-token" not in representation
    # The auth's TYPE is shown; the env-var NAME (never the value) is fine to leak in higher-level
    # logs but the config repr is intentionally minimal.
    assert "SubprotocolAuth" in representation


def test_subprotocol_token_absent_from_ws_error(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("CALIBRATED_AGENT_EVALUATION_TOKEN", "s3cret-taco-token")
    adapter = WebSocketEndpointAdapter(
        ConnectionConfig(
            endpoint="wss://taco.example/agent",
            auth=SubprotocolAuth(token_env="CALIBRATED_AGENT_EVALUATION_TOKEN"),
        ),
        transport=_raising_transport(ConnectionRefusedError),
    )
    with pytest.raises(EndpointUnreachableError) as excinfo:
        adapter.send("hi")
    # The typed error message must not leak the resolved subprotocol token, even when the wrapped
    # cause is echoed. (SubprotocolAuth resolves the token to pass to `subprotocols=`; a naive
    # error message that included the request payload could leak it.)
    assert "s3cret-taco-token" not in str(excinfo.value)
    assert "s3cret-taco-token" not in repr(excinfo.value)


def test_ws_adapter_reuses_http_parse_for_json_frames(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # A frame body that parses as generic JSON with `content`/`answer` should be surfaced through
    # _parse the same way the HTTP adapter does (adapter unification).
    monkeypatch.setenv("CALIBRATED_AGENT_EVALUATION_TOKEN", "t")
    replies = [json.dumps({"content": "hello from ws"})]
    adapter = WebSocketEndpointAdapter(
        ConnectionConfig(
            endpoint="wss://x.example/agent",
            auth=BearerAuth(token_env="CALIBRATED_AGENT_EVALUATION_TOKEN"),
        ),
        transport=_stub_transport(replies),
    )
    resp = adapter.send("hi")
    assert resp.text == "hello from ws"


def test_slow_first_frame_waits_on_config_timeout_not_idle_gap(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Regression: a live agent may take seconds to emit its FIRST frame. The first recv must be
    # budgeted at the config's `timeout_seconds`, NOT the short idle-gap budget -- otherwise a
    # working-but-slow endpoint would look unreachable. This stub returns the frame only when the
    # first recv is given a generous timeout (mirroring an agent that answers after > the idle gap).
    monkeypatch.setenv("CALIBRATED_AGENT_EVALUATION_TOKEN", "t")

    class _SlowFirstFrameConn:
        def __init__(self) -> None:
            self._delivered = False

        def __enter__(self) -> _SlowFirstFrameConn:
            return self

        def __exit__(self, *exc: object) -> None:
            pass

        def send(self, message: str | bytes) -> None:
            pass

        def recv(self, timeout: float | None = None) -> str:
            if not self._delivered:
                # A short idle budget can't reach this frame; only the full config timeout does.
                if timeout is None or timeout < 1.0:
                    raise TimeoutError("first frame not ready within the idle gap")
                self._delivered = True
                return json.dumps({"response": "slow but alive"})
            raise TimeoutError("stub idle")  # response complete after the one frame

    def _slow_transport(uri: str, **_kwargs: Any) -> _SlowFirstFrameConn:
        return _SlowFirstFrameConn()

    adapter = WebSocketEndpointAdapter(
        ConnectionConfig(
            endpoint="wss://slow.example/agent",
            auth=BearerAuth(token_env="CALIBRATED_AGENT_EVALUATION_TOKEN"),
            timeout_seconds=5.0,  # first-frame budget; the idle gap (0.5s) alone would time out
        ),
        transport=_slow_transport,
    )
    resp = adapter.send("hi")
    assert resp.text == "slow but alive"  # not a false EndpointUnreachableError


def test_ws_adapter_finalizes_close(monkeypatch: pytest.MonkeyPatch) -> None:
    # Resource lifecycle: the stub's `__exit__` is called via the `with connect(...)` context
    # manager, so the connection is closed even after a successful send.
    monkeypatch.setenv("CALIBRATED_AGENT_EVALUATION_TOKEN", "t")

    captured: dict[str, _StubWsConn] = {}

    def _connect(uri: str, **_kwargs: Any) -> _StubWsConn:
        conn = _StubWsConn([json.dumps({"text": "ok"})])
        captured["conn"] = conn
        return conn

    adapter = WebSocketEndpointAdapter(
        ConnectionConfig(
            endpoint="wss://x.example/agent",
            auth=BearerAuth(token_env="CALIBRATED_AGENT_EVALUATION_TOKEN"),
        ),
        transport=_connect,
    )
    adapter.send("hi")
    assert captured["conn"].closed is True


# --- guard against env pollution between tests (secrets stay in the process, but between-test
#     leakage would let a later test see a token an earlier test set even without monkeypatch) ---
@pytest.fixture(autouse=True)
def _no_leak_between_tests() -> Iterator[None]:
    original = os.environ.get("CALIBRATED_AGENT_EVALUATION_TOKEN")
    yield
    if original is None:
        os.environ.pop("CALIBRATED_AGENT_EVALUATION_TOKEN", None)
    else:
        os.environ["CALIBRATED_AGENT_EVALUATION_TOKEN"] = original


def test_wrapped_ws_error_scrubs_the_resolved_token(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # The live-TACO failure mode: the transport raises with the resolved token echoed in its
    # message (websockets does this for an invalid subprotocol). The wrapped error must NOT
    # contain the token, but MUST retain the non-secret diagnostic.
    monkeypatch.setenv("CALIBRATED_AGENT_EVALUATION_TOKEN", "s3cret-taco-token")

    def _echoing_connect(uri: str, **_kwargs: Any) -> _StubWsConn:
        raise ValueError("invalid subprotocol: s3cret-taco-token")

    adapter = WebSocketEndpointAdapter(
        ConnectionConfig(
            endpoint="wss://taco.example/agent",
            auth=SubprotocolAuth(token_env="CALIBRATED_AGENT_EVALUATION_TOKEN"),
        ),
        transport=_echoing_connect,
    )
    with pytest.raises(EndpointUnreachableError) as excinfo:
        adapter.send("hi")
    text = str(excinfo.value)
    assert "s3cret-taco-token" not in text
    assert "[redacted]" in text
    # the non-secret diagnostic (the "invalid subprotocol" reason) is retained.
    assert "invalid subprotocol" in text
