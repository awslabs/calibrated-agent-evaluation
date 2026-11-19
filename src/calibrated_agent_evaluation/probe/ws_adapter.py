# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""WebSocket endpoint adapter.

Twin of ``adapter.EndpointAdapter`` for the WebSocket transport: same public
``send(prompt) -> ProbeResponse`` shape, so the harness routes on URL scheme and every downstream
path (framing, `probed` Trace shape, authorization, redaction, circuit breaker) is transport-
agnostic. Uses ``websockets.sync.client.connect`` from the ``[probe-ws]`` optional extra (verified
against `websockets==16.0`). The `websockets` package is a network-transport library, in the same
category as stdlib `urllib`; the vendor-isolation boundary test allowlists it here and rejects
any real vendor-SDK import in `probe/`.

Auth is carried in ``additional_headers`` (bearer/api-key/header) or ``subprotocols``
(`SubprotocolAuth`; the TACO shape). A failed handshake / connect / recv is wrapped into
:class:`~calibrated_agent_evaluation.probe.errors.EndpointUnreachableError` chained ``from exc``;
never a silent empty-success. The consecutive-failure circuit breaker mirrors the HTTP adapter.
"""

from __future__ import annotations

import json
import time
import uuid
from collections.abc import Callable, Iterable
from typing import Any, Protocol, cast

from calibrated_agent_evaluation._install import install_hint
from calibrated_agent_evaluation.probe.adapter import ProbeResponse, _parse
from calibrated_agent_evaluation.probe.connection import ConnectionConfig, SubprotocolAuth
from calibrated_agent_evaluation.probe.errors import (
    EndpointUnreachableError,
    ProbeMissingExtraError,
)
from calibrated_agent_evaluation.probe.redact import scrub

# Recv budget: how long we wait for frames after `send` before deciding the response is complete.
# The connection can also close from the server side to signal completion; either path terminates.
_RECV_IDLE_SECONDS = 0.5
_RECV_MAX_FRAMES = 1024


def _missing_extra_message() -> str:
    return (
        "WebSocket probing requires the [probe-ws] extra. Install it with "
        f"`{install_hint('probe-ws')}` (adds `websockets>=16,<17`, BSD-3-Clause)."
    )


class _WsConn(Protocol):
    """The subset of ``websockets.sync.client.ClientConnection`` we depend on.

    Verified against ``websockets==16.0``: context-manager
    ``__enter__``/``__exit__``, ``send(str|bytes)``, ``recv() -> str|bytes``, ``close()``. Enough
    for one prompt frame + a bounded recv loop.
    """

    def __enter__(self) -> _WsConn: ...
    def __exit__(self, *exc: object) -> None: ...
    def send(self, message: str | bytes) -> None: ...
    def recv(self, timeout: float | None = ...) -> str | bytes: ...
    def close(self) -> None: ...


# A WS transport factory: (uri, *, additional_headers, subprotocols, open_timeout, close_timeout)
# -> a context-manager ClientConnection. Mirrors `websockets.sync.client.connect`'s signature
# (verified surface, `websockets==16.0`) so a stub factory in tests is a blessed test double.
# Return type is `Any` because the real signature is loaded lazily; the Protocol above is
# the *documentation* of the used surface, and the adapter only calls into it via duck-typing.
WsConnect = Callable[..., Any]


def _import_websockets_connect() -> WsConnect:
    """Lazy-load ``websockets.sync.client.connect``; raise on missing ``[probe-ws]`` extra."""
    try:
        from websockets.sync.client import connect
    except ImportError as exc:
        raise ProbeMissingExtraError(_missing_extra_message()) from exc
    return cast("WsConnect", connect)


def require_probe_ws_extra() -> None:
    """Refuse when WS is requested but ``[probe-ws]`` is not installed.

    Called by the harness *before* the authorization gate would send anything, so a missing extra
    fails closed rather than being masked by a later error.
    """
    _import_websockets_connect()


class WebSocketEndpointAdapter:
    """Send prompts over a WebSocket and parse the returned frames.

    Same public shape as ``EndpointAdapter``: ``send(prompt) -> ProbeResponse``. Reuses the HTTP
    adapter's ``_parse`` so JSON/SSE-shaped frame payloads flow through the same normalization.
    """

    def __init__(self, config: ConnectionConfig, *, transport: WsConnect | None = None) -> None:
        self._config = config
        self._transport = transport  # None => resolved lazily on first send (via the extra)
        self._consecutive_failures = 0
        self._session_id = (
            f"calibrated-agent-evaluation-{int(time.time() * 1000)}-{uuid.uuid4().hex}"
        )

    def send(self, prompt: str) -> ProbeResponse:
        """Open a WS connection, send ``prompt`` as one frame, recv until close/idle, parse.

        Opens the circuit (raises) once ``max_consecutive_failures`` is reached, matching the HTTP
        adapter. Any transport / handshake / recv failure is wrapped into
        :class:`EndpointUnreachableError` -- never returned as an empty success.
        """
        if self._consecutive_failures >= self._config.max_consecutive_failures:
            raise EndpointUnreachableError(
                f"circuit open: {self._consecutive_failures} consecutive failures against "
                f"{self._config.endpoint!r}; refusing to keep probing"
            )
        connect = self._transport or _import_websockets_connect()
        payload = json.dumps({"prompt": prompt, "session_id": self._session_id})
        additional_headers = self._resolve_additional_headers()
        subprotocols = self._resolve_subprotocols()
        try:
            with connect(
                self._config.endpoint,
                additional_headers=additional_headers or None,
                subprotocols=subprotocols or None,
                open_timeout=self._config.timeout_seconds,
                close_timeout=self._config.timeout_seconds,
            ) as ws:
                ws.send(payload)
                frames = _drain(ws, self._config.timeout_seconds)
        except EndpointUnreachableError:
            raise
        except ProbeMissingExtraError:
            # Extra check failed at send time; no failure to attribute to the endpoint.
            raise
        except Exception as exc:
            self._consecutive_failures += 1
            # A vendor library (e.g. websockets rejecting an invalid subprotocol) can echo
            # the resolved credential into its exception message; scrub it before re-raising so a
            # wrapped error never leaks the secret, while keeping the diagnostic (status/reason).
            raw = (
                f"websocket endpoint {self._config.endpoint!r} unreachable: "
                f"{type(exc).__name__}: {exc}"
            )
            raise EndpointUnreachableError(
                scrub(raw, self._resolved_secrets(additional_headers, subprotocols))
            ) from exc
        self._consecutive_failures = 0
        return _parse(frames)

    @staticmethod
    def _resolved_secrets(additional_headers: dict[str, str], subprotocols: list[str]) -> list[str]:
        """The resolved secret values to scrub from a wrapped error.

        Every subprotocol value is a raw token (the shape `websockets` echoes); every header value
        may carry a secret, and a ``Bearer <token>`` header also contributes its bare token so an
        echo of either the full header or just the token is masked.
        """
        secrets: list[str] = list(subprotocols)
        for value in additional_headers.values():
            secrets.append(value)
            secrets.append(value.removeprefix("Bearer ").strip())
        return secrets

    def _resolve_additional_headers(self) -> dict[str, str]:
        headers: dict[str, str] = dict(self._config.extra_headers)
        if self._config.auth is not None:
            headers.update(self._config.auth.headers())
        return headers

    def _resolve_subprotocols(self) -> list[str]:
        auth = self._config.auth
        if isinstance(auth, SubprotocolAuth):
            return auth.subprotocols()
        return []


def _drain(ws: _WsConn, first_frame_timeout: float) -> Iterable[bytes]:
    """Receive frames until the server closes or we hit an idle gap / frame cap.

    The FIRST frame is waited for up to ``first_frame_timeout`` (the config's ``timeout_seconds``):
    a live agent may take seconds to emit its first token, and a short idle budget here would make a
    working-but-slow endpoint look unreachable. Once at least one frame has arrived, subsequent
    frames use the short ``_RECV_IDLE_SECONDS`` budget so an idle gap terminates the response.
    """
    frames: list[bytes] = []
    for _ in range(_RECV_MAX_FRAMES):
        # Budget the first frame at the full config timeout; tighten to the idle gap thereafter.
        timeout = _RECV_IDLE_SECONDS if frames else first_frame_timeout
        try:
            msg: Any = ws.recv(timeout=timeout)
        except TimeoutError:
            # Idle gap after at least one frame received => response complete. A timeout before ANY
            # frame is a real unreachable/silence -> re-raise as a transport failure caught above.
            if frames:
                break
            raise
        except Exception:
            # Anything else (ConnectionClosed*, protocol errors) is either "server closed to signal
            # end of response" (if we already got frames) or a real failure. Either way, stop
            # draining; the outer send() decides whether the accumulated content is a success.
            if frames:
                break
            raise
        frames.append(msg.encode("utf-8") if isinstance(msg, str) else msg)
    return frames
