# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Endpoint connection config + auth.

The programmatic surface a caller (CLI, a future UI, or Python) constructs to point the
probing harness at an endpoint. Credentials are **never inlined**: each auth variant names an
environment variable, and the secret is resolved in-process at send time and redacted from every
``repr``/log/artifact. No vendor type appears here.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass

from calibrated_agent_evaluation.probe.errors import ProbeError


@dataclass(frozen=True)
class BearerAuth:
    """Bearer-token auth; the token is read from ``token_env`` at send time."""

    token_env: str

    def headers(self) -> dict[str, str]:
        """The ``Authorization: Bearer`` header, with the token read from the environment."""
        return {"Authorization": f"Bearer {_require_env(self.token_env)}"}


@dataclass(frozen=True)
class ApiKeyAuth:
    """API-key auth in a named header; the key is read from ``key_env`` at send time."""

    header: str
    key_env: str

    def headers(self) -> dict[str, str]:
        """The API-key header, with the key read from the environment."""
        return {self.header: _require_env(self.key_env)}


@dataclass(frozen=True)
class HeaderAuth:
    """Raw-header auth; ``headers_env`` holds a JSON object of header name -> value."""

    headers_env: str

    def headers(self) -> dict[str, str]:
        """The headers parsed from the JSON object in the environment variable."""
        raw = _require_env(self.headers_env)
        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise ProbeError(
                f"HeaderAuth env var {self.headers_env!r} is not valid JSON headers: {exc}"
            ) from exc
        if not isinstance(parsed, dict):
            raise ProbeError(f"HeaderAuth env var {self.headers_env!r} must be a JSON object")
        return {str(k): str(v) for k, v in parsed.items()}


@dataclass(frozen=True)
class SubprotocolAuth:
    """WebSocket auth carried in the ``Sec-WebSocket-Protocol`` handshake header.

    The token is read from ``token_env`` at send time and passed to
    ``websockets.sync.client.connect(..., subprotocols=[<token>])``. The TACO target uses this
    shape. Not usable on the HTTP path -- if paired with an ``http(s)://`` endpoint the WS-only
    :meth:`subprotocols` accessor is what the adapter reads, and the ``.headers()`` accessor
    intentionally returns an empty mapping (nothing to inject into HTTP headers).
    """

    token_env: str

    def headers(self) -> dict[str, str]:
        """No headers: this auth travels in the WebSocket subprotocol."""
        # WS-only: no header form. HTTP callers get an empty mapping (auth carried elsewhere).
        return {}

    def subprotocols(self) -> list[str]:
        """The token, read from the environment, as the single WebSocket subprotocol."""
        return [_require_env(self.token_env)]


Auth = BearerAuth | ApiKeyAuth | HeaderAuth | SubprotocolAuth


@dataclass(frozen=True)
class ConnectionConfig:
    """How to reach a target endpoint. Secrets are env-sourced and redacted.

    ``timeout_seconds`` is explicit (never a library default that could change); a dead peer can't
    hang forever. ``max_consecutive_failures`` bounds the circuit breaker.
    """

    endpoint: str
    auth: Auth | None = None
    extra_headers: tuple[tuple[str, str], ...] = ()
    timeout_seconds: float = 30.0
    max_consecutive_failures: int = 3

    def resolved_headers(self) -> dict[str, str]:
        """Resolve auth + extra headers to a plain dict (secrets pulled from env at call time)."""
        headers: dict[str, str] = dict(self.extra_headers)
        if self.auth is not None:
            headers.update(self.auth.headers())
        return headers

    def __repr__(self) -> str:
        # A config's repr must never expose a resolved secret. Auth is shown by TYPE +
        # env-var NAME only -- the value stays in the environment / process memory.
        auth_desc = type(self.auth).__name__ if self.auth is not None else "None"
        return (
            f"ConnectionConfig(endpoint={self.endpoint!r}, auth={auth_desc}, "
            f"timeout_seconds={self.timeout_seconds}, "
            f"max_consecutive_failures={self.max_consecutive_failures})"
        )


def _require_env(name: str) -> str:
    """Read a required credential from the environment, or raise a typed error (never log it).

    Strips leading/trailing whitespace: a copy-pasted token often carries a trailing
    newline, and a newline in a value used as a WS ``Sec-WebSocket-Protocol`` breaks the transport's
    subprotocol grammar before the request is even sent. An empty-after-strip value is treated as a
    missing credential -- the same typed error as an unset variable.
    """
    raw = os.environ.get(name)
    value = raw.strip() if raw is not None else None
    if not value:
        # The message names the VARIABLE, never a value -- there is no value to leak here anyway.
        raise ProbeError(
            f"credential env var {name!r} is not set (or is empty/whitespace-only); probing "
            "credentials are read from the environment (never inlined) -- export it before "
            "probing"
        )
    return value
