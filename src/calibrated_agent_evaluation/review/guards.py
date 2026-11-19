# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Request guards for the local review surface: trusted hosts and same-origin writes.

The surface binds to a local port and writes gold labels, so a page on any other site the labeler
has open must not be able to drive it. Two guards close that:

- a Host allowlist, so a DNS-rebinding page (an attacker's domain re-pointed at 127.0.0.1) is
  refused before it can read trace content;
- a same-origin rule for every request that changes state. The browser client sends an
  ``HX-Request`` header on every call, and a cross-origin page cannot add a custom header without
  a CORS preflight this app never approves. A browser-supplied ``Origin`` or ``Referer`` from
  another origin is refused even when the header is present.

Neither guard stops a non-browser client that can reach the port, which is why the CLI serves the
surface on a loopback address only.
"""

from __future__ import annotations

import secrets
import socket
from collections.abc import Iterable, Sequence
from typing import Any
from urllib.parse import urlsplit

from starlette.datastructures import Headers
from starlette.responses import PlainTextResponse
from starlette.types import ASGIApp, Receive, Scope, Send

from calibrated_agent_evaluation.review.addresses import host_of

__all__ = [
    "DEFAULT_ALLOWED_HOSTS",
    "SameOriginWrites",
    "TrustedHosts",
    "allowed_hosts_for",
    "sessionless",
]

DEFAULT_ALLOWED_HOSTS: tuple[str, ...] = ("127.0.0.1", "localhost", "::1")
"""Host names the surface answers to when no other bind host is configured."""


def sessionless() -> dict[str, Any]:
    """``FastHTML`` options that turn its session middleware off.

    The review surfaces keep no session state. Without a ``secret_key``, ``FastHTML`` reads or
    creates a ``.sesskey`` file in the current directory for a session cookie nothing uses; a
    throwaway key in memory stops that write.
    """
    return {"sess_cls": None, "secret_key": secrets.token_urlsafe(32)}


_SAFE_METHODS = frozenset({"GET", "HEAD", "OPTIONS"})
_WILDCARD_BINDS = frozenset({"0.0.0.0", "::", ""})  # noqa: S104  # names, not a bind


def allowed_hosts_for(bind_host: str | None, extra: Iterable[str] = ()) -> tuple[str, ...]:
    """The Host header names the surface should accept when served on ``bind_host``.

    Args:
        bind_host: The address the server binds to; ``None`` for the defaults only. A wildcard
            bind (``0.0.0.0`` or ``::``) adds this machine's host name and fully qualified name,
            the names other machines on the network would use.
        extra: More host names to accept.

    Returns:
        The defaults plus the bind host (or the machine's names) plus ``extra``, de-duplicated in
        order.
    """
    hosts = list(DEFAULT_ALLOWED_HOSTS)
    if bind_host is not None:
        if bind_host in _WILDCARD_BINDS:
            hosts += [socket.gethostname(), socket.getfqdn()]
        else:
            hosts.append(bind_host)
    hosts += list(extra)
    return tuple(dict.fromkeys(h.strip("[]").lower() for h in hosts if h))


class TrustedHosts:
    """ASGI middleware that answers only to the allowed Host names, which blocks DNS rebinding.

    Starlette's ``TrustedHostMiddleware`` splits the Host header on the first colon, so a
    bracketed IPv6 Host (``[::1]:5001``) never matches; this one parses it.
    """

    def __init__(self, app: ASGIApp, allowed_hosts: Sequence[str]) -> None:
        """Wrap ``app``.

        Args:
            app: The ASGI app to guard.
            allowed_hosts: The accepted host names, IPv6 with or without brackets.
        """
        self.app = app
        self.allowed = frozenset(h.strip("[]").lower() for h in allowed_hosts)

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        """Pass a request whose Host is allowed; answer 400 to anything else."""
        if scope["type"] not in ("http", "websocket"):
            await self.app(scope, receive, send)
            return
        if host_of(Headers(scope=scope).get("host", "")) in self.allowed:
            await self.app(scope, receive, send)
            return
        await PlainTextResponse("Invalid host header", status_code=400)(scope, receive, send)


class SameOriginWrites:
    """ASGI middleware that refuses a state-changing request unless it comes from this origin."""

    def __init__(self, app: ASGIApp) -> None:
        """Wrap ``app``.

        Args:
            app: The ASGI app to guard.
        """
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        """Pass a safe or same-origin request through; answer 403 to anything else."""
        if scope["type"] != "http" or scope["method"] in _SAFE_METHODS:
            await self.app(scope, receive, send)
            return
        reason = _cross_origin_reason(Headers(scope=scope), scope.get("scheme", "http"))
        if reason is None:
            await self.app(scope, receive, send)
            return
        response = PlainTextResponse(f"Refused: {reason}", status_code=403)
        await response(scope, receive, send)


def _cross_origin_reason(headers: Headers, scheme: str) -> str | None:
    """Why the request is not same-origin, or ``None`` when it is."""
    host = headers.get("host", "")
    own = f"{scheme}://{host}"
    origin = headers.get("origin")
    if origin is not None and origin != own:
        return f"a write from origin {origin!r} is not allowed; the review surface is {own}"
    referer = headers.get("referer")
    if origin is None and referer is not None and _origin_of(referer) != own:
        return "a write linked from another site is not allowed"
    if headers.get("hx-request") != "true":
        return "a write must come from the review page (the HX-Request header is missing)"
    return None


def _origin_of(url: str) -> str:
    parts = urlsplit(url)
    return f"{parts.scheme}://{parts.netloc}"
