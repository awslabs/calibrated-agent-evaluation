# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the stdlib endpoint adapter."""

from __future__ import annotations

import pytest

from _probe_helpers import unreachable_transport
from calibrated_agent_evaluation.probe import (
    ConnectionConfig,
    EndpointAdapter,
    EndpointUnreachableError,
)
from calibrated_agent_evaluation.probe.adapter import _urllib_transport, tool_names


def test_unreachable_endpoint_raises_typed_error_not_empty_success() -> None:
    # Negative: a dead endpoint raises EndpointUnreachableError, never a silent empty-success.
    config = ConnectionConfig(endpoint="https://dead")
    adapter = EndpointAdapter(config, transport=unreachable_transport())
    with pytest.raises(EndpointUnreachableError):
        adapter.send("hello")


def test_detects_the_payload_format_the_endpoint_accepts() -> None:
    # Positive: the adapter tries formats and locks onto the one the endpoint accepts.
    import json

    def picky_transport(url, headers, body, timeout):
        payload = json.loads(body)
        # Accept only the AgentCore SSE shape: {"prompt", "session_id"} with no nested "input".
        if "session_id" in payload and "input" not in payload:
            return 200, [b"data: " + json.dumps({"type": "response", "response": "hi"}).encode()]
        return 404, [b""]

    adapter = EndpointAdapter(ConnectionConfig(endpoint="https://a"), transport=picky_transport)
    resp = adapter.send("real prompt")
    assert adapter.detected_format == "agentcore_sse"
    assert resp.text == "hi"


@pytest.mark.parametrize("scheme", ["file", "ftp", "gopher", ""])
def test_default_transport_rejects_non_http_schemes(scheme: str) -> None:
    # Hardening: the stdlib transport must refuse file://, ftp://, etc. BEFORE opening the request,
    # so urllib's non-HTTP handlers (arbitrary local-file read) are never reachable via a mistyped
    # or malicious endpoint. Raises the typed error rather than performing the read.
    url = f"{scheme}://etc/passwd" if scheme else "etc/passwd"
    with pytest.raises(EndpointUnreachableError, match="scheme"):
        _urllib_transport(url, {}, b"{}", 1.0)


def test_default_transport_allows_https_scheme() -> None:
    # Positive counterpart: an https:// URL PASSES the scheme guard and proceeds to the real
    # urlopen, which then fails at connect with a transport error (URLError/OSError) — NOT the
    # typed scheme refusal. Proves the guard is scheme-specific, not a blanket refusal.
    # (The raw transport does not wrap into EndpointUnreachableError; _send_with does that.)
    import urllib.error

    with pytest.raises((urllib.error.URLError, OSError)) as exc_info:
        _urllib_transport("https://127.0.0.1:1/unreachable", {}, b"{}", 0.05)
    assert "scheme" not in str(exc_info.value)


def test_circuit_opens_after_consecutive_failures() -> None:
    # Negative/edge: after max_consecutive_failures the circuit opens and stops probing.
    config = ConnectionConfig(endpoint="https://dead", max_consecutive_failures=2)
    adapter = EndpointAdapter(config, transport=unreachable_transport())
    for _ in range(2):
        with pytest.raises(EndpointUnreachableError):
            adapter.send("hello")
    # Circuit now open: the next call raises the circuit-open error without another transport hit.
    with pytest.raises(EndpointUnreachableError, match="circuit open"):
        adapter.send("hello")


def test_tool_names_keep_string_values_once_in_first_seen_order() -> None:
    events: list[object] = [
        {"type": "tool_use", "tool_name": "refund_order"},
        {"type": "tool_use", "tool_name": "lookup_order"},
        {"type": "tool_use", "tool_name": "refund_order"},
        # Not a tool's name: a number, null, a list and an object.
        {"tool_name": 7},
        {"tool_name": None},
        {"tool_name": ["lookup_order"]},
        {"tool_name": {"name": "cancel_order"}},
        {"response": "no tool here"},
        # Not an object, so it has no tool_name key.
        "tool_name",
        ["tool_name", "cancel_order"],
        {"type": "tool_use", "tool_name": "cancel_order"},
    ]

    assert tool_names(events) == ("refund_order", "lookup_order", "cancel_order")
    assert tool_names([]) == ()


def test_a_redirect_is_refused_and_the_token_never_reaches_the_second_host() -> None:
    import threading
    from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

    seen_by_target: list[str | None] = []

    class _Target(BaseHTTPRequestHandler):
        def do_POST(self) -> None:
            seen_by_target.append(self.headers.get("Authorization"))
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(b'{"response": "hi"}')

        do_GET = do_POST  # urllib turns a redirected POST into a GET

        def log_message(self, format: str, *args: object) -> None:
            pass

    target = ThreadingHTTPServer(("127.0.0.1", 0), _Target)
    target_url = f"http://127.0.0.1:{target.server_address[1]}/stolen"

    class _Redirector(BaseHTTPRequestHandler):
        def do_POST(self) -> None:
            self.send_response(302)
            self.send_header("Location", target_url)
            self.send_header("Content-Length", "0")
            self.end_headers()

        def log_message(self, format: str, *args: object) -> None:
            pass

    redirector = ThreadingHTTPServer(("127.0.0.1", 0), _Redirector)
    servers = (target, redirector)
    threads = [threading.Thread(target=s.serve_forever, daemon=True) for s in servers]
    for thread in threads:
        thread.start()
    try:
        endpoint = f"http://127.0.0.1:{redirector.server_address[1]}/invocations"
        with pytest.raises(EndpointUnreachableError, match="redirect"):
            _urllib_transport(endpoint, {"Authorization": "Bearer canary-token"}, b"{}", 5.0)
        config = ConnectionConfig(
            endpoint=endpoint, extra_headers=(("Authorization", "Bearer canary-token"),)
        )
        with pytest.raises(EndpointUnreachableError, match="redirect"):
            EndpointAdapter(config).send("hello")
    finally:
        for server in servers:
            server.shutdown()
            server.server_close()
    assert seen_by_target == []


@pytest.mark.skipif(not __import__("os").path.isdir("/dev/fd"), reason="needs /dev/fd")
def test_the_stdlib_transport_closes_its_connection() -> None:
    import gc
    import threading
    from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
    from pathlib import Path

    class _Agent(BaseHTTPRequestHandler):
        def do_POST(self) -> None:
            self.send_response(200)
            self.send_header("Content-Length", "19")
            self.end_headers()
            self.wfile.write(b'{"response": "hi"}\n')

        def log_message(self, format: str, *args: object) -> None:
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), _Agent)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    url = f"http://127.0.0.1:{server.server_address[1]}/"
    try:
        _urllib_transport(url, {}, b"{}", 5.0)  # warm up imports and lazy fds
        gc.collect()
        before = len(list(Path("/dev/fd").iterdir()))
        kept = [_urllib_transport(url, {}, b"{}", 5.0) for _ in range(5)]
        after = len(list(Path("/dev/fd").iterdir()))
        assert all(list(lines) == [b'{"response": "hi"}\n'] for _, lines in kept)
    finally:
        server.shutdown()
        server.server_close()
    assert after - before < 5, f"{after - before} descriptors left open by 5 requests"
