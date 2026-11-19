# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""HTTP endpoint adapter over the Python standard library.

Talks to an unowned agent endpoint using only ``urllib.request`` (stdlib) -- no third-party HTTP
client, so ``probe/`` stays vendor-free. Multi-format payload **auto-detection** (generic JSON,
OpenAI-compatible, AgentCore SSE, nested-AgentCore SSE), SSE + plain-JSON parsing, and a
consecutive-failure circuit breaker. Any transport/HTTP error is wrapped into
:class:`EndpointUnreachableError` -- never a silent empty-success.

The transport is injectable (``transport=``) so unit tests run against a stub mirroring the real
``(status, line-iterator)`` surface without opening a socket; the default binds to ``urllib``.

:class:`ProbeAdapter` is what the probe needs from any adapter: one ``send`` per prompt. The HTTP
and WebSocket adapters satisfy it, and :func:`~calibrated_agent_evaluation.probe.harness.probe_with`
runs the probe over any other one. :func:`tool_names` is the rule every adapter uses to read the
tools a reply names.
"""

from __future__ import annotations

import json
import time
import urllib.error
import urllib.parse
import urllib.request
import uuid
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
from typing import Protocol

from calibrated_agent_evaluation.probe.connection import ConnectionConfig
from calibrated_agent_evaluation.probe.errors import EndpointUnreachableError
from calibrated_agent_evaluation.probe.redact import scrub

# A transport maps (url, headers, body) -> (status_code, response-body-lines). Stdlib by default;
# tests inject a fake that mirrors this exact surface (rubric-blessed stub).
Transport = Callable[[str, dict[str, str], bytes, float], "tuple[int, Iterable[bytes]]"]

_DETECT_PROMPT = "Hello, what can you help me with?"

# The stdlib default transport speaks only ``http``/``https``. ``urllib`` also honors ``file://``,
# ``ftp://`` and other schemes; a probe endpoint is always an HTTP(S) agent, so an explicit
# allowlist means a stray ``file://`` (e.g. a misconfigured endpoint or a path pasted where a URL
# was meant) can never turn into an arbitrary local-file read. Defense-in-depth: the endpoint is
# operator-supplied and authorization-gated, but the toolkit ships a hard boundary anyway. WS
# schemes route to the WebSocket adapter upstream (harness), so they never reach this transport.
_HTTP_SCHEMES: frozenset[str] = frozenset({"http", "https"})


class _RefuseRedirects(urllib.request.HTTPRedirectHandler):
    """Refuse every redirect.

    urllib's default handler re-sends the request headers, ``Authorization`` included, to whatever
    host the ``Location`` names, and will follow https to http. A probe endpoint is a fixed agent
    URL, so a redirect is treated as a misconfiguration and never followed.
    """

    def redirect_request(self, req, fp, code, msg, headers, newurl):  # noqa: ARG002 - urllib's signature
        fp.close()
        raise EndpointUnreachableError(
            f"endpoint {req.full_url!r} answered HTTP {code} redirect to {newurl!r}; the probe "
            "does not follow redirects, so its credentials never go to another URL. Point "
            "--endpoint at the final URL"
        )


_OPENER = urllib.request.build_opener(_RefuseRedirects)


@dataclass(frozen=True)
class PayloadFormat:
    """One request shape the adapter can speak; auto-detection picks the one the endpoint accepts.

    ``build_body`` renders the prompt to a JSON body; ``url_suffix`` is appended to the endpoint
    for formats that need it (e.g. AgentCore's ``?qualifier=DEFAULT``).
    """

    name: str
    build_body: Callable[[str, str], dict[str, object]]
    url_suffix: str = ""


# Ordered by commonality (the prior AgentSession's PAYLOAD_FORMATS, stdlib-ported).
PAYLOAD_FORMATS: tuple[PayloadFormat, ...] = (
    PayloadFormat("generic_message", lambda prompt, _sid: {"message": prompt}),
    PayloadFormat(
        "openai", lambda prompt, _sid: {"messages": [{"role": "user", "content": prompt}]}
    ),
    PayloadFormat("agentcore_sse", lambda prompt, sid: {"prompt": prompt, "session_id": sid}),
    PayloadFormat(
        "nested_agentcore",
        lambda prompt, sid: {
            "input": {"prompt": json.dumps({"prompt": prompt, "context": {}})},
            "session_id": sid,
        },
        url_suffix="?qualifier=DEFAULT",
    ),
)


@dataclass(frozen=True)
class ProbeResponse:
    """One parsed probe response (capability-discovery raw material).

    Attributes:
        status: The HTTP status of the reply.
        text: The reply's final text.
        observed_tools: The tools the reply named, in first-seen order, each once.
        events: The reply's decoded JSON objects, in order.
        trace_id: The ID the probe names this reply's trace by. ``None`` lets the probe name it
            after the endpoint and the prompt's position; the HTTP and WebSocket adapters give
            none.
    """

    status: int
    text: str
    observed_tools: tuple[str, ...] = ()
    events: tuple[dict, ...] = ()
    trace_id: str | None = None


class ProbeAdapter(Protocol):
    """Sends one probe prompt to a target and answers with the parsed reply.

    The probe calls nothing else, so any object with this method can be probed through
    :func:`~calibrated_agent_evaluation.probe.harness.probe_with`. :class:`EndpointAdapter` and
    :class:`~calibrated_agent_evaluation.probe.ws_adapter.WebSocketEndpointAdapter` satisfy it. An
    adapter raises only errors of the Calibrated Agent Evaluation family, so a failed prompt reaches
    its caller typed and explained.
    """

    def send(self, prompt: str) -> ProbeResponse:
        """Send ``prompt`` and return the parsed reply, or raise the adapter's own typed error.

        Args:
            prompt: The probe prompt.

        Returns:
            The parsed reply.

        Raises:
            CalibratedAgentEvaluationDataError: The prompt was refused or failed: an error of the
                :class:`~calibrated_agent_evaluation.schema.errors.CalibratedAgentEvaluationDataError`
                family, such as a :class:`~calibrated_agent_evaluation.probe.errors.ProbeError` or
                the adapter's own subclass, never a raw transport or SDK error. The probe passes it
                through unwrapped.
        """
        ...


def tool_names(events: Iterable[object]) -> tuple[str, ...]:
    """The tools a reply's events name: each ``tool_name`` string, in first-seen order, once.

    Only an object's ``tool_name`` key names a tool, and only when it holds a string; any other
    value there, and any event that is not an object, names none. Every probe route passes it a
    stream's events only: a reply that is a single JSON document is read for its text alone and
    names no tool, so one reply body reports the same tools on every route.

    Args:
        events: The reply's decoded values, in order.

    Returns:
        The tool names, without repeats.
    """
    observed: list[str] = []
    for event in events:
        tool = event.get("tool_name") if isinstance(event, Mapping) else None
        if isinstance(tool, str) and tool not in observed:
            observed.append(tool)
    return tuple(observed)


def _urllib_transport(
    url: str, headers: dict[str, str], body: bytes, timeout: float
) -> tuple[int, Iterable[bytes]]:
    """Default stdlib transport: POST ``body`` to ``url`` and stream response lines.

    Rejects any non-HTTP(S) scheme before opening the request so ``urllib``'s ``file://``/``ftp://``
    handlers can never be reached via a mistyped or malicious endpoint (arbitrary file read / SSRF
    hardening), and refuses redirects (see :class:`_RefuseRedirects`). WS schemes are routed away
    by the harness and never arrive here.
    """
    scheme = urllib.parse.urlsplit(url).scheme.lower()
    if scheme not in _HTTP_SCHEMES:
        raise EndpointUnreachableError(
            f"endpoint {url!r} uses unsupported URL scheme {scheme or '(none)'!r}; the HTTP "
            "probe transport allows only http/https"
        )
    # nosec B310 - scheme is allowlisted to http/https just above; endpoint is operator-supplied
    # and authorization-gated, not untrusted web input.
    req = urllib.request.Request(url, data=body, headers=headers, method="POST")  # noqa: S310 - operator-supplied endpoint, scheme-checked, not untrusted web input
    try:
        with _OPENER.open(req, timeout=timeout) as resp:  # nosec B310
            return resp.status, resp.readlines()
    except urllib.error.HTTPError as exc:
        exc.close()  # an HTTPError holds the response's socket open
        raise


class EndpointAdapter:
    """Sends prompts to a target endpoint and parses SSE/JSON responses.

    On the first send it auto-detects which :class:`PayloadFormat` the endpoint accepts and reuses
    it for subsequent sends.
    """

    def __init__(self, config: ConnectionConfig, *, transport: Transport | None = None) -> None:
        self._config = config
        self._transport = transport or _urllib_transport
        self._consecutive_failures = 0
        self._session_id = (
            f"calibrated-agent-evaluation-{int(time.time() * 1000)}-{uuid.uuid4().hex}"
        )
        self._format: PayloadFormat | None = None

    @property
    def detected_format(self) -> str | None:
        """The name of the payload format that worked, once detected (else ``None``)."""
        return self._format.name if self._format else None

    def send(self, prompt: str) -> ProbeResponse:
        """Send ``prompt`` and return the parsed response, or raise on failure.

        Auto-detects the payload format on first use. Opens the circuit (raises) once
        ``max_consecutive_failures`` is reached rather than hammering a dead endpoint.
        """
        if self._consecutive_failures >= self._config.max_consecutive_failures:
            raise EndpointUnreachableError(
                f"circuit open: {self._consecutive_failures} consecutive failures against "
                f"{self._config.endpoint!r}; refusing to keep probing"
            )
        if self._format is None:
            self._detect()  # raises EndpointUnreachableError (+increments) if nothing connects
        return self._send_with(self._format, prompt)  # type: ignore[arg-type]  # set by _detect

    def _detect(self) -> None:
        """Try each payload format until one returns a usable response (ported auto-detection)."""
        reasons: list[str] = []
        for fmt in PAYLOAD_FORMATS:
            try:
                # Success = the endpoint ACCEPTED this format (HTTP 200). A reachable endpoint that
                # returns empty/unparseable content is detected here and degraded later --
                # only a format the endpoint rejects (non-200) or a dead endpoint is skipped/failed.
                self._send_with(fmt, _DETECT_PROMPT, _count_failure=False)
            except EndpointUnreachableError as exc:
                # Keep WHY each format was rejected (HTTP status / transport error) so the aggregate
                # failure is debuggable -- 401/403 => auth, 404 => path, 400 => body shape.
                reasons.append(f"{fmt.name} -> {exc}")
                continue
            self._format = fmt
            self._consecutive_failures = 0
            return
        self._consecutive_failures += 1
        detail = "; ".join(reasons) if reasons else "no formats attempted"
        raise EndpointUnreachableError(
            f"could not connect to {self._config.endpoint!r} with any known payload format. "
            f"Per-format results: {detail}"
        )

    def _send_with(
        self, fmt: PayloadFormat, prompt: str, *, _count_failure: bool = True
    ) -> ProbeResponse:
        headers = {"Content-Type": "application/json", **self._config.resolved_headers()}
        url = self._config.endpoint + fmt.url_suffix
        body = json.dumps(fmt.build_body(prompt, self._session_id)).encode("utf-8")
        try:
            status, lines = self._transport(url, headers, body, self._config.timeout_seconds)
        except (urllib.error.URLError, OSError, TimeoutError) as exc:
            if _count_failure:
                self._consecutive_failures += 1
            # Scrub any resolved credential a transport error message might echo.
            raw = f"endpoint {url!r} unreachable: {type(exc).__name__}: {exc}"
            raise EndpointUnreachableError(scrub(raw, list(headers.values()))) from exc
        if status != 200:
            if _count_failure:
                self._consecutive_failures += 1
            raise EndpointUnreachableError(f"endpoint {url!r} returned HTTP {status}")
        if _count_failure:
            self._consecutive_failures = 0
        return _parse(lines)


def _parse(lines: Iterable[bytes]) -> ProbeResponse:
    """Parse a response stream as SSE (``data:`` lines) or a single JSON object."""
    text_parts: list[str] = []
    events: list[dict] = []
    decoded_lines: list[str] = []
    for chunk in lines:
        decoded_lines.extend(chunk.decode("utf-8", errors="replace").splitlines())

    saw_sse = False
    for line in decoded_lines:
        if not line.startswith("data: "):
            continue
        saw_sse = True
        try:
            event = json.loads(line[6:])
        except json.JSONDecodeError:
            continue
        events.append(event)
        for field_name in ("response", "content", "message", "text", "output", "answer", "data"):
            val = event.get(field_name)
            if isinstance(val, str):
                text_parts.append(val)
                break

    # Only stream events name tools; a single JSON document is read for its text alone.
    observed = tool_names(events)
    if not saw_sse:
        try:
            doc = json.loads("\n".join(decoded_lines))
        except json.JSONDecodeError:
            return ProbeResponse(status=200, text="\n".join(decoded_lines).strip())
        events.append(doc)
        for field_name in ("response", "content", "message", "text", "output", "answer"):
            val = doc.get(field_name)
            if isinstance(val, str):
                text_parts.append(val)
                break

    return ProbeResponse(
        status=200,
        text="".join(text_parts),
        observed_tools=observed,
        events=tuple(events),
    )
