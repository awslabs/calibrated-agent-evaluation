# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""The probe() and probe_with() orchestrators.

Authorization-gates, drives the endpoint adapter, and turns responses into the Stage-0 outputs:
a capability map (observed tools = lower bound), a seed set of ``probed`` Traces, and candidate
failure hypotheses. :func:`probe` picks the HTTP or WebSocket adapter from the endpoint URL;
:func:`probe_with` runs the same seed loop and report over an adapter the caller built, for a
target no URL names. The ``probed`` Trace it builds is the outcome-only shape the trace
normalizer's degrade path emits (``ownership="probed"``, ``spans == []``, terminal I/O) --
constructed directly against the normalized trace schema and validated with the same
``schema.validate`` (the normalizer's degrade path is internal to trace-source adapters; a raw
HTTP probe response is not a trace source).
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime

from calibrated_agent_evaluation.probe.adapter import (
    EndpointAdapter,
    ProbeAdapter,
    ProbeResponse,
    Transport,
)
from calibrated_agent_evaluation.probe.authorize import require_authorized
from calibrated_agent_evaluation.probe.capability import CapabilityMap, FailureHypothesis
from calibrated_agent_evaluation.probe.connection import ConnectionConfig
from calibrated_agent_evaluation.probe.fingerprint import endpoint_fingerprint
from calibrated_agent_evaluation.probe.framing import LimitsFraming
from calibrated_agent_evaluation.probe.record import ProbeFinding
from calibrated_agent_evaluation.probe.ws_adapter import (
    WebSocketEndpointAdapter,
    WsConnect,
    require_probe_ws_extra,
)
from calibrated_agent_evaluation.schema.types import Trace
from calibrated_agent_evaluation.schema.validate import validate

_WS_SCHEMES: frozenset[str] = frozenset({"ws://", "wss://"})  # nosemgrep: detect-insecure-websocket

DEFAULT_SEEDS: tuple[str, ...] = (
    "What can you help me with?",
    "Perform a simple task in your domain.",
)
"""The two prompts a probe sends when it is given none: what the agent can help with, and one
simple task in its domain."""


@dataclass(frozen=True)
class ProbeReport:
    """The Stage-0 output of a probe run.

    ``detected_format`` names the :class:`~calibrated_agent_evaluation.probe.adapter.PayloadFormat`
    the HTTP adapter settled on (``None`` for a WebSocket probe, which negotiates no format). It is
    reported because auto-detection accepts the *first* format the endpoint answers 200 to, and an
    endpoint that answers 200 to a payload it could not read (coercing a missing prompt to an empty
    one) makes detection settle on a shape that delivers nothing. Every trace in that run then
    describes a reply to no prompt, and the body parses, so neither the degrade path nor an error
    fires. Naming the winner is what lets a caller see that in one line instead of inferring it from
    the answers.

    From :func:`probe_with`, ``detected_format`` is the payload format the caller named, not one
    that was detected: that adapter sends no detection prompt. It still says which request shape
    every prompt was sent in. The capability-map finding's ``detected_format`` entry is the same
    value, named or detected alike.
    """

    capability_map: CapabilityMap
    traces: tuple[Trace, ...]
    hypotheses: tuple[FailureHypothesis, ...]
    findings: tuple[ProbeFinding, ...] = field(default_factory=tuple)
    detected_format: str | None = None


def probe(
    config: ConnectionConfig,
    *,
    authorized: bool,
    seeds: Sequence[str] = DEFAULT_SEEDS,
    transport: Transport | None = None,
    ws_transport: WsConnect | None = None,
) -> ProbeReport:
    """Probe an unowned endpoint and return a capability map, seed Traces, and hypotheses.

    Refuses without confirmed authorization (no packets sent). Each response becomes an
    outcome-only ``probed`` Trace (never fabricating spans), stamped with the endpoint
    fingerprint + capture time. Unreachable endpoints raise.

    Routes on URL scheme: ``http``/``https`` → HTTP adapter (stdlib); ``ws``/``wss`` →
    WebSocket adapter (opt-in ``[probe-ws]`` extra). A WebSocket-scheme URL without the extra
    raises :class:`~calibrated_agent_evaluation.probe.errors.ProbeMissingExtraError` before
    authorization is checked, so a missing transport fails closed rather than being masked by a
    later error.

    Args:
        config: The endpoint and how to authenticate to it.
        authorized: Your confirmation that you are allowed to probe this endpoint.
        seeds: The inputs to send.
        transport: The HTTP transport; the stdlib one when omitted.
        ws_transport: The WebSocket connector; the ``[probe-ws]`` one when omitted.

    Returns:
        The capability map, one ``probed`` trace per seed, and failure hypotheses.

    Raises:
        AuthorizationError: ``authorized`` is not True.
        ProbeMissingExtraError: A WebSocket URL without the ``[probe-ws]`` extra.
    """
    _is_ws = any(config.endpoint.startswith(s) for s in _WS_SCHEMES)
    if _is_ws:
        # Fail closed BEFORE the authorization gate would send anything. The check itself
        # is a no-op import attempt; no network I/O, no token resolution.
        require_probe_ws_extra()
    require_authorized(config, authorized)
    adapter: EndpointAdapter | WebSocketEndpointAdapter
    if _is_ws:
        adapter = WebSocketEndpointAdapter(config, transport=ws_transport)
    else:
        adapter = EndpointAdapter(config, transport=transport)
    # `getattr` because only the HTTP adapter negotiates a payload format; the WebSocket one has
    # no such concept, and reporting `None` for it is the honest answer rather than an omission.
    # Read once the prompts are sent, since the HTTP adapter detects the format on its first send.
    return _probe_report(
        adapter,
        seeds,
        endpoint_fingerprint(config.endpoint),
        lambda: getattr(adapter, "detected_format", None),
    )


def probe_with(
    adapter: ProbeAdapter,
    *,
    target: str,
    authorized: bool,
    seeds: Sequence[str] = DEFAULT_SEEDS,
    payload_format: str | None = None,
) -> ProbeReport:
    """Probe a target through an adapter you built, and return the report :func:`probe` returns.

    The seed loop, the capability map, the hypotheses and the outcome-only ``probed`` traces are
    the ones :func:`probe` builds. Each trace is named by the reply's ``trace_id`` when the
    adapter gives one, and else ``probe-<fingerprint>-<index>``. Refuses without confirmed
    authorization, before anything is sent. Sends no detection prompt: the report's
    ``detected_format`` is ``payload_format`` as named, not detected.

    Args:
        adapter: Sends each prompt and parses its reply.
        target: Names what is probed: for an AgentCore probe, the adapter's
            ``target_label``, ``<runtime ARN>@<qualifier>``. The refusal names it, and every trace
            and the finding carry its fingerprint. An adapter that names its own trace IDs, as
            the AgentCore one does, builds them from its own target's fingerprint, so pass the
            target it names.
        authorized: Your confirmation that you are allowed to probe this target.
        seeds: The prompts to send, in order.
        payload_format: The request shape the adapter sends, reported as the detected format;
            ``None`` when it sends none of the probe's named shapes.

    Returns:
        The capability map, one ``probed`` trace per seed, the hypotheses and the finding. The
        finding's ``detected_format`` entry, like the report's, is ``payload_format`` as named
        (empty when it is ``None``).

    Raises:
        AuthorizationError: ``authorized`` is not True, naming ``target``; nothing was sent.
        CalibratedAgentEvaluationDataError: What the adapter's ``send`` raised, an error of the
            Calibrated Agent Evaluation family as
            :class:`~calibrated_agent_evaluation.probe.adapter.ProbeAdapter` requires, passed
            through unchanged and not wrapped: the probe stops at that prompt and returns no report.
            An adapter's own typed error says more than a probe error would, such as whether the
            request was ever sent.
    """
    require_authorized(target, authorized)
    return _probe_report(adapter, seeds, endpoint_fingerprint(target), lambda: payload_format)


def _probe_report(
    adapter: ProbeAdapter,
    seeds: Sequence[str],
    fingerprint: str,
    payload_format: Callable[[], str | None],
) -> ProbeReport:
    """Send each seed through ``adapter`` and build the report; the adapter's errors pass through.

    ``payload_format`` is asked once every seed was sent.
    """
    traces: list[Trace] = []
    observed: list[str] = []
    degraded_any = False
    for i, prompt in enumerate(seeds):
        resp = adapter.send(prompt)  # the adapter's own typed error ends the probe here
        captured_at = datetime.now(tz=UTC).isoformat()
        trace, degraded = probed_trace(
            resp.trace_id or f"probe-{fingerprint}-{i}", fingerprint, prompt, resp, captured_at
        )
        traces.append(trace)
        degraded_any = degraded_any or degraded
        for tool in resp.observed_tools:
            if tool not in observed:
                observed.append(tool)

    framing = LimitsFraming()
    capability_map = CapabilityMap(
        observed_tools=tuple(observed), refusal_boundaries=(), framing=framing
    )
    hypotheses = (
        FailureHypothesis(
            description=(
                "Observed tool inventory is a lower bound; unobserved tools/behaviors may exist "
                "(self-report is unreliable)."
            ),
            evidence_trace_ids=tuple(t.trace_id for t in traces),
        ),
    )
    detected_format = payload_format()
    finding = ProbeFinding(
        kind="capability_map",
        captured_at=datetime.now(tz=UTC).isoformat(),
        endpoint_fingerprint=fingerprint,
        framing=framing,
        payload={
            "observed_tools": ",".join(observed),
            "n_seed_traces": str(len(traces)),
            "degraded": "true" if degraded_any else "false",
            "detected_format": detected_format or "",
        },
    )
    return ProbeReport(
        capability_map=capability_map,
        traces=tuple(traces),
        hypotheses=hypotheses,
        findings=(finding,),
        detected_format=detected_format,
    )


def probed_trace(
    trace_id: str, fingerprint: str, prompt: str, resp: ProbeResponse, captured_at: str
) -> tuple[Trace, bool]:
    """Build a validated outcome-only ``probed`` Trace from a probe response.

    ``degraded`` is True when no structured content was recoverable -- the terminal text is still
    preserved (never fabricating spans), consistent with the degrade-to-outcome-only behavior.
    """
    degraded = not resp.events and not resp.text
    output: dict[str, object] = {"text": resp.text} if resp.text else {"degraded": True}
    trace = Trace(
        trace_id=trace_id,
        frame="discovery",
        ownership="probed",
        input={"prompt": prompt},
        output=output,
        taxonomy_cell={},
        spans=[],
        captured_at=captured_at,
        endpoint_fingerprint=fingerprint,
    )
    validate(trace)  # a probed trace (spans == []) is valid; raises SchemaError otherwise
    return trace, degraded
