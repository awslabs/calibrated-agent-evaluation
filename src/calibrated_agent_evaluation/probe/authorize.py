# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Refuse-by-default authorization gate.

Probing a running endpoint requires an explicit human attestation of authorization -- distinct
from *authentication* (the credentials in :class:`ConnectionConfig`). A token proves you can reach
the endpoint, not that you are allowed to probe it. The tool cannot verify authorization, so it
records the attestation (an auditable :class:`ProbeFinding`) and refuses to send anything
until it is confirmed.
"""

from __future__ import annotations

from datetime import UTC, datetime

from calibrated_agent_evaluation.probe.connection import ConnectionConfig
from calibrated_agent_evaluation.probe.errors import AuthorizationError
from calibrated_agent_evaluation.probe.fingerprint import endpoint_fingerprint
from calibrated_agent_evaluation.probe.framing import LimitsFraming
from calibrated_agent_evaluation.probe.record import ProbeFinding


def require_authorized(target: ConnectionConfig | str, authorized: bool) -> None:
    """Raise :class:`AuthorizationError` unless authorization was explicitly confirmed.

    Called before any request is sent -- refuse-by-default means an unauthorized probe emits
    zero packets.

    Args:
        target: The endpoint about to be probed, or a label naming a target no URL names, such
            as ``<runtime ARN>@<qualifier>``.
        authorized: Your confirmation that you are allowed to probe it.

    Raises:
        AuthorizationError: ``authorized`` is not True, naming the endpoint or the label.
    """
    if not authorized:
        named = target.endpoint if isinstance(target, ConnectionConfig) else target
        raise AuthorizationError(
            f"probing {named!r} requires confirmed authorization; refusing to send any "
            "request. Pass authorized=True only if you are authorized to probe it."
        )


def confirm_authorization(config: ConnectionConfig) -> ProbeFinding:
    """Record the authorization attestation as an auditable finding."""
    return ProbeFinding(
        kind="authorization",
        captured_at=datetime.now(tz=UTC).isoformat(),
        endpoint_fingerprint=endpoint_fingerprint(config.endpoint),
        framing=LimitsFraming(),
        payload={"endpoint": config.endpoint, "authorized": "true"},
    )
