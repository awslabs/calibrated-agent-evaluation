# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Endpoint fingerprint for the non-stationarity stamp.

A stable, secret-free identifier for a probed endpoint: the SHA-256 of the name the probe gives it
(never the credentials). That name is the endpoint URL for an HTTP or WebSocket probe, and
``<runtime ARN>@<qualifier>`` for an AgentCore Runtime probe. Stamped on every Stage-0 output so
results are attributable to a specific endpoint at a specific time even though a remote agent is
non-stationary.
"""

from __future__ import annotations

import hashlib


def endpoint_fingerprint(endpoint: str) -> str:
    """Return a short, stable, secret-free fingerprint of ``endpoint``, the probed target's name."""
    return "ep-" + hashlib.sha256(endpoint.encode("utf-8")).hexdigest()[:16]
