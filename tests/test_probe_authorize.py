# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the refuse-by-default authorization gate."""

from __future__ import annotations

import pytest

from _probe_helpers import json_transport
from calibrated_agent_evaluation.probe import (
    AuthorizationError,
    ConnectionConfig,
    confirm_authorization,
    probe,
)


def test_unauthorized_probe_refuses_and_sends_nothing() -> None:
    # Negative: without confirmed authorization, no HTTP request is made. A transport that would
    # raise if called proves zero packets were sent.
    def exploding_transport(url, headers, body, timeout):
        raise AssertionError("transport must not be called when unauthorized")

    config = ConnectionConfig(endpoint="https://a")
    with pytest.raises(AuthorizationError):
        probe(config, authorized=False, transport=exploding_transport)


def test_authorized_probe_proceeds() -> None:
    config = ConnectionConfig(endpoint="https://a")
    report = probe(config, authorized=True, transport=json_transport())
    assert report.traces


def test_confirm_authorization_records_an_auditable_attestation() -> None:
    finding = confirm_authorization(ConnectionConfig(endpoint="https://agent.example/invoke"))
    assert finding.kind == "authorization"
    assert finding.payload["authorized"] == "true"
    assert finding.endpoint_fingerprint  # attributable
