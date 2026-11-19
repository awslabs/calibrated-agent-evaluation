# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for ConnectionConfig + auth credential handling."""

from __future__ import annotations

import pytest

from calibrated_agent_evaluation.probe import BearerAuth, ConnectionConfig, ProbeError


def test_repr_never_exposes_the_resolved_token(monkeypatch: pytest.MonkeyPatch) -> None:
    # Negative: the secret must not appear in repr (which lands in logs/artifacts).
    monkeypatch.setenv("CALIBRATED_AGENT_EVALUATION_PROBE_TOKEN", "super-secret-value")
    config = ConnectionConfig(
        endpoint="https://agent.example/invoke",
        auth=BearerAuth("CALIBRATED_AGENT_EVALUATION_PROBE_TOKEN"),
    )
    assert "super-secret-value" not in repr(config)
    assert "BearerAuth" in repr(config)  # type shown, value redacted


def test_token_is_read_from_env_not_a_field(monkeypatch: pytest.MonkeyPatch) -> None:
    # Positive: the resolved header carries the token (env-sourced), proving no inlined field.
    monkeypatch.setenv("CALIBRATED_AGENT_EVALUATION_PROBE_TOKEN", "tok-123")
    config = ConnectionConfig(
        endpoint="https://a", auth=BearerAuth("CALIBRATED_AGENT_EVALUATION_PROBE_TOKEN")
    )
    assert config.resolved_headers()["Authorization"] == "Bearer tok-123"


def test_missing_env_var_raises_typed_error(monkeypatch: pytest.MonkeyPatch) -> None:
    # Negative: an unset credential var raises a typed ProbeError (naming the var, not a value).
    monkeypatch.delenv("CALIBRATED_AGENT_EVALUATION_PROBE_TOKEN", raising=False)
    config = ConnectionConfig(
        endpoint="https://a", auth=BearerAuth("CALIBRATED_AGENT_EVALUATION_PROBE_TOKEN")
    )
    with pytest.raises(ProbeError):
        config.resolved_headers()


def test_subprotocol_auth_repr_never_exposes_the_token(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # SubprotocolAuth resolves its token for the WS handshake (`subprotocols=[<token>]`);
    # like BearerAuth, the resolved value must never appear in a config repr.
    from calibrated_agent_evaluation.probe import SubprotocolAuth

    monkeypatch.setenv("CALIBRATED_AGENT_EVALUATION_PROBE_TOKEN", "s3cret-taco-token")
    config = ConnectionConfig(
        endpoint="wss://taco.example/agent",
        auth=SubprotocolAuth(token_env="CALIBRATED_AGENT_EVALUATION_PROBE_TOKEN"),
    )
    representation = repr(config)
    assert "s3cret-taco-token" not in representation
    assert "SubprotocolAuth" in representation  # type shown, value redacted


def test_subprotocol_auth_missing_env_raises_typed_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Negative: an unset subprotocol-token env var raises a typed ProbeError, not KeyError.
    from calibrated_agent_evaluation.probe import SubprotocolAuth

    monkeypatch.delenv("CALIBRATED_AGENT_EVALUATION_PROBE_TOKEN", raising=False)
    auth = SubprotocolAuth(token_env="CALIBRATED_AGENT_EVALUATION_PROBE_TOKEN")
    with pytest.raises(ProbeError):
        auth.subprotocols()


def test_env_credential_is_whitespace_stripped(monkeypatch: pytest.MonkeyPatch) -> None:
    # A copy-pasted token with a trailing newline resolves to the trimmed value (positive).
    monkeypatch.setenv("CALIBRATED_AGENT_EVALUATION_PROBE_TOKEN", "tok-abc\n")
    config = ConnectionConfig(
        endpoint="https://a", auth=BearerAuth("CALIBRATED_AGENT_EVALUATION_PROBE_TOKEN")
    )
    assert config.resolved_headers()["Authorization"] == "Bearer tok-abc"


def test_whitespace_only_credential_is_treated_as_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # A whitespace-only value raises the same typed error as an unset var (negative).
    monkeypatch.setenv("CALIBRATED_AGENT_EVALUATION_PROBE_TOKEN", "   \n\t ")
    config = ConnectionConfig(
        endpoint="https://a", auth=BearerAuth("CALIBRATED_AGENT_EVALUATION_PROBE_TOKEN")
    )
    with pytest.raises(ProbeError):
        config.resolved_headers()
