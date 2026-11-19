# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Credential scrubbing."""

from __future__ import annotations

from calibrated_agent_evaluation.probe import scrub


def test_scrub_removes_every_occurrence_of_each_secret() -> None:
    msg = "invalid subprotocol: TOKEN123; retry with TOKEN123"
    out = scrub(msg, ["TOKEN123"])
    assert "TOKEN123" not in out
    assert out.count("[redacted]") == 2
    # non-secret text is retained (the diagnostic value survives).
    assert out.startswith("invalid subprotocol: ")


def test_scrub_with_no_secrets_returns_message_unchanged() -> None:
    msg = "server rejected WebSocket connection: HTTP 401"
    assert scrub(msg, []) == msg
    # an empty-string "secret" must not blank the whole message.
    assert scrub(msg, ["", "  "]) == msg


def test_scrub_masks_a_secret_that_contains_another() -> None:
    # Longer secrets are replaced first so a nested secret is fully masked (no residue).
    msg = "leaked BearerABC and ABC"
    out = scrub(msg, ["ABC", "BearerABC"])
    assert "BearerABC" not in out
    assert "ABC" not in out
