# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Scrub resolved credentials from a message before it is surfaced.

A vendor/transport library can echo an input back in its exception message -- e.g. ``websockets``
rejecting an invalid ``Sec-WebSocket-Protocol`` value includes that value (which is the resolved
token) in its ``ValueError``. The ``repr`` redaction on :class:`ConnectionConfig` did not cover
that path. ``scrub`` removes the *known resolved secret values* (which the adapter holds) from a
string before it is wrapped into a :class:`~calibrated_agent_evaluation.probe.errors.ProbeError`, so
a wrapped exception never leaks a secret -- while retaining the rest of the message (the HTTP status
/ reason that makes a failure debuggable).
"""

from __future__ import annotations

from collections.abc import Iterable

_REDACTED = "[redacted]"


def scrub(message: str, secrets: Iterable[str]) -> str:
    """Return ``message`` with every non-empty value in ``secrets`` replaced by ``[redacted]``.

    Precise by construction: only the *known resolved values* are removed (no regex heuristics, so
    no false positives and nothing over-redacted). Called with no secrets, the message is returned
    unchanged. Longer secrets are replaced first so a secret that contains another is fully masked.
    """
    scrubbed = message
    for secret in sorted((s for s in secrets if s), key=len, reverse=True):
        scrubbed = scrubbed.replace(secret, _REDACTED)
    return scrubbed
