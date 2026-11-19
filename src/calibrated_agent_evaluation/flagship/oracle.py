# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Oracle comparison for the flagship proof.

Each pipeline stage's output is checked against a committed ``fixtures/expected/*`` oracle -- a
known-good value the shipped stage must reproduce. The comparison is *semantic* (like the golden
comparison used elsewhere): a caller passes the subset of fields the oracle pins, and this module
compares them to the loaded oracle document, returning an
:class:`~calibrated_agent_evaluation.flagship.report.OracleCheck`.

A missing or corrupt oracle for a stage that claims one is a **proof-integrity failure**, not a
silent skip -- it raises :class:`~calibrated_agent_evaluation.flagship.errors.FlagshipError`. An
oracle-less stage (one proven structurally, e.g. ``define``/``generate``/``label``) passes
``oracle_path=None`` and the caller records the structural check it made in the ``OracleCheck.note``
upstream.

The harness never *computes* a value here; it only compares what a shipped stage produced against
the frozen oracle.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from calibrated_agent_evaluation.flagship.errors import FlagshipError
from calibrated_agent_evaluation.flagship.report import OracleCheck, StageName


def load_oracle(oracle_path: str | Path) -> dict[str, Any]:
    """Load a committed oracle document. Raises :class:`FlagshipError` if absent/corrupt.

    A missing oracle for a stage that claims one is a proof-integrity failure -- the proof cannot be
    trusted against an oracle that is not there -- so this refuses loudly rather than returning an
    empty dict that would let a stage "match" nothing.
    """
    path = Path(oracle_path)
    if not path.exists():
        raise FlagshipError(
            f"flagship oracle missing at {path}: a stage claiming an oracle cannot be proven "
            "against a file that is not there"
        )
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise FlagshipError(f"flagship oracle at {path} is unreadable/corrupt: {exc}") from exc


def _diverging_key(expected: Any, actual: Any, prefix: str = "") -> str | None:
    """Return a human path to the first field where ``actual`` diverges from ``expected``.

    ``expected`` drives the walk: only the keys the oracle pins are compared (a stage may carry
    extra fields the oracle does not constrain). Returns ``None`` when every pinned field matches.
    """
    if isinstance(expected, dict):
        if not isinstance(actual, dict):
            return prefix or "<root>"
        for key, exp_val in expected.items():
            if key not in actual:
                return f"{prefix}{key} (missing)"
            sub = _diverging_key(exp_val, actual[key], prefix=f"{prefix}{key}.")
            if sub is not None:
                return sub
        return None
    if isinstance(expected, list):
        if not isinstance(actual, list) or len(expected) != len(actual):
            return f"{prefix.rstrip('.')} (list mismatch)"
        for i, exp_val in enumerate(expected):
            sub = _diverging_key(exp_val, actual[i], prefix=f"{prefix}{i}.")
            if sub is not None:
                return sub
        return None
    if expected != actual:
        return f"{prefix.rstrip('.')}: expected {expected!r}, got {actual!r}"
    return None


def compare_oracle(
    stage: StageName, produced: dict[str, Any], oracle_path: str | Path
) -> OracleCheck:
    """Compare a stage's ``produced`` fields against the committed oracle.

    Only the fields the oracle pins are compared (semantic comparison, not byte-exact prose). A
    missing/corrupt oracle raises :class:`FlagshipError`. Returns an :class:`OracleCheck` whose
    ``matched`` is True iff every pinned field is reproduced, else ``matched=False`` naming the
    first divergence. An oracle that pins no field raises :class:`FlagshipError`: it would match
    any output, so it proves nothing.
    """
    oracle = load_oracle(oracle_path)
    if not isinstance(oracle, dict):
        raise FlagshipError(
            f"flagship oracle at {oracle_path} must be a JSON object of pinned fields, "
            f"not {type(oracle).__name__}"
        )
    # An oracle either wraps its pinned fields under "expected", or pins them at the top level
    # alongside a human-readable "note" (the shared oracle shape). Drop "note" so a prose
    # annotation is never compared as a pinned field.
    expected = oracle.get("expected", {k: v for k, v in oracle.items() if k != "note"})
    if not isinstance(expected, dict) or not expected:
        raise FlagshipError(
            f"flagship oracle at {oracle_path} pins no fields, so any output would match it; "
            "pin the fields the stage must reproduce (at the top level or under 'expected')"
        )
    divergence = _diverging_key(expected, produced)
    return OracleCheck(
        stage=stage,
        oracle_path=str(oracle_path),
        matched=divergence is None,
        divergence=divergence,
    )
