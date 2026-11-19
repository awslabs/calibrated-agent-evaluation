# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Risk-zone detection for synthetic inputs.

Risk-zone handling is tiered: synthetic *inputs* are warn-and-allow everywhere; synthetic
ground-truth *labels* are warn-and-block in high-stakes domains. This engine only ever produces
inputs (it has no label-writing path at all) so the "block labels" half is satisfied
**structurally** (there is nothing to block). This module implements the "warn on inputs" half:
a configurable keyword lexicon that raises :class:`RiskZoneWarning` when a taxonomy value names a
risk-zone domain. The lexicon is a plain ``frozenset`` the caller may override or disable.
"""

from __future__ import annotations

import warnings

from calibrated_agent_evaluation.sampling.types import FrameResult
from calibrated_agent_evaluation.synth.errors import RiskZoneWarning

# Built-in high-stakes / risk-zone domains. Matched case-insensitively as substrings
# of a taxonomy value token, so "medical_advice" or "Legal" both trip. Callers override via the
# `lexicon` argument; this default is deliberately broad and default-safe.
DEFAULT_RISK_LEXICON: frozenset[str] = frozenset(
    {
        "legal",
        "medical",
        "health",
        "financial",
        "finance",
        "high-stakes",
        "high_stakes",
        "low-resource",
        "low_resource",
        "underrepresented",
        "safety",
        "clinical",
    }
)


def scan_risk_zones(
    frame: FrameResult,
    *,
    lexicon: frozenset[str] = DEFAULT_RISK_LEXICON,
) -> dict[str, list[dict[str, str]]]:
    """Scan a frame's taxonomy cells for risk-zone domains.

    Returns a mapping ``matched-domain -> list of cells that named it``. If any match is found,
    emits a single :class:`RiskZoneWarning` naming the matched domain(s). An empty ``lexicon``
    disables scanning entirely (the caller opted out). Never blocks: inputs are
    warn-and-allow; the warning is the whole behavior.
    """
    matches: dict[str, list[dict[str, str]]] = {}
    if not lexicon:
        return matches

    for row in frame.rows:
        for value in row.cell.values():
            token = value.casefold()
            for domain in lexicon:
                if domain.casefold() in token:
                    matches.setdefault(domain, [])
                    if row.cell not in matches[domain]:
                        matches[domain].append(row.cell)

    if matches:
        warnings.warn(
            f"synthetic inputs touch risk-zone domain(s) "
            f"{sorted(matches)}; synthetic inputs are allowed, but validate their distribution "
            f"against production before trusting any synthetic-derived rate",
            RiskZoneWarning,
            stacklevel=2,
        )
    return matches
