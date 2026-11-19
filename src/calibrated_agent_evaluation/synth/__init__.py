# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Input synthesis & generation engine: the Generate stage.

Turns a taxonomy frame into natural-language inputs and (optionally) routes them through
the real agent to produce frame-tagged, taxonomy-attributable Traces. Two-step generation
(tuple → NL) with dedup/diversity guards; inputs-only (never fabricates an output or a label).
The vendor SDK (``strands_evals``) is confined to :mod:`calibrated_agent_evaluation.synth.borrow`
and never imported at package-import time (the ``[synth]`` extra is opt-in).

No synth artifact is persisted through the dump/load registry: the engine emits in-memory
``SynthInput`` objects (consumed immediately) and ``Trace`` objects (the schema layer already
owns Trace codecs). So there is no new codec to register here.
"""

from __future__ import annotations

from calibrated_agent_evaluation.synth.errors import (
    DiversityFloorWarning,
    RepresentativenessWarning,
    RiskZoneWarning,
    SynthError,
)
from calibrated_agent_evaluation.synth.generate import generate, generate_and_run
from calibrated_agent_evaluation.synth.guards import (
    DEFAULT_DEDUP_THRESHOLD,
    dedup,
    enforce_diversity_floor,
)
from calibrated_agent_evaluation.synth.risk import DEFAULT_RISK_LEXICON, scan_risk_zones
from calibrated_agent_evaluation.synth.types import Embedder, Renderer, SynthInput, TaskBuilder

__all__ = [
    "DEFAULT_DEDUP_THRESHOLD",
    "DEFAULT_RISK_LEXICON",
    "DiversityFloorWarning",
    "Embedder",
    "Renderer",
    "RepresentativenessWarning",
    "RiskZoneWarning",
    "SynthError",
    "SynthInput",
    "TaskBuilder",
    "dedup",
    "enforce_diversity_floor",
    "generate",
    "generate_and_run",
    "scan_risk_zones",
]
