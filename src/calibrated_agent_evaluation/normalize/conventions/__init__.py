# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""OTel GenAI trace-convention adapters.

Each convention adapter turns a list of raw OTLP-JSON spans (one session) into the
vendor-free :class:`~calibrated_agent_evaluation.normalize.tree.Intermediate`, so the shared tree
builder + ownership gate + schema validation produce the normalized ``Trace``. The generic OTel
adapter already ships the ``gen_ai/legacy`` and ``gen_ai/latest``
(``invoke_agent``/``execute_tool``) grammars inside ``otel_adapter``; this package adds the three
widely-emitted conventions the wild actually uses; ``gen_ai/messages`` (Claude Code / current
OpenLLMetry), legacy ``openllmetry`` (Traceloop dotted-index), and ``openinference``
(Arize/Phoenix); plus a hybrid :mod:`detect` front-door and a :mod:`registry` so an unshipped
convention is addable without editing core.

The convention layer is **shared**: both a single OTLP file (``OTelSpans``) and a dataset
row (``SessionSpans``) route through the same adapters, so a convention is reconstructed by
exactly one adapter regardless of how the spans arrived. This package is vendor-free;
it reads plain OTLP-JSON attribute keys via ``otlp_json``; no ``openllmetry``/``openinference``
runtime dependency exists.
"""

from __future__ import annotations

from calibrated_agent_evaluation.normalize.conventions.registry import (
    ConventionAdapter,
    get_adapter,
    register_convention,
    registered_kinds,
)
from calibrated_agent_evaluation.normalize.tree import AgentNode, Intermediate, ToolInvocation

__all__ = [
    "AgentNode",
    "ConventionAdapter",
    "Intermediate",
    "ToolInvocation",
    "get_adapter",
    "register_convention",
    "registered_kinds",
]
