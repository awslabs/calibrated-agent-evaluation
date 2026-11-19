# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Hybrid convention detection.

Given a session's raw OTLP-JSON spans, pick the convention from **distinctive attribute-key
signals**: ``openinference.span.kind`` / ``llm.input_messages.*`` → OpenInference; dotted-index
``gen_ai.prompt.{i}.*`` / ``gen_ai.completion.{i}.*`` → legacy OpenLLMetry; the
``gen_ai.input.messages`` / ``gen_ai.output.messages`` pair → the ``gen_ai/messages`` convention.

**Precedence:** when agent-boundary operations (``invoke_agent``/``execute_tool``) are present
alongside an attribute-only grammar, ``gen_ai/latest`` wins; boundary spans are strictly more
information than attributes alone, since they carry the agent tree the attribute-only conventions
have to collapse into a single node, and the attribute content is recovered regardless by
:mod:`~calibrated_agent_evaluation.normalize.content_placements`.

When **no** signal is present, or more than one attribute-only convention family matches with no
boundary ops to break the tie, it raises a typed
:class:`~calibrated_agent_evaluation.normalize.errors.NormalizationError` naming the candidates it
considered. It never silently guesses one (the no-silent-failure rule). A caller that already knows
its convention bypasses all of this by declaring ``semconv`` on the source.
"""

from __future__ import annotations

from typing import Any

from calibrated_agent_evaluation.normalize import otlp_json
from calibrated_agent_evaluation.normalize.errors import NormalizationError

# The distinctive per-convention signals. A convention is a candidate if ANY of its signal keys
# appears (exact key, or a dotted-prefix for the indexed grammars) on any span attribute.
_OPENINFERENCE_EXACT = frozenset({"openinference.span.kind"})
_OPENINFERENCE_PREFIX = ("llm.input_messages.", "llm.output_messages.")
_MESSAGES_EXACT = frozenset({"gen_ai.input.messages", "gen_ai.output.messages"})
_OPENLLMETRY_PREFIX = ("gen_ai.prompt.", "gen_ai.completion.")
# Built-in grammar signals: the invoke_agent/execute_tool operation names (gen_ai/latest) and the
# legacy gen_ai.* message events (gen_ai/legacy). Auto-detect returns these so a single OTLP file in
# the built-in grammar normalizes with no declared semconv too (the invoke_agent front-door).
_F003_OPS = frozenset({"invoke_agent", "execute_tool"})
_LEGACY_EVENT_NAMES = frozenset(
    {"gen_ai.user.message", "gen_ai.assistant.message", "gen_ai.tool.message", "gen_ai.choice"}
)


def detect(spans: list[dict[str, Any]]) -> str:
    """Return the convention kind for ``spans``, or raise naming the candidates.

    Recognizes the three convention-layer conventions AND the two built-in grammars
    (``gen_ai/latest`` for the ``invoke_agent``/``execute_tool`` span grammar, ``gen_ai/legacy``
    for the legacy ``gen_ai.*`` message-event grammar), so ``semconv="auto"`` is the single
    front-door for any OTLP source.
    """
    keys: set[str] = set()
    ops: set[str] = set()
    event_names: set[str] = set()
    for span in spans:
        attrs = otlp_json.attributes(span)
        keys.update(attrs.keys())
        op = attrs.get("gen_ai.operation.name")
        if isinstance(op, str):
            ops.add(op)
        events = span.get("events")
        # A dataset row may carry `events` as a JSON string (e.g. "[]") rather than a list; only
        # walk genuine list-shaped span events, so a string never gets iterated char-by-char.
        if isinstance(events, list):
            for ev in events:
                name = ev.get("name") if isinstance(ev, dict) else None
                if isinstance(name, str):
                    event_names.add(name)

    candidates: set[str] = set()
    if keys & _OPENINFERENCE_EXACT or _any_prefixed(keys, _OPENINFERENCE_PREFIX):
        candidates.add("openinference")
    if keys & _MESSAGES_EXACT:
        candidates.add("gen_ai/messages")
    if _any_prefixed(keys, _OPENLLMETRY_PREFIX):
        candidates.add("openllmetry")
    if ops & _F003_OPS:
        candidates.add("gen_ai/latest")
    elif event_names & _LEGACY_EVENT_NAMES:
        # Only claim legacy when there is no stronger structured signal (the elif on ops keeps a
        # latest invoke_agent run from also matching legacy on its message events).
        candidates.add("gen_ai/legacy")

    if len(candidates) == 1:
        return next(iter(candidates))
    if len(candidates) > 1 and "gen_ai/latest" in candidates:
        # Boundary ops beat any attribute-only grammar. Stated over the general
        # distinction, never over a list of vendors: `invoke_agent`/`execute_tool` spans are
        # strictly MORE information than attributes alone, because they carry the agent tree that
        # every attribute-only convention has to collapse into one node. Choosing the grammar with
        # the tree therefore loses nothing, and the content the attribute-only grammar would have
        # read is recovered anyway by `content_placements`.
        #
        # This shape is the commonest real one, not a corner: a Traceloop-instrumented crew emits
        # `invoke_agent` + `gen_ai.agent.name` AND `gen_ai.input.messages`, so before this rule it
        # raised "ambiguous" and `calibrated-agent-evaluation ingest` skipped the file. The rejected
        # alternative; keep refusing and make the caller declare `semconv=`; is worse, because a
        # wrong declaration silently empties the content instead of failing.
        return "gen_ai/latest"
    if not candidates:
        raise NormalizationError(
            "could not detect an OTel GenAI convention: no gen_ai.input/output.messages, "
            "openinference.span.kind/llm.*_messages, gen_ai.prompt/completion.*, "
            "invoke_agent/execute_tool, or legacy gen_ai.* message-event signal found. "
            "Declare `semconv=` on the source (`calibrated-agent-evaluation ingest --convention "
            "<name>`) to force one, "
            "or "
            "register a custom convention."
        )
    raise NormalizationError(
        f"ambiguous OTel GenAI convention: signals for multiple conventions present "
        f"({sorted(candidates)}). Declare `semconv=` on the source (`calibrated-agent-evaluation "
        "ingest --convention "
        f"<name>`) to disambiguate."
    )


def _any_prefixed(keys: set[str], prefixes: tuple[str, ...]) -> bool:
    return any(k.startswith(p) for k in keys for p in prefixes)
