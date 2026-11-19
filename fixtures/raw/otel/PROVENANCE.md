# Provenance: per-convention OTLP-JSON fixtures for dataset ingestion

> Every convention fixture under `openinference/`, `openllmetry/`, `messages/` is built from **real
> vendor output** (repo rule: fixture traces are genuine, never hand-shaped). This file records the
> exact source, license, and revision so the authenticity is auditable. Regenerate with
> `uv run python fixtures/raw/otel/generate_convention_fixtures.py`.

The generator **only wraps** these real attribute sets in the canonical OTLP-JSON envelope
(`resourceSpans → scopeSpans → spans`, `KeyValue`/`AnyValue`) that an OTLP/HTTP exporter emits — the
envelope is transport, not content. No attribute key or value was invented.

## `messages/swebench_tool_call.otlp.json`: `gen_ai/messages` convention

- **Source:** the committed real dataset slice `fixtures/raw/hf/swebench-slice.parquet` (from
  `Exgentic/agent-llm-traces` v1 at revision `70036b93a04e61b0ea2706a68b962f4f26774587`). The first
  `chat` span of session `e3bfdef4aca3_c0427304` that carries a `tool_call` in its
  `gen_ai.output.messages`; the messages and request model are extracted verbatim.
  License: CDLA-Permissive-2.0 (see `fixtures/raw/hf/README.md`).
- **Derived attribute:** the v1 rows have no `gen_ai.provider.name`, so `gen_ai.system` is taken
  from the provider prefix of the LiteLLM model string (`azure/DeepSeek-V3.2` gives `azure`). The
  generator refuses a span whose `service.name` is not `litellm` or whose model has no prefix,
  instead of falling back to a default. `service.name` (`litellm`) is the span's own resource
  attribute.

## `openinference/weather_tool_call.otlp.json` — OpenInference (Arize/Phoenix) convention

- **Source:** `Arize-ai/openinference`,
  `js/packages/openinference-genai/test/__fixtures__/openinference_client_span_tool_calls.json`.
- **License:** Apache-2.0.
- **Revision:** commit `930e41adaab8318e093ed24a68cbeea098e41b13`.
- **What:** the published captured-span attribute set, copied **verbatim** (`openinference.span.kind`,
  `llm.input_messages.{i}.message.*`, `llm.*_messages.*.tool_calls.{j}.tool_call.function.*`,
  `input.value`/`output.value`). The one edit: `input.value`/`output.value` blobs were truncated to a
  short valid-JSON stub (they are large opaque request/response echoes irrelevant to attribute
  parsing) — the reconstruction-relevant keys are untouched.

## `openllmetry/weather_tool_call.otlp.json` — OpenLLMetry / Traceloop legacy convention

- **Grammar source:** `traceloop/openllmetry` tag `v0.33.12`,
  `packages/opentelemetry-instrumentation-openai/tests/traces/test_functions.py` (+ the
  `opentelemetry-semantic-conventions-ai` `SpanAttributes` constants) — the exact dotted-index keys
  `gen_ai.prompt.{i}.{role,content}`, `gen_ai.completion.{i}.{role,finish_reason,tool_calls.{j}.{id,name,arguments}}`.
- **Value source:** the real recorded VCR cassette in the same repo/tag,
  `tests/traces/cassettes/test_functions/test_open_ai_function_calls_tools.yaml` — a genuine recorded
  OpenAI API interaction: user `"What's the weather like in Boston?"`, response `gpt-4-0613`,
  `finish_reason=tool_calls`, tool_call `call_b4lU3wKOqwO7Xb2Ksok2XWyB`
  `get_current_weather({"location":"Boston"})`.
- **License:** Apache-2.0.

## Other fixtures in this tree

- `degraded_v2.otlp.json` — a `gen_ai/messages` span whose `gen_ai.input/output.messages` are
  **empty lists** (a detectable convention, but no recoverable per-step content) → the messages
  adapter degrades to `probed` with a reason. Minimal by construction (it tests the
  *absence* of recoverable content), not a fabricated trajectory. Distinct from `unknown/`, which
  carries **no** convention signal at all and so raises an unrecognized-convention error rather
  than degrading.
- `unknown/unrecognized.otlp.json` — an OTLP span whose attributes match **no** known convention
  (detection raises). Also minimal by construction.
- `ORD-*.otlp.json`: the trajectory normalizer's pre-existing real captures (the `gen_ai/legacy` + `gen_ai/latest`
  regression corpus), unchanged.

## Names rewritten after recording

The names in these recordings that carried the project's earlier working name (resource names,
session ids, span attribute keys and scope names) were rewritten in place by
`scripts/rewrite_recordings.py`; see "Names rewritten after recording" in `fixtures/README.md`.
Nothing else in them changed.
