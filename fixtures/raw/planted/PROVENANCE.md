# Provenance: planted-bad inputs for the ingest gate

Each file here is a real frozen capture with one structural fault introduced, so the ingest
gate's no-silent-loss refusals have a committed input to run against (the two planted-input
refusal rows of the acceptance checklist, and `tests/test_ingest_gate.py`). Regenerate with
`uv run python fixtures/raw/planted/generate_planted.py`.

| File | Source capture | The one edit | Expected refusal |
|---|---|---|---|
| `_bad_unlinked_tools.otlp.json` | `fixtures/raw/otel/ORD-1001.attrs.otlp.json` | every `execute_tool` span's `parentSpanId` points at `AAAAAAAAAAA=`, a span id the trace does not contain | tool calls lost: 3 tool-execution spans, 0 tool calls recovered, an adapter/parent-linking mismatch |
| `_bad_dup_span.otlp.json` | `fixtures/raw/strands/CLM-4471.otlp.json` | the `policy-checker` agent span takes the `intake` agent span's id, and its child cycle span points at that id | duplicate span ids, naming the agent and the distinct `name=` fix |

The files sit outside `fixtures/raw/otel/` and `fixtures/raw/strands/` on purpose: one acceptance step globs
the top-level OTel captures and another walks `fixtures/raw/otel` recursively, and both expect
only good inputs there.

## Names rewritten after recording

The names in these recordings that carried the project's earlier working name (resource names,
session ids, span attribute keys and scope names) were rewritten in place by
`scripts/rewrite_recordings.py`; see "Names rewritten after recording" in `fixtures/README.md`.
Nothing else in them changed.
