# Calibrated Agent Evaluation shared fixture corpus

> The **real system** the test suite and the flagship proof run against. Every pipeline stage is tested against the slice of this corpus it needs.

## What this is

A version-pinned corpus of **OrderBot** (single-agent) and **ClaimsCrew** (multi-agent crew) traces, the worked examples of the flagship end-to-end proof, plus their human **gold labels** and per-stage **oracles**. The corpus grew stage by stage into the substrate the flagship proof (`just flagship`) runs on, rather than being assembled only at the end.

## What this is NOT

- **Not invented data.** Raw traces are *genuinely emitted* by the runnable reference agents in `corpus/` . Random-shaped traces benefit nobody.
- **Not shipped.** `fixtures/` is committed to the repo but excluded from the `calibrated-agent-evaluation` wheel (which packages only `src/calibrated_agent_evaluation`). Frozen fixtures are read with **zero runtime dependencies**.

## The `normalized/` oracle is hand-transcribed from the real `raw/` captures, on purpose

The component that turns a raw trace into a normalized `Trace` is the **trajectory normalizer**. The `normalized/*.json` files were written before it existed, as **faithful hand-transcriptions of the real captured `raw/` traces** into the data plane's normalized schema: real content, hand-*mapped* from the actual captured bytes, never invented shape.

> **Re-authored while the normalizer was implemented (a recorded spec correction).** These oracles were first drafted with the data plane, from the documented worked examples *before* the real `raw/` corpus existed, and had drifted into an idealized shape (an invented `output.action="REFUND_ISSUED"` classification that appears **nowhere** in `raw/`; a CLM `payout` `issue_payment` tool call that never happened; a `policy-checker` with one lookup when the real run called two). They have since been re-transcribed field-for-field from `fixtures/raw/` so the normalizer has an independently authored target. The terminal `output` is now the run's **real free-text answer**, and classification (REFUND / PAID-DENIED verdicts) is left to a downstream **judge**.

The hand-authored `normalized/` files are the **gold the normalizer must reproduce**, authored independently of the code that has to match it. The normalizer *generates* a `Trace` from each `raw/` capture and the golden test (`tests/normalize/test_normalize_golden.py`) diffs it against these files on the **structural/semantic** fields, which makes it a contract test.

## Layout

```
fixtures/
  VERSION                 # corpus schema version; MUST equal calibrated_agent_evaluation.SCHEMA_VERSION ("1")
  README.md               # this file
  validate_corpus.py      # stdlib-only validator for normalized/, labels/ and expected/
  normalized/             # Trace oracle, hand-transcribed from the real raw/ captures (faithful, not idealized)
    ORD-1001.json         # OrderBot: out-of-window over-refund (the headline failure), real free-text answer
    ORD-1002.json         # OrderBot: in-window refund, clean PASS (the negative control)
    CLM-4471.json         # ClaimsCrew: manifestation != origination (intake drops "vacant")
    CLM-4473/4475/4477    # ClaimsCrew: culprit=policy-checker, outcome-pass/span-fail, disputed culprit
  labels/
    gold.json             # human gold Labels (outcome + critique + crew attribution)
  raw/                    # real emitted artifacts, produced by the corpus/ agents against Bedrock
    otel/                 # OrderBot #1 (events and attributes placements), #2 (Anthropic+MCP), a degraded run
    strands/              # ClaimsCrew Graph, claims-triage Swarm, OrderBot Agent-as-Tools
    inspect/              # real Inspect .eval logs
    cassettes/            # committed model-call recordings for offline re-capture
    hf/                   # small parquet/JSONL dataset slices for dataset ingestion
    planted/              # deliberately malformed OTLP files the ingest refusals are tested on
    quickstart/           # the OrderBot runs and judge cassette behind `calibrated-agent-evaluation quickstart --certify`
  calibration/            # a synthetic calibration set and the script that makes it
  judges/                 # judge definitions for the enforcement layer (one valid, one refused)
  judge-files/            # CLI judge files (`calibrated-agent-evaluation calibrate/bias/judge --judge FILE`; `dataset run --judge`)
  scenarios/              # hand-written simulated slices for `calibrated-agent-evaluation dataset run`
  bias/                   # bias probe files and per-model thresholds (see bias/README.md)
  expected/               # per-stage oracles, one directory per stage that transforms the corpus
```

## Validate the corpus (stdlib only, no install)

```sh
python3 fixtures/validate_corpus.py
```

This validator encodes the data plane's normalized-schema contract (`calibrated_agent_evaluation.schema`) and checks every `normalized/` trace and every gold label against it. **It is the schema pressure-test**: if a documented real trace cannot be represented losslessly, the validator fails, and that is a gap in the data-plane schema to fix *before* downstream stages build on the contract. It deliberately depends only on the Python standard library, so it runs before any of `calibrated-agent-evaluation` is implemented.

## The `raw/` corpus: real captured vendor output

`raw/` holds the **genuinely-emitted** traces the trajectory normalizer is coded and tested against. They
are produced by the runnable reference agents in [`corpus/`](../corpus/README.md), run against
**Bedrock (Claude Sonnet 4.6)**, and frozen here. Each file is the real wire format its tool produces:

- `otel/*.otlp.json`: **canonical OTLP-JSON** (`resourceSpans → scopeSpans → spans`, `KeyValue`/
  `AnyValue` attributes, content in span `events` or `attributes`). What a production OTLP/HTTP
  exporter emits, which differs from the OTel SDK's debug `to_json()` shape. Read with stdlib `json` (zero deps).
- `strands/*.otlp.json`: Strands-native OTLP-JSON sharing one span-name grammar
  (`invoke_agent {name}` / `execute_event_loop_cycle` / `chat` / `execute_tool {name}` + a
  multiagent root `invoke_{instance}`), `gen_ai.system="strands-agents"`, content in `gen_ai.*`
  events (legacy semconv). Names are **verified against these captured files**, replacing the
  span names first derived from the Strands docs. Three real multi-agent patterns are
  frozen so the normalizer's Strands path is proven on each, not assumed:
  - `CLM-4471.otlp.json`: a **`Graph`**: `invoke_graph` root → `invoke_agent {name}` nodes; handoffs
    are graph edges (no handoff span).
  - `CLM-7720-swarm.otlp.json`: a **`Swarm`**: `invoke_swarm` root + a real
    `execute_tool handoff_to_agent` span (the handoff is the framework's *tool*, **not** a span
    primitive) → the normalizer lands it as a `ToolCall`, never a phantom agent span.
  - `ORD-3310-as-tool.otlp.json`: **Agent-as-Tools**: a sub-agent's `invoke_agent` span nested
    **under** the parent's `execute_tool {child}` span (alongside a plain leaf `execute_tool`) → the
    sub-agent discriminator (sub-agent ⇒ child `Span`; leaf tool ⇒ `ToolCall`).
- `inspect/*.eval`: real Inspect AI eval logs (read with `inspect_ai.log.read_eval_log_samples`;
  the span tree is built via `inspect_ai.event.event_tree`). Needs the `[inspect]` extra to read.
- `cassettes/*.cassette.json`: committed recordings of the live model calls. They let
  `python -m corpus.generate.record` re-freeze the whole corpus **offline and deterministically**
  (no Bedrock), and are the record/replay layer the executor reuses.

### Re-capturing `raw/`

Re-capture is a **developer action**, not an install step. From the repo root:

```sh
uv sync --group corpus                                   # the corpus deps (`just install` includes them)
uv run python -m corpus.generate.record                # replay committed cassettes (offline)
AWS_PROFILE=<your-profile> CALIBRATED_AGENT_EVALUATION_CASSETTE_MODE=record \
  uv run python -m corpus.generate.record              # re-record against live Bedrock
```

See [`corpus/README.md`](../corpus/README.md) for the agents, the capture harness, and the rule
that these traces are *real emitted output*, never hand-shaped.

> **Note on `raw/` ↔ `normalized/`.** The `normalized/` oracle is now a **faithful transcription of
> the real `raw/` captures** (re-authored while the normalizer was implemented; see above), not an idealized
> shape. The real agents emit **free-text final answers**, so the terminal `output` in the oracle is
> `{"answer": "<real prose>"}`; there is **no invented `action`/classification field** (a REFUND or
> PAID/DENIED verdict is a downstream *judge*'s job, not the normalizer's). Per-source fidelity
> differs and the oracle records it: the Inspect `.eval` carries only the **bare message
> string** as its input (`{"message": "..."}`), whereas the OTLP/Strands captures carry a **richer
> structured input** (narrative + claim_id + amount). Because the model's prose varies run-to-run,
> the golden test compares the **structural/semantic** fields (tool-call name sequence, tool
> `args`/`result` content, notably `check_refund_policy.within_window=true` and
> `occupancy_status=null`, the span tree, ownership), **not** the exact natural-language text or any
> synthesized `action`. The normalizer regenerates a `Trace` from each `raw/` capture and
> the golden test diffs it against these files on those structural/semantic fields (modulo timestamps + free
> text).

## Names rewritten after recording

The recordings were captured while the project had an earlier working name, which appeared in AWS
resource names (runtimes, evaluators, online configs, Gateways, Policy engines and policies),
session ids, span attribute keys (`calibrated_agent_evaluation.agent.role` and the like),
instrumentation scope names, metadata keys and cassette schema strings. Re-recording would cost
live calls and could change the models' replies, so those names were rewritten in place instead,
by `scripts/rewrite_recordings.py`. The script rewrites every spelling of the old word and nothing
else, inside `.gz` and `.xz` files too, and recomputes each cassette entry's fingerprint with the
library's own `fingerprint` function wherever a rewrite changed the request it keys.

The same pass made one other literal replacement: the test account's agent name in
`raw/agentcore/evaluation_results.json` is now `SupportAgent`. Every other recorded value is as
captured.

## Current slice (what is built today)

The corpus is the flagship proof's substrate. It holds the full end-to-end set:

- **`normalized/`**: the Trace oracle: OrderBot `ORD-1001` (headline over-refund FAIL) + `ORD-1002` (PASS control); ClaimsCrew `CLM-4471` (culprit=intake) / `4473` (culprit=policy-checker) / `4475` (outcome-pass/span-fail) / `4477` (disputed culprit).
- **`labels/gold.json`**: the human-ratified gold (every label `kind="human"`, because humans own the gold) + the crew attribution fields (culprit/manifestation/handoff).
- **`raw/`**: the frozen real captures (OTel / Strands / Inspect `.eval` / cassettes) the reference agents in `corpus/` emit.
- **`calibration/`, `judges/`, `judge-files/`, `bias/`**: the stage fixtures for calibration, judge enforcement, the CLI judge files and the bias check.
- **`expected/`**: the committed per-stage oracles the flagship and the tests reproduce: `enforce/`, `run/`, `improve/`, `bias/`, `probe/`, and, **authored when the flagship proof completed the corpus**, `cluster/{orderbot,claimscrew}.json` (the human-ratified failure-mode taxonomy) and `attribution/claimscrew.json` (the counterfactual walk-back transform: culprit ≠ manifestation).

The stdlib `validate_corpus.py` validates every `normalized/` Trace and `gold.json` Label against the data-plane schema, refuses a non-human gold label lacking a `ratified_by` human-ratification record (humans own the gold), and checks every `expected/*` oracle is a well-formed JSON object. Run `python3 fixtures/validate_corpus.py`. `just flagship` exercises the whole corpus end to end against the local store, and against Langfuse when `--local-only` is dropped.
