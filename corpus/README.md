# `corpus/`: runnable reference agents and the live-capture harness

> **This is development / test infrastructure, not user-facing examples.** If you're looking for
> "how do I use Calibrated Agent Evaluation on my own agent," see the documentation site and the quickstart instead of this
> directory. `corpus/` builds the frozen test-fixture corpus that the library's own test suite runs
> against.

> **What this is.** Three reference agents that exercise the trace formats the library's trajectory
> normalizer must ingest, plus a one-time capture harness that runs them against **Bedrock
> (Claude Sonnet 4.6)** and freezes their genuinely-emitted traces into [`fixtures/raw/`](../fixtures/README.md).
>
> **It is built first, on purpose.** The normalizer is coded against the *real captured vendor
> output* in `fixtures/raw/`, not against hand-shaped fixtures: "assume nothing" about the wire
> shapes. These agents produce that ground truth.

## The golden rule for this directory

**The agents conform to their frameworks; Calibrated Agent Evaluation conforms to the agents.**

Each agent is the kind of application a real team would build with that framework (a real Strands
`Graph`, a real Anthropic Messages + stdio-MCP loop, a real Inspect `Task`). We capture *exactly what
they emit*. We do **not** reshape an agent, rewrite its output, or inject a post-hoc "defect" to make
a fixture convenient. When you add or change an agent, keep it idiomatic: if a real Strands user
wouldn't do it, don't do it here.

Determinism (so the corpus is reproducible and the tests run offline) comes only from things real apps and
tests already do:
- **stubbed tool backends** with fixed example data,
- **`temperature=0`**, and
- a **VCR-style cassette** that records the real model calls once and replays them offline: a test
  harness *around* the app, not a modification of it.

### How the worked-example failures are real bugs (not injected)

| Trace | Failure | The genuine bug |
|---|---|---|
| **ORD-1001** | over-refund on an out-of-window return | `check_refund_policy` ([`orderbot_backends.py`](agents/orderbot_backends.py)) has a real stub bug: its window check was never implemented, so it always returns `within_window: true`. A correctly-behaving agent then refunds on bad tool data. |
| **CLM-4471** | claim wrongly paid despite the vacancy exclusion | intake's `extract_claim_fields` is a **naive extractor** that never parses "in Florida for the last month" into a vacancy signal, so it returns `occupancy_status: null`. Downstream agents are correct-but-doomed. |
| **ORD-1002** | *(none; control)* | the SAME buggy `check_refund_policy`, but the order is genuinely in-window, so the refund is correct. The bug only *manifests* out-of-window. |

> **A finding worth knowing:** Sonnet 4.6 is well-aligned: a merely *permissive system prompt* did
> **not** make it over-refund (it respected an "out of window" tool result). The failure had to come
> from a broken tool. That is itself a realistic, common production failure mode (an agent
> faithfully trusting a buggy tool), and it's why ORD-1001's bug lives in the tool.

## Layout

```
corpus/
  agents/
    orderbot_backends.py        # OrderBot's real tool implementations (check_refund_policy is buggy)
    orderbot_otel.py            # #1 hand-coded agent loop, instrumented with the OTel SDK
    orderbot_mcp.py             # #2 Anthropic Messages loop + a real stdio-MCP server + a skill
    orderbot_mcp_server.py      #    the stdio MCP server (spawned as a subprocess by #2)
    claimscrew_strands.py       # #3 a real Strands Graph crew (supervisor -> intake -> ... -> payout)
    claimscrew_swarm.py         # #3b a real Strands Swarm (triage -> adjuster via handoff_to_agent)
    orderbot_agent_as_tool.py   # #3c a Strands Agent-as-Tools build (sub-agent under execute_tool)
    strands_cassette_model.py   #    a custom Strands Model provider that records/replays via cassette
    orderbot_inspect.py         # OrderBot + ClaimsCrew as Inspect AI Tasks (mints the .eval logs)
    inspect_cassette_provider.py#    re-export shim → calibrated_agent_evaluation.execute.cassette_provider (shipped in src/)
    common.py                   # the shared Converse tool-use loop + ExecutionRecord
    otel_emit.py                # render an ExecutionRecord into real OTel spans (events|attributes)
  generate/
    scenarios.py                # the worked-example scenarios, derived from fixtures/normalized/
    cassette.py                 # the record/replay cassette (reused by the execution plane)
    bedrock_client.py           # a cassette-backed Bedrock Converse client
    otel_capture.py             # capture OTel SDK spans -> canonical OTLP-JSON
    record.py                   # THE ENTRY POINT: capture everything -> fixtures/raw/
    quickstart_set.py           # capture and build the `calibrated-agent-evaluation quickstart --certify` labeled set
    quickstart_judge.py         # record the quickstart judges' live verdicts into their cassette
```

## The three reference agents

All three implement the **same observable behavior** where they overlap, in **different builds**.
This is the *target-agnosticism* property: one gold label must apply regardless of how the agent was
built.

1. **OrderBot #1: hand-coded + OTel** (`orderbot_otel.py`). A plain Python tool-use loop over
   Bedrock Converse, instrumented by hand with the OpenTelemetry SDK. Emitted in **two content
   placements** (span *events* and span *attributes*) because a generic OTLP source can put
   message content in either, and the normalizer must handle both.
2. **OrderBot #2: Anthropic SDK + stdio-MCP + skill** (`orderbot_mcp.py` + `orderbot_mcp_server.py`).
   The same behavior via a genuinely different stack: the **Anthropic Messages API** (a different
   wire shape than Converse) driving tools served by a **real stdio MCP server subprocess**, with a
   **skill** pulled from the server as an MCP prompt. Emits OTel via the same emitter as #1.
3. **ClaimsCrew: Strands `Graph`** (`claimscrew_strands.py`). A real Strands multi-agent `Graph`
   (`supervisor → intake → policy-checker → fraud-scorer → payout`) emitting **Strands-native OTel**
   (`gen_ai.system="strands-agents"`, content in span events). The CLM-4471 "manifestation ≠
   origination" worked example.

The Strands adapter claims to support **all** Strands multi-agent patterns, not just `Graph`, so two
more real Strands builds are captured (*assume nothing, verify each against real output*). All
Strands patterns share one span-name grammar (`invoke_agent`/`execute_event_loop_cycle`/`chat`/
`execute_tool`/`invoke_{instance}`), so the adapter is one pattern-agnostic code path; these prove it:

3b. **Claims triage: Strands `Swarm`** (`claimscrew_swarm.py`). A real `Swarm` (`triage → adjuster`)
    where coordination is the framework's registered **`handoff_to_agent` tool**, so the capture
    carries an `invoke_swarm` root and a real `execute_tool handoff_to_agent` span (the handoff is a
    *tool*, not a span primitive). Proves the normalizer lands it as a `ToolCall`, never a phantom
    agent span.

3c. **OrderBot: Strands Agent-as-Tools** (`orderbot_agent_as_tool.py`). A parent agent that invokes
    a sub-agent via `child.as_tool()`, so the sub-agent's `invoke_agent` span nests **under** the
    parent's `execute_tool {child}` span (alongside a plain leaf tool). Proves the discriminator: a
    tool wrapping an agent ⇒ child `Span`; a leaf tool ⇒ `ToolCall`.

Each OrderBot/ClaimsCrew build is also run under **Inspect AI** (`orderbot_inspect.py`) to mint
`.eval` logs, the third trace format and the execution-plane path.

## Running it

Everything goes through **`corpus/generate/record.py`**. It needs the `corpus` dependency group,
which `just install` syncs along with every extra. The group is never part of the package's
install, and `src/calibrated_agent_evaluation` imports none of it. Outside the dev environment:

```sh
uv sync --group corpus
```

### Replay the committed corpus offline (default: no Bedrock, no AWS, deterministic)

```sh
uv run python -m corpus.generate.record
```

This replays the committed cassettes in [`fixtures/raw/cassettes/`](../fixtures/raw/cassettes) and
re-freezes `fixtures/raw/`. A fresh clone runs this with **zero credentials**. Use it to
regenerate the corpus after changing an agent or the emitter.

### Re-capture against live Bedrock (developer action; re-records the cassettes)

```sh
AWS_PROFILE=<your-profile> CALIBRATED_AGENT_EVALUATION_CASSETTE_MODE=record uv run python -m corpus.generate.record
```

- Requires Bedrock access to Claude Sonnet 4.6 in `us-west-2` (override with `AWS_REGION` /
  `CALIBRATED_AGENT_EVALUATION_BEDROCK_MODEL_ID`, a bare model id with no `bedrock/` prefix). Credentials are read from the environment / the named profile and
  **never** written to a cassette.
- `--only otel|mcp|strands|swarm|as-tool|inspect` captures just one target.
- `CALIBRATED_AGENT_EVALUATION_CASSETTE_MODE`: `replay` (default; offline, a miss is a hard error), `record` (always
  call live + re-record), `auto` (replay on hit, record on miss).

## How the cassette works (and why it's the execution-plane seam)

`cassette.py` records each model request → response keyed by a stable content fingerprint, and
replays by fingerprint. The same record/replay layer is what the execution plane reuses to run
agents offline and deterministically in tests. Three integration points, all **composition, never
monkey-patching**:

- **OrderBot #1** calls the cassette-backed `BedrockConverseClient` directly.
- **OrderBot #2** wraps the Anthropic Messages call through the async cassette.
- **ClaimsCrew** uses `CassetteModel`, a custom **Strands `Model` provider**; **Inspect** uses
  `cassette/...`, a custom **Inspect `@modelapi` provider**. Both are the frameworks' own documented
  extension points.

> A model request is fingerprinted on its *semantic* content only. Volatile per-process fields (e.g.
> Inspect assigns each message a random `id`) are stripped, so replay is deterministic across runs.

## Adding a new reference agent

1. Build it as a **real** application with its framework (don't bend it to the library's schema).
2. Drive its model through the cassette via the framework's own provider/client extension point.
3. Back its tools with deterministic example data; if you need a failure, make it a **genuine bug**
   in the app (a buggy tool, a misaligned prompt), never a post-hoc output rewrite.
4. Add a `record_*` step in `record.py` that freezes its emitted trace into `fixtures/raw/<format>/`.
5. Re-capture once against Bedrock, then confirm `uv run python -m corpus.generate.record` (replay)
   reproduces it offline.

## Constraints this directory honors

- **Corpus deps are `corpus/`-only** and never enter the shipped runtime tree. `src/calibrated_agent_evaluation` imports
  none of them; they live in the `corpus` dependency group.
- **`fixtures/raw/` is real, emitted output**, produced by these agents, never hand-shaped.
- **Bedrock is the one external dependency**; capture is a developer step, the output is committed
  frozen and replays offline.
