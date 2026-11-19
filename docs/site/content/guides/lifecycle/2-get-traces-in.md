# 2. Get traces in

With the `run-error-analysis` skill, ask:

> Ingest the traces in traces/ for my_agent.

The skill picks the route that matches what you have and runs it. The commands, by route:

```bash
calibrated-agent-evaluation ingest traces/                              # OTLP, Inspect .eval, JSONL or parquet files
calibrated-agent-evaluation ingest --from-langfuse "my-agent-prod"      # a Langfuse project
calibrated-agent-evaluation capture --agent my_agent:run_once           # one in-process run of an instrumented agent
calibrated-agent-evaluation generate --run --frame frame.json \
  --executor strands --agent my_agent:build        # route the frame's inputs through your agent
```

[Choose your path](../../integrate/index.md) decides between them in one to three questions, and
each route has its own page with the extras and flags it needs. Start every new source with
`--dry-run`, which reports what would be stored and what would be refused without writing anything.

What you'll see (`calibrated-agent-evaluation ingest --dry-run` on a folder of OpenTelemetry
exports):

<div class="wrap-output" markdown>

```text
--8<-- "fixtures/expected/docs/calibrated-agent-evaluation-ingest-dry-run.txt"
```

</div>

Ingest refuses a file it cannot convert without losing content, and says which and why. Fix the
export or pass `--convention`; do not edit the trace to make it fit.

`generate --run --multi-turn` holds a conversation per frame row instead of one turn, with a
simulated person writing the user turns ([Route your agent through](../../integrate/route-your-agent.md#hold-a-conversation-instead-of-one-shot-multi-turn)).

## Two kinds of traces, kept apart

Every trace carries a **frame**: why it was collected. The rest of the guide depends on it.

| Frame | What goes in it | Used for |
| --- | --- | --- |
| `discovery` | traces you picked to find failures: the generated frame, escalations, complaints, edge cases | error analysis (phases 3 and 4) and the judge's calibration labels (phase 5) |
| `production-representative` | a random draw from real traffic, nothing hand-picked | the failure rate (phases 6 and 8) |

A hunted set over-represents failures on purpose, so a rate measured on it is wrong by construction.
`calibrated-agent-evaluation run` leaves verdicts on discovery traces out of the rate and says so.
Both frames can sit in one store.

Ingested traces whose spans carry no frame are stamped `discovery` unless you pass `--frame`. The
random draw for the rate is `calibrated-agent-evaluation ingest --sample N --seed S`, which ingests
a seeded random sample of the files or Langfuse project and tags it `production-representative`
([phase 8](8-production.md#the-scheduled-sample) shows it). If you control the agent's OpenTelemetry
setup, set the frame on the root span with
`calibrated_agent_evaluation.normalize.guidance.tag_span`, together with a readable trace id and the
taxonomy cell ([Name traces at instrumentation
time](../../integrate/ingest-existing-traces.md#name-traces-at-instrumentation-time)). Traces from
`calibrated-agent-evaluation generate --run` carry the frame and cell of the row that produced them.

!!! warning "Mark each labeling round with a cell"
    Phase 5 checks each judge version on labels it has never seen. Give each batch of hunted traces
    a taxonomy attribute that names its round, for example
    `calibrated_agent_evaluation.taxonomy.label_round=1`, so later commands can select one round
    with `--cell label_round=1`. The bundled kit does this.

## Owned and probed

A trace from an agent you instrument shows every step and tool call. A trace from an endpoint you
can only call (`calibrated-agent-evaluation probe --endpoint URL --authorized`) shows the outcome
alone, so you can gate it but cannot say which step failed. `calibrated-agent-evaluation ingest`
reports a trace that arrives with only its input and output, and `--require-owned` refuses it, for a
corpus you plan to analyze step by step. [What you get depends on what the library can
see](../../integrate/index.md#what-you-get-depends-on-what-the-library-can-see) has the full table.

## How many traces

For error analysis, start with enough to cover every row of the frame at least once, plus the hunted
traces you already have. Phase 4 tells you whether to collect more. For the gate in phase 6,
`calibrated-agent-evaluation plan --gate` gives the traces each mode needs. For the corrected rate
in phase 8, the judge needs at least 150 production-representative traces to score.

Next: [3. Look and label](3-look-and-label.md).
