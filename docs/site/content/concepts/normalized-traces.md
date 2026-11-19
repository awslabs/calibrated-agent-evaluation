# Normalized traces

Every source is normalized to one `Trace` shape, so the later stages never branch on how the agent
was built, and a single agent and a multi-agent crew are evaluated the same way.

## Sources

However an agent is built, its execution reduces to the same normalized shape:

- a hand-coded Python tool-use loop,
- an Anthropic Messages + stdio-MCP composition with a skill,
- a Strands `Graph` / `Swarm` / Agent-as-Tools crew,
- an Inspect AI task.

Calibrated Agent Evaluation's normalizer ingests each source (convention-aware OTel, Inspect `.eval`
logs, framework-native traces) and produces a `Trace` of `Span`s with a consistent vocabulary: agent
invocations, tool calls, handoffs, and content. Downstream stages (labeling, calibration, judging,
gating) see only the normalized `Trace` and never need to know how the agent was built.

## One label across builds

Because the representation is shared, one gold label applies regardless of build. The bundled
corpus shows this: the same OrderBot behavior is captured as a hand-coded OTel loop,
an Anthropic+MCP loop, and an Inspect task, and they normalize to equivalent traces with the same
outcome and failure-mode label. You can change how your agent is implemented without invalidating
your evaluation.

## Crews

A crew is a `Trace` with a deeper span tree: one span per agent step, linked by `parent`, where
each span's `input` is the handoff it received and its `output` the handoff it emitted.
Attribution (culprit span vs. manifestation span) is an operation over the normalized tree, and the same judges and gates work for
a single agent and a five-agent crew alike. See
[Evaluate a multi-agent crew](../guides/evaluate-a-crew.md).

## Outcome-only traces

Not every source gives full trajectory detail. When only the final outcome is observable (a
black-box endpoint), the normalizer degrades to an outcome-only trace rather than fabricating
structure, and the tooling states that limit of inference.

## Export

Every artifact (traces, labels, scores, records) round-trips to plain CSV + JSON via
`calibrated-agent-evaluation dump` / `calibrated-agent-evaluation load`, so it is not tied to a
proprietary format or a single backend.

`dump` writes `dump-manifest.json` after every other file. It lists each data file the dump wrote
with its record count and sha256. `load` reads nothing from a directory that does not match its
manifest: a dump that was interrupted (no manifest), a file changed after the dump or left by
another dump (a digest that differs), or a `.csv` or `.json` file the manifest does not list.
Dumping into a directory that holds an earlier dump first removes that dump's manifest and the
files it lists. Hidden files and other files, such as a README, are left alone and ignored.

## The contract

The schema is five frozen Pydantic models in `calibrated_agent_evaluation.schema`: `Trace`, `Span`
and `Label`, plus `ToolCall` and `TokenUsage` inside a span. Every backend stores them, and
`calibrated-agent-evaluation dump` writes them, through one mapping
(`calibrated_agent_evaluation.schema.serde`), so the JSON below is what a stored trace is.

This is the trace `calibrated-agent-evaluation capture` stored in [Capture and
ingest](../integrate/capture-and-ingest.md), serialized with
`calibrated_agent_evaluation.schema.serde.to_dict`:

```json
{
  "trace_id": "ORD-A1001",
  "frame": "production-representative",
  "ownership": "owned",
  "input": {
    "request": "Refund order A-1001, it arrived broken."
  },
  "output": {
    "decision": "refund_approved",
    "amount": 59.0
  },
  "taxonomy_cell": {},
  "spans": [
    {
      "span_id": "ORD-A1001/orderbot",
      "trace_id": "ORD-A1001",
      "agent": "orderbot",
      "role": "specialist",
      "parent": null,
      "seq": 0,
      "status": "completed",
      "input": {
        "request": "Refund order A-1001, it arrived broken."
      },
      "tool_calls": [
        {
          "name": "lookup_order",
          "args": {
            "order_id": "A-1001"
          },
          "result": {
            "order_id": "A-1001",
            "status": "delivered",
            "days_since_delivery": 41
          }
        }
      ],
      "output": {
        "decision": "refund_approved",
        "amount": 59.0
      },
      "started_at": "2026-10-16T02:07:31.214607+00:00",
      "ended_at": "2026-10-16T02:07:31.214665+00:00",
      "usage": null,
      "occurrence": 1
    }
  ],
  "captured_at": null,
  "endpoint_fingerprint": null,
  "schema_version": "1"
}
```

### `Trace`

| Field | Type | Meaning |
| ----- | ---- | ------- |
| `trace_id` | `str` | The id every span and label refers to. |
| `frame` | `"discovery"` or `"production-representative"` | Which sample the trace belongs to. Rates are only estimated on production-representative traces; discovery traces are for finding failures. |
| `ownership` | `"owned"` or `"probed"` | `owned` has a span tree. `probed` is input and output only, with `spans == []`. |
| `input`, `output` | `dict` | The terminal request and the final answer. |
| `taxonomy_cell` | `dict[str, str]` | The situation this trace covers, such as `{"feature": "refund_handling", "persona": "P1_consumer"}`. |
| `spans` | `list[Span]` | The agent steps. Empty for a probed trace. |
| `captured_at` | `str` or `None` | When the trace was captured, if the source said. |
| `endpoint_fingerprint` | `str` or `None` | For a probed trace, which endpoint was observed. |
| `session_id` | `str` or `None` | The conversation or runtime session the trace ran in, as its source recorded it; traces with the same value are turns of one session. `None` when the source recorded none, and Calibrated Agent Evaluation never makes one up. An empty or whitespace-only value is refused. Left out of the JSON when unset, so a trace with no session is written exactly as before the field existed; the example above has none. AgentCore Runtime runs and AgentCore ingest fill it today; the Langfuse store also groups the trace under it in its Sessions view when the value is US-ASCII and under 200 characters. |
| `schema_version` | `str` | `"1"`. A reader refuses any other value. |

`Trace` has no `tool_calls` field. A single agent is a one-span tree, and tool calls always live on
a span, so the stages read tool calls from one place whatever the agent's shape.

### `Span`

| Field | Type | Meaning |
| ----- | ---- | ------- |
| `span_id` | `str` | Unique within the trace; normalizers write `<trace_id>/<agent>`, suffixed when an agent runs more than once. |
| `trace_id` | `str` | The owning trace. |
| `agent` | `str` | The agent's name. |
| `role` | `"specialist"`, `"supervisor"` or `"tool"` | The agent's part in the run. |
| `parent` | `str` or `None` | The `span_id` of the span that handed off to this one; `None` for the root. |
| `seq` | `int` | Execution order. Stored explicitly because not every store preserves order. |
| `status` | `"completed"` or `"errored"` | Whether the step finished. |
| `input` | `dict` | The handoff payload the step received. |
| `tool_calls` | `list[ToolCall]` | Each call's `name`, `args` and `result`, in order. |
| `output` | `dict` | The handoff payload the step emitted. |
| `started_at`, `ended_at` | `str` or `None` | Timestamps, if the source carried them. |
| `usage` | `TokenUsage` or `None` | Token counts the source reported for this step (`input_tokens`, `output_tokens`, `total_tokens`, each optional). `None` means none were reported; a total is never computed from its parts. |
| `occurrence` | `int` | Which execution of this agent the span is: 1 the first time, 2 when a retry or a cycle runs it again. |

### A crew

A crew's trace has the same fields with more spans. These are the spans of the corpus's ClaimsCrew
trace CLM-4471 after `calibrated-agent-evaluation ingest fixtures/raw/strands`, printed as `seq,
agent, role, parent, number of tool calls`:

```text
0 supervisor supervisor None 0
1 fraud-scorer specialist CLM-4471/supervisor 1
2 intake specialist CLM-4471/supervisor 1
3 payout specialist CLM-4471/supervisor 0
4 policy-checker specialist CLM-4471/supervisor 2
```

There is no separate handoff object. A handoff is a `parent` edge, and what crossed it is the
child span's `input`. A crew label names a handoff by its two agents (`handoff_implicated`, such as
`"intake->policy-checker"`) and a span by `span_id` (`culprit_span`, `manifestation_span`).

### `Label`

A label refers to a `trace_id`, and to a `span_id` for a span-level label. Human gold, judge
verdicts and code-scorer verdicts are all labels, told apart by `kind`.

| Field | Type | Meaning |
| ----- | ---- | ------- |
| `trace_id` | `str` | The trace the label is about. |
| `span_id` | `str` or `None` | The span, for a span-level label; `None` for a trace-level one. |
| `kind` | `"human"`, `"llm-judge"`, `"code"` or `"policy-engine"` | Who produced it: a person, a judge, a code scorer, or an AgentCore Policy engine's recorded deny. Only `human` labels are gold. |
| `outcome` | `"PASS"` or `"FAIL"` | The verdict. There is no third value. |
| `critique` | `str` | Required and non-empty: why the verdict was given, written before it. |
| `failure_mode` | `str` or `None` | The ratified mode a FAIL belongs to. |
| `culprit_span` | `str` or `None` | In a crew, the first span that produced a wrong output from the input it got. |
| `manifestation_span` | `str` or `None` | In a crew, the span where the failure became visible. |
| `handoff_implicated` | `str` or `None` | A handoff named by its two agents, such as `"intake->policy-checker"`. |
| `span_defect` | `bool` | A span-level FAIL that does not fail the trace, so it never gates. |
| `codebook_version` | `str` or `None` | The codebook version the label was written under; agreement is measured within one version. |
| `expected` | `dict` or `None` | A person's correct output for an objective FAIL, which `calibrated-agent-evaluation score` compares against. Never a substitute for the critique. |
| `rater_id` | `str` or `None` | Who applied a human label. `None` means not recorded, never anonymous. |

[The stage map](stage-map.md) shows a label with an example.

The ingestion adapters that produce these from OTel, Strands and Inspect sources are in
`calibrated_agent_evaluation.normalize`. See the [Python API reference](../reference/index.md).

## Next

- [Evaluate a multi-agent crew](../guides/evaluate-a-crew.md): attribution over the trace tree.
- [The stage map](stage-map.md): where normalization feeds the stages.
