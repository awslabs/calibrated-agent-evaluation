# OpenTelemetry GenAI spans

You need an agent that emits spans with the OpenTelemetry GenAI semantic conventions
(`gen_ai.*` attributes), and a way to write them to OTLP-JSON files. No extra.

## Export the spans

If your agent runs in a Python process you control, Calibrated Agent Evaluation's `SpanCapture`
collects every span it emits and writes them as OTLP-JSON (it needs the `strands` extra, which
carries the OTel SDK):

```python
from pathlib import Path

from calibrated_agent_evaluation.execute.strands_capture import SpanCapture, write_otlp_json

with SpanCapture("my-agent") as capture:
    capture.register_global()      # before the agent's first span
    run_my_agent()                 # your agent, emitting gen_ai.* spans as usual
write_otlp_json(capture.finished_spans, Path("exports/run-1.otlp.json"))
```

Any other exporter that writes one OTLP/HTTP JSON document per file works too.

## Ingest

```bash
calibrated-agent-evaluation ingest exports/
```

On a span recorded in the `gen_ai.input.messages` / `gen_ai.output.messages` style (the
`gen_ai/messages` convention, as Claude Code and current OpenLLMetry write it):

```text
--8<-- "fixtures/expected/docs/integrate-ingest-messages.txt"
```

`calibrated-agent-evaluation ingest` reads three generations of the GenAI conventions and picks the
right one per file:

| Convention | What the spans look like |
| ---------- | ------------------------ |
| `gen_ai/latest` | `invoke_agent` and `execute_tool` spans, content on attributes or events |
| `gen_ai/messages` | `gen_ai.input.messages` / `gen_ai.output.messages` with structured tool-call parts |
| `gen_ai/legacy` | older per-message `gen_ai.*` events |

## What you get

An owned trace: each `invoke_agent` span becomes an agent step, and each tool call keeps its
arguments and result. Nested agents become a crew tree.

## Limits

- A trace whose spans carry only the outer input and output degrades to probed, and ingest says
  why. Use `--require-owned` to refuse those instead.
- A file whose signals match two conventions is skipped with a reason. Name one with
  `--convention`.
- Call `register_global()` before the agent creates its first span. A span is captured when it
  starts inside the capture, so spans from before that call, or from unrelated work in the same
  process, are left out. You can open as many captures as you like in one process.
