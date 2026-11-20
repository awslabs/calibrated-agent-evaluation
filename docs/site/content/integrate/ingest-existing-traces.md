# Ingest trace files

If your agent already exports traces, `calibrated-agent-evaluation ingest` reads them with no change
to the agent. Here it reads the repository's Strands crew captures:

```text
--8<-- "fixtures/expected/docs/integrate-ingest-strands.txt"
```

Each line is one trace: the file it came from, the id it was stored under, the format it was read
as, how many agent steps (spans) and tool calls were recovered, and whether it is owned (Calibrated
Agent Evaluation can see its steps) or probed (only the outer input and output survived). A trace id
is the file name up to its first dot, so `CLM-4471.otlp.json` becomes `CLM-4471`.

## What it reads

| Your traces are | Example file | Extra |
| --------------- | ------------ | ----- |
| OpenTelemetry spans exported as OTLP-JSON (GenAI semantic conventions, OpenLLMetry/Traceloop, OpenInference/Phoenix) | `session.otlp.json` | none |
| Binary OTLP protobuf | `session.otlp` | `otel` |
| A Strands OTel export | `run.otlp.json` | none |
| Inspect `.eval` logs | `2026-07-23_orderbot.eval` | `inspect` |
| A dataset of traces, one per record | `week-30.jsonl`, `slice.parquet` | none for JSONL, `datasets` for parquet |
| A HuggingFace dataset repo | `Exgentic/agent-llm-traces` with `--format hf` | `hf` and `datasets` |

`calibrated-agent-evaluation ingest` takes files and directories, and walks directories recursively.
It works out the format from each file, and for OpenTelemetry files it also detects which attribute
convention the spans use. The [integration gallery](gallery/index.md) has a recipe for each source
with its real output.

```bash
calibrated-agent-evaluation ingest exports/                        # every trace file under exports/
calibrated-agent-evaluation ingest captures/week-30.jsonl          # a dataset, one trace per line
calibrated-agent-evaluation ingest --format hf Exgentic/agent-llm-traces
calibrated-agent-evaluation --backend langfuse ingest exports/     # store into Langfuse instead of ./.calibrated-agent-evaluation
```

## Preview first with --dry-run

`--dry-run` reports what would happen and writes nothing. Run it on a directory you have not
ingested before. The repository's OTel fixture directory deliberately mixes good, degraded and bad
files:

```text
--8<-- "fixtures/expected/docs/calibrated-agent-evaluation-ingest-dry-run.txt"
```

Every file ends in one of three states:

- `[ok]` would be stored. Two of these come out probed because the files carry only the outer
  input and output, and the lines at the top say why.
- `[REFUSED]` would lose data if stored. Here two files resolve to the same trace id, so storing
  both would overwrite one with the other. The reason names both files and two ways to fix it.
- `[skipped]` could not be read as a trace at all. The reason lists the signals it looked for.

A dry run always exits 0. A real run exits 1 if any file was refused or skipped, so a script can
tell a partial ingest from a complete one.

When files in one batch share the part of the name before the first dot (the three `ORD-1001.*`
files above), each is stored under its OpenTelemetry trace id instead, which is why those lines
show hex ids. To keep readable ids, give each file a distinct prefix or tag the spans at
instrumentation time (below).

## Draw a sample instead of taking everything

A failure rate describes production only when it is measured on traces that stand for production
traffic. A directory of exports or a Langfuse project usually doesn't: whoever exported it chose
what went in. `--sample N --seed S`
draws a seeded random sample of the population, stores those traces tagged
`production-representative`, and writes a record of the draw:

```text
--8<-- "fixtures/expected/docs/integrate-ingest-sample.txt"
```

The seed is required so anyone can repeat the draw and get the same traces. The record under
`<root>/records/sample/` keeps the seed, the population size, the ids drawn and the ids stored, and
`calibrated-agent-evaluation dump` carries it with everything else. Asking for more traces than the
population holds stops with an error that gives the population size.

For a Langfuse project, `--cell KEY=VALUE` draws within one taxonomy cell, which is a stratified
sample: run it once per cell and each draw gets its own record with its own population size.

```bash
calibrated-agent-evaluation ingest --from-langfuse my-project --sample 200 --seed 7
calibrated-agent-evaluation ingest --from-langfuse my-project --sample 60 --seed 7 --cell channel=chat
```

A trace whose taxonomy cell is unknown cannot belong to a stratum, so it is left out of the
population and counted on the record. Over files, `--cell` is refused: a file's cell is only known
once it is normalized.

## Refuse outcome-only traces

If you plan to analyze the traces step by step, a trace that carries only its outer input and
output is of little use. `--require-owned` refuses it instead of storing it as probed:

```text
--8<-- "fixtures/expected/docs/integrate-ingest-require-owned.txt"
```

## When a convention is ambiguous

If a file's spans match more than one OpenTelemetry convention, `calibrated-agent-evaluation ingest`
skips it and says so rather than guess. The one exception: when the GenAI conventions' agent and
tool spans are present, they win, because they carry the agent tree the others lack. Otherwise name
the one to use:

```bash
calibrated-agent-evaluation ingest --convention openinference exports/phoenix/
```

The choices are `gen_ai/latest`, `gen_ai/legacy`, `gen_ai/messages`, `openllmetry` and
`openinference`.

## From Python

The CLI wraps a two-step library path: `normalize()` turns one source into a `Trace`, and a
backend stores it.

```python
from pathlib import Path

from calibrated_agent_evaluation.backends import get_backend
from calibrated_agent_evaluation.normalize import OTelSpans, normalize

trace = normalize(OTelSpans(path=Path("exports/session-8f2c.otlp.json")))

with get_backend() as backend:      # the local store at ./.calibrated-agent-evaluation
    backend.put_trace(trace)        # storing the same trace id again replaces it
```

`normalize()` returns a complete, validated `Trace` with every span, or raises
`NormalizationError`. Note one difference from the CLI: the library takes the
trace id from the spans (pass `trace_id=` to set one), while the CLI names a trace after its file.

Each source has its own descriptor:

```python
from pathlib import Path

from calibrated_agent_evaluation.normalize import InspectEvalLog, SessionSpans, StrandsSession, normalize, normalize_all
from calibrated_agent_evaluation.normalize.datasets import JSONLDataset, load_traces

normalize(StrandsSession(path=Path("exports/claims-run.otlp.json")))

# An Inspect log holds one sample per dataset row; normalize_all returns one Trace per sample.
traces = normalize_all(InspectEvalLog(path=Path("logs/2026-07-23_orderbot.eval")))

# Spans you already hold in memory, for example from your own exporter.
normalize(SessionSpans(spans=my_span_dicts))

# A dataset streams one Trace per record.
for trace in load_traces(JSONLDataset(path=Path("captures/week-30.jsonl")), limit=500):
    ...
```

`normalize()` on a multi-sample Inspect log refuses rather than read only the first sample. Use
`normalize_all()`, or pass `sample_id=` to pick one.

`calibrated_agent_evaluation.normalize.datasets` also has `ParquetDataset`, `HFDataset` and
`DatasetMapping`, which names the columns a differently shaped dataset keeps its session id, spans
and taxonomy fields in.

## Name traces at instrumentation time

If you control the agent's OpenTelemetry setup, stamp three things on the root span so they
survive ingest instead of being filled in with defaults:

- a readable trace id, such as an order or ticket number;
- the frame, which says why the trace was collected: `production-representative` for a sample of
  real traffic, `discovery` for traces gathered to find bugs. Rates are only ever estimated from the
  first kind;
- the taxonomy cell, the combination of input dimensions the run exercised (channel, tier,
  feature), so failures can be broken down by it.

```python
from calibrated_agent_evaluation.normalize.guidance import tag_span

tag_span(
    span.set_attribute,             # your OTel span's set_attribute
    trace_id="ORD-2026-07-0042",
    frame="production-representative",
    taxonomy_cell={"channel": "chat", "tier": "premium"},
)
```

This is optional. Without it, ingest uses the file name or OpenTelemetry trace id, and the
`discovery` frame unless you pass `--frame production-representative` (or draw a sample with
`--sample N --seed S`, which tags the drawn traces `production-representative`).

## My framework is not listed

There are no adapters per framework. LangGraph, CrewAI, AutoGen and the rest reach Calibrated Agent
Evaluation as OpenTelemetry spans. If your framework emits OpenTelemetry natively, or has an
OpenLLMetry or OpenInference instrumentation you can turn on, export OTLP-JSON and run
`calibrated-agent-evaluation ingest`.

For a dialect none of the built-in conventions reads, `register_convention` adds one from Python.
It is library-only, since the CLI cannot load your module, and you reach it through
`SessionSpans`:

```python
from calibrated_agent_evaluation.normalize import SessionSpans, normalize
from calibrated_agent_evaluation.normalize.conventions.registry import register_convention
from calibrated_agent_evaluation.normalize.tree import AgentNode, Intermediate, ToolInvocation


def my_dialect(session: SessionSpans) -> Intermediate:
    attrs = session.spans[0]["attributes"]
    node = AgentNode(
        agent=attrs["myco.agent"],
        role="specialist",
        parent=None,
        input={"message": attrs["myco.prompt"]},
        output={"answer": attrs["myco.reply"]},
        tool_calls=[ToolInvocation(name=attrs["myco.tool"], args={}, result={})],
    )
    return Intermediate(
        trace_id=session.trace_id or "myco-trace",
        frame=session.frame,
        taxonomy_cell=dict(session.taxonomy_cell),
        trace_input=node.input,
        trace_output=node.output,
        nodes=[node],
    )


register_convention("myco", my_dialect)
trace = normalize(SessionSpans(spans=my_span_dicts, semconv="myco"))
```

Registering a built-in name, or one already registered, raises instead of replacing it. If your
dialect only keeps content under different attribute keys, `register_content_placements` in
`calibrated_agent_evaluation.normalize.content_placements` adds those keys to what the built-in
readers try, with no new adapter.

## Next

- [Capture in process](capture-and-ingest.md) if your agent does not export traces yet.
- [Trace format](../trace-format.md) for what a stored trace contains.
- The [`calibrated_agent_evaluation.normalize` API
  reference](../reference/calibrated_agent_evaluation/normalize/index.md) for every descriptor
  field.
