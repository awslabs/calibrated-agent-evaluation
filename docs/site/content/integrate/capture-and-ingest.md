# Capture in process

`calibrated-agent-evaluation capture` runs one instrumented run of your agent and stores the trace
it emitted. Use it when your agent emits OpenTelemetry spans but you have no exporter writing them
to files yet.

It needs the `strands` extra, which carries the OpenTelemetry SDK and the OTLP encoder the capture
uses:

```bash
uv sync --inexact --extra strands
source .venv/bin/activate    # puts `calibrated-agent-evaluation` on your PATH
```

## The agent

The agent is any Python function that takes no arguments and emits spans through the global
OpenTelemetry tracer. This is a complete one, `orderbot.py`, with one agent span and one tool span
using the OpenTelemetry GenAI attribute names:

```python
--8<-- "docs/site/examples/integrate/orderbot.py"
```

An agent built on a framework that already emits GenAI spans (Strands, or anything under an
OpenLLMetry or OpenInference instrumentor) needs no hand-written spans. The function just runs the
agent once.

## Capture it

These four commands were run from the directory holding `orderbot.py`, with `PYTHONPATH=.` so the
installed `calibrated-agent-evaluation` script can import it. An agent inside an installed package
needs no `PYTHONPATH`.

```text
--8<-- "fixtures/expected/docs/integrate-capture.txt"
```

In order, they show:

1. `--dry-run` runs the agent and reports what would be stored, without storing it. The id is the
   capture's OpenTelemetry trace id, which differs on every run.
2. `--trace-id` stores the trace under a name you choose. The line reports the convention the spans
   were read with, the agent steps and tool calls recovered, and that the trace is owned (its steps
   are visible).
3. A second capture under the same id stops with an error and keeps the stored trace, since a
   capture can't be re-run to get the old one back. Pass `--overwrite` to replace it on purpose.
4. A function that emits no spans stops with an error, and nothing is stored.
   `os:getcwd` is a real function that emits nothing.

If the function raises, `calibrated-agent-evaluation capture` prints the `--agent` value and the
exception, exits 1 and stores nothing.

## Why it has to run in the same process

Capture works by installing an OpenTelemetry tracer provider in the process that runs the agent, so
`calibrated-agent-evaluation capture` imports your module and calls the function itself. An agent
that runs as a separate service, CLI or job cannot emit into that provider. For those, have the
agent's process export OTLP-JSON and [ingest the file](ingest-existing-traces.md). Calibrated Agent
Evaluation's `SpanCapture` does the export in a few lines:

```python
# in the agent's process, with the strands extra installed
from pathlib import Path

from calibrated_agent_evaluation.execute.strands_capture import SpanCapture, write_otlp_json

with SpanCapture("orderbot") as capture:
    capture.register_global()      # before the agent's first span
    run_once()                     # your agent, emitting spans as usual
write_otlp_json(capture.finished_spans, Path("exports/run-1.otlp.json"))
```

```bash
calibrated-agent-evaluation ingest exports/run-1.otlp.json
```

Any exporter that writes one OTLP/HTTP JSON document (`resourceSpans`, `scopeSpans`, `spans`) per
file works.

## From Python

`capture_and_ingest` is the function behind the command:

```python
--8<-- "docs/site/examples/integrate/capture_from_python.py:capture"
```

```text
--8<-- "fixtures/expected/docs/integrate-capture-python.txt"
```

The full program is `docs/site/examples/integrate/capture_from_python.py`.

You can call `capture_and_ingest` as many times as you like in one process; each call stores only
the spans its agent emitted. A span belongs to the capture it started inside, or to the capture
that holds its parent span, so two captures running on two threads keep their spans apart, and
host telemetry that started outside the capture (a web request, a database call) stays out of the
trace. If your application already installed its own OpenTelemetry SDK tracer provider, capture
attaches to it. While a capture is open, Strands' tracer emits only to the capture, so a Strands
run is not also exported raw by whatever exporter the application's provider carries. A provider
that is not the OpenTelemetry SDK's (some
vendor distributions) is refused with an error that names it, because capture cannot attach to
it; export OTLP-JSON from that process and ingest the file instead.

## Next

With traces in the store, the next step is labeling. `calibrated-agent-evaluation label` without
`--serve` prints the command that opens the review surface on your store:

<div class="wrap-output" markdown>

```text
--8<-- "fixtures/expected/docs/integrate-label-hint.txt"
```

</div>

See [Review traces in the UI](../how-to/review-ui.md) for the labeling session itself.
