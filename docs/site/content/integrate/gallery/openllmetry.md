# OpenLLMetry (Traceloop)

You need an agent instrumented with Traceloop's
[OpenLLMetry](https://github.com/traceloop/openllmetry) instrumentations, exporting OTLP-JSON. No
extra.

## Export the spans

OpenLLMetry instrumentors emit into the global tracer provider. In a Python process you control,
open a Calibrated Agent Evaluation `SpanCapture` and call `register_global()` first, then turn the
instrumentation on. This is how the repository's external agent run captured a LangGraph crew under
the LangChain instrumentor (`opentelemetry-instrumentation-langchain` 0.62.3):

```python
from pathlib import Path

from opentelemetry.instrumentation.langchain import LangchainInstrumentor

from calibrated_agent_evaluation.execute.strands_capture import SpanCapture, write_otlp_json

with SpanCapture("my-crew") as capture:
    capture.register_global()
    LangchainInstrumentor().instrument()
    run_my_crew()
write_otlp_json(capture.finished_spans, Path("exports/run-1.otlp.json"))
```

## Ingest

```bash
calibrated-agent-evaluation ingest exports/
```

On an OpenLLMetry OpenAI span with a tool call (OpenLLMetry's own recorded test values):

```text
--8<-- "fixtures/expected/docs/integrate-ingest-openllmetry.txt"
```

Calibrated Agent Evaluation reads the older dotted-index grammar (`gen_ai.prompt.{i}.*`,
`gen_ai.completion.{i}.*`) as `openllmetry`. Newer instrumentations that emit `invoke_agent` spans
are read with the GenAI conventions instead: the LangGraph crew captured under the LangChain
instrumentor 0.62.3 reads as `gen_ai/latest`, with its `execute_task` spans as the crew's agents.
Detection picks the reader per file, so you do not need to know which one applies.

## What you get

An owned trace with the model's messages and tool calls. A crew becomes a tree of agents.

## Limits

- A completion consumed inside an agent step, such as a crew picking its next role, is on the wire
  only as an LLM span. A trace holds agent steps and tool calls, so those calls are not separate
  nodes.
- Traceloop emits empty duplicate `invoke_agent LangGraph` spans; ingest counts them as nodes with
  no content and reports the count rather than hiding them.
