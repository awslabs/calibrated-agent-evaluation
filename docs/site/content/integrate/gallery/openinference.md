# OpenInference (Arize Phoenix)

You need an agent instrumented with an [OpenInference](https://github.com/Arize-ai/openinference)
instrumentor (the instrumentation Arize Phoenix collects), exporting its spans as OTLP-JSON. No
extra.

## Export the spans

Turn on the OpenInference instrumentor for your framework as its documentation describes, and send
the spans to a file instead of, or as well as, Phoenix. In a Python process you control, open a
Calibrated Agent Evaluation `SpanCapture` and call `register_global()` before the instrumented code
runs, then write the spans out, as in the [GenAI recipe](otel-genai.md#export-the-spans).

## Ingest

```bash
calibrated-agent-evaluation ingest exports/phoenix/
```

On an OpenInference LLM span with a tool call (an attribute set published in the OpenInference
repository):

```text
--8<-- "fixtures/expected/docs/integrate-ingest-openinference.txt"
```

Calibrated Agent Evaluation recognizes the convention by `openinference.span.kind` and reads the
messages and tool calls from `llm.input_messages.*` and `llm.output_messages.*`.

## What you get

An owned trace with the conversation and each tool call the model made, with its arguments.

## Limits

- Tool results are read only where the spans record them.
- If the same file also matches another convention, it is skipped with a reason. Force this one
  with `calibrated-agent-evaluation ingest --convention openinference`.
