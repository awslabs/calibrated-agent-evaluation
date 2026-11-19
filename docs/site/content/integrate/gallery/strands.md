# Strands Agents

You need a [Strands Agents](https://strandsagents.com) `Agent`, `Graph` or `Swarm`. Ingesting
an export needs no extra; driving the agent needs the `strands` extra.

There are two ways in.

## Ingest a Strands OTel export

Strands emits OpenTelemetry spans through the global tracer provider. Export them as OTLP-JSON
(for example with `SpanCapture`, as in the [GenAI recipe](otel-genai.md#export-the-spans)) and
ingest the files. Graph, Swarm and agents-as-tools topologies share one span grammar:

```text
--8<-- "fixtures/expected/docs/integrate-ingest-strands.txt"
```

The four `CLM-*` traces are a claims crew built as a Graph, `CLM-7720-swarm` is a Swarm, and
`ORD-3310-as-tool` is an agent that calls another agent as a tool.

From Python, `normalize(StrandsSession(path=...))` reads one export; pass `mode="legacy"` for an
export written with Strands' older span attributes.

## Let Calibrated Agent Evaluation drive it

Give `StrandsExecutor` a function that builds your topology from a model, and it runs the agent,
records the model calls to a cassette and returns traces. Replayed from the repository's cassette:

```text
--8<-- "fixtures/expected/docs/integrate-route-strands.txt"
```

The code, and how to record your own cassette, is on
[Route your agent through Calibrated Agent Evaluation](../route-your-agent.md).

## What you get

An owned trace per run: a supervisor span with one child per specialist, each with its tool calls,
arguments and results. The taxonomy cell you pass to the task is copied onto the trace.

## Limits

- Specialists that run in parallel finish in a different order on each run, so compare traces by
  agent, not by step order.
- A live run calls Bedrock and needs AWS credentials. Replay needs none: the Bedrock model is
  only built when a cassette miss needs a live call.
- The executor sends `temperature=0.0` unless you pass `temperature=None`; the newest Anthropic
  models reject an explicit temperature.
