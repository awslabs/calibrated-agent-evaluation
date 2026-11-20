# Inspect AI

You need [Inspect AI](https://inspect.aisi.org.uk) `.eval` logs, and the `inspect` extra.

## Ingest the logs

```bash
uv sync --inexact --extra inspect
source .venv/bin/activate    # puts `calibrated-agent-evaluation` on your PATH
calibrated-agent-evaluation ingest logs/
```

On a one-sample log of the repository's OrderBot:

```text
--8<-- "fixtures/expected/docs/integrate-ingest-inspect.txt"
```

A log holds one sample per dataset row, and ingest stores one trace per sample. From Python, use
`normalize_all(InspectEvalLog(path=...))` for every sample; `normalize()` on a multi-sample log
raises an error unless `sample_id=` names one, so you always know which samples were read.

## Or let Calibrated Agent Evaluation run the task

`InspectExecutor` runs an Inspect task with your agent as the solver and returns its traces.
Replayed from the repository's cassette:

```text
--8<-- "fixtures/expected/docs/integrate-route-inspect.txt"
```

See [Route your agent through Calibrated Agent Evaluation](../route-your-agent.md) for the code.

## What you get

An owned trace per sample: the agent's step with every tool call, its arguments and results. The
sample's metadata can carry the trace id and taxonomy cell.

## Limits

- An agent run through Inspect's agent bridge stores long strings as `attachment://` references in
  the log. Check that the content you need is in the trace before relying on it.
- The trace above is the OrderBot failure the bundled judge kit is built on: the order was outside
  the 30-day return window, the broken policy tool said it was within it, and the agent refunded
  $89.99 on its word. Ingest records what happened; judging it comes later.
