# Read from Langfuse

If your team already sends agent traces to [Langfuse](https://langfuse.com), Calibrated Agent
Evaluation can read them straight from the project. It works with an unmodified Langfuse v3
deployment, self-hosted or not, and needs the `langfuse` extra.

```bash
uv sync --inexact --extra langfuse
source .venv/bin/activate    # puts `calibrated-agent-evaluation` on your PATH

export LANGFUSE_HOST="https://langfuse.your-company.internal"   # default: http://localhost:3000
export LANGFUSE_PUBLIC_KEY="pk-lf-..."
export LANGFUSE_SECRET_KEY="sk-lf-..."

calibrated-agent-evaluation ingest --from-langfuse "My Project"
```

That reads every trace in the project, normalizes each one and writes the result to the local
store (`./.calibrated-agent-evaluation`). The traces in Langfuse are not changed.

Without the keys, the command stops before contacting anything:

```text
--8<-- "fixtures/expected/docs/integrate-langfuse-no-keys.txt"
```

## Why the project name is a check

Langfuse keys belong to one project, and its API cannot read another. So `--from-langfuse` takes
the project name (or id) as a check rather than a choice: if it does not match the project the
keys belong to, the command stops and names the project the keys do belong to. That catches a key
pasted from the wrong project before a single trace is read.

## What it reads

Traces your application wrote to Langfuse, with the Langfuse SDK or with an OpenTelemetry exporter
pointed at Langfuse's OTLP endpoint, are read like a trace file:

- each Langfuse observation becomes a span, and the OpenTelemetry attributes it recorded are kept;
- without those attributes, an `agent` observation is an agent step, a `tool` observation a tool
  call and a `generation` an LLM call;
- a trace with no `agent` observation treats its outermost observation as the agent.

The same checks as file ingest apply. A trace whose tool calls cannot be attached to an agent is
refused rather than stored with them missing, and a trace that cannot be read at all is reported
as skipped while the rest of the project still comes in. The command exits 1 if anything was
refused or skipped.

Traces the library itself stored in Langfuse come back exactly as they were written.

## Use Langfuse as the store

`--backend langfuse` makes Langfuse the store every stage reads and writes, instead of the local
directory. It is a global flag, so it goes before the subcommand:

```bash
calibrated-agent-evaluation --backend langfuse ingest exports/        # ingest files into Langfuse
calibrated-agent-evaluation --backend langfuse run --ci               # gate over the verdicts stored in Langfuse
```

An unreachable host fails at once with a typed error, so a dead backend cannot look like an empty
one. Langfuse writes are asynchronous; a command waits for each write to become readable before a
later stage depends on it.

From Python:

```python
# needs the langfuse extra
from calibrated_agent_evaluation.backends import LangfuseBackend

with LangfuseBackend.from_credentials(
    public_key="pk-lf-...",
    secret_key="sk-lf-...",
    host="https://langfuse.your-company.internal",
) as backend:
    trace = backend.get_trace("ORD-1001")
```

## Try it against a local Langfuse

The repository ships a stock Langfuse v3 stack for development under `docker/langfuse/`, with
telemetry off and a provisioned dev project:

```bash
just langfuse-up     # web UI on http://localhost:3000
export LANGFUSE_PUBLIC_KEY=pk-lf-calibrated-agent-evaluation-dev LANGFUSE_SECRET_KEY=sk-lf-calibrated-agent-evaluation-dev
calibrated-agent-evaluation --backend langfuse ingest fixtures/raw/strands
```

The dev keys in `docker/langfuse/.env.example` are throwaway values for a local instance only.

## Next

- [Ingest trace files](ingest-existing-traces.md) for exports that never reach Langfuse.
- [Look and label](../guides/lifecycle/3-look-and-label.md) for what to do with the traces once
  they are in.
