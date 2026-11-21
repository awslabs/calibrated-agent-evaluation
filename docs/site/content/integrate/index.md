# Connect your agent

Everything in the toolkit works on what your agent did, so the first job is getting its runs into
the store as traces. A trace is one run: the input, every agent step and tool call along the way,
and the output. Once traces are in the store, every later step (labeling, judging, the CI gate)
works the same way whichever path brought them in.

There are five ways in. Answer one to three questions to find yours:

--8<-- "docs/site/content/includes/diagrams/integration-decision.md"

<div class="grid cards" markdown>

-   **[Ingest trace files](ingest-existing-traces.md)**

    Your agent already exports OpenTelemetry spans, Inspect `.eval` logs or a dataset of traces.
    Point `calibrated-agent-evaluation ingest` at the files. No change to your agent.

-   **[Read from Langfuse](langfuse.md)**

    Your traces already land in a Langfuse project. `calibrated-agent-evaluation ingest
    --from-langfuse` pulls them in, including traces the library did not write.

-   **[Capture in process](capture-and-ingest.md)**

    Your agent is instrumented with OpenTelemetry and you can call one run as a Python function.
    `calibrated-agent-evaluation capture` runs it and stores what it emitted.

-   **[Route your agent through](route-your-agent.md)**

    Your agent is a Strands topology, an Inspect solver or an agent deployed to Amazon Bedrock
    AgentCore Runtime, and you want the library to drive it over generated inputs and record each run.

-   **[Probe a black-box endpoint](probe-black-box.md)**

    You can only reach the agent over HTTP or WebSocket, or invoke it on AgentCore Runtime.
    `calibrated-agent-evaluation probe` sends prompts and stores the replies.

-   **[Bedrock AgentCore](agentcore.md)**

    Run, probe, audit or judge an agent deployed to Amazon Bedrock AgentCore Runtime.

-   **[Integration gallery](gallery/index.md)**

    One short recipe per framework and trace format, each with real output.

</div>

## What you get depends on what the library can see

Every path ends in the same kind of trace, but a trace from an agent you instrument carries more
than one from an endpoint you can only call. The first kind is called owned and the second
probed, and each trace records which one it is.

| You can evaluate | Owned (you instrument the agent) | Probed (endpoint only) |
| ---------------- | :------------------------------: | :--------------------: |
| Whether the whole run passed or failed | yes | yes |
| Which step went wrong, and its tool calls with arguments and results | yes | no |
| Which agent in a crew caused a failure, and where it showed up | yes | no |
| Failure rates per agent in a crew | yes | no |
| The CI gate, a calibrated judge and the before/after fix check | yes | yes, on outcomes |
| Capabilities you can rule in | everything in the trace | only what the replies show, as a lower bound |

How the agent is built does not change this table. A hand-written loop, an LLM with MCP tools, a
LangGraph or AutoGen crew and a Strands graph all reduce to the same trace. What changes the table
is whether the library can see the steps.

A trace with steps can still arrive as probed if its spans carry only the outer input and output.
`calibrated-agent-evaluation ingest` says so when it happens, and `--require-owned` stops the
ingest instead, for a corpus you plan to analyze step by step.

## Install and check

```bash
pip install calibrated-agent-evaluation
calibrated-agent-evaluation doctor   # your Python, the extras present, and the install command for any you are missing
```

The base install has one runtime dependency (pydantic) and works offline. Each path needs at most
one or two [extras](../how-to/install.md#extras):

| Path | Extra |
| ---- | ----- |
| Ingest OTLP-JSON files, Strands exports, JSONL datasets | none |
| Ingest binary `.otlp` protobuf | `otel` |
| Ingest Inspect `.eval` logs | `inspect` |
| Ingest parquet datasets | `datasets` |
| Ingest a HuggingFace dataset repo | `hf` and `datasets` (the repo's shards are parquet) |
| Read from Langfuse | `langfuse` |
| Capture in process | `strands` (it carries the OTel SDK the capture uses) |
| Route through Strands / Inspect / AgentCore Runtime | `strands` / `inspect` / `agentcore` |
| Probe an HTTP endpoint | none |
| Probe a WebSocket endpoint | `probe-ws` |
| Probe an agent on AgentCore Runtime | `agentcore` |
| Run the safety benchmark battery against a probed endpoint | `probe` |
| Check an AWS account is ready for AgentCore | `agentcore` |

## CLI or library

Every path has a CLI command, and that is the shortest way in. The CLI writes to a local store under
`./.calibrated-agent-evaluation` by default (`--root` picks another directory, and `--backend
langfuse` stores into Langfuse instead). A command that can't finish exits 1 and a usage error
exits 2, so each command drops into a script or a CI job as it is.

The library is for what the CLI does not take: spans you already hold in memory, a custom
OpenTelemetry dialect, or a whole loop in one Python file. The top-level
`calibrated_agent_evaluation` package exports only `__version__`; import from the subpackage that
owns each piece, such as `from calibrated_agent_evaluation.normalize import normalize, OTelSpans` or
`from calibrated_agent_evaluation.backends import get_backend`.

## Next

With traces in the store, [Read and label traces](../walkthrough/label.md) picks up the walkthrough
from labeling, through a checked judge, to the CI gate. The [trace format](../trace-format.md) is
the reference for what a stored trace holds.
