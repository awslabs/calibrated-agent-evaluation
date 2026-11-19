# Run fully offline

Calibrated Agent Evaluation works without a network by default. The core install has one third-party
dependency (pydantic), the local store is a directory on disk, and nothing needs an API key.

## The offline default

```bash
uv sync              # from a checkout; see Installation
source .venv/bin/activate
calibrated-agent-evaluation quickstart     # seeds and analyzes the sample corpus
calibrated-agent-evaluation doctor         # "Local backend: writable at .calibrated-agent-evaluation (offline path ready)"
```

The local backend (`--backend local`, the default) keeps traces, labels and records in a SQLite
file under `--root`. Ingesting files, labeling, clustering, calibrating with `--cassette`, the bias
check, scoring with `--cassette`, the CI gate and dump/load all run with no network calls.

## Which extras reach the network

Most extras are local code: `[stats]`, `[sampling]`, `[scorers]`, `[datasets]`,
`[otel]` and `[ui]` (the review server listens on your machine). These extras can reach a
network, and only when you use the part that does:

| Extra | Reaches | When |
| --- | --- | --- |
| `[inspect]` | the judge's model endpoint | a judge or an Inspect agent runs without `--cassette` |
| `[bedrock]` | Amazon Bedrock | `calibrated-agent-evaluation generate --model` and `calibrated-agent-evaluation cluster --embedder bedrock` |
| `[strands]` | the agent's model provider | a Strands agent runs live |
| `[synth]` | its model provider | multi-turn input synthesis |
| `[agentcore]` | your AWS account (AgentCore, CloudWatch, X-Ray) | an AgentCore module is used; importing the package without one makes no call |
| `[probe]` | benchmark datasets and the endpoint you probe | the benchmark battery, from the library |
| `[probe-ws]` | the WebSocket endpoint you probe | `calibrated-agent-evaluation probe --endpoint ws://...` |
| `[langfuse]` | your Langfuse server | `--backend langfuse` or `ingest --from-langfuse` |
| `[hf]` | the Hugging Face Hub | resolving a dataset by id |

`calibrated-agent-evaluation probe` against an `http(s)://` endpoint needs no extra; it contacts
only the endpoint you name, and only with `--authorized`. Reading Inspect `.eval` logs with
`[inspect]` stays local. A cassette replays recorded model replies and never falls back to a live
call, so a run with `--cassette` stays offline even with these extras installed.

## Bring your own trace store

If you run a self-hosted Langfuse, point the CLI at it instead of the local store:

```bash
uv sync --inexact --extra langfuse
export LANGFUSE_HOST=http://your-langfuse:3000
export LANGFUSE_PUBLIC_KEY=...   # from your Langfuse project
export LANGFUSE_SECRET_KEY=...
calibrated-agent-evaluation --backend langfuse run --ci
```

You run the server, so no trace data leaves your network.

## A judge on a local model

The judge file's `model` field is an Inspect model string, so a judge can run on a model on your
own hardware (`"model": "ollama/llama3.1"`), and only its model call leaves the process.
[Run a judge with no network](../recipes/air-gapped-judge.md) walks through it end to end.

## Next

- [The extras model](../concepts/extras.md): why the core is empty and what each extra adds.
- [Installation](../getting-started/installation.md): the extras each command needs.
