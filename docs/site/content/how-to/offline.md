# Run fully offline

The toolkit works without a network by default. The core install has one third-party dependency
(pydantic), the local store is a directory on disk, and nothing needs an API key.

```bash
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
`[inspect]` stays local. A run with `--cassette` replays recorded model replies and makes no live
call, even with these extras installed.

## Bring your own trace store

If you run a self-hosted Langfuse, point the CLI at it instead of the local store:

```bash
pip install "calibrated-agent-evaluation[langfuse]"
export LANGFUSE_HOST=http://your-langfuse:3000
export LANGFUSE_PUBLIC_KEY=...   # from your Langfuse project
export LANGFUSE_SECRET_KEY=...
calibrated-agent-evaluation --backend langfuse run --ci
```

You run the server, so no trace data leaves your network.

all.

## Replay a judge from a cassette

A cassette records a judge model's replies once and replays them afterwards, with no credentials
and no network. The [walkthrough](../walkthrough/judge.md) runs this way. A prompt the cassette
never recorded stops the run with an error (see [Troubleshooting](troubleshooting.md)); it does
not fall back to the live model. Use replay in CI and to reproduce a past run.

To record a cassette, run the same command once where the judge's model is reachable, with
`--cassette PATH --cassette-mode record`. It calls the live model for every judge call (so it
needs `inspect` and the model's credentials), overwrites any reply the file already holds for the
same prompt, and writes each reply as it arrives. A run that stops part way keeps what it
recorded. Commit the file, then replay it everywhere else with `--cassette PATH`.

## Run the judge on your own hardware

### 1. Serve a model

Run a model behind an OpenAI-compatible endpoint, for example with Ollama, vLLM or the llama.cpp
server.

### 2. Point the judge file at it

The `model` field is an Inspect model string:

```json
--8<-- "docs/site/examples/guides/local-judge.json"
```

`ollama/<model>` goes to `http://localhost:11434/v1` unless `OLLAMA_BASE_URL` says otherwise. For
any other OpenAI-compatible server use `openai-api/<service>/<model>`, with `<SERVICE>_BASE_URL`
and `<SERVICE>_API_KEY` set. Bump `prompt_version` too: a new model is a new judge and has to be
checked again.

### 3. Check the file, and where it will send requests

Neither check calls the model:

```text
--8<-- "fixtures/expected/docs/guide-local-judge-check.txt"
```

```text
--8<-- "fixtures/expected/docs/guide-local-judge-endpoint.txt"
```

### 4. Install and run

Install the `[inspect]` extra and the `openai` package (3.4 or later), which Inspect uses for
these providers. Then calibrate, run the bias check and score as usual, without `--cassette`.

## Pitfalls

### A smaller local model may not pass the check

`calibrate` decides that the same way it does for a hosted model. If the judge isn't certified, the
output says how many more labels would settle it; a model that is wrong too often won't pass at any
number of labels.

### Tool calling

Only Bedrock's Claude models are sent a response schema. Every other model answers through one
forced tool call whose arguments are the verdict, so the model and server you choose must support
tool calling.

### The bias check needs the model too

It runs the judge on its probes, so it also goes to your endpoint.

## Next

- [Install](install.md): the extras each command needs.
- [Configure the CI job](ci-config.md): replaying a judge in CI.
