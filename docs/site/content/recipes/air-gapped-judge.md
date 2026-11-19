# Run a judge with no network

Calibrate and score with a judge whose model runs on your own hardware, or with no model call at
all.

## Option 1: replay a cassette

A cassette records a judge model's replies once and replays them afterwards. Every command on the
[Calibrate a judge](../guides/calibrate-a-judge.md) page runs this way, with no credentials and no
network. Replay never falls back to a live call: a prompt that wasn't recorded is refused (see
[Troubleshooting](../guides/troubleshooting.md)). Use this in CI and for reproducing a past run.

To record a cassette, run the same command once where the judge's model is reachable, with
`--cassette PATH --cassette-mode record`. It calls the live model for every judge call (so it
needs `inspect` and the model's credentials), overwrites any reply the file already holds for the
same prompt, and writes each reply as it arrives. A run that stops part way keeps what it
recorded. Commit the file, then replay it everywhere else with `--cassette PATH`.

## Option 2: a model on your own hardware

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
and `<SERVICE>_API_KEY` set. Bump `prompt_version` too: a new model is a new judge for
calibration.

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

### A smaller local model may not certify

Calibration decides that, the same way it does for a hosted model. If it is refused, the output
says how many more labels would settle it; a model that is wrong too often won't certify at any
size.

### Tool calling

Only Bedrock's Claude models are sent a response schema. Every other model answers through one
forced tool call whose arguments are the verdict, never free text, so the model and server you
choose must support tool calling.

### The bias check needs the model too

It runs the judge on its probes, so it also goes to your endpoint.
