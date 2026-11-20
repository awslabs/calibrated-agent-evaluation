# Install

You need Python 3.11.5 or later.

```bash
pip install calibrated-agent-evaluation
```

The base install has one runtime dependency, pydantic, and stores traces and labels in a local
SQLite file. Everything that needs a third-party package is an extra you add when you reach the
command that uses it. Most people start with the judge statistics and the review UI:

```bash
pip install "calibrated-agent-evaluation[stats,ui]"
```

Check what you have:

```bash
calibrated-agent-evaluation --version
calibrated-agent-evaluation doctor
```

`doctor` prints your Python version, every extra the package declares, which ones are installed,
and the command to add any that are missing.

=== "uv"

    ```bash
    uv add "calibrated-agent-evaluation[stats]"        # into a uv project
    uv tool install calibrated-agent-evaluation        # the CLI alone, in its own environment
    ```

=== "From a checkout"

    ```bash
    git clone https://github.com/awslabs/calibrated-agent-evaluation.git
    cd calibrated-agent-evaluation
    uv sync --extra stats --extra ui
    source .venv/bin/activate
    ```

    Inside a checkout you can prefix commands with `uv run` instead of activating the environment.
    The example programs under `docs/site/examples/` need a checkout.

## What each command needs

| Command | Extras |
| --- | --- |
| `quickstart`, `ingest` (trace files), `label`, `doctor` | none |
| `dump`, `load` | none for traces and labels; `stats` once the store holds run, bias or improve records |
| `calibrate`, `bias`, `judge`, `run`, `improve`, `plan`, `results` | `stats` |
| a live judge model (not a recorded cassette) | `stats` and `inspect` |
| `cluster` | `stats`; `bedrock` for `--embedder bedrock` |
| `define` | `sampling` |
| `generate` | `sampling` and `stats` |
| `score` with a compare-to-expected scorer (exact, levenshtein, numeric, JSON) | `scorers` |
| `label --serve` (the review UI) | `ui` |
| `capture` | `strands` |
| `ingest --from-langfuse`, `--backend langfuse` | `langfuse` |
| `dataset publish`, `dataset pull`, `online`, `audit abtest`, `audit recommendation`, `audit arms` | `agentcore` |
| `dataset answers`, `dataset build`, `audit report` | none |
| `audit retest` | `stats` |

A command that needs a missing extra stops and prints the command that installs it, written for
how you installed the package:

```text
--8<-- "fixtures/expected/docs/ts-missing-stats.txt"
```

| Installed with | It prints |
| --- | --- |
| `pip` | `python -m pip install "calibrated-agent-evaluation[stats]"`, naming your interpreter |
| `uv pip` | `uv pip install --python PYTHON "calibrated-agent-evaluation[stats]"` |
| `uv sync` in a checkout | `uv sync --inexact --extra stats` (`--inexact` keeps the extras you have) |
| `uv tool install` | `uv tool install "calibrated-agent-evaluation[ui,stats]"`, repeating the extras the tool already has |
| a local wheel | the same command with `"calibrated-agent-evaluation[stats] @ <wheel>"` |
| another editable install | `... install -e ".[stats]"` |

## Extras

| Extra | Adds |
| --- | --- |
| `stats` | checking a judge, the corrected failure rate, the CI gate, and failure-mode clustering |
| `ui` | the trace-review web app (`label --serve`) |
| `langfuse` | Langfuse as the trace store (`--backend langfuse`) |
| `sampling` | the test-input plan (`define`, `generate`) |
| `scorers` | deterministic compare-to-expected scorers |
| `inspect` | reading Inspect AI `.eval` logs, and running Inspect agents and judge models |
| `otel` | reading binary OTLP protobuf (`.otlp`); OTLP-JSON needs no extra |
| `strands` | running Strands agents programmatically, and `capture` |
| `synth` | multi-turn input synthesis through `strands-agents-evals` |
| `datasets` / `hf` | ingesting local parquet / Hugging Face Hub datasets |
| `probe` | the agent-security benchmark battery |
| `probe-ws` | probing WebSocket (`ws://`, `wss://`) endpoints |
| `bedrock` | the Bedrock text renderer and Titan embedder (`generate`, `cluster`) |
| `agentcore` | the Amazon Bedrock AgentCore plug-in; see [Bedrock AgentCore](../integrate/agentcore.md) |

There is no `all` extra. In a checkout, `uv sync --all-extras` installs every one.

The adapters for third-party SDKs (Langfuse, Inspect, Strands, Bedrock, AgentCore) live in their
own packages at the edge of the library. The code that computes rates and runs the gate imports no
vendor type, so you can swap a trace store or an executor without changing a number.

## Bedrock model IDs and Regions

The Bedrock defaults in these docs, such as `global.anthropic.claude-sonnet-4-6`, are cross-Region
inference profiles. An ID that starts with `us.`, `eu.` or `apac.` keeps processing inside that
geography; a `global.` profile routes to supported commercial Regions worldwide. Judge, renderer
and agent prompts carry trace content, so if you have data-residency requirements, pass an
in-Region model ID or the geographic profile that fits. See
[Supported Regions and models for inference profiles](https://docs.aws.amazon.com/bedrock/latest/userguide/inference-profiles-support.html).

## Licensing

The package is Apache-2.0. The default install is permissive (BSD, MIT, Apache). The extras are
too, apart from two unmodified MPL-2.0 dependencies: certifi (required by httpx and requests) and
tqdm (required by huggingface-hub). Nothing in the dependency tree is GPL, LGPL or AGPL, and a
license scan in CI fails the build if that changes.

## Next

[The walkthrough](../walkthrough/index.md) runs the whole loop on the sample agent in a few minutes.
