# Installation

Calibrated Agent Evaluation requires Python 3.11.5 or later. The base install has one runtime dependency
(pydantic), and the local SQLite/CSV backend works fully offline with nothing else. Every
capability that needs a third-party package is an opt-in extra, so you install only what your
workflow uses.

## Install

Install the package from PyPI:

```bash
pip install calibrated-agent-evaluation
```

Add extras in brackets, for example the judge statistics and the review UI:

```bash
pip install "calibrated-agent-evaluation[stats,ui]"
```

??? note "Other ways to install"
    === "uv"

        ```bash
        uv add "calibrated-agent-evaluation[stats]"        # into a uv project
        uv tool install calibrated-agent-evaluation        # the CLI alone, in its own environment
        ```

    === "From a checkout"

        ```bash
        git clone https://github.com/awslabs/calibrated-agent-evaluation.git
        cd calibrated-agent-evaluation
        uv sync                      # creates .venv with the base install
        source .venv/bin/activate    # puts `calibrated-agent-evaluation` on your PATH
        ```

        Inside a checkout, prefix a command with `uv run` if you prefer not to activate the
        environment (`uv run calibrated-agent-evaluation doctor`). Some pages run scripts from
        `docs/site/examples/`, which need a checkout.

Verify the install and see what's available:

```bash
calibrated-agent-evaluation --version
calibrated-agent-evaluation doctor      # reports Python, installed extras, and the install line for any you're missing
```

## What each command needs

The base install runs the first commands with no extras at all. Add an extra when you reach the
command that needs it:

| Command | Extras |
| --- | --- |
| `quickstart`, `ingest` (trace files), `label`, `doctor` | none |
| `dump`, `load` | none for traces and labels; `stats` once the store holds run, bias or improve records |
| `run`, `calibrate`, `judge`, `bias`, `improve` | `stats` |
| a live judge model (not a recorded cassette) | `stats` and `inspect` |
| `cluster` | `stats`; `bedrock` for `--embedder bedrock` |
| `define`, `generate` | `sampling` |
| `label --serve` (the review UI) | `ui` |
| `capture` | `strands` |
| `ingest --from-langfuse`, `--backend langfuse` | `langfuse` |

A command that needs a missing extra stops with the exact install command to run.

## Extras

Install extras with the same command as the package, listing every extra you want:

```bash
pip install "calibrated-agent-evaluation[ui,stats,langfuse]"
```

In a checkout, add them with uv, one `--extra` flag each (`uv sync --all-extras` installs every
one). `uv sync` removes any extra you don't list, so pass `--inexact` to add extras and keep the
ones you have (`uv sync --inexact --extra ui`).

When `calibrated-agent-evaluation doctor` or an error message names an extra to install, it prints
the command that matches how you installed the package:

| Installed with | It prints |
| --- | --- |
| `pip` or `uv pip` from PyPI | `uv pip install "calibrated-agent-evaluation[stats]"` |
| `uv sync` in a checkout | `uv sync --inexact --extra stats` (`--inexact` keeps the extras you have) |
| `uv tool install` | `uv tool install "calibrated-agent-evaluation[ui,stats]"` (with `@ <wheel>` for a wheel), repeating the extras the tool already has |
| `uv pip install` of a wheel | `uv pip install "calibrated-agent-evaluation[stats] @ <wheel>"` |
| another editable install | `uv pip install -e ".[stats]"` |

| Extra        | Enables                                                       |
| ------------ | ------------------------------------------------------------- |
| `stats`      | the judge checks, the corrected failure rate, and failure-mode clustering & saturation (`calibrated-agent-evaluation cluster`) |
| `ui`         | the FastHTML trace-review app (`calibrated-agent-evaluation label --serve`)          |
| `langfuse`   | the Langfuse trace-store backend (`--backend langfuse`)       |
| `sampling`   | covering-array discovery frames (`calibrated-agent-evaluation define`, `calibrated-agent-evaluation generate`) |
| `scorers`    | structured-target scorers (exact-match, JSON, numeric)        |
| `inspect`    | read Inspect AI `.eval` logs; run Inspect agents and judge models |
| `otel`       | read binary OTLP protobuf (`.otlp`) exports; OTLP-JSON needs no extra |
| `strands`    | run Strands agents programmatically, and `calibrated-agent-evaluation capture`       |
| `synth`      | multi-turn input synthesis via `strands-agents-evals`         |
| `datasets` / `hf` | ingest local parquet / HuggingFace-hub datasets          |
| `probe`      | the agent-security benchmark battery                          |
| `probe-ws`   | probe WebSocket (`ws://`, `wss://`) endpoints                 |
| `bedrock`    | the Bedrock LLM renderer + Titan embedder (`calibrated-agent-evaluation generate`, `calibrated-agent-evaluation cluster`) |
| `agentcore`  | the Amazon Bedrock AgentCore plug-in (`calibrated_agent_evaluation.agentcore`); see [AgentCore](../integrate/gallery/agentcore.md) |

There is no `all` extra. In a checkout, `uv sync --all-extras` installs every one, which is what
`just docs` and `just flagship` use.

!!! tip "Not sure what you have?"
    `calibrated-agent-evaluation doctor` lists every extra the installed package declares, read from its metadata, as
    installed or not, with the install command for any that are missing. A command that
    needs a missing extra fails with that line, not a bare `ImportError`. If a command still refuses, see
    [Troubleshooting](../guides/troubleshooting.md).

## Bedrock model IDs and Regions

The Bedrock defaults in these docs, such as `global.anthropic.claude-sonnet-4-6`, are cross-Region
inference profiles. A model ID that starts with `us.`, `eu.`, `apac.` or `global.` names a profile,
and Bedrock may process the prompt in a Region other than the one you call. A geographic profile
(`us.`, `eu.`, `apac.`) keeps processing inside that geography. A `global.` profile routes requests
to supported commercial Regions worldwide. Prompts sent to a judge, renderer or agent carry trace
content, so if you have data-residency requirements, pass an in-Region model ID or the geographic
profile that fits them. See
[Supported Regions and models for inference profiles](https://docs.aws.amazon.com/bedrock/latest/userguide/inference-profiles-support.html)
for what each model offers.

## Licensing

Every extra is license-scanned before it can ship. The default install is permissive (BSD / MIT /
Apache); the optional extras are too, apart from two unmodified MPL-2.0 dependencies (certifi and
tqdm). Nothing in the tree is copyleft (GPL / LGPL / AGPL). The package itself is Apache-2.0.

## Next

Run the [Quickstart](quickstart.md) to see the pipeline produce a real result in about five minutes.
