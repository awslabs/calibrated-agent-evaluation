<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="docs/site/content/assets/readme/banner-dark.svg">
    <img alt="Calibrated Agent Evaluation" src="docs/site/content/assets/readme/banner-light.svg" width="100%">
  </picture>
</p>

<h3 align="center">Check your LLM judge before you trust its numbers.</h3>

<p align="center">
  <a href="https://pypi.org/project/calibrated-agent-evaluation/"><img alt="PyPI" src="https://img.shields.io/pypi/v/calibrated-agent-evaluation"></a>
  <a href="https://pypi.org/project/calibrated-agent-evaluation/"><img alt="Python versions" src="https://img.shields.io/pypi/pyversions/calibrated-agent-evaluation"></a>
  <a href="https://github.com/awslabs/calibrated-agent-evaluation/actions/workflows/ci.yml"><img alt="CI" src="https://img.shields.io/github/actions/workflow/status/awslabs/calibrated-agent-evaluation/ci.yml?branch=main&label=CI"></a>
  <a href="LICENSE"><img alt="License: Apache-2.0" src="https://img.shields.io/github/license/awslabs/calibrated-agent-evaluation"></a>
</p>

<p align="center">
  <a href="https://awslabs.github.io/calibrated-agent-evaluation/">Documentation</a> ·
  <a href="https://awslabs.github.io/calibrated-agent-evaluation/why/">Why check your judge</a> ·
  <a href="https://awslabs.github.io/calibrated-agent-evaluation/walkthrough/">Walkthrough</a> ·
  <a href="https://awslabs.github.io/calibrated-agent-evaluation/cli/">CLI reference</a> ·
  <a href="CONTRIBUTING.md">Contributing</a>
</p>

---

Most teams grade their AI agent with an LLM judge and ship on what it says, without ever checking
the judge. Calibrated Agent Evaluation has people on your team label a sample of runs, names the
failure modes they find, and measures the judge against those labels. Only a judge that agrees with
your team often enough and passes a bias check can score runs, and `run --ci` refuses verdicts from
any other judge. Those verdicts gate CI per failure mode. The reported failure rate has the judge's
measured mistakes taken out.

It grades what the agent did (a normalized trace), so a custom loop, an LLM with MCP tools, a
Strands agent and a multi-agent crew all go through the same pipeline. It is a library and CLI you
run yourself, offline on a laptop or inside a corporate network. There is no hosted service.

## What it checks that a raw judge score skips

- **Judge certification.** A judge is certified only when the 95% lower bounds on both its
  true-positive and true-negative rates reach 0.80, measured on gold labels that are used once.
- **Corrected failure rates.** The judge's measured rate is corrected for its error
  (Rogan–Gladen), with an interval that carries the uncertainty of the human gold and the judge.
  It is reported beside the CI decision and does not change it. The correction is refused when the
  judge is too close to chance.
- **Enough labels.** Good–Turing coverage tells you when labeling has stopped finding new failure
  modes, and a planner sizes the gold set and the CI sample before you start.
- **Fixes that held.** A fix is accepted against a target registered before the change, with
  McNemar tests and a Holm-corrected panel of guard metrics.
- **Crews.** In a multi-agent run, the span that caused a failure is kept apart from the span
  where it showed.

It does not label traces for you (a human writes the gold). For subjective qualities such as tone,
`calibrated-agent-evaluation prefer` ranks outputs instead of gating them, and `run --ci --atoms
FILE` refuses to gate any mode the file doesn't declare objective.

## Install

```bash
pip install calibrated-agent-evaluation
```

Or `uv add calibrated-agent-evaluation`, or `uv tool install calibrated-agent-evaluation` for the
CLI alone. Python 3.11.5 or later. The base install depends only on pydantic. Everything that needs
a third-party package is an opt-in extra:

```bash
pip install "calibrated-agent-evaluation[ui,stats,langfuse]"
```

| Extra       | Adds                                                                  |
| ----------- | --------------------------------------------------------------------- |
| `ui`        | the trace-review web app for human labeling                           |
| `stats`     | judge calibration, corrected rates and failure-mode clustering        |
| `langfuse`  | Langfuse as the trace store                                           |
| `inspect`   | Inspect AI logs, agents and judge models                              |
| `strands`   | running and capturing Strands agents                                  |
| `bedrock`   | Amazon Bedrock input generation and embeddings                        |
| `agentcore` | the Amazon Bedrock AgentCore plug-in                                  |

`calibrated-agent-evaluation doctor` lists what is installed. The full set is in the
[install guide](https://awslabs.github.io/calibrated-agent-evaluation/how-to/install/).

## Quickstart

The bundled sample corpus runs end to end in a temporary directory, with no keys or network:

```console
$ calibrated-agent-evaluation quickstart
Seeded 6 agent traces + 9 gold labels into an offline local store

First-pass result: 5 of 6 traces have a labeled failure.
Failure modes found:
    - trusted_broken_policy_tool: 2
    - dropped_material_fact: 1
    - misread_amount: 1
    - vacancy_threshold_margin_mishandled: 1
```

`calibrated-agent-evaluation quickstart --certify` goes further. It seeds 848 traces and writes a
judge kit with recorded model replies, then prints the commands that certify the judge, run its
bias battery and gate CI, all offline. Then bring your own traces from
OpenTelemetry (OpenInference or OpenLLMetry), Inspect AI, Strands, Langfuse, JSONL or Parquet:
[connect your agent](https://awslabs.github.io/calibrated-agent-evaluation/integrate/).

## How it works

<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="docs/site/content/assets/readme/pipeline-dark.svg">
    <img alt="The pipeline in order: probe, define, generate, label and cluster for error analysis; calibrate and bias as the two trust gates; judge and run for scoring; improve to test a fix" src="docs/site/content/assets/readme/pipeline-light.svg" width="100%">
  </picture>
</p>

Each stage is one subcommand, run in this order. Error analysis comes first: you look at real
traces, label them, and name the failure modes before anything is scored. A judge then has to pass
both trust gates before `judge` will run it, and `run` fails the build when a failure mode's
pass-rate 95% lower bound falls below the threshold. The
[walkthrough](https://awslabs.github.io/calibrated-agent-evaluation/walkthrough/) covers each stage on one sample agent, and
[methods and citations](https://awslabs.github.io/calibrated-agent-evaluation/methods/) lists the statistics behind it.

## Skills plugin

`skills/` is a Claude Code and Kiro plugin that helps a coding agent drive the pipeline: it drafts
the taxonomy, the judge prompt or the review session, then hands off to the CLI command that
enforces the gate. Load it for one session with `claude --plugin-dir <path-to-checkout>`, or
install it from inside Claude Code:

```text
/plugin marketplace add <path-to-checkout>
/plugin install calibrated-agent-evaluation@calibrated-agent-evaluation
```

For Kiro, see [`skills/README.md`](skills/README.md).

## Documentation

- [Why check your judge](https://awslabs.github.io/calibrated-agent-evaluation/why/), with where it fits next to other tools
- [The walkthrough](https://awslabs.github.io/calibrated-agent-evaluation/walkthrough/), from defining behavior to watching production
- [Put a judge on probation](https://awslabs.github.io/calibrated-agent-evaluation/walkthrough/judge/)
  and [gate your pull requests](https://awslabs.github.io/calibrated-agent-evaluation/walkthrough/gate/)
- [Methods and citations](https://awslabs.github.io/calibrated-agent-evaluation/methods/)
- [Python API](https://awslabs.github.io/calibrated-agent-evaluation/reference/) and
  [CLI](https://awslabs.github.io/calibrated-agent-evaluation/cli/) reference

The project is at 0.1.0. Before 1.0, a minor release may change the API
([versioning policy](https://awslabs.github.io/calibrated-agent-evaluation/versioning/)).

## Contributing

[CONTRIBUTING.md](CONTRIBUTING.md) covers the development setup, the code layout and the checks a
change must pass (`just check`). Coding agents should read [AGENTS.md](AGENTS.md).

## Security

See [CONTRIBUTING](CONTRIBUTING.md#security-issue-notifications) for how to report a security
issue, and [SECURITY.md](SECURITY.md) for the security policy and the security model.

## License

This project is licensed under the Apache-2.0 License. See [LICENSE](LICENSE) and
[NOTICE](NOTICE). Vendored code and third-party fixture data are listed with their copyright
notices and full license texts in [THIRD-PARTY-LICENSES](THIRD-PARTY-LICENSES).
