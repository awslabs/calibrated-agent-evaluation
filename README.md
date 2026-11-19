# Calibrated Agent Evaluation

Calibrated Agent Evaluation is a toolkit for error analysis of AI agents. It grades an agent's
observable behavior (a normalized trace) rather than its implementation, and treats single
agents and multi-agent crews the same way.

A custom agent loop, an LLM+MCP+skills composition, a single agent, and a multi-agent crew all
reduce to one normalized trace, so the same pipeline, judges, and gates apply to all of them.
It runs the same on a laptop (no required cloud) as inside a corporate network.

It ships as three things:

- a Python library and CLI (`calibrated-agent-evaluation`) exposing one pipeline:
  `probe → define → generate → label → cluster → calibrate → bias → judge → run → improve`;
- a thin trace-review web UI (FastHTML); and
- a Claude Code skills plugin (Markdown).

It reuses [Langfuse](https://langfuse.com) (trace store) and
[Inspect AI](https://inspect.aisi.org.uk) (execution) for those jobs. It is a library you run
yourself: there is no hosted instance.

Status: version 0.1.0, pre-1.0, so minor releases may still break the API (see
[versioning](https://awslabs.github.io/calibrated-agent-evaluation/versioning/)).
Distribution: `calibrated-agent-evaluation` · CLI: `calibrated-agent-evaluation` ·
import name: `calibrated_agent_evaluation` ·
[documentation](https://awslabs.github.io/calibrated-agent-evaluation/)

## What makes it different

The toolkit takes the statistical steps that raw LLM-judge workflows usually skip:

- A judge is certified only when the 95% lower bound on both its true-positive and true-negative
  rate clears the gate, which is fixed at 0.80, on gold labels that are used once.
- Failure rates are corrected for the judge's error (Rogan–Gladen), with an interval drawn from
  Jeffreys posteriors on TPR, TNR and the measured rate, so it carries the uncertainty of both the
  human gold and the judge verdicts. The correction is refused when Youden's J is too low.
- Good–Turing coverage and related estimators decide when you have labeled enough, and a planner
  sizes the gold set and the CI sample before you label.
- Every reported number carries its interval, and families of comparisons are corrected (BY-FDR or
  Holm).
- For multi-agent crews, the culprit span is kept apart from the span where the failure showed,
  and gates run on the workflow-level outcome rate.

The methodology plane is a single language (Python) with no vendor types in it. The runtime
dependency tree is permissive apart from two unmodified MPL-2.0 dependencies that some extras
pull in (certifi and tqdm), with zero copyleft (GPL/LGPL/AGPL), and it is license-scanned per
optional extra.

What it does not do: it does not label traces for you (a human writes the gold), it does not ship
a hosted service, and it does not gate on subjective qualities such as tone. `calibrated-agent-evaluation prefer`
ranks those, and the ranking is never a gate.

## The pipeline

One subcommand per stage, in workflow order. Error-analysis stages come before scoring:

| Stage       | What `calibrated-agent-evaluation <stage>` does                                                     |
| ----------- | ---------------------------------------------------------------------------- |
| `probe`     | sends prompts to an endpoint you are authorized to test and stores what it shows; `--battery` runs safety benchmarks against it |
| `define`    | turns a taxonomy into a pairwise covering-array discovery frame              |
| `generate`  | renders one input per frame row; `--run` routes them through your agent, `--multi-turn` holds a simulated conversation |
| `label`     | serves the hotkey review UI (`--serve`) where a human labels PASS/FAIL; `--agreement` measures inter-rater agreement on the gold |
| `cluster`   | reports Good–Turing saturation and taxonomy blind spots; drafts modes with `--embedder`, adopts them with `--ratify` |
| `calibrate` | certifies a judge against the human gold (both 95% lower bounds must reach 0.80); `--check` re-checks a certified judge on fresh gold |
| `bias`      | runs a certified judge's CALM bias battery over the probe files you pass     |
| `judge`     | scores traces with a judge that holds a certification and a clear bias run   |
| `score`     | grades traces with a deterministic code scorer (exact, Levenshtein, numeric, JSON) |
| `run`       | the CI gate on each mode's pass-rate lower bound; `--judge` adds the corrected rate, `--drift-baseline` the input-drift report, `--publish` a dashboard |
| `improve`   | pre-registers a fix's target, then accepts or rejects the fix from the before and after stores (McNemar, Holm guard panel) |

Around the stages:

- `quickstart` runs a bundled sample corpus end to end in a temp directory, offline, with no keys.
- `doctor` reports the Python version, the installed extras and whether the store is writable;
  `--agentcore` checks an AWS account is ready for the AgentCore plug-in, read-only.
- `ingest` and `capture` bring your own traces in; `ingest --sample N --seed S` draws a seeded
  production-representative sample.
- `plan --gate` sizes the sample `calibrated-agent-evaluation run` gates on; `results` writes every stored estimate to
  one CSV.
- `prefer` imports pairwise preferences on subjective dimensions and ranks traces.
- `dump` / `load` round-trip the store and records as plain CSV and JSON; `dump --judges-only`
  exports what another store needs to score with your certified judges.
- `dataset` turns the failures your reviewers caught into test scenarios and, with the
  `agentcore` extra, publishes and pulls them as AgentCore evaluation dataset versions.
  `dataset run` holds a simulated conversation per scenario with your agent and can judge each
  one with an AgentCore session-level evaluator.

## Requirements

- Python 3.11.5+ (development targets 3.13; `just matrix` runs the tests on 3.11–3.14 locally,
  and GitHub Actions runs them on 3.11 and 3.14)
- for development, [uv](https://docs.astral.sh/uv/) for environments and dependencies

## Install

```bash
pip install calibrated-agent-evaluation
calibrated-agent-evaluation quickstart   # the bundled sample corpus, end to end, offline, no keys
calibrated-agent-evaluation doctor       # which optional extras are installed

# With the trace-review web UI and the Langfuse trace-store backend
pip install "calibrated-agent-evaluation[ui,langfuse]"
```

With uv, `uv add calibrated-agent-evaluation` (or `uv tool install calibrated-agent-evaluation` for
the CLI alone) works the same way. To work on the code, see
[Quick start (development)](#quick-start-development).

The default install has one runtime dependency, pydantic: the local SQLite/CSV backend works
fully offline. Every capability that needs a third-party package is an opt-in extra, each
license-scanned before it can ship. Common extras:

| Extra        | Enables                                              |
| ------------ | ---------------------------------------------------- |
| `ui`         | the FastHTML trace-review app (`calibrated-agent-evaluation label --serve`) |
| `langfuse`   | the Langfuse trace-store backend (`--backend langfuse`) |
| `stats`      | the calibration engine and failure-mode clustering (scipy, statsmodels, numpy) |
| `scorers`    | structured-target scorers for `calibrated-agent-evaluation score` (autoevals) |
| `inspect`    | read Inspect AI `.eval` logs; run Inspect agents and judge models |
| `strands`    | run Strands agents programmatically, and `calibrated-agent-evaluation capture` |
| `probe`      | the agent-security benchmark battery (`calibrated-agent-evaluation probe --battery`) |
| `bedrock`    | the LLM renderer for `calibrated-agent-evaluation generate` and the Titan embedder for `calibrated-agent-evaluation cluster --embedder` |
| `agentcore`  | the Amazon Bedrock AgentCore plug-in (`calibrated_agent_evaluation.agentcore`, `calibrated-agent-evaluation doctor --agentcore`) and the `strands-agents-evals` failure detector: `failure_vocabulary`, `FailureDetector`, and `DetectorDraftNamer` for draft cluster names |

The other extras are `sampling`, `otel`, `synth`, `datasets`, `hf` and `probe-ws`. See
`pyproject.toml` for the full set and the reasoning behind each pin, or run `calibrated-agent-evaluation doctor`.

## Quick start (development)

```bash
git clone https://github.com/awslabs/calibrated-agent-evaluation.git
cd calibrated-agent-evaluation

# Every extra plus the dev and corpus groups (installs Python 3.13 if needed)
just install

# Run the CLI
uv run calibrated-agent-evaluation --help

# Seed the shared fixture corpus into the local backend, then walk a stage
uv run python scripts/seed_fixture.py --backend local
uv run calibrated-agent-evaluation cluster
uv run calibrated-agent-evaluation run --ci --verdicts human   # the CI gate over the gold; non-zero if it fails

# The flagship proof: both reference agents through probe, define, generate, label, cluster,
# judge, run and improve, checked against committed oracles
just flagship --local-only  # hermetic; start Langfuse (`just langfuse-up`) and drop
                            # --local-only to add the Langfuse leg

# Common tasks (`just --list` shows the full catalog)
just test         # run the test suite
just lint         # Ruff lint
just fix          # Ruff autofix + format
just typecheck    # Pyright
just check        # lint, format check, types, tests, license scan, id scan
just matrix       # full test matrix across Python versions (nox)
just license-scan # fail on any copyleft node in the resolved tree
```

### Live AWS tests

The live AgentCore and Bedrock tests are opt-in and run from a maintainer's machine with
`just test-agentcore-live`, as the role in [`infra/agentcore-live-test-role.yaml`](infra/README.md).
CI does not run them yet. The manual `integration-aws.yml` workflow is in place, but that role
trusts only an IAM role or user, so the workflow can't assume it. Running the live tests from
GitHub Actions needs a GitHub OIDC provider in the test account and a trust statement for this
repository's `aws-integration` environment.

### Git hooks

Install the pre-commit hooks with:

```bash
just hooks
```

The secrets scan runs a locally installed `gitleaks` binary, so install it first
(`brew install gitleaks`). Commits fail until it is on your `PATH`.

## Skills plugin

The `skills/` directory is a Claude Code / Kiro skills plugin that helps a coding agent drive
the pipeline. A skill produces the artifacts the pipeline consumes (a taxonomy, a judge prompt, a review
session) and then hands off to the enforced `calibrated-agent-evaluation` command that actually gates them; it never
re-implements a gate or bypasses one. The five canon skills: `audit-eval-pipeline`,
`run-error-analysis`, `build-binary-judge`, `validate-judge`, `build-annotation-ui`. See
[`skills/README.md`](skills/README.md).

The repository is its own Claude Code marketplace (`.claude-plugin/marketplace.json`), so install
the plugin from a local checkout. To load it for one session, start Claude Code with `claude --plugin-dir <path-to-checkout>`. To install it, add the checkout as
a marketplace from inside Claude Code:

```text
/plugin marketplace add <path-to-checkout>
/plugin install calibrated-agent-evaluation@calibrated-agent-evaluation
```

The skills run `calibrated-agent-evaluation`, so install the package in the project you evaluate as
well (`pip install calibrated-agent-evaluation`, or `uv add --editable <path-to-checkout>`). Kiro reads the same
`SKILL.md` files: copy each `skills/<name>/` directory into your project's `.kiro/skills/` or into `~/.kiro/skills/`.

## Package map

The methodology plane lives under `src/calibrated_agent_evaluation/`:

| Package        | Responsibility                                                          |
| -------------- | ----------------------------------------------------------------------- |
| `schema/`      | normalized `Trace`/`Span`/`Label` Pydantic models; the serialization contract |
| `backends/`    | trace-store abstraction; local SQLite/CSV + Langfuse adapters           |
| `normalize/`   | trajectory normalizer (Inspect / Strands / OTel → normalized `Trace`)   |
| `ingest/`      | `calibrated-agent-evaluation ingest` / `calibrated-agent-evaluation capture`: files, directories, datasets, Langfuse → store; seeded `--sample` draws |
| `dumpload/`    | the CSV+JSON dump/load registry behind `calibrated-agent-evaluation dump` / `calibrated-agent-evaluation load`        |
| `execute/`     | execution plane: Inspect + Strands executors, judge models, Bedrock renderer and embedder, benchmark tasks, cassette record/replay |
| `stats/`       | statistics primitives (Wilson, Rogan–Gladen, McNemar, agreement, drift, multiplicity) and the coverage studies |
| `calibrate/`   | the judge-calibration engine (the first trust gate), sample-size planners, single-use gold ledger |
| `enforce/`     | tiered lint / enforcement policy layer and judge admission              |
| `sampling/`    | covering-array discovery + stratified production frames                 |
| `cluster/`     | failure-mode clustering, coverage & saturation                          |
| `synth/`       | input synthesis + dedup/diversity guards                                |
| `attribution/` | multi-agent walk-back attribution                                       |
| `judges/`      | judge-execution runtime; workflow/span/process judge levels             |
| `subjective/`  | subjective outcomes: atom decomposition, preference pairs, Bradley–Terry / Elo |
| `scorers/`     | heuristic scorers: exact match, Levenshtein, numeric diff, JSON diff, valid JSON |
| `scenarios/`   | test scenarios and dataset slices built from human FAILs and an answer key (vendor-free) |
| `probe/`       | black-box probing toolkit                                               |
| `bias/`        | CALM bias battery (the second trust gate)                               |
| `run/`         | the CI gate, corrected-rate trend, drift report, staleness and judge re-check records |
| `improve/`     | causal before/after guards, the pre-registration ledger, crew evaluation |
| `review/`      | the FastHTML review web app (optional `[ui]` extra), codebooks, the agreement gate |
| `cli/`         | the `calibrated-agent-evaluation` pipeline CLI                                                  |
| `flagship/`    | the end-to-end proof orchestrator (composes the above; `just flagship`) |
| `agentcore/`   | the opt-in Amazon Bedrock AgentCore plug-in (`[agentcore]`): the account check (`calibrated-agent-evaluation doctor --agentcore`), the window ingest (`calibrated-agent-evaluation ingest --source agentcore`), the Runtime executor (`calibrated-agent-evaluation generate --run --executor agentcore`), dataset publish, pull and simulated runs with session-level judging (`calibrated-agent-evaluation dataset publish`,
`pull`, `run`), and the `strands-agents-evals` failure detector with its vocabulary and draft namer (live model calls capped by `max_live_calls`, 0 by default); imports no AWS SDK until used |

## Toolchain

| Concern            | Tool                                  |
| ------------------ | ------------------------------------- |
| Env & dependencies | uv (+ `uv.lock`)                      |
| Build backend      | Hatchling                             |
| Lint & format      | Ruff                                  |
| Type checking      | Pyright                               |
| Tests & coverage   | pytest, pytest-cov, hypothesis        |
| Git hooks          | pre-commit                            |
| Multi-version tests | nox (`just matrix`, run locally)     |
| Task catalog       | just                                  |
| License scan       | pip-licenses (no copyleft, reviewed MPL) |

## Documentation

The full usage guide, concepts, and API/CLI reference live in the documentation site (built with
MkDocs Material under [`docs/site/`](docs/site/); run `just docs` to serve it locally). For
contributors:

- [`CONTRIBUTING.md`](CONTRIBUTING.md): development setup and the checks a change must pass.
- [`AGENTS.md`](AGENTS.md): a short guide for coding agents working in this repository.
- [`skills/README.md`](skills/README.md): the skills-plugin contract.

## Security

See [CONTRIBUTING](CONTRIBUTING.md#security-issue-notifications) for how to report a security
issue, and [SECURITY.md](SECURITY.md) for the security policy and the security model.

## License

This project is licensed under the Apache-2.0 License. See [LICENSE](LICENSE) and
[NOTICE](NOTICE). Vendored code and third-party fixture data are listed with their copyright
notices and full license texts in [THIRD-PARTY-LICENSES](THIRD-PARTY-LICENSES).
