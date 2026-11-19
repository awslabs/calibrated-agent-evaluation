---
name: run-error-analysis
description: Walk the observe, open-code and cluster error-analysis loop on real agent traces using the `calibrated-agent-evaluation` CLI. Use when a user has an AI agent (single-agent or crew) and wants to find and name its failure modes from real traces before reaching for any metric, the "look at your data" habit. Drives `calibrated-agent-evaluation ingest`, `calibrated-agent-evaluation define`, `calibrated-agent-evaluation generate`, `calibrated-agent-evaluation label` and `calibrated-agent-evaluation cluster`; never scores or gates directly (that is the calibrated-judge path).
allowed-tools: Bash, Read, Write, Edit
---

# run-error-analysis

Help the user do error analysis first: read what the agent did, open-code what went wrong, and cluster the critiques into named failure modes before any metric. This is the Calibrated Agent Evaluation method's primary loop. The skill writes the taxonomy file the pipeline consumes and runs the `calibrated-agent-evaluation` stages over it; the statistics stay in `calibrated-agent-evaluation`.

## What this skill does NOT do

- It does **not** compute or display a score, a rate, or a gate decision. Scoring requires a calibrated judge (`build-binary-judge`, then `validate-judge`); recommend those instead of printing a number.
- It does **not** write a gold label. A human labels in the review surface (`calibrated-agent-evaluation label`); an LLM may only pre-fill.
- It contains no statistics of its own. Every computation is a `calibrated-agent-evaluation` CLI or library call.

## Steps

1. **Orient.** Run `calibrated-agent-evaluation --help` for the stage list and `calibrated-agent-evaluation doctor` for the installed extras. Every stage reads and writes one store: `./.calibrated-agent-evaluation` by default, or the directory given with the global `--root PATH`, which goes before the subcommand (`calibrated-agent-evaluation --root PATH cluster`).
2. **Get traces into the store.** Pick the route that matches what the user has:
   - trace files (OTLP JSON or protobuf, Inspect `.eval`, JSONL, parquet): `calibrated-agent-evaluation ingest PATH`;
   - an instrumented agent that runs in-process: `calibrated-agent-evaluation capture --agent module:callable`;
   - a deployed endpoint the user is authorized to test: `calibrated-agent-evaluation probe --endpoint URL --authorized`, with `--auth-bearer-env ENV_VAR` naming the variable that holds the token. Always pass `--endpoint` and `--authorized`: without `--endpoint`, `calibrated-agent-evaluation probe` asks for its setup interactively when it has a terminal, which blocks an agent's shell.
3. **Frame the inputs (optional).** To cover the input space on purpose, draft a taxonomy with the user and write it as a JSON object of dimension to values, for example `{"feature": ["refunds", "exchanges"], "persona": ["new", "returning"]}`. Then `calibrated-agent-evaluation define --taxonomy taxonomy.json --out frame.json` builds the pairwise covering array, and `calibrated-agent-evaluation generate --frame frame.json --out inputs.jsonl` synthesizes one input per row (offline templates by default; `--model` uses an LLM renderer and needs the `[bedrock]` extra). To route the inputs through the user's agent and store the traces, run `calibrated-agent-evaluation generate --run --frame frame.json --executor strands --agent module:callable`. With `--executor inspect`, also pass `--agent-model`. The agent is the user's code: ask for its `module:callable`, never invent one.
4. **Look at the data.** `calibrated-agent-evaluation label` prints the command that serves the review surface. The human runs `calibrated-agent-evaluation label --serve --rater RATER_ID` with their own rater id (needs the `[ui]` extra), reads each trace, and records PASS or FAIL with a critique and, for a FAIL, the failure mode it shows.
5. **Cluster.** `calibrated-agent-evaluation cluster` reports the ratified failure modes, whether they are saturated (Good-Turing coverage with its lower bound, a Chao1 richness interval) and which taxonomy cells have no failures yet. `calibrated-agent-evaluation cluster --embedder bedrock --out draft.json` also drafts candidate modes from the FAIL critiques; it makes paid Bedrock calls, so ask before running it. A draft is unratified: the human names and ratifies its modes before tagging labels with them.
6. **Hand off.** Once the failure modes are named, recommend `build-binary-judge` to draft a judge for one mode, then `validate-judge` to take it through the calibration gate before it may score or gate anything.

## Guardrails to state to the user

- "10 of 10 passed" says little: zero failures in ten is consistent with a true failure rate near 28%. Recommend a calibrated judge and `calibrated-agent-evaluation run`, which reports each mode's pass rate with its lower bound.
- Saturation is a coverage stop rule that `calibrated-agent-evaluation cluster` computes, not "nothing new in the last 20". Keep labeling while it reports NOT saturated.
- Never present a rate without its confidence treatment; if the user asks for one, route them to the judge and run path.
