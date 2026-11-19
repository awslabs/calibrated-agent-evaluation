---
name: build-annotation-ui
description: Stand up the Calibrated Agent Evaluation hotkey trace-review surface so a human can read and label traces quickly (the "look at your data" step). Use when a user is ready to label traces with binary pass/fail and a critique, using hotkeys and auto-advance, for error analysis or to build a gold calibration set. Drives `calibrated-agent-evaluation label` (the browser review surface); the human stays on the gold path, and the skill never labels a trace itself.
allowed-tools: Bash, Read
---

# build-annotation-ui

Help the user open the review surface: hotkeys and auto-advance, so reading 50 traces takes under 30 minutes and error analysis stops being the step teams skip. This skill launches the shipped surface; a human does the labeling. It has no Write tool on purpose: its only artifact is the labels, and those come from the human through the surface.

## Steps

1. **Check the store has traces.** The surface shows the traces in the store (`./.calibrated-agent-evaluation`, or the global `--root PATH`). If it is empty, recommend `run-error-analysis` step 2 to bring traces in (`calibrated-agent-evaluation ingest`, `calibrated-agent-evaluation capture`, or `calibrated-agent-evaluation generate` with `--run`).
2. **Install the UI extra** if `calibrated-agent-evaluation doctor` reports it missing, with the install command `calibrated-agent-evaluation doctor` prints (it matches how Calibrated Agent Evaluation was installed, for example `uv sync --extra ui` in a uv-managed checkout). Without it, serving the surface exits with the same install hint; surface that rather than working around it.
3. **Open the surface.** `calibrated-agent-evaluation label` prints the exact command and URL for the current backend and root. The human runs it with their own rater id: `calibrated-agent-evaluation label --serve --rater RATER_ID`. Global options go before the subcommand, so a Langfuse store is `calibrated-agent-evaluation --backend langfuse label --serve --rater RATER_ID`. `--rater` is required: ask the human for their id rather than inventing one. The server binds `127.0.0.1:5001` unless `--host` or `--port` say otherwise.
4. **Scope the session (optional).** `--cell KEY=VALUE` reviews only traces in one taxonomy cell, `--frame production-representative` only production traces, and `--limit N` at most N traces still to label. A rater resumes where they stopped; `--relabel` includes traces they already labeled.
5. **Label with hotkeys.** The human records PASS or FAIL with a critique per trace, and for a FAIL the failure mode it shows; auto-advance moves to the next trace. For a crew, the human also records attribution (the culprit span and the span where the failure shows).
6. **Use the labels.** Recommend the next step: `calibrated-agent-evaluation cluster` to check failure-mode saturation and coverage, or, for a calibration set, `validate-judge`.

## The gold boundary

- **A human produces gold labels.** An LLM may pre-fill a suggested label through the library, but the human ratifies it in the surface before it enters any gold or calibration set. Never write an unratified LLM label into gold: that invalidates every downstream confidence number.
- The surface writes labels only through the configured trace-store backend; it introduces no statistics and no second language.

## What this skill does NOT do

- It does **not** label, score, or gate. It opens the surface for a human; scoring is the calibrated-judge path (`validate-judge`, then `calibrated-agent-evaluation judge`).
