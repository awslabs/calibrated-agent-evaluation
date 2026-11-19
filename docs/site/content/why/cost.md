# What it costs you

Calibrated Agent Evaluation asks for human labels; everything else runs locally. This page lists
what each stage needs from you.

## Labels

The labels are the main cost. These counts come from the planner in the shipped code, so they
change only when the code does.

| What you're doing | Human labels you need |
| --- | --- |
| Find failure modes in your traces | as many as it takes; `calibrated-agent-evaluation cluster` tells you when new labels stop turning up new modes |
| Accept a judge you expect to be right 90% of the time | 55 failures and 55 passes |
| Accept a judge you expect to be right 95% of the time | 80 failures and 80 passes |
| Accept a judge you expect to be right 85% of the time | 208 failures and 208 passes |
| Be sure of a corrected failure rate however accurate the judge turns out to be | 162 of each kind (208 if you expect 85%) |
| The rate sample the judge scores | no labels, but at least 150 randomly drawn traces, separate from the ones you labeled |

A more accurate judge doesn't always need fewer labels. At 95% it would be accepted on 28 of each
kind, but its rarer mistakes make the corrected rate harder to pin down, so the plan asks for 80.
The 162 row covers that: it is enough for a corrected rate at any accuracy from 88% to 99%, so you
don't have to guess the judge's accuracy right before you start labeling.

| Rule | Why |
| --- | --- |
| The labels come from people. | A judge is never checked against another model's verdicts. |
| The traces you label to check the judge are not the traces you measure the rate on. | Labeling usually hunts for failures, so that set is skewed on purpose. The rate comes from a separate random sample: `calibrated-agent-evaluation run` leaves verdicts on the hunted traces out of the rate and says so. |

If you don't know how good your judge is yet, start with the 90% row. When you run
`calibrated-agent-evaluation calibrate` short of the plan, the refusal tells you how many more of
each kind to collect.

## Time

| Stage | Time |
| --- | --- |
| Labeling a trace in the review UI | not measured yet. A timed labeling session hasn't been run. It will measure the median seconds per trace in `calibrated-agent-evaluation label --serve`, for single-agent and crew traces separately, and this row will show that number. |
| Everything Calibrated Agent Evaluation computes | seconds on a laptop. The bundled examples on this site each run in under 10 seconds. |

## Installs, credentials and model calls

| Stage | Extra to install | Credentials | Model calls |
| --- | --- | --- | --- |
| `quickstart`, `ingest` (files), `label`, `doctor` | none | none | none |
| `ingest --from-langfuse` | `langfuse` | Langfuse keys | none |
| `capture` | `strands` | none | whatever your agent makes |
| `label --serve` (the review UI) | `ui` | none | none |
| `define`, `generate` | `sampling` (and `stats` for `generate`) | none (a live renderer needs `bedrock` and AWS) | none by default |
| `cluster` | `stats`, plus `bedrock` for `--embedder` | AWS credentials for `--embedder` | one embedding per FAIL critique with `--embedder`, none otherwise |
| `calibrate`, `bias`, `judge` | `stats`, plus `inspect` for a live judge model | model provider credentials for a live judge; none when replaying a recorded cassette | one per labeled trace (`calibrate`), a few per bias trial, since each check repeats an input (`bias`), one per trace scored (`judge`) |
| `run`, `improve` | `stats` | none | none |

`calibrated-agent-evaluation doctor` shows which extras you have and prints the install line for any
you're missing. [Installation](../getting-started/installation.md) has the full list.

## What the model calls cost

The bundled judge kit ([Calibrate a judge](../guides/calibrate-a-judge.md)) was recorded on Bedrock in October 2026, in two
rounds. Replaying it from the recorded cassette costs nothing.

| What ran | Calls | Tokens in / out | Cost |
| --- | --- | --- | --- |
| Round 1: OrderBot (Claude Sonnet 4.6), 580 customer conversations | not recorded | 1.89M / 0.22M | about $9.0 |
| Round 2: OrderBot (Claude Sonnet 4.6), 460 customer conversations | 1,379 | 1.50M / 0.18M | $7.12 |
| Round 2: the v2 judge (Claude Haiku 4.5), calibration on 324 labels, the bias checks, 200 scored traces | 944 | 0.86M / 0.25M | $2.11 |

The v1 judge's 492 calls were made in round 1 and replayed in round 2; their cost is not broken
out. Round 2 was capped at $15 of live spend before it ran, and the counts come from the recording
run's token usage.

| Judge cost per unit | Value |
| --- | --- |
| Per 1,000 judge calls | about $2.24 |
| Input tokens per call (average) | 906 (855,419 in total) |
| Output tokens per call (average) | 266 (251,511 in total) |

These are short single-agent traces. Longer traces, a crew, or a larger judge model cost
proportionally more.

## What you don't pay for

- No hosted service and no account. The default store is a local SQLite file.
- No network unless you choose a backend or model that needs one. See
  [Run fully offline](../guides/offline.md).
- No copyleft dependencies. Every extra is license-scanned.
