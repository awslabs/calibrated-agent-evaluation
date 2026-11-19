# 0. Before you start

## Roles

| Role | Who | What they do in this guide |
| --- | --- | --- |
| Agent owner | the engineer who ships the agent | gets traces in, wires the CI job and the scheduled run, writes fixes |
| Domain expert | one person who knows what a correct answer is | owns the expected-behavior spec and the codebook, and labels |
| Second rater | another person with the same domain knowledge | double-codes the calibration labels so agreement can be measured |
| Eval lead | the person who answers for the numbers | sets the gate threshold, signs off pre-registered targets and the re-certification policy |

One person can hold more than one role, except the two raters. The second rater exists so that the
labels a judge is checked against are shown to mean the same thing to two people before anyone
trusts a judge certified on them.

!!! note "If you lead the team"
    The domain expert's time is the cost that decides the schedule. Book it before phase 3, and
    expect the first labeling round to take longer than later ones: the codebook is still moving.

## What it costs

Labels are the main cost. For one failure mode and a judge you expect to be right 90% of the time,
the planner in the shipped code asks for:

| You want | Human labels |
| --- | --- |
| A judge that may score and gate | 55 failures and 55 passes (65 of each when two raters double-code them) |
| The same judge, plus a corrected failure rate you can quote | 162 failures and 162 passes |
| The rate itself | no labels, but at least 150 randomly drawn production traces for the judge to score |

When the second rater double-codes those labels, as [phase 3](3-look-and-label.md) recommends, the
plan can grow so the agreement check has enough shared traces; `calibrated-agent-evaluation
calibrate` prints the size it needs. Each further failure mode with its own judge needs its own
labels. Error analysis (phases 3 and 4) needs as many labels as it takes for
`calibrated-agent-evaluation cluster` to stop finding new modes. [What it costs
you](../../why/cost.md) has the installs and credentials per stage, and the measured model spend of
the bundled judge (about $2.24 per 1,000 judge calls on short traces).

## Install

Install from a checkout ([Installation](../../getting-started/installation.md)), then check what
you have:

```bash
uv sync
source .venv/bin/activate
calibrated-agent-evaluation doctor
```

`calibrated-agent-evaluation doctor` lists the optional extras you have and prints the install line
for each one you are missing. [The extras model](../../concepts/extras.md) says which stage needs
which extra.

Every command reads and writes one store: `./.calibrated-agent-evaluation` unless you set
`CALIBRATED_AGENT_EVALUATION_LOCAL_ROOT` or pass `--root PATH` before the subcommand
(`calibrated-agent-evaluation --root evals/store label`). Pick one path for your agent's working
store now and keep it:

```bash
export CALIBRATED_AGENT_EVALUATION_LOCAL_ROOT=evals/store
```

## The skills

To drive the phases from Claude Code, install the skills plugin ([Claude Code
skills](../../integrate/skills.md)). The skills run the same `calibrated-agent-evaluation` commands
this guide shows. They do not write gold labels or report a pass that a gate did not give.

If you already evaluate the agent some other way, start with the audit:

> Audit how we evaluate my_agent.

`audit-eval-pipeline` reads your eval scripts, judge prompts and CI config and lists each practice
Calibrated Agent Evaluation would refuse (an uncalibrated judge used as a gate, 1 to 5 scores, a raw
judge count reported as a rate, "10 of 10 passed" with no bound), each with the phase of this guide
that replaces it. It runs no `calibrated-agent-evaluation` command and gives no verdict of its own.

Next: [1. Define behavior](1-define-behavior.md).
