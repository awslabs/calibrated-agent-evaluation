# Use the Claude Code skills

The repository is a Claude Code plugin with five skills, so a coding agent can run the loop with
you: review your current eval setup, help you find failure modes, draft a judge and take it through
the checks. Every step of the [walkthrough](../walkthrough/index.md) shows the skill next to the CLI
command in a "With Claude Code" tab.

Each skill works by running `calibrated-agent-evaluation` commands. A skill does not write labels
for you and does not compute any number itself; it reports what the command printed.

## Install

Add the repository as a plugin marketplace from inside Claude Code, then install the plugin:

```text
/plugin marketplace add awslabs/calibrated-agent-evaluation
/plugin install calibrated-agent-evaluation@calibrated-agent-evaluation
```

From a local checkout, pass the checkout path to `/plugin marketplace add` instead, or load the
plugin for one session with `claude --plugin-dir <path-to-checkout>`.

The skills call the CLI, so install the package in the project you are evaluating as well (see
[Install](install.md)). Claude Code names the skills `calibrated-agent-evaluation:<skill>`.

For Kiro, copy each `skills/<name>/` directory from the repository into your project's
`.kiro/skills/` or into `~/.kiro/skills/`.

## The five skills

| Skill | Ask it to | It runs |
| ----- | --------- | ------- |
| `audit-eval-pipeline` | Review how you evaluate today: unchecked judges gating CI, 1 to 5 scores, raw judge counts reported as rates, "10/10 passed" with no range, fixes accepted on the traces they were tuned on | Reads only, then recommends the command or skill that fixes each finding |
| `run-error-analysis` | Find and name your agent's failure modes from real traces, before any metric | `ingest` (files), `capture`, `probe`, `define`, `generate`, `cluster`; prints the `label --serve` command for you to run |
| `build-binary-judge` | Turn one named failure mode into a judge file: one pass/fail question, critique before verdict | Writes the judge file and checks it with `judge --check` |
| `validate-judge` | Find out whether a judge can be trusted to score, gate CI or report a corrected rate | `calibrate`, `bias`, `judge`, `run` |
| `build-annotation-ui` | Get the review app ready so a person can label traces with hotkeys | checks the store and the `ui` extra, then prints the `label --serve` command for you to run |

A test in the repository checks every command and flag the skills mention against the shipped
CLI, so a renamed flag fails the build.

## A first session

Starting from traces you already have:

1. "Audit how we evaluate the refunds agent." `audit-eval-pipeline` reads your eval code and lists
   what to fix.
2. "Ingest last week's traces and help me find the failure modes." `run-error-analysis` ingests your
   exports, prints the command that opens the review app, and runs `cluster` once you have labeled a
   sample.
3. "Write a judge for the trusted_broken_policy_tool failure mode." `build-binary-judge` drafts the judge file.
4. "Can we trust it?" `validate-judge` runs `calibrate` and `bias` and tells you what they decided,
   including how many more labels you need if the judge isn't ready.
