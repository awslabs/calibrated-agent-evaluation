# Claude Code skills

The repository ships a skills plugin for Claude Code (and Kiro) so a coding agent can run the
workflow for you: review your current eval setup, find failure modes in your traces, draft a judge
and take it through calibration. Each skill does its work with the `calibrated-agent-evaluation`
CLI and hands the gated decisions to the command that enforces them. Skills do not write gold labels or compute
statistics; each reports what the CLI decided.

## Install

Install the plugin from a local checkout of the repository. To load it for one session:

```bash
claude --plugin-dir <path-to-checkout>
```

To install it, add the checkout as a marketplace from inside Claude Code:

```text
/plugin marketplace add <path-to-checkout>
/plugin install calibrated-agent-evaluation@calibrated-agent-evaluation
```

The skills run `calibrated-agent-evaluation`, so install the package in the project you are
evaluating too (`pip install calibrated-agent-evaluation`, or `uv add --editable
<path-to-checkout>`). Claude Code names the skills `calibrated-agent-evaluation:<skill>`.

For Kiro, copy each `skills/<name>/` directory from the checkout into your project's
`.kiro/skills/` or into `~/.kiro/skills/`.

## The five skills

| Skill | Ask it to | It runs |
| ----- | --------- | ------- |
| `audit-eval-pipeline` | Review an existing eval setup for practices Calibrated Agent Evaluation refuses: uncalibrated judges used as gates, 1 to 5 scores, raw judge counts reported as rates, "10/10 passed" with no bound, fixes accepted on the traces they were tuned on | Reads only, then recommends the command or skill that fixes each finding |
| `run-error-analysis` | Find and name your agent's failure modes from real traces before any metric | `calibrated-agent-evaluation ingest`, `define`, `generate`, `label`, `cluster` |
| `build-binary-judge` | Turn a named failure mode into a judge file: one pass/fail question, critique before verdict | Writes the judge file, checks it with `calibrated-agent-evaluation judge --check` |
| `validate-judge` | Find out whether a judge can be trusted to score, gate CI or report a corrected rate | `calibrated-agent-evaluation calibrate`, `bias`, `judge`, `run` |
| `build-annotation-ui` | Open the review surface so a person can label traces with hotkeys | `calibrated-agent-evaluation label --serve`; the person labels, never the agent |

A test in the repository checks every `calibrated-agent-evaluation` command and flag the skills mention against the
shipped CLI, so a skill cannot drift from the commands it runs.

## A session

A typical first session with a coding agent, starting from traces you already have:

1. "Audit how we evaluate the refunds agent." The agent runs `audit-eval-pipeline` over your eval
   code and lists what to fix.
2. "Ingest last week's traces and find the failure modes." `run-error-analysis` runs
   `calibrated-agent-evaluation ingest` on your exports, then opens the review surface for you to label a sample and
   runs `calibrated-agent-evaluation cluster` on your labels.
3. "Write a judge for the out-of-window refund failure." `build-binary-judge` drafts the judge
   file.
4. "Can we trust it?" `validate-judge` runs `calibrated-agent-evaluation calibrate` and `calibrated-agent-evaluation bias` and reports what
   they decided, including a refusal and how many more labels it needs.
