# Calibrated Agent Evaluation skills plugin

A Claude Code and Kiro skills plugin that helps a coding agent drive the Calibrated Agent
Evaluation pipeline. Each skill writes an input the pipeline reads, or recommends a command, and leaves every
gate to the `calibrated-agent-evaluation` command that enforces it.

## The contract

These skills are **atomic** and **recommendation-based**:

- **They drive the shipped surface, never re-implement it.** Every skill invokes the `calibrated-agent-evaluation` CLI
  or the `calibrated_agent_evaluation` Python package. A skill contains **no statistics** and **no second
  implementation language**: the methodology plane is one language, Python. A skill is
  prose and `calibrated-agent-evaluation` commands.
- **They write the artifacts the pipeline consumes and hand off to the stage that gates them.** A
  skill helps you produce a taxonomy file or a judge file, then recommends the enforced `calibrated-agent-evaluation`
  command that reads it. The skills that write a file have the Write and Edit tools;
  `audit-eval-pipeline` only reads, and `build-annotation-ui` only opens the surface where a human
  writes the labels.
- **They never bypass an enforcement gate.** A skill must not write a gold
  label, emit a gate decision, or present a headline number without its confidence treatment.
  Where a decision is a gated one, the skill **recommends the corresponding `calibrated-agent-evaluation` command** and
  defers to the enforced pipeline. A refusal from that command is the answer to report.

## The five canon skills

Mirrors the error-analysis canon's skill set, adapted for an agent-centric scope:

| Skill | What it helps you do | Hands off to |
|---|---|---|
| `audit-eval-pipeline` | Review an existing eval setup for the anti-patterns the pipeline refuses | the skill or `calibrated-agent-evaluation` command that fixes each finding |
| `run-error-analysis` | Walk observe, open-code and cluster on real traces | `calibrated-agent-evaluation ingest`, `calibrated-agent-evaluation define`, `calibrated-agent-evaluation generate`, `calibrated-agent-evaluation label`, `calibrated-agent-evaluation cluster` |
| `build-binary-judge` | Write a binary, critique-first, single-axis judge file and check it | `calibrated-agent-evaluation judge --check`, then `validate-judge` |
| `validate-judge` | Take a judge through the calibration and bias gates, then score with it | `calibrated-agent-evaluation calibrate`, `calibrated-agent-evaluation bias`, `calibrated-agent-evaluation judge`, `calibrated-agent-evaluation run` |
| `build-annotation-ui` | Open the hotkey review surface for a human to label traces | `calibrated-agent-evaluation label` (the review UI) |

The tests in `tests/test_skills_cli_surface.py` check every `calibrated-agent-evaluation` command these files name
against the shipped parser, so a renamed flag or a missing required option fails the build.

## Distribution

The repository root is a Claude Code plugin (`.claude-plugin/plugin.json`, named `calibrated-agent-evaluation`) and a
marketplace that lists it (`.claude-plugin/marketplace.json`). Claude Code finds the skills in this
`skills/` directory, the default location for a plugin's skills, and names them
`calibrated-agent-evaluation:<skill>`.

Install from a local checkout of the repository. Start Claude Code with
`claude --plugin-dir <path-to-checkout>` to load the plugin for one session, or add the checkout as
a marketplace and install from it:

```text
/plugin marketplace add <path-to-checkout>
/plugin install calibrated-agent-evaluation@calibrated-agent-evaluation
```

For Kiro, copy each `skills/<name>/` directory into `.kiro/skills/` or `~/.kiro/skills/`.
