---
name: build-binary-judge
description: Draft a binary, critique-first, single-axis LLM-judge file for a named failure mode, ready for the calibration gate. Use when a user has a named failure mode (from error analysis) and wants to turn it into an LLM judge the right way, with one binary axis, the critique before the verdict, and no 1-5 vibe scores. Writes the judge file `calibrated-agent-evaluation calibrate`, `calibrated-agent-evaluation bias`, `calibrated-agent-evaluation judge` and `calibrated-agent-evaluation run` read; recommends `validate-judge` before the judge may gate anything.
allowed-tools: Bash, Read, Write, Edit
---

# build-binary-judge

Help the user write a judge file that the toolkit's enforcement accepts: a binary verdict, a non-empty `critique` ordered before the verdict, and one question per judge. This skill drafts the file; whether the judge may score is decided by `calibrated-agent-evaluation calibrate` and `calibrated-agent-evaluation bias`.

## The rules the judge must satisfy

- **Binary verdict.** `pass` or `fail`, never a 1-5 Likert score. The judge model fills a forced schema whose verdict is `pass` or `fail`, so the file cannot ask for a score.
- **Critique before verdict.** The structured output places a non-empty `critique` before the `verdict`. A file whose `output_fields` put the verdict first is refused before any model call.
- **One axis.** One judge answers one question about one failure mode. If the prompt asks about helpfulness and correctness, that is two judges.
- **Not a generic quality metric.** A generic generation-quality score can never be a CI gate. A judge measures a specific named failure mode.

## The judge file

One JSON object. `failure_mode` is the mode the judge detects, as the human named it in error analysis. `level` is `workflow` (the judge reads the whole trace) or `span` (it reads one crew agent's spans, so `agent` names that agent). Change `prompt_version` whenever the prompt changes: a calibration is pinned to the model and the prompt version, and a new pair needs a new calibration.

```json
{
  "judge_id": "orderbot-refund-resolved",
  "model": "bedrock/global.anthropic.claude-sonnet-4-6",
  "prompt_version": "v1",
  "level": "workflow",
  "prompt_template": "You are grading whether ... Write your critique first, then give a binary verdict: 'pass' if ..., 'fail' if ...",
  "failure_mode": "trusted_broken_policy_tool",
  "agent": null
}
```

`output_fields` is optional and defaults to `["critique", "verdict"]`; leave it out. Any other field is refused.

## Steps

1. **Start from a named failure mode** (from `run-error-analysis` and `calibrated-agent-evaluation cluster`). If none exists, recommend running error analysis first; do not invent a mode.
2. **Draft the judge file** with the user: the question for that one mode, the pass and fail conditions in the prompt, and one or two reference examples quoted from real traces in the store, never made up.
3. **Check the file.** Run the check `calibrated-agent-evaluation calibrate` and `calibrated-agent-evaluation judge` apply when they load a judge. It reads nothing from the store, calls no model and writes nothing:

   ```bash
   calibrated-agent-evaluation judge --check judge.json
   ```

   A missing or unknown field, a bad `level`, a span judge with no `agent`, or a verdict ordered before the critique exits with a typed error. Fix the file and run it again until it prints `judge file ok`. The check cannot read the prompt's intent: a prompt that asks for a score or for two criteria still passes it, so read the prompt yourself and split it if it asks more than one question.
4. **Hand off to calibration.** A judge file that passes the check is still not trusted. Recommend `validate-judge`, which runs `calibrated-agent-evaluation calibrate` against human gold and then `calibrated-agent-evaluation bias`, before the judge may score or gate. Say this explicitly.

## What this skill does NOT do

- It does **not** calibrate the judge or emit a score. That is `validate-judge`, then `calibrated-agent-evaluation judge`. `calibrated-agent-evaluation judge` refuses a judge with no certifying calibration record and no CLEAR bias battery; recommend the calibration step, never bypass it.
- It does **not** run `calibrated-agent-evaluation calibrate` to check the file. Calibration consumes the gold set as a single-use locked test, so a run made only to check the file spends it.
- It contains no statistics. Every check is a `calibrated-agent-evaluation` CLI or library call.
