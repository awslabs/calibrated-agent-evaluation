---
name: audit-eval-pipeline
description: Audit an existing AI-agent eval setup for the anti-patterns Calibrated Agent Evaluation refuses, such as uncalibrated judges used as gates, 1-5 vibe scores, raw judge counts reported as rates, generic generation-quality metrics gating CI, "10/10 passed" claims with no confidence bound, and fixes accepted on the traces they were tuned on. Use when a user wants their current eval approach reviewed against the disciplined method. Recommends the `calibrated-agent-evaluation` command or skill that fixes each gap; it diagnoses and recommends, and never rubber-stamps.
allowed-tools: Bash, Read, Grep, Glob
---

# audit-eval-pipeline

Review a user's existing eval setup against the Calibrated Agent Evaluation method and produce a **findings list with the `calibrated-agent-evaluation` fix for each**. This skill diagnoses and recommends; the enforced fixes live in the `calibrated-agent-evaluation` pipeline.

## The anti-patterns to look for (each maps to an enforced Calibrated Agent Evaluation gate)

| Anti-pattern found | Why it's wrong | Recommend |
|---|---|---|
| An LLM judge used as a gate with no calibration | Certifying a judge against itself makes every number theater | `validate-judge`: `calibrated-agent-evaluation calibrate` certifies only when the 95% lower bound on both TPR and TNR is at least 0.80, an intersection-union test that each judge passes on its own gold |
| 1-5 / Likert "quality" scores | Vibe scoring is the named anti-pattern | `build-binary-judge`: one binary, critique-first axis |
| Raw judge count reported as the failure rate | A 90/90 judge on a true 6% rate reports about 15%. In the toolkit's committed coverage study, a Wilson interval on the raw rate of a 0.90 judge calibrated on 55 per class covers the true 6% rate in 0.105 of 2000 replicates (100 eval traces) | `calibrated-agent-evaluation run --judge FILE` reports the raw rate beside the Rogan-Gladen corrected rate, whose interval draws from Jeffreys posteriors on the judge's TPR and TNR; the same cell covers 0.941 |
| A corrected rate from a judge certified on a small gold set | The gate can certify on fewer labels than the corrected rate's interval needs to cover; a judge right on 32 of 35 already needs 80 per class | `calibrated-agent-evaluation calibrate` marks such a certification rate-unsafe and `calibrated-agent-evaluation run` refuses its corrected rate. `calibrated-agent-evaluation calibrate --judge FILE --rate-safe` plans for it: 162 per class at the 0.80 gate |
| A generic generation-quality metric gating CI | Banned: it doesn't measure a named failure mode | `run-error-analysis`, then a mode-specific calibrated judge |
| "10 of 10 passed" with no bound | Consistent with a true failure rate near 28% | `calibrated-agent-evaluation run`, whose CI gate reads each mode's pass-rate 95% lower bound |
| Watching many slices, flagging any that fails | Watch 20 slices and one fails by chance about 64% of the time | `calibrated-agent-evaluation run --ci` passes only when every mode's lower bound clears the threshold, so adding modes makes the gate stricter. For exploratory per-slice flags, `calibrated_agent_evaluation.run.flags.flag_slices` applies Benjamini-Yekutieli FDR at q = 0.10; `calibrated-agent-evaluation run` computes no drift flags |
| A fix accepted because its target metric moved | A target picked after the fact, or measured on the traces the fix was tuned on, finds an improvement in noise; a fix can also break another mode | `calibrated-agent-evaluation improve --register METRIC --min-reduction X` before the fix exists, then `calibrated-agent-evaluation improve --evaluate METRIC --before-root A --after-root B --development FILE`: a fix evaluated on any of its development traces is refused, and every other mode is a guard slice tested with McNemar, Holm-corrected at 0.05 |
| Evaluate-on-few-shot / re-run after a locked test | Contaminates the estimate | `calibrated-agent-evaluation calibrate` treats the gold as a single-use locked test and refuses a second run on it |

The two coverage figures come from `fixtures/expected/coverage/rg_interval_coverage.json` (cell p̂ 0.90, 55 per class, prevalence 0.06, 100 eval traces). Across that map's 168 planner-reachable cells, no cell of the corrected interval is off 0.95 after Holm across the map, and the raw Wilson interval undercovers in 144. Quote these figures only from that file; re-read it rather than this table if the study is re-run.

## Steps

1. **Gather the setup.** Read the user's eval scripts, judge prompts and CI config (Grep, Glob, Read).
2. **Diagnose.** For each anti-pattern present, record a finding: what it is, why it's invalid, and the exact `calibrated-agent-evaluation` command or skill that fixes it (table above).
3. **Report, don't rubber-stamp.** Produce the findings list. Where the user's setup is already sound, say so; where it isn't, recommend the enforced path. Do **not** declare an eval "good" by inspection: soundness comes from the calibration and run gates, which this audit does not run.

## What this skill does NOT do

- It does **not** emit a score, a gate decision, or a "PASS" for the pipeline. It recommends the `calibrated-agent-evaluation` commands that do, under enforcement. It contains no statistics and no second implementation language.
