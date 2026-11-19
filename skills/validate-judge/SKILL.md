---
name: validate-judge
description: Take an LLM judge through the two trust gates of Calibrated Agent Evaluation. `calibrated-agent-evaluation calibrate` certifies a judge against human gold only when the 95% lower bound on both TPR and TNR clears 0.80, and `calibrated-agent-evaluation bias` then runs its bias battery. Use when a user has a binary judge file and wants to know whether it can be trusted to score, gate CI, or report a corrected failure rate. Runs `calibrated-agent-evaluation calibrate`, `calibrated-agent-evaluation bias`, `calibrated-agent-evaluation judge` and `calibrated-agent-evaluation run`; it reports the gates' verdicts and never fabricates a pass.
allowed-tools: Bash, Read, Write, Edit
---

# validate-judge

Help the user run a judge through the trust gates. A judge is not trusted until it is measured against human gold and its confidence bound clears the bar, and then shown not to be biased. This skill sizes the gold set, runs the gates and reports what they decide; **it does not decide the verdict**.

## The calibration gate

- A judge is certified only when the **95% lower confidence bound on BOTH TPR and TNR is at least 0.80**. A point estimate at the threshold does not certify. The two bounds are an intersection-union test: both must clear, so neither needs a multiplicity correction, and each judge's certification stands on its own gold.
- How many gold labels that takes depends on how accurate the judge turns out to be, so **do not hardcode a number; ask the planner**, which inverts the gate's own Wilson bound:

  ```bash
  python -c "from calibrated_agent_evaluation.calibrate import plan_calibration_size; print(plan_calibration_size(0.90, 0.80))"
  ```

  At an expected judge accuracy of `p̂=0.90` against the `τ=0.80` gate this is **55 per class**. Certifying the judge is steep in `p̂` (0.95 needs 28, 0.90 needs 55, 0.85 needs 208), and the planner also sizes for the corrected rate's interval, which needs more at high accuracy: it returns 80 at 0.95, 130 at 0.98 and 162 at 0.99. If the judge is expected to be more accurate on one class than the other, pass both rates: `plan_calibration_size(tpr, 0.80, tnr_hat=tnr)` certifies at the lower and sizes the interval at the higher (a 0.90/0.95 judge needs 80 per class). If two raters will double-code the gold so the agreement check can run, pass `--raters 2` (or `raters=2`): the plan rises to 65 per class at the defaults, because agreement needs 130 shared traces. Plan with the accuracy you actually expect, and if the bound comes in under 0.80, recommend labeling more. Never lower the bar.
- **If the user wants a corrected rate (`calibrated-agent-evaluation run --judge`), plan for rate-safety.** The plan above certifies the judge, but the record is rate-unsafe when the gold is below the coverage floor at the accuracy the judge *showed*. `plan_calibration(p̂, 0.80, rate_safe=True).per_class`, or `calibrated-agent-evaluation calibrate --judge judge.json --rate-safe`, is the smallest size at which every accuracy that would certify meets its floor: **162 per class** from 0.90 up to a perfect judge (208 at 0.85, where certification itself needs more). Tell the user both numbers: the certification-only plan and the rate-safe one.
- Gold labels are **human**-produced. An LLM may pre-fill in the review surface; a human ratifies. Only labels of kind `human` count as gold.
- The gold set is a **single-use locked test**. `calibrated-agent-evaluation calibrate` refuses a second run on the same gold, so decide on `--rate-safe` before the first run, and never run it just to see what happens.
- Reported rates from a certified judge are **Rogan-Gladen prevalence-corrected**, with an interval drawn from Jeffreys posteriors on the judge's TPR and TNR, and the correction is refused when Youden's J is below 0.5. Never present a raw judge count as the failure rate.

## Steps

1. **Check the judge file** with the check in `build-binary-judge`. A file that fails it cannot be calibrated.
2. **Assemble the human gold.** A human labels traces in the review surface (`build-annotation-ui`, `calibrated-agent-evaluation label`): a FAIL for the judge's failure mode, or a PASS. Size the target with the planner above, for the accuracy the user expects, and report the count you have. A judge written for one agent or feature is calibrated on its cell only: `calibrated-agent-evaluation calibrate --judge judge.json --cell feature=refunds`.
3. **Run the calibration gate**, with `--rate-safe` if the user wants a corrected rate. `--expected-accuracy` (and `--expected-tnr` for an asymmetric judge) set the plan the output compares the gold against; the default is 0.90.

   ```bash
   calibrated-agent-evaluation calibrate --judge judge.json --rate-safe
   ```

   Report its verdict verbatim: CERTIFIED, or REFUSED with the bounds and the shortfall it prints. If a CERTIFIED run prints a `rate-unsafe` line, the judge can score, but `calibrated-agent-evaluation run` refuses its corrected rate until the judge is calibrated on more gold. **Do not restate a REFUSED as a soft pass.** It exits 1 on a refusal and writes the record either way.
4. **Run the bias battery** on a certified judge. It needs a directory of probe files and a thresholds file with a ceiling per bias for the judge's model; a must-pass bias with no ceiling fails closed. The format is the one in the Calibrated Agent Evaluation repository's `fixtures/bias/` (its `README.md` describes it); write the user's probes in that format with them, from their domain.

   ```bash
   calibrated-agent-evaluation bias --judge judge.json --probes PROBES_DIR --thresholds THRESHOLDS_FILE
   ```

   Report CLEAR or BIASED and the biases it names.
5. **Score and report.** With a CERTIFIED record and a CLEAR battery, `calibrated-agent-evaluation judge` admits the judge and writes its verdicts, and `calibrated-agent-evaluation run` gates them and reports the corrected rate for the judged mode:

   ```bash
   calibrated-agent-evaluation judge --judge judge.json
   calibrated-agent-evaluation run --judge judge.json --ci
   ```

   The judge's live model needs the `[inspect]` extra. On a refusal at any step, recommend the concrete fix (more gold, a better prompt, a different mode) and a new run on new gold. Never override a gate.

## What this skill does NOT do

- It does **not** compute a bound or a rate itself; the `calibrated-agent-evaluation` CLI and library do. This skill reports their verdicts.
- It does **not** write a gold label or a calibration or bias record. Those come from a human and from `calibrated-agent-evaluation`.
- It does **not** promote a judge a gate refused. The refusal is a feature; recommend the fix and re-run.
