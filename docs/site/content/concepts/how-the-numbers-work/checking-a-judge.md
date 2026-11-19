# Checking a judge against your labels

Calibration checks a judge's TPR and TNR against human gold labels (pass or fail, with a written
critique), as you would check a thermometer. Only a certified judge writes scores.

## Why 20 labels is not enough

Twenty traces cannot tell a 90% judge from a lucky 75% one, so the gate reads the **Wilson lower
bound**: the lowest TPR and TNR consistent with the labels at 95% confidence. Both must reach
0.80, fixed in the code (a low TNR wrecks the rate, see the [previous page](raw-rate.md)).

A judge about 90% right, by labels per class:

| Labels per class | Correct | Lower bound | Result |
| --- | --- | --- | --- |
| 20 | 18 | 0.738 | refused |
| 35 | 32 (91.4%) | 0.804 | certified (31 correct, exactly 90%: 0.768, refused) |
| 48 | 43 | 0.801 | exactly 90% first clears |
| 51 to 54 | 45 to 48 | 0.788 to 0.799 | the dip: refused again |
| 55 | 49 | 0.803 | the plan: clears at every larger size too |

90% of some sizes rounds down, so a plan of 48 would let three more labels un-certify the judge.
The planner picks the smallest size from which every larger one clears:

```text
--8<-- "fixtures/expected/docs/example-calibration-sizing.txt"
```

## How many labels to collect

`plan_calibration` returns the larger of the certification size and the **coverage floor**, the
smallest size at which the corrected rate's range was measured to hold (35 at 0.90, so the plan
is 55). Refusals print how many more labels each class needs.

![Labels needed to certify a judge](../../assets/charts/calibration-size-light.svg#only-light)
![Labels needed to certify a judge](../../assets/charts/calibration-size-dark.svg#only-dark)

| `plan_calibration(accuracy, 0.80)` | 0.85 | 0.88 | 0.90 | 0.92 | 0.95 | 0.98 | 0.99 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| To certify | 208 | 86 | 55 | 42 | 28 | 20 | 20 |
| The plan | 208 | 86 | 55 | 80 | 80 | 130 | 162 |

Above 0.90 the floor wins: an accurate judge's error rates rest on two or three mistakes
([why](refusals.md#not-enough-labels-for-a-corrected-rate)).

## The labels the judge is checked on

Calibration traces form a **locked test set**:

| Rule | What Calibrated Agent Evaluation enforces |
| --- | --- |
| Single use | Used up at the first check. A recheck after a prompt change (even as a copied judge file) is refused, as is a set reusing its traces. A set refused as too small can grow and be rechecked, judge unchanged. |
| No prompt examples | Traces used as examples in the judge's prompt are refused. |
| Human labels only | A model may pre-fill a label in the review UI; only a human-recorded one is gold. |

```text
--8<-- "fixtures/expected/docs/example-discipline-refusals.txt"
```

A certificate covers one model, one prompt (version and text) and one failure mode; change any,
re-calibrate.

## When two people label

`calibrated-agent-evaluation label --agreement --raters A,B` scores rater agreement with
Krippendorff's alpha (`calibrated_agent_evaluation.review.agreement.inter_rater_agreement`) and
exits 0 only when the gold set is usable; [Review
traces](../../guides/review-traces.md#measure-agreement-between-raters) shows a run.

| Agreement (alpha) | What it means for the gold set |
| ----------------- | ------------------------------ |
| lower bound at least 0.80 | usable |
| estimate at least 0.80, lower bound below | label more shared traces, recompute |
| estimate 0.667 to 0.80 | revise the definitions, re-label the disagreements |
| estimate below 0.667 | the definitions are not working |

Judge-to-gold agreement is the TPR and TNR; `calibrated-agent-evaluation calibrate` does not report
Cohen's kappa (`calibrated_agent_evaluation.enforce.display.agreement_display`). Qualities that are
not pass or fail, such as tone, are tracked with pairwise comparisons over time and never gated.

## The formula

Wilson score bound (Wilson 1927), one-sided 95%, *k* correct of *n*, *z* = 1.645:

```text
lower = (p̂ + z²/2n - z × sqrt(p̂(1 - p̂)/n + z²/4n²)) / (1 + z²/n),   p̂ = k/n
```

```python
from calibrated_agent_evaluation.stats import wilson_lower

wilson_lower(49, 55)   # 0.8026
```

Next: [correcting the rate](correcting-the-rate.md).
