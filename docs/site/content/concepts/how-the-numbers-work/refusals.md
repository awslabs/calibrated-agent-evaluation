# When the library refuses, and why

Each refusal below says what triggered it and what to do; the CLI prints the same next step.

| Refusal | What triggered it | What to do |
| ------- | ----------------- | ---------- |
| Judge not certified | The lower bound on TPR or TNR is below 0.80 | Label more traces of the class that is short (the refusal says how many), or improve the judge |
| Few-shot pool used as a test set | The traces you calibrated on are the judge's prompt examples | Calibrate on a separate locked set |
| Locked test reused | This locked set already certified a judge once | Build a fresh locked set from traces the judge has not been checked on |
| Rate measured on the calibration set | The rate sample is the same set the judge was calibrated on | Score a random sample of production traffic |
| Rate sample too small | Fewer than 150 traces in the rate sample | Score more production traces (the refusal says how many) |
| Not enough labels for a corrected rate | The gold labels per class are below what this judge's accuracy needs for a trustworthy range | Label more, or plan with `--rate-safe` (`rate_safe=True` in the library) next time |
| Judge too weak to correct | TPR + TNR - 1 (Youden's J) is below 0.5 | Use the reported range, or improve the judge |
| Flag rate outside what the judge can produce | The judge flagged fewer traces than its own false-alarm rate, or more than its catch rate | Check the rate sample; the range is still reported |
| Judge biased | The judge changes its verdict when something irrelevant about the input changes | Fix the judge prompt, then re-calibrate and re-run the bias check |

## Not enough labels for a corrected rate

A judge can be certified to score and still not be trusted for a corrected rate. The range from
[the correcting page](correcting-the-rate.md) needs a minimum number of gold labels per class, and
that minimum depends on how accurate the judge turned out to be:

| Judge accuracy shown | 0.85 | 0.88 | 0.90 | 0.92 | 0.95 | 0.98 | 0.99 | up to 0.995 | 0.995 and up |
| -------------------- | ---- | ---- | ---- | ---- | ---- | ---- | ---- | ----------- | ------------ |
| Gold labels per class | 28 | 28 | 35 | 80 | 80 | 130 | 162 | 162 | 11 |

A more accurate judge needs more labels, which looks backwards until you remember that its range is
built from its mistakes. A 98% judge on 50 labels makes about one mistake per class, and one mistake
is too little to estimate an error rate from. The gate checks the size at the accuracy you planned
for and again at the accuracy the judge showed, and the larger applies. A judge certified below this
size has a **rate-unsafe** record: it may still score, and `calibrated-agent-evaluation run` reports
its raw verdicts but refuses the corrected rate.

To be safe whatever accuracy the judge turns out to have, plan with `--rate-safe`
(`rate_safe=True` in the library). That asks for
162 per class at any planned accuracy from 0.88 to 0.99, and 208 at 0.85, where certification alone
needs more.

These sizes were measured by simulation, and a stronger rerun found that the range is still too
narrow in a few cases they admit. [Why the range matters](the-range.md#where-the-promise-does-not-hold)
explains where.

## The bias check

Calibration shows a judge agrees with your labels on ordinary traces. It does not show whether the
judge is swayed by things that should not matter. Before a certified judge may score, it runs a bias
battery (based on the CALM set of biases in Ye et al. 2025): pairs of inputs that differ in one
irrelevant way, such as a longer answer or a user insisting they are right. A judge that changes its
verdict across the pair is showing that bias.

| Bias | Gates scoring? |
| ---- | -------------- |
| length, self-preference, sycophancy | yes, must pass |
| distraction, bandwagon, compassion fade | no, recorded for you to read |
| positional | only for judges that compare two answers; skipped for single-verdict judges |

LLM judges are not perfectly consistent, so some verdicts flip even when nothing changed. To keep
that from counting as bias, every trial comes with a **placebo**: the unchanged input judged again.
The bias is how much more often the judge flips on the changed input than on the placebo. The check
puts a 95% upper bound on that excess and compares it with a threshold set for the judge's model.
A must-pass bias with no threshold configured refuses; no default is borrowed.

A worked number: in 12 trials, a judge flips its verdict 3 times when the answer is made longer and
0 times on the placebo. The excess is 3 in 12, 0.25, and its upper bound is 0.487. Against a
threshold of 0.35 the judge is biased. Twelve trials is not many: a judge with no flips at all in 12
trials still has an upper bound of 0.184, so a threshold below that can never be met. Forty clean
trials bring the bound down to 0.063.

The bound is Newcombe's interval for a paired difference (Newcombe 1998), since a trial and its
placebo are judged on the same item. With no placebo flips it equals the Wilson upper bound on the
trial flip rate.

Counting flips misses one case: a judge that is noisy on the plain answer but always passes the
padded one flips about as often on the placebo as on the trial, so the excess looks like zero.
Length and sycophancy probes therefore also say which verdict the change pushes toward (PASS for a
padded answer, the asserted verdict for sycophancy). The check counts how often the changed input
moves the verdict that way, compares it with how often the placebo makes the same move, and puts
the same upper bound on the excess. That bound must stay under 0.20, set before the test was run
on any judge. The quickstart's v1 judge fails it on both biases (sycophancy moved 2 of 12 trials
toward the asserted verdict, none of the placebos did); v2 moves none of 30.

Next: [the CI gate](ci-gate.md).
