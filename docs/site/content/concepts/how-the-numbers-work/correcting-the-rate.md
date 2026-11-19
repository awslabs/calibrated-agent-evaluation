# Correcting the rate

Once you know a judge's TPR and TNR, you can undo its known mistakes. Go back to the
[1,000-trace example](raw-rate.md): the judge flagged 14.8%, and you know from calibration that it
wrongly flags 10% of good traces and catches 90% of bad ones.

1. Take off the false alarms. A judge with TNR 0.90 would flag 10% of traces even if none failed,
   so start from 14.8% - 10% = 4.8%.
2. Scale up for the misses. The judge only catches 90% of failures, and each point of real failure
   rate adds 0.90 - 0.10 = 0.80 points to the flag rate. So 4.8% / 0.80 = 6%.

That recovers the true 6%. This is the Rogan–Gladen correction (Rogan and Gladen 1978), used for
decades to correct disease prevalence measured with an imperfect test.

```text
corrected = (flagged - (1 - TNR)) / (TPR - (1 - TNR))
          = (0.148 - 0.10) / (0.90 - 0.10)
          = 0.06
```

## The real thing, with its range

TPR and TNR are themselves estimates from a limited number of gold labels, and the flag rate is an
estimate from a limited number of production traces. So the corrected rate is reported with a
range that carries both uncertainties. Here is the library on a 90/90 judge, calibrated on 80
labels per class and run over 1,000 production traces:

```text
--8<-- "fixtures/expected/docs/example-rate-correction.txt"
```

The judge flagged 14.8%. The corrected rate is 6.0%, likely between 0% and 12.6%. The range is
wide because 1,000 traces is still a small sample at a 6% failure rate:

![The judge's flag rate against the corrected rate](../../assets/charts/rate-correction-light.svg#only-light)
![The judge's flag rate against the corrected rate](../../assets/charts/rate-correction-dark.svg#only-dark)

The second half of that output is a refusal. When the judge flags fewer traces than its own false
alarm rate would produce (5% flagged against a 10% false-alarm floor), the formula would return a
negative failure rate. The number is refused and the range is still reported.
[When a number is refused](refusals.md) lists every case like this.

## Two samples, two jobs

The correction needs two different sets of traces, and they must not be the same set.

--8<-- "docs/site/content/includes/diagrams/two-samples.md"

- **The calibration set** is the gold-labeled traces from the [previous page](checking-a-judge.md).
  You usually build it by hunting for failures, so it is far more failure-heavy than production.
  That is fine for measuring TPR and TNR, which do not depend on how common failures are.
- **The rate sample** is drawn at random from real traffic and judged, not hand-labeled. Its flag
  rate is what gets corrected. It has to look like production, because its rate is the one you
  report.

If you pass the calibration set as the rate sample, the command refuses with `SampleMisuseError`:
the hunted set's failure rate is whatever you made it, not production's. The rate sample also needs
at least 150 traces. Below that the range was measured to be too narrow (see [Why the range
matters](the-range.md)).

When a judge scores several spans in one trace, those verdicts are not independent: a bad trace
tends to produce several bad spans. Pass the trace ids and the rate sample is counted at its
effective size, the number of traces' worth of independent information it holds (Kish 1965), not
the number of verdicts.

## When the judge is too weak to correct

Look at the denominator in the formula: TPR - (1 - TNR), which equals TPR + TNR - 1. Statisticians
call it Youden's J (Youden 1950). For the 90/90 judge it is 0.80. For a judge at 0.70 and 0.75 it is
0.45, and dividing by a small number magnifies every error in the flag rate. Below J = 0.5 the
corrected point estimate is refused and the range is still reported, which is wide enough to show
how little the judge tells you.

## How the range is computed

Calibrated Agent Evaluation draws many plausible values of TPR, of the false-alarm rate 1 - TNR, and
of the flag rate, each from the spread its sample size allows. It runs each draw through the formula
and keeps the middle 95% of the results. In technical terms, each of the three comes from a Jeffreys
posterior, `Beta(x + 1/2, n - x + 1/2)` (Brown, Cai and DasGupta 2001), and 9,999 draws are used.

A larger worked example, a judge with TPR and TNR of 154/162 on 1,000 traces:

```python
from calibrated_agent_evaluation.stats.rogan_gladen import rg_interval, rogan_gladen

cal_pos = [1] * 154 + [0] * 8    # flagged 154 of 162 true FAILs: TPR 0.951
cal_neg = [1] * 8 + [0] * 154    # flagged 8 of 162 true PASSes: TNR 0.951
rate = [1] * 150 + [0] * 850     # flags 150 of 1,000 production traces: 15.0%

print(round(rogan_gladen(0.150, 154 / 162, 154 / 162), 4))
print(rg_interval(cal_pos, cal_neg, rate, seed=0))
```

```text
0.1116
RGInterval(lo=0.06243283370392709, hi=0.14825398042693932, n_draws=9999, n_nonidentifiable=0, eval_design_effect=1.0)
```

The judge flagged 15.0%; the corrected rate is 11.2%, between 6.2% and 14.8%.

One adjustment applies at the edges. When a judge makes no mistakes at all on a class (162 correct
out of 162), or the rate sample has no flags or nothing but flags, the draws above would still put
a little weight away from zero and pull the range off the truth. In those cases the matching edge
of the range is computed with that rate at exactly zero (or one), following the modified Jeffreys
rule of Brown, Cai and DasGupta (2001). The rule only ever widens the range, and away from those
edge counts it changes nothing.

Next: [why the range matters](the-range.md).
