# Why the range matters

"6%, likely between 0% and 13%" makes a promise: if you repeated the whole exercise many times
(new gold labels, a new rate sample, a new range each time), about 95 of every 100 ranges would
contain the true failure rate. That share is called **coverage**, and a range that claims 95% is
only useful if its coverage really is close to 95%.

## Measuring whether the promise holds

Calibrated Agent Evaluation does not assume its range covers 95% of the time. It was checked by
simulation: pick a true failure rate, simulate a judge, gold labels and a rate sample, compute the
range, and record whether it contained the truth. Repeat 2,000 times and count.

![Coverage of the corrected-rate interval against a naive interval](../../assets/charts/interval-coverage-light.svg#only-light)
![Coverage of the corrected-rate interval against a naive interval](../../assets/charts/interval-coverage-dark.svg#only-dark)

The chart shows a 90% judge, 55 gold labels per class and 1,000 production traces, at seven true
failure rates. The corrected range contains the true rate between 94.25% and 95.6% of the time. A
standard interval drawn around the judge's raw flag rate contains it in at most 0.15% of studies at
failure rates up to 20%, and in about half at 35%. That interval estimates the judge's flag rate,
not the failure rate.

Across the full study, 168 combinations of judge accuracy (0.85 to 0.98), gold-label count,
failure rate (2% to 50%) and rate-sample size (100 to 5,000 traces), the range covered between 93.2%
and 96.2% of the time, and none of the 168 differed from 95% by more than chance allows. A further
840 combinations where the judge is better on one class than the other covered between 92.95% and
96.5%, again none beyond chance.

## Where the promise does not hold

A later, larger test found gaps the first study was too small to see. Rerun with 3,000 simulations
per combination, the range covers less often than 95% in two situations:

- **Very small rate samples.** For one accurate judge (99%, 600 labels per class) with 100
  production traces at a 2% failure rate, coverage fell to about 92%. From 150 traces it holds, so a corrected rate
  from fewer than 150 is refused.
- **A judge with only two or three mistakes per class.** When a class of gold labels holds about two
  or three judge errors, the range is too narrow, with coverage as low as 91% in the worst case. For
  most accuracies this happens below the label counts the planner asks for, so you never see it. Two
  cases are inside what the gate allows: a 98% judge on 130 labels per class and a 99% judge on 260,
  both at failure rates of 5% or less with 100,000-trace rate samples, where coverage was about
  92%.

Nothing in the library fixes the second gap yet. It needs a different way of computing the range
at low error counts, which has not been built. If you report a corrected rate from a 98% or 99% judge on a
failure rarer than 5%, treat the range as a few points of coverage too narrow.

Two more limits apply everywhere:

- Failure rates below 2% were not simulated, so the 150-trace minimum is not validated there.
- The simulation assumes each verdict is an independent coin flip with the judge's TPR or TNR. A
  judge whose mistakes cluster on particular kinds of trace may cover differently.

The exact cells, seeds and coverage values are in the
[reviewer notes](../reviewer-notes.md#how-the-coverage-floor-was-validated-and-what-a-stronger-test-found).

Next: [when a number is refused, and why](refusals.md).
