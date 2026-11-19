# Why a judge's raw rate is wrong

Take 1,000 production traces where 60 really failed, a true failure rate of 6%. Run a judge over
them that is right 90% of the time on failures and 90% of the time on passes.

|                      | Judge says FAIL | Judge says PASS | Total |
| -------------------- | --------------- | --------------- | ----- |
| Trace really failed  | 54              | 6               | 60    |
| Trace really passed  | 94              | 846             | 940   |
| Total                | 148             | 852             | 1,000 |

The judge flags 148 traces, so it reports a 14.8% failure rate. The true rate is 6%.

The error comes from the bottom row. The judge wrongly flags 10% of the 940 good traces, which is
94 false alarms, and that outweighs the 54 real failures it catches. The rarer the failure, the
larger the pool of good traces producing false alarms, so the further the raw rate drifts from the
truth.

![A 90/90 judge reports 14.8% when the true rate is 6%](../../assets/charts/rate-correction-light.svg#only-light)
![A 90/90 judge reports 14.8% when the true rate is 6%](../../assets/charts/rate-correction-dark.svg#only-dark)

The left panel draws this for every true rate from 0 to 50%. The judge reports too high below
50%, and the two lines cross only at 50%, where false alarms and misses happen to cancel. Most
agent failure modes sit far below 50%, in the region where the raw rate is most wrong. The right
panel shows the fix, which the next pages explain.

## Two numbers describe a judge

The two rows of the table are the two things to know about any judge:

- **TPR** (true-positive rate): of the traces that really failed, the share the judge flags. Here
  54 of 60, or 0.90.
- **TNR** (true-negative rate): of the traces that really passed, the share the judge leaves
  alone. Here 846 of 940, or 0.90.

A single "accuracy" number hides which kind of mistake the judge makes, and the two kinds pull the
reported rate in opposite directions. A judge with TPR 0.99 and TNR 0.80 looks excellent at
catching failures and still reports a rate several times too high when failures are rare.

TPR and TNR are properties of the judge, not of the traffic. They stay the same whether 1% or 40%
of traces fail, which is what makes them worth measuring once and reusing. The raw flag rate does
not have that property.

## The formula

With true failure rate *p*, the share of traces a judge flags is

```text
flagged = TPR × p + (1 - TNR) × (1 - p)
```

For the table above: 0.90 × 0.06 + 0.10 × 0.94 = 0.054 + 0.094 = 0.148.

Next: [checking a judge against your labels](checking-a-judge.md) measures TPR and TNR, and
[correcting the rate](correcting-the-rate.md) runs this formula backwards.
