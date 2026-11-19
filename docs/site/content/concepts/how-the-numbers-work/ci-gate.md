# The CI gate

`calibrated-agent-evaluation run --ci` turns verdicts into a pass or fail for your build. It does not average everything
into one accuracy number. It looks at each failure mode separately and asks: is this mode's pass
rate high enough, given how many traces we have?

```text
--8<-- "fixtures/expected/docs/calibrated-agent-evaluation-run-ci.txt"
```

## One mode, one lower bound

Like [judge calibration](checking-a-judge.md), the gate reads a lower bound, not the raw pass rate.
A mode passes when the 95% lower bound on its pass rate reaches the threshold, 0.80 unless you set
`--threshold`. A mode where 9 of 10 traces passed has a raw rate of 90% and a lower bound of 0.65,
so it fails: ten traces cannot rule out a mode that fails one time in three.

That means each mode needs a minimum number of traces before it can pass at all, even with no
failures:

```text
--8<-- "fixtures/expected/docs/integrate-gate-sample-size.txt"
```

A mode with no failures passes from 11 traces. A mode that really passes 90% of the time needs 55
before the gate can see that it clears 0.80. A mode that truly sits at 0.80 never passes, because
the lower bound of a rate sits below the rate itself.

## Every mode has to pass

The build passes only when every gated mode passes. One failing mode fails the build and names the
mode.

You might expect that gating ten modes means ten chances for a fluke pass, and so a correction for
checking many modes. It does not, because the gate is the opposite of a search for a significant
result. A build passes only if all ten checks pass, and a truly bad mode on its own already gets
through at most 5% of the time. Adding more modes can only make passing harder. This kind of gate is
an intersection-union test (Berger 1982), and it needs no multiple-comparison correction.

--8<-- "docs/site/content/includes/diagrams/ci-gate.md"

## What the gate does not decide on

Some signals sit next to the gate decision without changing it:

- **The corrected-rate trend**, with `--judge FILE`: each judged mode's corrected failure rate, or
  the reason it wasn't reported. `calibrated-agent-evaluation run` prints it.
- **Drift**, with `--drift-baseline STORE --drift-feature F` (`calibrated_agent_evaluation.run.input_drift`): whether
  the traces in scope look different from a baseline store's production sample.
- **Staleness**, with `--drift-baseline` (`calibrated_agent_evaluation.run.check_staleness`): a warning for each judge
  given with `--judge` when input drift is flagged or its calibration no longer certifies it.
  Drift and staleness never flip the result; `--fail-on-drift` turns either into exit code 3.
- **Exploratory flags**, from `calibrated_agent_evaluation.run.flag_slices`: slices worth a look. Because many slices are
  scanned at once, these are corrected for multiple comparisons (Benjamini and Yekutieli 2001, at a
  false-discovery rate of 10%). A flag blocks nothing.

The drift report compares inputs three ways (population stability index, Jensen–Shannon
divergence, and a scaled Wasserstein distance), each with a permutation test. Only the first and
last can raise a flag. Jensen–Shannon divergence is computed on the same bins as the population
stability index, so their tests agree almost every time, and counting both would test the same
thing twice. It is shown for reference. The report deliberately skips the Kolmogorov–Smirnov test,
which flags shifts too small to matter once samples are large, and for the same reason a
significant Wasserstein distance flags only when the shift is at least a tenth of the typical
spread.

Next: [did the fix work?](did-the-fix-work.md)
