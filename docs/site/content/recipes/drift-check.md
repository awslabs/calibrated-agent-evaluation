# Check for drift

Find out whether production traffic still looks like the traces your judge was calibrated on.

A judge is certified on one set of traces. If production drifts away from that set (longer
conversations, a new product line, late returns), the judge's measured accuracy may no longer
apply. `calibrated-agent-evaluation run --drift-baseline STORE --drift-feature input-chars|turns|tool-calls` runs this
check on every run against a stored baseline (see [Gate a PR in
CI](../guides/gate-in-ci.md#check-drift-against-the-last-release)). This recipe calls the library
directly, for an input dimension the CLI does not offer as a `--drift-feature`.

## Steps

Pick an input dimension you can compute from every trace, collect it for the calibration traces
and for a production sample, and call `calibrated_agent_evaluation.run.input_drift`. This program does that for the
bundled judge kit, on two dimensions:

```python
--8<-- "docs/site/examples/guides/drift_check.py"
```

```text
--8<-- "fixtures/expected/docs/recipe-drift-check.txt"
```

??? note "The same check, with the numbers behind each verdict"

    Run the program with `--detail` to see each measure, its p-value and the bar it must clear.

    ```text
    --8<-- "fixtures/expected/docs/recipe-drift-check-detail.txt"
    ```

Message length looks the same in both sets. Days since delivery does not: the calibration traces
were hunted for failures, and late returns are a common failure, so they lean late. That is
expected for a hunted calibration set, and it is the reason a judge's rate is measured on a
separate random sample.

## Reading the output

Each input gets three measures of how far apart the two samples are. Two of them can raise a flag:
PSI, which compares how the values spread across ten bins, and a Wasserstein distance, which
measures how far the values moved. A measure flags only when the difference is too large to be
chance, after allowing for checking two measures at once, and also big enough to matter. For the
distance, "big enough" is a shift of a tenth of the typical spread, so a very large sample that
picks up a trivial shift doesn't flag it. The third measure, Jensen–Shannon divergence, is shown
for reference only: it reads the same bins as PSI, so it would count the same evidence twice.

`calibrated_agent_evaluation.run.judge_drift` compares a judge's agreement with human labels between two calibration
records on separate samples, and `judge_drift_paired` compares two runs on the same traces. Both
report the change in how often the judge catches failures and clears passes, each with a range.
`calibrated-agent-evaluation calibrate --check` runs the first from the command line, against fresh labels
([Re-check a certified judge](../guides/calibrate-a-judge.md#re-check-a-certified-judge)).

## Pitfalls

### A flag on a small sample can be noise

On two samples drawn from the same distribution, the check flags PSI or Wasserstein about 6% of
the time: between 5.8% and 7.0% across five kinds of data, each measured over 2,000 random splits
of 100 against 100 traces (or 50 against 50). The committed measurement is
`fixtures/expected/drift/null_calibration.txt`. Confirm a flag on a
fresh sample before acting on it.

### Small samples miss small shifts

In the same file's power check, a shift of half a standard deviation between two samples of 100 is
flagged about 90% of the time, and a shift of 0.2 standard deviations about 24% of the time. At
1,000 per side a 0.2 shift is flagged 93% of the time. A shift below the bar is never flagged
however large the sample.

### Drift is not failure

A flagged dimension says the inputs changed. Check the judge against fresh human labels on the new
traffic before trusting its rate.

### Each side needs enough traces

With the default 10 bins, fewer than 50 values per side is refused rather than reported as no
drift.

See the [`calibrated_agent_evaluation.run` reference](../reference/calibrated_agent_evaluation/run/index.md) for the full signatures.
