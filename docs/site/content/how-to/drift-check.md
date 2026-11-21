# Check for drift

Find out whether production traffic still looks like the traces your judge was checked on.

A judge is checked on one set of traces. If production moves away from that set (longer
conversations, a new product line, more late returns), how often the judge is right may change too.
`calibrated-agent-evaluation run --drift-baseline STORE --drift-feature input-chars|turns|tool-calls`
runs this check on every run (see [Configure the CI job](ci-config.md#compare-with-the-last-release)).
This page calls the library directly, for an input the CLI does not offer as a `--drift-feature`.

## Steps

Pick an input you can compute from every trace, collect it for the calibration traces
and for a production sample, and call `calibrated_agent_evaluation.run.input_drift`. This program does that for the
bundled judge kit, on two dimensions:

```python
--8<-- "docs/site/examples/guides/drift_check.py"
```

```text
--8<-- "fixtures/expected/docs/recipe-drift-check.txt"
```

??? note "The same check, with the numbers behind each verdict"

    Run the program with `--detail` to see each measure and the bar it must clear.

    ```text
    --8<-- "fixtures/expected/docs/recipe-drift-check-detail.txt"
    ```

Message length looks the same in both sets. Days since delivery does not: the calibration traces
were picked to find failures, and late returns are a common failure, so they lean late. That is
expected for a hand-picked calibration set, and it is why a judge's failure rate is measured on a
separate random sample.

## Reading the output

The first block is the answer for each input. The detail gives three measures of how far apart the
two samples are. Two of them can raise a flag: `psi` compares how the values spread across ten
bins, and `wasserstein` measures how far the values moved. A measure flags only when the difference
is larger than chance would produce and also large enough to matter (for `wasserstein`, a shift of
a tenth of the width of the middle half of the baseline's values), so a sample of thousands does not flag a trivial shift. The third,
`jsd`, reads the same bins as `psi` and is shown for reference only. [Methods](../methods.md#drift)
names each test and its source.

To check whether the judge itself changed, rather than its inputs, run
`calibrated-agent-evaluation calibrate --check` against fresh labels (see
[Watch production](../walkthrough/production.md)). In the library,
`calibrated_agent_evaluation.run.judge_drift` compares two calibration records on separate samples
and `judge_drift_paired` compares two runs on the same traces.

## Pitfalls

### A flag on a small sample can be noise

When nothing changed, the check still flags about 6% of the time: between 5.8% and 7.0% across five
kinds of data, each measured over 2,000 random splits of 100 against 100 traces (or 50 against 50).
The committed measurement is `fixtures/expected/drift/null_calibration.txt`. Confirm a flag on a
fresh sample before acting on it.

### Small samples miss small shifts

In the same file, a shift of half a standard deviation between two samples of 100 is flagged about
90% of the time, and a shift of a fifth of a standard deviation about 24% of the time. At 1,000 per side the
smaller shift is flagged 93% of the time. A shift below the bar is not flagged however large the
sample.

### Drift is not failure

A flagged dimension says the inputs changed. Check the judge against fresh human labels on the new
traffic before trusting its rate.

### Each side needs enough traces

With the default 10 bins, fewer than 50 values per side gets a "not measured" note, not a "no
drift" answer.

See the [`calibrated_agent_evaluation.run` reference](../reference/calibrated_agent_evaluation/run/index.md) for the full signatures.
