# 7. After the gate

`calibrated-agent-evaluation run --ci` exits 0 when every gated mode clears the bar and 1 when one
does not or when it refuses. With `--fail-on-drift` it exits 3 when the gate did not fail but drift
or a stale judge was flagged ([phase 8](8-production.md#drift)). Each run is recorded under
`<root>/records/run/`, whatever the result.

## The gate passed

Merge. If you want the numbers in a spreadsheet or a report, `calibrated-agent-evaluation results`
writes every stored estimate (bounds, rates, drift, refusals with their reason) to one CSV, a row
each, with its method, level and provenance:

```bash
calibrated-agent-evaluation results --out results.csv
```

What you'll see, on the kit's store after the whole chain:

```text
--8<-- "fixtures/expected/docs/guide-results.txt"
```

## The gate failed

The failing exit code blocks the merge. Do not re-run the job hoping for a pass, and do not lower
`--threshold` to get one. Work through the failure instead.

### 1. Triage the failing mode

The output names each failing mode. `--explain` adds the counts behind each one:

??? example "The kit's failing run with `--explain`"

    <div class="wrap-output" markdown>

    ```text
    --8<-- "fixtures/expected/docs/tutorial-run-ci-explain.txt"
    ```

    </div>

Every judge verdict is stored with the judge's critique. To read the critiques of the FAIL
verdicts, dump the store and open `label.csv`, where those rows have `kind` `llm-judge`:

```bash
calibrated-agent-evaluation dump triage/
```

Read a sample of the failing traces and decide which of two things happened:

- The agent is wrong. The critiques describe real breaches of the spec. Go on to step 2.
- The judge is wrong on this kind of trace. Have the domain expert label the traces in question.
  If the judge disagrees with them often, the judge needs a new version and a fresh calibration
  ([phase 5](5-build-and-certify-a-judge.md#revise-then-certify-on-fresh-labels)). Do not merge on
  the grounds that the judge is probably wrong.

A failing mode can also mean the fixed input set has too few traces of that mode to show a pass
rate of 80% at all ([phase 6](6-gate-in-ci.md#how-big-the-fixed-input-set-must-be) has the minimums). The output
says "label more traces where a mode has too few"; add inputs for that mode to the fixed set.

### 2. Pre-register the fix and its bar

Before you write the fix, register the mode it targets and the smallest drop in its failure rate
that would count:

```bash
calibrated-agent-evaluation improve --register wrong_refund --min-reduction 0.10 --code-ref "$(git rev-parse HEAD)"
```

What you'll see, in the kit:

<div class="wrap-output" markdown>

```text
--8<-- "fixtures/expected/docs/tutorial-improve-register.txt"
```

</div>

`--min-reduction 0.10` is an absolute drop of 10 percentage points. The registration is frozen: a
later evaluation with a different bar is refused. `--code-ref` records the commit of the agent the
target was set against.

!!! note "If you lead the team"
    Sign off the bar before the fix exists. A bar chosen after seeing the result makes almost any
    result a win.

### 3. Fix, keeping your development traces out

While you write the fix you will look at some failing traces and re-run the agent on them. List
their trace ids, one per line, in `dev-ids.txt`. The evaluation refuses a fix measured on any of
them, because a fix tuned on a trace looks better on that trace than it is.

### 4. Evaluate the fix on the same inputs

Run the agent before and after the fix over the same inputs into two stores, score both with the
certified judge, and compare them trace by trace:

```bash
calibrated-agent-evaluation --root before judge --judge judges/wrong_refund.json
calibrated-agent-evaluation --root after judge --judge judges/wrong_refund.json
calibrated-agent-evaluation improve --evaluate wrong_refund \
  --before-root before --after-root after --development dev-ids.txt
```

What you'll see, from an example program that builds the two stores from 40 corpus traces, labels
them for one mode and a guard mode, and has the fix cut 14 failures to 2
(`docs/site/examples/guides/improve_crew.py --single`):

<div class="wrap-output" markdown>

```text
--8<-- "fixtures/expected/docs/guide-improve-single.txt"
```

</div>

The tests behind a verdict, from a second example program that builds the before and after
verdicts directly (40 held-out traces go from 18 failures to 4, for two different fixes):

<div class="wrap-output" markdown>

```text
--8<-- "fixtures/expected/docs/example-improve-guard.txt"
```

</div>

??? note "The test behind each verdict"

    ```text
    --8<-- "fixtures/expected/docs/example-improve-guard-detail.txt"
    ```

A fix is accepted when McNemar's test on the traces whose outcome changed shows the failure rate
fell, the whole interval of the drop clears the registered minimum, and no other mode got worse.
Every other mode is a guard, tested the same way and Holm-corrected together. For a crew,
`--register --crew` targets the crew's end-to-end outcome rate, and `--evaluate` with
`--crew-file CREW.json --touched-agent NAME` adds a guard slice per other agent and per handoff,
so a fix that only moves the failure from one agent to another is rejected, as the second fix
above is ([Evaluate a multi-agent crew](../evaluate-a-crew.md)).
[Did the fix work?](../../concepts/how-the-numbers-work/did-the-fix-work.md) explains the tests,
and [The improve request file](../../recipes/improve-request-file.md) covers verdicts measured
outside the library.

### 5. Re-gate

Push the fix. The pull request's CI job runs the [phase 6](6-gate-in-ci.md) gate again on the
fixed input set, and the merge waits for it to pass.

## When the library refuses

A refusal exits 1 and prints a next step.

| You see | Why | What to do |
| --- | --- | --- |
| `Not enough labels yet to trust this judge` | the calibration round is below the plan | label the per-class count it names, then calibrate again with the judge unchanged |
| calibrate prints REFUSED with enough labels | a lower bound of TPR or TNR is below 0.80 | improve the prompt, bump `prompt_version`, certify on a fresh round |
| `These labels were already used to test a judge for this failure mode` | the calibration set is single-use | label a fresh round with its own cell |
| `Bias check failed` | the verdict moves on a must-pass bias beyond its ceiling | revise the prompt for the named bias, bump the version, certify on a fresh round, rerun `calibrated-agent-evaluation bias` |
| `no calibration, no score` | no certifying calibration or no clear bias record in this store | run phase 5 here, or `calibrated-agent-evaluation load` the judges-only export ([phase 6](6-gate-in-ci.md#export-the-judges-once)) |
| `the judge's prompt changed since it was certified` | the prompt was edited under the same `prompt_version` | bump the version and certify on a fresh round |
| `no recorded verdict (cassette miss)` | replay met a trace the cassette never saw | record the new traces with `--cassette-mode record`, or run the judge live |
| `no judge verdicts with a failure mode ... to gate` | nothing scored in this store | run `calibrated-agent-evaluation judge`, or gate another source with `--verdicts` |
| `failure rate: not estimated (too few scored traces ...)` | under 150 production-representative traces scored | score the number it names with `calibrated-agent-evaluation judge --frame production-representative` |
| `failure rate: not estimated (the judge needs more labels ...)` | the record is rate-unsafe | label what `--explain` names, or plan with `--rate-safe` next time |
| `failure rate: not estimated (the judge isn't accurate enough ...)` | TPR + TNR - 1 is below 0.5 | improve the judge; the raw verdicts and the gate still stand |
| `verdict(s) on hand-picked traces were left out of the rate` | discovery traces were scored | not an error; the rate reads the random sample only |
| agreement check: top up overlap, revise codebook or codebook broken | too few double-coded traces, or the two raters do not agree well enough | follow the table in [phase 3](3-look-and-label.md#double-code-the-calibration-gold) |
| `ConfirmationContaminationError` from `improve` | a paired trace is in `--development` | evaluate on traces the fix was not developed on |
| `improve` refuses a different threshold | the bar was frozen at registration | evaluate against the registered bar, or register a new target |

The rate refusals withhold only the number. The judge still scores and the gate still sets the
exit code. A refusal is not a failing agent: it means the evidence for a verdict or a number is
not there yet. [When the library refuses](../../concepts/how-the-numbers-work/refusals.md) gives the reason
for each rule, and [Troubleshooting](../troubleshooting.md) shows the full output of each refusal.

Next: [8. Run it in production](8-production.md).
