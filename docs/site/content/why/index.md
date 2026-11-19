# The problem, with numbers

Most agent evaluation ends with a line like "the judge failed 15% of traces." Three things can make
that number wrong without anyone noticing. Each section below shows one of them on a real run that
the docs build re-executes, so the output always matches the code.

## The judge's own mistakes end up in the rate

An LLM judge that agrees with your labels 90% of the time sounds trustworthy. Run it over 1,000
traces and it flags 148 of them, so the dashboard says 14.8%. But the judge also wrongly flags
about 1 in 10 good traces, and on a mostly healthy agent those false alarms make up most of what it
flagged.

```text
--8<-- "fixtures/expected/docs/example-rate-correction.txt"
```

??? note "The same run with the statistical detail"

    ```text
    --8<-- "fixtures/expected/docs/example-rate-correction-explain.txt"
    ```

Calibrated Agent Evaluation takes the judge's measured mistakes back out of what it flagged. The
failure rate here is about 6%, which is the true rate of this synthetic sample. The second sample
shows the other case: when the judge flags fewer traces than its own false alarms would produce,
there is no honest single number to report, so only the range is reported.

![The rate a 90% judge reports compared with the true rate](../assets/charts/rate-correction-light.svg#only-light)
![The rate a 90% judge reports compared with the true rate](../assets/charts/rate-correction-dark.svg#only-dark)

## A handful of labels can't vouch for a judge

To check a judge you compare it with human labels: some traces a person marked as failures, some
marked as passes. With too few of each, a judge that looks 90% right could easily be much worse,
and a small set can't tell the difference.

??? note "The program's output, with the statistical detail"

    ```text
    --8<-- "fixtures/expected/docs/example-calibration-sizing.txt"
    ```

A judge right on 18 of 20 labels of each kind is refused. Right on 32 of 35, a little over 90%, it
is accepted. A judge right on 90% of its labels, rounded down, first clears the bar at 48 labels of
each kind, but the chart shows it can dip back under between 51 and 54. So the planner takes the
safe case: at an expected 90% agreement it asks for 55 labels of each kind, the smallest number from
which collecting more can never flip a passing judge back to refused. If you have fewer, the refusal
tells you how many more of each kind to collect.

![Labels per class needed before a judge is accepted](../assets/charts/calibration-size-light.svg#only-light)
![Labels per class needed before a judge is accepted](../assets/charts/calibration-size-dark.svg#only-dark)

## Reading the same verdicts three ways gives different decisions

Take a CI gate that fails a build when more than 20% of traces fail. Here is one illustrative
scenario with two failure modes, read three ways: the raw fraction as a bare number, the raw
fraction with its likely range, and the rate after removing the judge's mistakes.

<div class="wrap-output" markdown>

```text
--8<-- "fixtures/expected/docs/example-decision-flips.txt"
```

</div>

- `trusted_broken_policy_tool`: the bare 16% is under the limit, so it passes. Read with its range,
  the same data could be as high as 21.7%, so it fails. With the judge's mistakes removed, the range
  sits under the limit again and the mode passes.
- `intake/dropped_material_fact`: the bare 50% fails outright. But this judge agrees with the labels
  only 70% of the time, so it isn't trusted, and the gate refuses to decide on its verdicts.

Which reading you use changes the decision. The gate always uses the last one, and refuses when
it can't compute it.

## What the library does about it

- It checks the judge against your labels before the judge scores anything, and tells you how many
  more labels to collect when it can't vouch for it yet.
- It reports failure rates with the judge's error removed and a range, or refuses when the judge
  is too weak.
- It rejects a judge that changes its verdict when an answer gets longer or someone pushes back.
- It grades what the agent did, from its traces. A single agent and a multi-agent crew go through
  the same steps, and for a crew it separates the agent that caused a failure from the one where it
  showed up.

## Who it's for

- You own your agent and can export its traces (every step, tool call and handoff), and want a
  failure you noticed turned into a named, measured, gated failure mode.
- You're handed a black-box endpoint with no source and no traces. The probe samples it and
  says plainly what it can't infer from the outside.

## Next

- [What it costs you](cost.md): labels, extras and credentials at each stage.
- [Compared with alternatives](alternatives.md): how Calibrated Agent Evaluation fits next to
  Inspect, Langfuse, Ragas and others.
- [Quickstart](../getting-started/quickstart.md): see it on the bundled sample corpus.
- [How the numbers work](../concepts/how-the-numbers-work/index.md): the statistics behind each
  section.
