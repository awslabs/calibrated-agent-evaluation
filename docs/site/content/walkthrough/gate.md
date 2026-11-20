# 5. Gate your pull requests

A developer on the OrderBot team rewrites the part of the system prompt that explains the return
window. They try ten refund requests by hand, all ten look right, and they merge. A week later the
refunds queue shows the agent approving returns outside the window more often than before.

Ten requests a person happened to pick could not have shown that. The gate replaces "it looked
fine" with a check on enough runs, drawn from real traffic and scored by the judge that passed its
probation in [step 4](judge.md).

## What the gate decides

=== "CLI"

    On the kit from step 4, after scoring its production sample with the revised judge:

    ```bash
    calibrated-agent-evaluation run --judge demo/quickstart/judge-v2.json --ci
    ```

=== "With Claude Code"

    With the `validate-judge` skill, ask:

    > Score this branch's runs with the refund-policy judge and gate them.

    The skill runs `calibrated-agent-evaluation judge` and `calibrated-agent-evaluation run --ci`
    and reports the gate's result as printed.

<div class="wrap-output" markdown>

```text
--8<-- "fixtures/expected/docs/tutorial-run-ci.txt"
```

</div>

The gate looks at each failure mode on its own and asks whether the mode's pass rate is high
enough, given how many runs there are:

- It counts the judge's verdicts for the mode as they are: here 162 of 200 replies passed, 81%.
- It takes a cautious estimate of that pass rate, the low end of what 200 runs can show. For 162 of
  200 that is about 76%.
- The mode passes when that cautious estimate reaches the bar, 80% unless you set `--threshold`.
  This one does not, so the build fails.
- The build passes only when every gated mode passes. Adding a mode can only make the gate
  stricter.

The line under the bar, "failure rate: 20% (likely between 14% and 26%)", is the corrected rate
from [step 4](judge.md#what-the-judge-is-for), printed because you passed `--judge`. It is the
number for a dashboard, and it does not change the exit code.
[Step 4](judge.md#what-the-judge-is-for) compares it with the kit's true rate.

[Methods](../methods.md) has how the cautious estimate is computed.

## Which verdicts are gated

The gate reads the judge verdicts stored for the runs in scope. Each verdict names the judge
version that made it, and the gate checks that this judge passed both parts of its probation in
step 4: its label check and its bias check are on record in the store. Verdicts from any other
judge, including ones imported with `calibrated-agent-evaluation load` without their records, stop
the run with the judge's name and what it is missing.

With `--judge FILE`, the gate reads only the verdicts that judge made as it is now, and when none of
the stored verdicts came from it (say, all were made by an earlier prompt version), it stops and
asks you to score the runs again. If nothing has been scored, the run fails with the step to take:

```text
--8<-- "fixtures/expected/docs/ts-nothing-to-gate.txt"
```

The gate can also read code scorers' verdicts (`--verdicts scorer`) or human labels
(`--verdicts human`). It gates the source you name, and when that source has no verdicts, it says
so and names the others.

## Only random-sample runs count

The cautious estimate assumes the runs are a random draw. Runs you hunted for because they looked
likely to fail are not one, so the gate reads only traces tagged `production-representative` and
leaves out `discovery` traces, saying how many it left out. Traces from
`calibrated-agent-evaluation generate --run` are tagged `discovery`, so they are not gated.

Build the pull request's input set from real traffic instead: draw it at random once with
`calibrated-agent-evaluation ingest --sample N --seed S`, run the pull request's agent over those
inputs in your own harness, and ingest its traces with `--frame production-representative`.
[Configure the CI job](../how-to/ci-config.md) has the job itself, the exit codes, and how to
carry the certified judge's records into a fresh CI store.

## How many runs each mode needs

Each failure mode needs a minimum number of runs before it can pass at all, even with no failures:

```text
--8<-- "fixtures/expected/docs/integrate-gate-sample-size.txt"
```

A mode with no failures passes from 11 runs. A mode that really passes 90% of the time needs 55
before the gate can see that it clears 80%. A mode that sits at exactly 80% never passes, because
the cautious estimate always sits below the rate itself.

For your threshold and the pass rate you expect, `plan --gate` gives the size:

```bash
calibrated-agent-evaluation plan --gate --threshold 0.8 --expected-pass 0.95
```

<div class="wrap-output" markdown>

```text
--8<-- "fixtures/expected/docs/guide-plan-gate.txt"
```

</div>

The size holds at that number and every larger one, so an input set that grows never drops back
under the bar by chance.

## Keep the input set hard

An input set that every mode clears by a wide margin, run after run, has stopped finding
regressions. When that happens, add the failures review has found since the set was built, or a
fresh random draw of recent traffic. Make each addition in its own pull request, so a change to the
agent and a change to the test never land together.

!!! warning "What a pass means for shipping"
    The gate is a regression check on a fixed input set. A pass means every failure mode cleared
    the bar on that set. The failure rate to quote for production comes from the scheduled sample
    in [step 7](production.md).

## What you have now

A CI step that fails the build and names the failure mode when any mode can't be shown to clear
your bar.

Next: [6. Check that a fix worked](improve.md).
