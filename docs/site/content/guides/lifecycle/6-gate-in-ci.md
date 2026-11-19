# 6. Gate in CI

Every pull request that touches the agent runs the agent over a fixed set of inputs, scores the
runs with the certified judges, and fails when any failure mode passes too rarely.

With the `validate-judge` skill, ask "score this branch's runs and gate them"; it runs
`calibrated-agent-evaluation judge` and `calibrated-agent-evaluation run --ci` and reports the verdict as printed.

## Export the judges once

The job's store starts empty, and `calibrated-agent-evaluation judge` admits a judge only with its certifying
calibration record and clear bias record in that store. After phase 5, export just those from
your working store and commit the directory. It carries no traces and no labels:

```bash
calibrated-agent-evaluation dump evals/judges --judges-only --judge judges/wrong_refund.json
```

What you'll see, for the kit's v2 judge:

<div class="wrap-output" markdown>

```text
--8<-- "fixtures/expected/docs/guide-dump-judges-only.txt"
```

</div>

A judge the trust chain would not admit is refused here, so a committed export always admits.

## The job

```bash
export CALIBRATED_AGENT_EVALUATION_LOCAL_ROOT=.calibrated-agent-evaluation
calibrated-agent-evaluation load evals/judges                                       # the judges' records, nothing else
calibrated-agent-evaluation generate --run --frame frame.json \
  --executor strands --agent my_agent:build                    # this PR's agent over the fixed inputs
calibrated-agent-evaluation judge --judge judges/wrong_refund.json                  # live judge, every trace in the store
calibrated-agent-evaluation run --judge judges/wrong_refund.json --ci               # exit 0 pass, 1 fail or refusal
```

`calibrated-agent-evaluation load` restores the export into the empty store:

<div class="wrap-output" markdown>

```text
--8<-- "fixtures/expected/docs/guide-load-judges-only.txt"
```

</div>

So `calibrated-agent-evaluation judge` scores this PR's runs and nothing else. A judge whose records do not match the
judge file in the pull request (a changed prompt, model or `prompt_version`) is refused in CI
exactly as it is locally.

What you'll see from the gate, for the kit's v2 judge on its 200-trace sample:

<div class="wrap-output" markdown>

```text
--8<-- "fixtures/expected/docs/tutorial-run-ci.txt"
```

</div>

## What the gate decides

For each failure mode, the gate takes the mode's pass rate over the verdicts in the store and
passes the mode only when the 95% lower bound of that rate is at least 0.80 (`--threshold` changes
the bar). The build passes only when every gated mode passes, so adding a mode can only make the
gate stricter. The kit's run above fails: 162 of 200 passed, 81%, but 200 traces can only show a
pass rate of at least about 76%.

The bar is read from the judge's verdicts as they are. The corrected failure rate printed with
`--judge` is a separate number and does not change the exit code.

One verdict counts per trace, mode and rater: when a judge scores a trace again, its newest
verdict replaces the older one. With `--verdicts human`, a person's PASS on a whole trace counts as
a pass for every gated mode, since it says the run had none of them, and a FAIL for one mode is no
evidence about another.

## How big the fixed input set must be

Each mode needs a minimum number of traces before it can pass at all. For your threshold and the
pass rate you expect the mode to have, `calibrated-agent-evaluation plan --gate` gives the size:

```bash
calibrated-agent-evaluation plan --gate --threshold 0.8 --expected-pass 0.95
```

<div class="wrap-output" markdown>

```text
--8<-- "fixtures/expected/docs/guide-plan-gate.txt"
```

</div>

The size holds at that number and every larger one, so a set that grows never falls back under the
bar by chance. A pairwise frame from `calibrated-agent-evaluation define` can be only a handful of rows, so build the
fixed set from the frame rows plus hunted inputs until each gated mode has at least this many
traces. [The CI gate](../../concepts/how-the-numbers-work/ci-gate.md) explains the bound and why
no multiple-comparison correction is needed.

## Keep the set hard

A set that keeps passing with headroom has stopped telling you much. When every mode clears the
bar by a wide margin run after run, add harder or fresher cases: the failures review found since
the set was built, recent production traces, or taxonomy cells the set does not cover yet. Each
addition is a change to the fixed set, so make it in its own pull request.

## Verdicts and model calls

| Source | Flag | Model calls in CI |
| --- | --- | --- |
| certified LLM judges | `--verdicts judge` (default) | one per trace scored, live or replayed |
| code scorers from `calibrated-agent-evaluation score` | `--verdicts scorer` | none |
| human labels | `--verdicts human` | none |

A source with no verdicts in the store is refused, never swapped for another.

A judge's model call can be replayed from a cassette or made live:

- Replay suits a set whose traces do not change between runs, such as a committed regression store.
  Record the cassette once with `--cassette PATH --cassette-mode record` (it calls the live model
  and overwrites what the file holds), commit it, and pass `--cassette PATH` in the job; replay is
  the default mode. A trace the cassette never saw fails instead of reaching a model.
- A pull request that changes the agent changes its runs, and a changed run is a cassette miss. Run
  the judge live for those: install the `inspect` extra in the job and get model credentials from
  your CI's identity federation (for Bedrock, an assumed IAM role scoped to the judge model), never
  long-lived keys in the repository.

??? failure "A cassette miss"

    ```text
    --8<-- "fixtures/expected/docs/ts-cassette-miss.txt"
    ```

The agent's own model calls can be recorded and replayed the same way with
`calibrated-agent-evaluation generate --run --cassette PATH`, which keeps a run of the unchanged agent offline.

## A committed regression store

For a pull request that changes the inputs, the threshold or the job itself but not the agent, you
can gate a committed store instead of new runs. `calibrated-agent-evaluation dump` writes a store's traces, labels and
records to a directory you commit, and `calibrated-agent-evaluation load` restores it as the job's first step:

```text
--8<-- "fixtures/expected/docs/integrate-ci-seed-and-gate.txt"
```

Dump a store that holds the fixed set's traces and the judges' records but no calibration rounds
and no judge verdicts, so the job scores only the fixed set. Replay the judge from its cassette
there, since the traces do not change.

[Gate a PR in CI](../gate-in-ci.md) has the GitHub Actions and GitLab CI jobs, and how to install
the wheel in the job.

!!! note "If you lead the team"
    The CI gate is a regression check on a fixed input set, chosen to cover the taxonomy. It is not
    the production failure rate: the fixed set is not a random draw from traffic. Quote the rate
    from the scheduled production sample in [phase 8](8-production.md).

Next: [7. After the gate](7-after-the-gate.md).
