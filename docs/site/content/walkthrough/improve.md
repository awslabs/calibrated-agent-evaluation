# 6. Check that a fix worked

The gate in [step 5](gate.md) has failed on `trusted_broken_policy_tool`. A developer on the OrderBot
team adds a line to the agent's prompt telling it to count the days since delivery itself instead
of trusting the policy tool. They re-run the replies they had been reading while they wrote the
change, and the failures on those replies drop from nearly half to one in ten. They merge.

Two things can make a result like that wrong, and neither shows up in the before and after numbers.

The first is that the fix was measured on the replies it was written against. Every reply the
developer read while writing the change shaped the prompt, and a prompt shaped around a reply does
better on it than on the next one. On new replies the drop is smaller.

The second is that the failures moved. A prompt that makes OrderBot distrust its policy tool can
also make it decline refunds the policy allows, such as a damaged item outside the return window.
That is a different failure mode, and a check that only counts the targeted mode misses it.

`calibrated-agent-evaluation improve` checks a fix against both before it counts.

## Decide what counts as a win, before the fix exists

Register the failure mode the fix targets and the smallest drop in its failure rate that would
count:

```bash
calibrated-agent-evaluation improve --register trusted_broken_policy_tool \
  --min-reduction 0.10 --code-ref "$(git rev-parse HEAD)"
```

<div class="wrap-output" markdown>

```text
--8<-- "fixtures/expected/docs/tutorial-improve-register.txt"
```

</div>

`--min-reduction 0.10` means the failure rate has to fall by at least 10 percentage points. The
evaluation always reads the bar from this registration, and registering the same mode a second time
is refused, so the bar can't be moved after the result is in. `--code-ref` records the commit the
target was set against. Have whoever signs off on releases agree the bar at this point, before the
fix is written.

## Keep the replies you worked on out of the test

While writing the fix you will read failing traces and re-run the agent on them. Write their trace
ids, one per line, to `dev-ids.txt`. The evaluation refuses to count a fix on any trace in that
file, so the test runs on replies the fix was never shaped around.

## Run the same inputs before and after

Run the agent before and after the fix over the same inputs, into two stores, and score both with
the certified judge. The evaluation pairs each input's verdict before with its verdict after:

```bash
calibrated-agent-evaluation --root before judge --judge judges/refund-policy.json
calibrated-agent-evaluation --root after judge --judge judges/refund-policy.json
calibrated-agent-evaluation improve --evaluate trusted_broken_policy_tool \
  --before-root before --after-root after --development dev-ids.txt
```

The OrderBot kit has no before-and-after stores, so the output below comes from an example program
on the ClaimsCrew sample instead. It builds the two stores from 40 corpus traces, records human
labels on them for one claims-handling mode (`wrong_claim_decision`) and a guard mode, and has the
fix cut 14 failures to 2 (`docs/site/examples/guides/improve_crew.py --single`):

<div class="wrap-output" markdown>

```text
--8<-- "fixtures/expected/docs/guide-improve-single.txt"
```

</div>

A fix is accepted when all of these hold:

- The drop is larger than chance would explain. Only inputs whose verdict changed count: an input
  that failed before and passes after is evidence for the fix, and one that passed before and
  fails after is evidence against it.
- Even the least favorable end of the likely improvement clears the bar you registered.
- No other failure mode got worse. Every other mode with verdicts in both stores is checked the
  same way, as a guard.
- When the verdicts come from a judge, a person has labeled every input whose verdict changed. The
  judge's call on exactly those inputs is the one the result rests on, so a human verdict
  replaces it.

[Methods](../methods.md) has the tests behind each rule.

## When the failures moved

Two fixes can produce identical before and after numbers for the agent that was changed and still
deserve different verdicts. In this example program, both take 40 held-out traces from 18 failures
to 4. The first is labeled as an OrderBot fix. The second is a change to the intake agent of
ClaimsCrew, the multi-agent claims workflow that ships with the toolkit:

```text
--8<-- "fixtures/expected/docs/example-improve-guard.txt"
```

The intake agent failed less after its fix, but the crew's end-to-end outcome did not improve. The
failures surfaced at a later step instead of going away. For a crew, register the fix with
`--crew` so the target is the crew's outcome, and evaluate it with `--crew-file CREW.json
--touched-agent NAME`, which adds a guard for each other agent and each handoff.
[Evaluate a multi-agent crew](../how-to/evaluate-a-crew.md) shows the labels this check reads.

## After the fix is accepted

Push it. The pull request's CI job runs the [step 5](gate.md) gate again, and the merge waits for
it. If the fix was measured outside the toolkit, [Write an improve request](../how-to/improve-request-file.md)
covers handing it the paired verdicts directly.

## What you have now

An accepted fix, with its registered bar and evaluation stored next to the traces it was tested on.

Next: [7. Watch production](production.md).
