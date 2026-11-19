# Evaluate a multi-agent crew

In a crew, the agent that causes a failure is often not the one where it shows up. Calibrated Agent
Evaluation records both, so a fix lands on the right agent and a fix that only moves the failure is
caught.

The examples on this page use ClaimsCrew, an insurance-claims crew in the sample corpus: a
supervisor hands a claim to intake, a policy checker, a fraud scorer and payout.

## Where the harm appears and what caused it

On claim CLM-4471 the customer writes that they have been in Florida for the last month, which
matters because the policy excludes damage to a home left empty that long. The intake agent drops
that fact and passes `occupancy_status: null` on. Every later agent does its job correctly with
what it was given, and payout pays a claim it should have denied.

The harm appears at payout. It was caused at intake. Here is the trace, with the culprit drawn
with a solid red border and the span where the harm appeared with a dashed one:

--8<-- "fixtures/expected/docs/crew-span-tree-clm-4471.md"

When you label a crew trace FAIL in the [review surface](review-traces.md), the form asks for two
spans: the one that caused the failure (the culprit) and the one where it showed up (the
manifestation). If the handoff between two agents is to blame, you can name the handoff too.

## Check the labels against the trace

Calibrated Agent Evaluation checks every culprit a reviewer names against the real span tree. A
label is refused if it names a span that isn't in the trace, blames a span that ran after the harm
appeared, or names a handoff the crew doesn't have. This program checks three ClaimsCrew failures:

```python
--8<-- "docs/site/examples/crew_attribution.py"
```

```text
--8<-- "fixtures/expected/docs/example-crew-attribution.txt"
```

CLM-4477 is disputed. Its reviewer could not choose between intake and the policy checker, so
both go forward as culprits and the label says so, instead of one being picked at random.

## Gate the crew on its outcome

The CI gate for a crew reads the outcome of the whole workflow, not each agent's local accuracy. A
crew where every agent is locally correct but the claim is still paid wrongly fails the gate, as it
should. In the library, `calibrated_agent_evaluation.attribution` also computes a corrected failure
rate per agent, for finding where to look.

## A fix that moves the failure

A change to one agent can make that agent look better while the crew does no better: the failure
moves to a later agent. `calibrated-agent-evaluation improve` checks for this. Both fixes below take
a held-out set of 40 traces from 18 failures to 4. The OrderBot fix is accepted. The ClaimsCrew fix
to intake is rejected, because intake fails less but the crew's outcome rate did not improve:

```text
--8<-- "fixtures/expected/docs/example-improve-guard.txt"
```

??? note "The same verdicts, with the test behind them"

    ```text
    --8<-- "fixtures/expected/docs/example-improve-guard-detail.txt"
    ```

The program that produced this is `docs/site/examples/improve_guard.py` in the repository.
[Did the fix work?](../concepts/how-the-numbers-work/did-the-fix-work.md) explains the test.

## Evaluate a crew fix from two runs

To check a real fix, run the same claims through the crew before and after the change, into two
local stores, and label one workflow failure mode in both, such as `wrong_claim_decision`. Each
FAIL names its culprit span as usual. Then describe the crew in a file:

```json
--8<-- "fixtures/expected/docs/guide-improve-crew-file.json"
```

Register the metric with `--crew` before the fix, and evaluate it with the crew file
(`--crew-file`) and the agent you changed (`--touched-agent`). In this example, intake caused 14
of 40 wrong decisions before the fix. After it, intake causes 2 and the policy checker 12, on the
same claims:

```text
--8<-- "fixtures/expected/docs/guide-improve-crew.txt"
```

The output also lists what the guard could not check. Escalation never runs on these claims, so
the reviewer's labels say nothing about it or the handoff into it. Those two slices stay in the
panel at p = 1 and are listed as not measured, so nobody reads them as clean. Judge verdicts leave
more unmeasured: a judge writes no handoff, and an agent with no span judge has no verdicts.

??? note "The same evaluation, with the counts and the test behind it"

    ```text
    --8<-- "fixtures/expected/docs/guide-improve-crew-explain.txt"
    ```

Failures count per trace. A trace counts once for an agent when a FAIL on it blames that agent: a
reviewer's label through its culprit span, a judge's verdict through the span it judged. This is the
same rule `calibrated-agent-evaluation run` uses for per-agent rates. A FAIL whose culprit is
disputed, like CLM-4477's, counts for no agent but still counts as a wrong decision, and the output
names the trace. The program that produced this is `docs/site/examples/guides/improve_crew.py`.

## In the library

Crew attribution is `calibrated_agent_evaluation.attribution`: `build_crew_definition`,
`validate_walkback`, per-agent corrected rates and the relocation check. See the
[`calibrated_agent_evaluation.attribution`
reference](../reference/calibrated_agent_evaluation/attribution/index.md).

## Next

- [Normalized traces](../concepts/normalized-traces.md): why a crew and a single agent look the
  same to the pipeline.
- [Review traces in the UI](review-traces.md): labeling culprit and manifestation spans.
