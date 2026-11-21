# 3. Name what goes wrong

Before this step, the OrderBot team tracked one number: a "helpfulness" score from an off-the-shelf
judge, averaged over every reply. A prompt change made OrderBot warmer and more apologetic, and the
score went up. Refunds against policy went on at the same rate, because nothing in "helpfulness"
asks whether a refund was allowed. The team shipped the change as an improvement.

A generic score averages every way a reply can be good or bad into one number, so a gain on one
side hides a loss on the other. This step replaces it with named failure modes: specific,
countable ways the agent goes wrong, each one a yes-or-no question about one run. "Refunded an
order the policy forbids" is one. "Poor customer experience" is several questions at once, and no
judge can answer it consistently.

??? example "The same pattern in a captured A/B test report"

    This is a report on a different agent, a SQL assistant on Bedrock AgentCore, from the
    [AgentCore page](../integrate/agentcore.md). Two prompt versions ran side by side:

    <div class="wrap-output" markdown>

    ```text
    --8<-- "fixtures/expected/docs/integrate-agentcore-audit-abtest.txt"
    ```

    </div>

    The generic Helpfulness score rose from 0.71 to 0.78, and the report says the rise is unlikely to
    be chance. Correctness
    barely moved. Neither says which failure, if any, the new prompt fixed. The toolkit prints the
    vendor's numbers as a claim, and the AgentCore page shows the re-test that measures one named
    failure (`missing_date_range`) on traces both versions answered.

## Group the failures

=== "CLI"

    ```bash
    calibrated-agent-evaluation cluster
    ```

    <div class="wrap-output" markdown>

    ```text
    --8<-- "fixtures/expected/docs/calibrated-agent-evaluation-cluster.txt"
    ```

    </div>

=== "With Claude Code"

    > Cluster OrderBot's failures and tell me whether we have seen them all.

    The `run-error-analysis` skill runs `calibrated-agent-evaluation cluster` and reads the result
    back to you.

This is the six-trace sample from `calibrated-agent-evaluation quickstart`. The output answers
three questions.

### Which failures have you seen?

The first line lists the modes named so far and how many labeled failures each has. Everything
after this step keys on these names: a judge looks for one mode, and the gate passes or fails each
mode on its own.

### Should you keep labeling?

The second line answers a question every team asks after the first round: have we seen most of the
ways this agent fails, or will the next batch turn up new ones? The toolkit works it out from how
often each mode has come up. When several modes have been seen only once, new ones are probably
still out there. It calls labeling done when all three of these hold:

- even a cautious estimate says the next failure has at least a 95% chance of being a mode you have
  already named,
- at least 30 failures are labeled,
- at least two different modes have been seen.

Until then the line says "keep labeling failures", and you go back to [step 2](label.md) for
another round. Six traces are far from enough, which is what the sample shows. A team that stops
early writes judges for the failures it happened to see first, and the rest reach customers with no
judge looking for them.
[Methods](../methods.md) has the estimate behind this rule.

### Where has nobody looked?

The third line counts the combinations of your [grid](define.md#the-grid-of-situations) where no
failure has been labeled. Each one is a blind spot: either the agent is safe there or nobody has
tested it. `--explain` lists them. Generate or hunt traces for those cells ([step 1](generate.md))
before you call the list done.

## Draft names from the reasons

Start the list from the "Must never" lines of the [behavior spec](define.md#the-behavior-spec). To
find modes the spec missed, `cluster` can group the written reasons on FAIL labels by meaning and
draft clusters from them:

```bash
calibrated-agent-evaluation cluster --embedder bedrock --out draft.json
```

This makes paid calls to an embedding model on Bedrock (the `bedrock` extra), so the skill asks
before it runs it, and this page shows no captured output for it. The draft lists each cluster
under a placeholder name with its reasons.

A draft is only a suggestion. The person who owns the spec reads each cluster, merges two that are
the same failure, and renames each to the term the spec uses.

## Agree the list

The owner writes the decisions as a ratification file: the list's version, who agreed it, and the
mode for each FAIL already in the store.

```json
--8<-- "docs/site/examples/guides/ratify.json"
```

=== "CLI"

    ```bash
    calibrated-agent-evaluation cluster --ratify ratify.json
    ```

    <div class="wrap-output" markdown>

    ```text
    --8<-- "fixtures/expected/docs/guide-cluster-ratify.txt"
    ```

    </div>

=== "With Claude Code"

    > Write a ratification file that names ORD-1001 as trusted_broken_policy_tool and CLM-4471 as
    > dropped_material_fact, version 1, ratified by alice. Then apply it.

    Claude Code writes the file with you and runs `calibrated-agent-evaluation cluster --ratify`.
    The names and the decision to ratify are yours.

The list is stored as version 1, which the review page then offers when you label with
`--codebook-version 1` ([step 2](label.md#name-the-failure-as-you-label)). The same file settles
modes people proposed while labeling: `adopt` adds a proposal to the version, and `merge` folds it
into an existing mode and retags its labels ([Review traces in the
UI](../how-to/review-ui.md#label-with-a-codebook) has the full format).

!!! warning "Renaming a mode changes what its labels mean"
    A renamed, merged or split mode gets a new version, and every judge checked against the old
    version has to be checked again ([step 7](production.md)).

## Decide who answers each mode

Not every mode needs an LLM judge:

| Who decides | Use it when | Gate with |
| --- | --- | --- |
| an LLM judge | deciding needs someone to read the trace | `--verdicts judge` (the default) |
| code | the mode is checkable mechanically: an exact value, valid JSON, a number within tolerance | `--verdicts scorer` |
| people | the mode is rare or serious enough that a person reviews every case | `--verdicts human` |

`trusted_broken_policy_tool` needs reading the dates against the policy, so OrderBot's team gives
it a judge in [step 4](judge.md). A mode that code can check gets a scorer, which needs no model and
no checking against labels. Here a code check, on an example mode called `wrong_reply`, compares
each reply with an expected answer:

```bash
calibrated-agent-evaluation score --scorer exact --mode wrong_reply --key answer --expected expected-replies.jsonl
```

<div class="wrap-output" markdown>

```text
--8<-- "fixtures/expected/docs/guide-score-exact.txt"
```

</div>

The scorers that compare with an expected value are `exact`, `levenshtein`, `numeric`, `json-diff`
and `valid-json`, and all five need the `scorers` extra. For agents that write code or SQL,
`--scorer code` runs the output against tests and `--scorer sql` runs the query against a seeded
database.

## What you have now

A versioned, agreed list of failure modes, with a decision for each on whether a judge, code or
people will answer it.

Next: [4. Put a judge on probation](judge.md).
