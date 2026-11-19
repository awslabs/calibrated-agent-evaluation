# 4. Name failure modes

With the `run-error-analysis` skill, ask:

> Cluster my_agent's failures and tell me whether we have seen them all.

The skill runs `calibrated-agent-evaluation cluster` and reads the result back to you. It reports no
rate and no score.

```bash
calibrated-agent-evaluation cluster
```

What you'll see, on the six-trace sample corpus from `calibrated-agent-evaluation quickstart`:

<div class="wrap-output" markdown>

```text
--8<-- "fixtures/expected/docs/calibrated-agent-evaluation-cluster.txt"
```

</div>

The output answers two questions about your FAIL labels.

The first line names the modes found so far and how many labeled failures each has. These names
are what the rest of the guide keys on: a judge detects one mode, and the gate passes or fails
each mode separately.

The second line is the stopping rule. Calibrated Agent Evaluation calls labeling **saturated** when
the 95% lower bound of the Good–Turing estimate (the chance that the next failure is in a mode you
have already named) is at least 0.95, at least 30 failures are labeled, and at least two modes have
been seen. Until then the line reads "keep labeling failures", and you go back to [phase
3](3-look-and-label.md) for another round. A mode seen exactly once is the signal that more modes
are out there.

The third line lists combinations of the [taxonomy](1-define-behavior.md#the-input-taxonomy) where
no failure has been labeled. Each is a blind spot: either the agent is safe there or nobody has
looked. Generate or hunt traces for those cells before you call the codebook done.

??? example "The same run with `--explain`"

    <div class="wrap-output" markdown>

    ```text
    --8<-- "fixtures/expected/docs/calibrated-agent-evaluation-cluster-explain.txt"
    ```

    </div>

    [Error clusters](../../concepts/how-the-numbers-work/error-clusters.md) explains the coverage
    estimate, the Chao1 richness range and the blind-spot list.

## Draft the codebook

Start the codebook from the must-never list in your [expected-behavior
spec](1-define-behavior.md#the-expected-behavior-spec). To find modes the list missed,
`calibrated-agent-evaluation cluster` can embed the FAIL critiques and draft clusters from them.
This makes paid Bedrock calls and needs the `bedrock` extra, so the skill asks before it runs it:

```bash
calibrated-agent-evaluation cluster --embedder bedrock --out draft.json
```

The embedding calls go to Bedrock and have no offline replay, so this guide shows no captured
output for them. The draft lists each cluster under a placeholder name with its critiques.

A draft is unratified. The domain expert reads each cluster's critiques and decides the names:
merge two clusters that are one failure and rename a cluster to the term the spec uses. Only a
ratified codebook may tag a label with a mode, and a changed codebook gets a new version.

A good mode is one a judge can answer as a single yes-or-no question about one run. "Refunded an
order the policy forbids" works. "Poor customer experience" does not: it is several questions, and
a generic quality score can never gate CI.

## Ratify the codebook and tag the labels

Write the decisions as a ratification map: the codebook version, who ratified it, and the mode for
each FAIL already in the store (`docs/site/examples/guides/ratify.json`):

```json
--8<-- "docs/site/examples/guides/ratify.json"
```

```bash
calibrated-agent-evaluation cluster --ratify ratify.json
```

What you'll see, on the sample corpus:

<div class="wrap-output" markdown>

```text
--8<-- "fixtures/expected/docs/guide-cluster-ratify.txt"
```

</div>

Each entry tags that rater's newest FAIL on the trace, and the list of modes is stored as the
codebook for version 1, which `calibrated-agent-evaluation label --serve --codebook-version 1` then
offers in the review form's picker ([phase 3](3-look-and-label.md#name-the-mode-as-you-label)). The
same map settles the modes raters proposed while labeling: `adopt` makes a proposal a mode of the
version, and `merge` folds it into an existing mode and retags its labels ([Label with a
codebook](../review-traces.md#label-with-a-codebook) has the full format). The whole map is checked
before anything is written.

!!! note "If you lead the team"
    The codebook is the domain expert's to own. A renamed, merged or split mode changes what every
    label under it means, which is why it gets a new version and why
    [phase 8](8-production.md#when-to-re-certify) treats a codebook change as a re-certification
    trigger.

## Which modes get a judge

You do not need a judge for every mode. For each one, pick how its verdicts will be produced:

| Verdict source | Use it when | Gate flag |
| --- | --- | --- |
| an LLM judge | the mode needs reading the trace to decide | `--verdicts judge` (the default) |
| a code scorer | the mode is checkable by code: an exact value, valid JSON, a numeric tolerance | `--verdicts scorer` |
| human labels | the mode is rare or high-stakes enough that people label every case | `--verdicts human` |

Code scorers come from `calibrated-agent-evaluation score`, which compares each trace's output with
an expected value and writes one verdict per trace for a named mode, with no model calls. The
expected value is the one a person recorded on a FAIL label for the mode (the form's "expected
correct output" field), or a JSONL file of `trace_id` and `expected`:

```bash
calibrated-agent-evaluation score --scorer exact --mode wrong_reply --key answer --expected expected-replies.jsonl
```

What you'll see, on the sample corpus with `docs/site/examples/guides/expected-replies.jsonl`:

<div class="wrap-output" markdown>

```text
--8<-- "fixtures/expected/docs/guide-score-exact.txt"
```

</div>

The compare-to-expected scorers are `exact`, `levenshtein`, `numeric`, `json-diff` and `valid-json`
(the `scorers` extra). For agents that write code or SQL, two scorers grade by *running* the output
instead: `calibrated-agent-evaluation score --scorer code --expected tests.jsonl` runs the trace's
code against the task's tests in a sandbox, and `--scorer sql` executes the trace's query against a
seeded database and compares result sets to the gold query. `--scorer code --sandbox agentcore` runs
the code on the Bedrock AgentCore Code Interpreter. The default local sandbox is best-effort and
not a security boundary, so prefer `agentcore` when the code comes from a model you do not trust.

Phase 5 builds and certifies the LLM judges.

Next: [5. Build and certify a judge](5-build-and-certify-a-judge.md).
