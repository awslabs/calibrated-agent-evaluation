# Rank subjective qualities

Some qualities of an answer have no PASS or FAIL, such as whether a refund denial reads as
courteous. You declare which parts of a criterion are objective (judged PASS or FAIL, checked, and
gated with `calibrated-agent-evaluation run`) and which are subjective (compared two traces at a
time and ranked). Rankings are tracked over time; no command fails a build on them.

## Declare the split

The split is one JSON file per criterion, passed with `--atoms` (the toolkit calls it an atom set):

```json
--8<-- "docs/site/examples/guides/prefer-atoms.json"
```

`objective` lists the pass/fail parts, `subjective` the dimensions you compare on. A name may
appear in only one of the two lists, and unknown keys are rejected.

## Import preferences

Each line of a pairs file says which of two stored traces was better on one subjective dimension:

```json
{"trace_id_a": "CLM-4473", "trace_id_b": "CLM-4471", "dimension": "tone", "winner": "b", "judge_kind": "human"}
```

`winner` is `a` or `b`; `judge_kind` is `human`, `llm-judge` or `code`. Import the file:

```text
--8<-- "fixtures/expected/docs/guide-prefer-import.txt"
```

The preferences go to `<root>/records/preference/` and travel with `calibrated-agent-evaluation
dump` and `calibrated-agent-evaluation load`. Importing the same file again adds nothing.

Every line is checked against the `--atoms` file, and one bad line rejects the whole file. A preference
on an objective part is the common mistake:

```text
--8<-- "fixtures/expected/docs/guide-prefer-objective.txt"
```

A trace compared with itself, a dimension the `--atoms` file does not declare, and a trace id the store
does not hold are rejected the same way.

## Record preferences in the review UI

```bash
calibrated-agent-evaluation label --serve --pairwise --atoms atoms.json --rater <your-id> --cell feature=refund_handling
```

The A/B page shows two traces side by side and asks which is better on one dimension. Press `a` (or
`1`) for the left trace, `b` (or `2`) for the right, `s` to skip a pair you cannot call. The queue
is every pair of traces in scope on every subjective dimension, in a fixed shuffled order with the
sides shuffled too. Your choices are stored as human preferences in the same records
`calibrated-agent-evaluation prefer --rank` reads, and a restarted session leaves out the pairs you
already did.

Narrow the scope with `--cell`, `--frame` or `--limit` so both traces answer the same kind of
input. Like the labeling page, it is served on this machine only (see
[Review traces in the UI](review-ui.md)).

## Rank a dimension

```text
--8<-- "fixtures/expected/docs/guide-prefer-rank.txt"
```

Each trace gets a rating from all the stored comparisons on the dimension, with a range. The scale
reads like chess ratings: a 400-point gap means the higher trace wins about 10 times out of 11.
When two traces' ranges overlap, the comparisons don't separate them yet; collect more.

Preferences from different criteria or different judge kinds are ranked separately, so a human's
choices and an LLM judge's never share a ranking.

`--explain` adds the method and a running chess-style rating for comparison:

```text
--8<-- "fixtures/expected/docs/guide-prefer-rank-explain.txt"
```

The running rating is for display only: it changes with the order the preferences were stored in
and has no range. The fitted rating is the one stored. [Methods](../methods.md#ranking-subjective-qualities)
names both.

!!! warning "A trace that never lost is not ranked"
    When a trace won (or lost) every comparison it was in, there is no finite rating for it.
    `calibrated-agent-evaluation prefer --rank` stops with exit 1 for that group and names it.
    Compare it against more traces.

## Next

- Gate the objective parts the usual way: [Put a judge on probation](../walkthrough/judge.md),
  then [Gate your pull requests](../walkthrough/gate.md).
- The flags: [`calibrated-agent-evaluation prefer`](../cli/prefer.md) and
  [`calibrated-agent-evaluation label`](../cli/label.md).
