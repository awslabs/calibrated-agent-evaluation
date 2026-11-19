# Rank subjective qualities

Some qualities of an answer have no PASS or FAIL: whether a denial letter reads as courteous, say.
Calibrated Agent Evaluation keeps those apart from the failure modes it gates on. You declare which
parts of a criterion are objective (judged PASS or FAIL, calibrated, gated with
`calibrated-agent-evaluation run`) and which are subjective (compared two traces at a time and
ranked). Rankings are tracked, never gated: no command fails a build on them.

## Declare the split

An atom set is one JSON file per criterion:

```json
--8<-- "docs/site/examples/guides/prefer-atoms.json"
```

`objective` lists the binary atoms, `subjective` the dimensions you compare on. A name may sit in
one lane only, and unknown keys are refused.

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

Every line is checked against the atom set, and one bad line refuses the whole file. A preference
on an objective atom is the common mistake:

```text
--8<-- "fixtures/expected/docs/guide-prefer-objective.txt"
```

A trace compared with itself, a dimension the atom set does not declare, and a trace id the store
does not hold are refused the same way.

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
[Review traces in the UI](review-traces.md)).

## Rank a dimension

```text
--8<-- "fixtures/expected/docs/guide-prefer-rank.txt"
```

Each rating comes from a Bradley–Terry fit (Bradley and Terry 1952) to every stored preference on
the dimension, with a 95% Wald interval. The scale reads like Elo: a 400-point gap means 10:1 odds
that the higher trace wins. Overlapping intervals mean the comparisons do not yet separate the two
traces; collect more.

Preferences from different criteria or different judge kinds are ranked separately. A ranking
that mixed a human's choices with an LLM judge's would report neither.

`--explain` adds the method and a sequential Elo for comparison:

```text
--8<-- "fixtures/expected/docs/guide-prefer-rank-explain.txt"
```

Sequential Elo is display only. It changes with the order the preferences were stored in and has
no interval, so the Bradley–Terry fit is the number of record.

!!! warning "A trace that never lost is not ranked"
    When some trace won (or lost) every comparison it was in, its Bradley–Terry rating is infinite
    (Ford 1957). `calibrated-agent-evaluation prefer --rank` refuses that group with exit 1 instead
    of printing an unbounded number. Compare it against more traces.

## Next

- Gate the objective atoms the usual way: [Calibrate a judge](calibrate-a-judge.md), then
  [Gate a PR in CI](gate-in-ci.md).
- The flags: [`calibrated-agent-evaluation prefer`](../cli/prefer.md) and
  [`calibrated-agent-evaluation label`](../cli/label.md).
