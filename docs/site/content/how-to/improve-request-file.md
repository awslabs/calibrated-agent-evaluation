# Write an improve request

Check a fix whose before-and-after results you measured outside this toolkit.

`calibrated-agent-evaluation improve` decides whether a fix really lowered a failure rate (see
[Check that a fix worked](../walkthrough/improve.md)). When the
before and after verdicts both live in this library's stores, `calibrated-agent-evaluation improve
--evaluate` pairs them for you. When you measured them elsewhere, write them to a request file and
pass it with `--request`.

## Steps

### 1. Register the bar before the fix exists

This freezes the failure mode and the smallest drop in failure rate that counts as a fix. The bar
is read from the stored record and not from the request, so seeing the results can't move it.

### 2. Run the same held-out traces before and after the fix

Write one pair per trace.

### 3. Evaluate the request

```text
--8<-- "fixtures/expected/docs/recipe-improve.txt"
```

This is the request file it evaluated:

```json
--8<-- "fixtures/expected/docs/recipe-improve-request.json"
```

## The fields

| Field | What it holds |
| --- | --- |
| `target_metric` | The registered failure mode. |
| `pairs` | One object per held-out trace: `trace_id`, `before_fail` and `after_fail` (booleans), and `labels_are_judge_produced`. |
| `confirmation_ids` | The traces the decision is read from. Optional; defaults to every trace in `pairs`. |
| `development_ids` | Required. The traces you looked at while writing the fix. They must not appear among the confirmation traces. |
| `slices` | The guard panel: other things the fix must not make worse. At least one is required. |

Each slice has a `name`, a `kind` (`target`, `agent`, `handoff` or `other`), a `regression_p`,
and `rate_before` and `rate_after`. `regression_p` is your own test's answer to "did this slice
get worse?", as a probability: small means it probably did. Each rate is an object with the value
(`point`), its likely range (`lo`, `hi`), how that range was computed (`method`, `level`, and
`sided`, which is `two-sided`, `lower` or `upper`) and the number of traces behind it (`n`).

When `labels_are_judge_produced` is true, every pair where the verdict changed needs a human
label on that trace in the store, which settles it, so a judge doesn't grade its own fix.

## The detail behind the verdict

```text
--8<-- "fixtures/expected/docs/recipe-improve-explain.txt"
```

The fix is accepted when the whole range of the change in failure rate sits past the registered
drop (here, the range ends at -0.029, past the -0.020 bar), and no guard slice got worse, allowing
for the fact that several slices were checked at once.

## Pitfalls

### `slices` can't be empty

With nothing to check against, the request can't show that nothing got worse, so `improve` stops
and asks for at least one slice.

### Registering after the fact

A metric registers once. Register before you start on the fix.

### Overlapping development and confirmation traces

A trace in both lists stops the evaluation: a fix scored on the traces it was written against says
nothing about new traffic.

### Crews

A request file cannot carry what a crew check needs (which agent owned each failure before and
after), so `--request` does not accept a crew metric. Evaluate a crew fix from two stores with
`--evaluate`, `--crew-file` and `--touched-agent`;
[Evaluate a multi-agent crew](evaluate-a-crew.md) shows how.
