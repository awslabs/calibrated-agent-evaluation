# Re-certify after a prompt change

You changed a judge's prompt or model and need it certified again before it scores.

A calibration record covers one model and one prompt version. Change either and the record no longer
covers the judge, so `calibrated-agent-evaluation judge` refuses until the new version is certified.
The new version also needs labels the old version never saw: the labels that showed you what to fix
are the ones the new prompt was written against.

## Steps

### 1. Bump `prompt_version`

Bump it in the judge file (or change `model`). Keep `judge_id` the same, so the records for both
versions sit under one judge.

### 2. Label a fresh set

Label traces for the new version, both FAIL and PASS for the judge's failure mode, as many per
class as the plan asks (55 at an expected accuracy of 0.90, 162 with `--rate-safe`).

### 3. Keep the two sets apart

Give the new traces a taxonomy cell. Set an attribute such as
`calibrated_agent_evaluation.taxonomy.label_round=2` on their spans when you capture or export them,
then pass `--cell label_round=2` to `calibrate`. The bundled kit does this.

### 4. Calibrate, run the bias check, then score

```text
--8<-- "fixtures/expected/docs/guide-certify-v2.txt"
```

## What happens if you skip step 2

Calibrating a revised judge on labels a version of it was already calibrated on is refused:

```text
--8<-- "fixtures/expected/docs/guide-reuse-gold.txt"
```

## Pitfalls

### Changing the prompt without bumping `prompt_version`

Calibrated Agent Evaluation refuses the edited judge, because each record stores a fingerprint of
the prompt it tested. Bump the version on every prompt edit, including small ones, so the records
say which wording they cover.

### Writing the fix while looking at the new labels

The new set is a fair test only if the prompt was finished before you labeled it.

### Expecting the old record to carry over to a new model

A model change is a new judge as far as calibration is concerned.

See [Calibrate a judge](../guides/calibrate-a-judge.md) for the full walk through the kit.
