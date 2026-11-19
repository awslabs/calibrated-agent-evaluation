# 5. Build and certify a judge

A judge scores at scale only after it passes two checks: it agrees with your labels on both
failures and passes, and its verdict does not move when something irrelevant about the input
changes. This phase writes the judge, sizes and labels its calibration set, and runs both checks.

## Write the judge file

With the `build-binary-judge` skill, ask:

> Write a judge for the wrong_refund failure mode.

The skill drafts one JSON file per failure mode with you. The pass and fail conditions come from
the policies and tool rules in your [expected-behavior spec](1-define-behavior.md); the verdict is
PASS or FAIL, and the judge writes its critique before the verdict. The shape:

```json
{
  "judge_id": "my-agent-wrong-refund",
  "model": "<an Inspect model string>",
  "prompt_version": "v1",
  "level": "workflow",
  "prompt_template": "You are grading whether ... <the rules from the spec> ... Write your critique first, then give a binary verdict: 'pass' if ..., 'fail' if ...",
  "failure_mode": "wrong_refund"
}
```

The bundled kit's v2 judge, field by field:

--8<-- "fixtures/expected/docs/guide-judge-file.md"

Check the file the way every scoring command loads it. This reads no store and calls no model:

```bash
calibrated-agent-evaluation judge --check judges/wrong_refund.json
```

What you'll see, for a judge file that points at a local model:

```text
--8<-- "fixtures/expected/docs/guide-local-judge-check.txt"
```

The check catches a missing or unknown field, a bad `level`, and a verdict ordered before the
critique. It cannot read the prompt's intent, so read the prompt yourself: if it asks two
questions, it is two judges.

## Size the calibration set

How many labels the judge needs depends on how accurate you expect it to be. The planner in the
shipped code inverts the gate's own bound:

```text
--8<-- "fixtures/expected/docs/example-calibration-sizing.txt"
```

At an expected accuracy of 0.90, 55 labels per class certify the judge. 55 is the smallest size
from which a judge that is right 90% of the time clears the gate at that size and every larger
one; at 48 the bound first clears, but 51 to 54 fall short again, so a plan of 48 could be undone
by a few more labels. When two raters double-code the gold, `calibrated-agent-evaluation calibrate` plans for the
agreement check as well: pass `--raters 2` and the plan rises to 65 per class, because agreement
needs 130 shared traces.

Certifying is not enough for a corrected failure rate. The rate's range needs more labels when the
judge turns out more accurate than planned, because its range is built from its mistakes.
`--rate-safe` plans for that: 162 per class covers any accuracy that would certify, up to a
perfect judge.

| You want | Labels per class at 0.90 | Flag |
| --- | --- | --- |
| a judge that may score and gate | 55 | (default) |
| the same, with gold two raters double-code | 65 | `--raters 2` |
| the same, plus a corrected rate you can quote | 162 | `--rate-safe` |

Decide on `--rate-safe` before the first run, because the calibration set can be used once.
`--expected-accuracy` and `--expected-tnr` change the planned accuracy.
[Checking a judge](../../concepts/how-the-numbers-work/checking-a-judge.md) has the table for
other accuracies.

!!! note "If you lead the team"
    The 162 row is the one to budget for if anyone will quote the failure rate outside the team.
    The 55 row is enough for a CI gate, which reads the judge's verdicts as they are.

## Label the calibration round

Write the judge prompt first, then label a fresh round of hunted traces for it, both FAIL and PASS
for the judge's mode, to the size the plan asks for. Give the round its own cell
(`label_round=2`) so it stays apart from the error-analysis labels the prompt was written from.
The second rater double-codes this round ([phase 3](3-look-and-label.md#double-code-the-calibration-gold)).

Three rules hold for this set:

- It is single-use. Once it has tested a judge for this mode, a second run on it is refused,
  whatever the result and whatever the judge file is called. The one exception: a set refused for
  having too few labels can grow and be checked again with the judge unchanged. The record counts
  each look, and `--explain` shows the count.
- No trace quoted in the judge prompt may be in it.
- Only human labels count.

## Run the calibration gate

With the `validate-judge` skill, ask:

> Can we trust judges/wrong_refund.json?

The skill runs the commands below in order, reports each verdict as printed, and stops at a
refusal.

```bash
calibrated-agent-evaluation calibrate --judge judges/wrong_refund.json --rate-safe --cell label_round=2
```

A live judge model needs the `inspect` extra and credentials for its provider. To record the
judge's replies as you go, so CI can replay them later without credentials:

```bash
calibrated-agent-evaluation calibrate --judge judges/wrong_refund.json --rate-safe --cell label_round=2 \
  --cassette judges/wrong_refund.cassette.json --cassette-mode record
```

Record mode calls the live model for every judge call and overwrites any reply the cassette
already holds; replay, the default, answers from the cassette only and calls no model.

??? example "Recording against a local stand-in model"

    This run points a copy of the kit's v1 judge at a small OpenAI-compatible server on
    127.0.0.1 that fails every trace (`docs/site/examples/guides/record_cassette_local.py`), so it
    records offline. A judge that fails everything is refused, and the cassette still holds a
    reply for every trace it called:

    <div class="wrap-output" markdown>

    ```text
    --8<-- "fixtures/expected/docs/guide-cassette-record.txt"
    ```

    </div>

What you'll see, for the kit's v1 judge on its first round:

<div class="wrap-output" markdown>

```text
--8<-- "fixtures/expected/docs/tutorial-calibrate-v1.txt"
```

</div>

The judge is certified when the 95% lower bound of both its true-positive rate (failures caught)
and its true-negative rate (passes cleared) reaches 0.80. If the round is short, the refusal says
how many more of each class to label:

??? example "Not enough labels"

    <div class="wrap-output" markdown>

    ```text
    --8<-- "fixtures/expected/docs/guide-calibrate-not-enough.txt"
    ```

    </div>

??? example "A second run on the same labels"

    <div class="wrap-output" markdown>

    ```text
    --8<-- "fixtures/expected/docs/guide-reuse-gold.txt"
    ```

    </div>

A certified record can still be rate-unsafe: certified on fewer labels than the accuracy the judge
showed needs for a rate. The judge may score and gate, and `calibrated-agent-evaluation run` refuses its corrected rate
and names the labels that would fix it.

## Run the bias check

A judge can agree with your labels on ordinary traces and still change its verdict when an answer
gets longer, or when someone in the conversation insists on a verdict.

```bash
calibrated-agent-evaluation bias --judge judges/wrong_refund.json \
  --probes judges/bias-probes --thresholds judges/bias-thresholds.json
```

`--probes` has no default: you write the probes from your own domain, for the judge's failure
mode. Each trial is a group of inputs whose fair verdict is the same, varying only the biasing
detail (a padded answer, an asserted verdict, a different claimed author). The thresholds file
sets a ceiling per bias for the judge's model, and a must-pass bias with no ceiling fails closed
instead of borrowing a default. [Bias probe files](../../bias-probe-format.md) has the format, the
kit's two shipped sets as worked examples, how to write probes for your domain, and what a ceiling
means.

The kit's v1 judge fails it:

<div class="wrap-output" markdown>

```text
--8<-- "fixtures/expected/docs/tutorial-bias-v1.txt"
```

</div>

??? example "The counts behind the refusal (`--explain`)"

    <div class="wrap-output" markdown>

    ```text
    --8<-- "fixtures/expected/docs/tutorial-bias-v1-explain.txt"
    ```

    </div>

    Every trial is paired with a placebo, the unchanged input judged again, and only the flips
    beyond the placebo's count. Twelve trials per check give a wide range, which is part of why v1
    fails. [The bias check](../../concepts/how-the-numbers-work/refusals.md#the-bias-check)
    explains the bound and the direction test.

## Revise, then certify on fresh labels

A refused judge goes back to its prompt. Read the critiques where it went wrong, change the prompt,
bump `prompt_version`, label a fresh round (`label_round=3`), and run both checks again. The kit's
v2 added four sentences to v1's prompt and changed nothing else:

```diff
--8<-- "fixtures/expected/docs/tutorial-prompt-revision.txt"
```

It was certified and cleared on its own round:

<div class="wrap-output" markdown>

```text
--8<-- "fixtures/expected/docs/tutorial-calibrate-v2.txt"
```

</div>

<div class="wrap-output" markdown>

```text
--8<-- "fixtures/expected/docs/tutorial-bias-v2.txt"
```

</div>

Bump `prompt_version` on every prompt edit, however small. Each record stores a fingerprint of the
prompt text, model, output fields and sampling settings, and a judge edited in place under the same
version is refused. Here the kit's certified v2 judge has one sentence added to its prompt
(`docs/site/examples/guides/edit_judge_prompt.py`) and the same `prompt_version`:

<div class="wrap-output" markdown>

```text
--8<-- "fixtures/expected/docs/guide-judge-fingerprint.txt"
```

</div>

!!! warning "Write the revision before you look at the new round"
    The fresh round is a fair test only if the prompt was finished before anyone labeled it. A
    prompt tuned while reading the new labels has been tuned on its test.

Once the judge has a certifying calibration record and a clear bias record in the store,
`calibrated-agent-evaluation judge` admits it. [Re-certify after a prompt change](../../recipes/recertify-after-prompt-change.md)
is the short version of this loop, and [Calibrate a judge](../calibrate-a-judge.md) walks the kit
through it with every output.

Next: [6. Gate in CI](6-gate-in-ci.md).
