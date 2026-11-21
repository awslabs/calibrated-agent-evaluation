# 7. Watch production

The OrderBot team certified its judge in spring, on spring traffic. By December the store runs a
holiday returns promotion, conversations are longer, more of them are about late returns, and the
agent has a new tool for exchanges. Nothing in the judge file changed, so nothing told the team to
check it again. The dashboard kept printing a failure rate computed from how often the judge was
right on spring's replies.

Two things go stale in production: the failure rate, which needs measuring on current traffic, and
the judge's accuracy, which was measured on traffic that no longer looks like this. This step puts
both on a schedule.

## Score a random sample on a schedule

Nightly or weekly, draw a random sample of production runs, score it with the certified judge, and
record the corrected rate, the gate and input drift. Give each scheduled run its own store, seeded
with the judge's records, so one sample's verdicts stay apart from another's. Here `evals/judges` is
a judges-only dump made with `dump --judges-only` ([Configure the CI
job](../how-to/ci-config.md#seed-the-jobs-store) covers it):

```bash
export CALIBRATED_AGENT_EVALUATION_LOCAL_ROOT=prod/2026-12-01
calibrated-agent-evaluation load evals/judges
calibrated-agent-evaluation ingest --from-langfuse "orderbot-prod" --sample 200 --seed 20261201
calibrated-agent-evaluation judge --judge judges/refund-policy.json --frame production-representative
calibrated-agent-evaluation run --judge judges/refund-policy.json --ci \
  --drift-baseline prod/baseline --drift-feature input-chars \
  --fail-on-drift --publish langfuse
```

`--sample N --seed S` stores a seeded random draw of N traces, tagged `production-representative`,
and records the seed and the population size, so the same seed over the same population draws the
same sample. On six exported traces:

<div class="wrap-output" markdown>

```text
--8<-- "fixtures/expected/docs/integrate-ingest-sample.txt"
```

</div>

Draw at least as many as the corrected rate needs ([step 4](judge.md#what-the-judge-is-for) gives
the minimum); with fewer, the output says how many more to score.

The kit's 200-reply production sample, scored and run with a drift baseline and a local dashboard
(`--publish local`):

<div class="wrap-output" markdown>

```text
--8<-- "fixtures/expected/docs/tutorial-judge-v2.txt"
```

</div>

<div class="wrap-output" markdown>

```text
--8<-- "fixtures/expected/docs/guide-run-production.txt"
```

</div>

"20% (likely between 14% and 26%)" is the number to report. The judge's raw share of FAIL verdicts
is not: it still carries the judge's misses and false alarms, which [step 4](judge.md#what-the-judge-is-for)
showed can put it well away from the truth.

## Notice when traffic changes

`--drift-baseline` names a local store whose `production-representative` traces are the baseline,
and `--drift-feature` names what to compare: `input-chars` (the length of the input), `turns`
(agent steps) or `tool-calls`. There is no default. Use as the baseline the production sample you
scored when the judge was last certified, so the question is: does traffic still look like it did
when we trusted this judge? Keep that store with `calibrated-agent-evaluation dump` and restore it
with `calibrated-agent-evaluation load`.

A shift is flagged only when it is larger than chance would produce and large enough to matter.
Each side needs at least 50 traces; with fewer, the output names the shortfall. When drift is
flagged, the run also warns that each judge you passed may be stale. A judge whose certification no
longer matches its file is not admitted, and the run prints no corrected rate for it. Drift and
staleness don't change the gate result. Two samples of the same traffic are sometimes flagged by
chance ([Check for drift](../how-to/drift-check.md) says how often), so treat a single flag as a
reason to run the re-check below. The same page covers comparing an input the CLI does not offer as a feature.

## Alert on the exit code

Alerting is your scheduler's job. `--publish langfuse` sends the run's rates and drift to the Langfuse
project in the `LANGFUSE_*` environment variables, and `--publish local` writes them under
`<root>/dashboard/<run_id>/`. The run is recorded before publishing, so a failed publish still
leaves the run on disk. Alert from your scheduler:

| Exit | Means |
| --- | --- |
| 0 | the run finished and the gate passed (without `--ci`, a failing gate also exits 0) |
| 1 | with `--ci`, a mode failed the gate; or a command refused, or publishing failed |
| 3 | with `--fail-on-drift`, the gate passed but drift or a stale judge was flagged |

A failing gate exits 1 even when drift is flagged too, so a scheduler that treats any non-zero code
as an alert needs no special case.

A failed scheduled gate on production traffic means a failure mode is happening to customers now.
Route it to the agent's owner like any production alert; the response is [step 6](improve.md).

## When to check the judge again

A certification covers one judge prompt, one model and one failure mode, measured on replies from
one version of the agent. Some changes are caught for you; the rest depend on your team's habits.

| Change | Caught by the toolkit? | What you do |
| --- | --- | --- |
| the judge's prompt, model, `prompt_version` or output settings | yes: `calibrated-agent-evaluation judge` refuses the judge | re-certify on a fresh round ([step 4](judge.md#what-ends-a-certification)) |
| a failure mode renamed, merged, split or redefined | in effect: the new name has no labels, and a redefined mode changes the judge prompt | re-certify on a fresh round |
| OrderBot's prompt, model or tools | no: the judge file is unchanged | the re-check below |
| drift flagged, or a staleness warning | warned, exit 3 with `--fail-on-drift` | the re-check below |
| a quarter with none of the above | no | the re-check below; it also catches a provider updating a model behind an unchanged name |

The agent row is the one teams miss. The judge's catch rate was measured on the old agent's
replies, and a new agent can fail in ways the judge reads differently.

### The re-check

Label fresh replies from the current agent, hunted the same way as the original round, in their
own cell, and measure the judge on them:

=== "CLI"

    ```bash
    calibrated-agent-evaluation label --serve --rater alice --frame discovery --cell label_round=4 --codebook-version 1
    calibrated-agent-evaluation calibrate --check --judge judges/refund-policy.json
    ```

=== "With Claude Code"

    The `build-annotation-ui` skill opens the review surface for the labeling; the person labels.
    Then, with the `validate-judge` skill, ask:

    > Re-check judges/refund-policy.json on the new labels.

`--check` measures the judge on labels no calibration of its mode has used and records the result
as a separate check. It does not change the judge's certification, so it is safe to run in your
working store. It needs enough fresh labels to notice a judge whose catch rate has fallen to 80%,
and says how many before it calls a model. In the kit, where every labeled reply already went into
a calibration:

<div class="wrap-output" markdown>

```text
--8<-- "fixtures/expected/docs/guide-calibrate-check.txt"
```

</div>

At that minimum, a judge that fell to exactly 80% is noticed about half the time, and one that
fell further more often. Label more than the minimum to notice smaller drops. A noticed drop exits
1 and tells you to certify again; the old certification stands until you do.

## The schedule

| When | What | Blocks on failure |
| --- | --- | --- |
| every pull request | the gate on the input set ([step 5](gate.md)) | the merge |
| nightly or weekly | production sample: corrected rate, gate, drift, publish | alerts through your scheduler |
| a judge prompt or model change | full re-certification | `judge` refuses until done |
| a failure mode redefined | full re-certification on a fresh round | in effect |
| an agent prompt, model or tool change | the re-check, then re-certification if it flags | no, team habit |
| drift or a staleness warning | the re-check | exit 3 with `--fail-on-drift` |
| every quarter | the re-check | no, team habit |
| the input set keeps passing with room to spare | add harder or fresher cases ([step 5](gate.md#keep-the-input-set-hard)) | no, team habit |

The rows marked "team habit" are the ones that slip. Put the quarterly re-check on a calendar with
an owner, and make "re-ran the judge check" part of done for any change to the agent's prompt,
model or tools.

## What you have now

A scheduled job that measures the failure rate on current traffic and alerts through its exit
code. When the scheduled sample turns up a new kind of failure, go back to [step 2](label.md).
