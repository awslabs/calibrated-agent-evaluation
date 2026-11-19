# 8. Run it in production

The CI gate checks a fixed input set. Production needs two more things: a failure rate measured on
real traffic, on a schedule, and a rule for when a certified judge has to be checked again.

## The scheduled sample

Nightly or weekly, draw a random sample of production runs, score it with the certified judges,
and record the corrected rate, the gate and input drift. Each scheduled run gets its own store,
seeded with the judges-only export from [phase 6](6-gate-in-ci.md#export-the-judges-once), so one
sample's verdicts never mix with another's:

```bash
export CALIBRATED_AGENT_EVALUATION_LOCAL_ROOT=prod/2026-10-26
calibrated-agent-evaluation load evals/judges
calibrated-agent-evaluation ingest --from-langfuse "my-agent-prod" --sample 200 --seed 20261026
calibrated-agent-evaluation judge --judge judges/wrong_refund.json --frame production-representative
calibrated-agent-evaluation run --judge judges/wrong_refund.json --ci \
  --drift-baseline prod/baseline --drift-feature input-chars \
  --fail-on-drift --publish langfuse
```

`--sample N --seed S` ingests a seeded random draw of N traces from the project (or from the files
you name) instead of every one, tags it `production-representative`, and records the seed and the
population size, so the same seed and population always draw the same sample. With
`--from-langfuse`, `--cell KEY=VALUE` draws from one taxonomy cell at a time, for a stratified
sample. On six exported traces:

<div class="wrap-output" markdown>

```text
--8<-- "fixtures/expected/docs/integrate-ingest-sample.txt"
```

</div>

Draw at least 150: under 150 scored traces the corrected rate is withheld and the output names how
many more to score.

What you'll see from `calibrated-agent-evaluation judge`, for the kit's 200-trace production sample:

<div class="wrap-output" markdown>

```text
--8<-- "fixtures/expected/docs/tutorial-judge-v2.txt"
```

</div>

And from `calibrated-agent-evaluation run`, with drift against a baseline store and a local
dashboard (`--publish local`):

<div class="wrap-output" markdown>

```text
--8<-- "fixtures/expected/docs/guide-run-production.txt"
```

</div>

The plain output gives each mode's corrected failure rate with its range, "20% (likely between
14% and 26%)" in the kit. That is the number to report. The raw share of FAIL verdicts is not: a
judge's misses and false alarms bias it, and the run corrects it with the judge's measured TPR and
TNR ([Correcting the rate](../../concepts/how-the-numbers-work/correcting-the-rate.md),
[Why the range matters](../../concepts/how-the-numbers-work/the-range.md)). The kit's labels come
from a rule, so it can also say what the true rate was:

--8<-- "fixtures/expected/docs/home-quickstart-line.md"

## Drift

`--drift-baseline` names a local store whose production-representative traces are the baseline, and
`--drift-feature` names what to compare: `input-chars` (the length of the trace input), `turns`
(agent steps) or `tool-calls`. There is no default feature. The taxonomy-cell mix is compared on
every run and reported without ever flagging. Use as the baseline the production sample you scored
when the judge was last certified, so the question is "does traffic still look like it did when we
trusted this judge?". `calibrated-agent-evaluation dump` that store's root and
`calibrated-agent-evaluation load` it into `prod/baseline` to keep it.

Drift never changes the gate. Each measure flags only when the difference is too large to be
chance after correcting for the two measures tested, and large enough to matter. With fewer than
50 values on a side the output names the shortfall instead of reporting no drift. When drift is
flagged, each admitted judge gets a staleness warning. [Check for drift](../../recipes/drift-check.md)
explains the measures and how often they flag by chance (about 6% of the time on two samples from
the same traffic).

## Dashboards and alerts

`--publish langfuse` sends the run's rates and drift to the Langfuse project in the `LANGFUSE_*`
environment variables, whatever `--backend` the store uses. `--publish local` writes them under
`<root>/dashboard/<run_id>/` as CSV and JSON. The run is recorded before publishing, so a failed
publish exits 1 with "Run recorded as <id>; publish failed: ..." and the gate verdict is still
printed.

Calibrated Agent Evaluation sends no messages. Alert from your scheduler on the exit code:

| Exit | Means |
| --- | --- |
| 0 | the run finished and the gate passed |
| 1 | a mode failed the gate, a command refused, or publishing failed |
| 3 | with `--fail-on-drift`: the gate passed, but input drift or a stale judge was flagged |

A failing gate exits 1 even when drift is flagged too, so a scheduler that treats any non-zero code
as an alert needs no special case, and one that wants drift as a warning can route exit 3
separately.

!!! note "If you lead the team"
    A failed scheduled gate on production traffic is a production incident for that failure mode,
    not a flaky test. Route it to the agent owner the way you route any other alert, and expect the
    response to be [phase 7](7-after-the-gate.md).

## When to re-certify

A certificate covers one judge prompt, one model and one failure mode, measured on traces from one
version of the agent. Some changes are caught automatically; others depend on your team's process.

| Trigger | Enforced by the library? | What happens | What you do |
| --- | --- | --- | --- |
| judge prompt, model, `prompt_version` or output fields change | enforced | `calibrated-agent-evaluation judge` refuses: the record no longer matches the judge's fingerprint | full re-certification (phase 5) on a fresh round |
| codebook change: a mode renamed, merged, split or redefined | effectively enforced | a renamed or merged mode has no gold under its new name, and a redefined mode's judge prompt changes, which is the row above; the calibration set is single-use, so either way you label a fresh round | full re-certification on a fresh round |
| agent prompt, model or tools change | process | nothing: the judge's record still matches the judge | the re-check below; full re-certification only if it flags |
| drift flagged by the scheduled run, or a staleness warning | process (exit 3 with `--fail-on-drift`) | a warning in the run output | the re-check below |
| no trigger for a quarter | process | nothing | the re-check below, which catches a provider updating a model behind an unchanged id |

The agent row matters because a judge's TPR and TNR were measured on the old agent's runs. A new
agent can fail in ways the judge reads differently, and nothing in the judge file changes to
tell you.

### The re-check

Label fresh traces from the current agent, hunted the same way as the original calibration round,
in their own cell (`label_round=4`), and re-check the judge on them:

```bash
calibrated-agent-evaluation label --serve --rater alice --frame discovery --cell label_round=4 --codebook-version 1
calibrated-agent-evaluation calibrate --check --judge judges/wrong_refund.json
```

`--check` measures the judge on gold no calibration of its mode used, records the change in TPR
and TNR as a separate judge-check record, and never writes or changes a calibration record, so it
can run in your working store. The fresh labels it needs per class are derived from the certifying
record: the smallest number at which a judge whose rate fell to 0.80 is flagged, at that size and
every larger one. Short gold is refused before any model call, with the count to label. In the kit,
where every labeled trace already went into a calibration:

<div class="wrap-output" markdown>

```text
--8<-- "fixtures/expected/docs/guide-calibrate-check.txt"
```

</div>

At that size a judge that fell to exactly 0.80 is flagged about half the time, and one that fell
further more often. Label more than the minimum if you need to see smaller drops. A flagged drop
exits 1 and says to certify again on new labels; the old certification stands until you do.
[Calibrate a judge](../calibrate-a-judge.md#re-check-a-certified-judge) has the details.

If the check flags, re-certify: revise the judge if it needs it, bump `prompt_version`, and
certify on a fresh round, as in [phase 5](5-build-and-certify-a-judge.md).

## Cadence

| When | What | Commands | Blocks on failure |
| --- | --- | --- | --- |
| every pull request | CI gate on the fixed input set | `calibrated-agent-evaluation judge`, `calibrated-agent-evaluation run --ci` | yes, the merge |
| nightly or weekly | production sample: corrected rate, gate, drift, publish | `calibrated-agent-evaluation ingest --sample`, `judge --frame production-representative`, `run --ci --drift-baseline --fail-on-drift --publish` | alerts through your scheduler |
| judge prompt or model change | full re-certification | `calibrated-agent-evaluation calibrate`, `calibrated-agent-evaluation bias` | yes, `calibrated-agent-evaluation judge` refuses |
| codebook change | full re-certification on a fresh round | `calibrated-agent-evaluation label`, `calibrate`, `bias` | yes, in effect |
| agent prompt, model or tool change | re-check, then full re-certification if it flags | `calibrated-agent-evaluation label`, `calibrated-agent-evaluation calibrate --check` | no, process |
| drift flagged or staleness warning | re-check | as above | exit 3 with `--fail-on-drift` |
| quarterly | re-check without a trigger | as above | no, process |
| the CI set keeps passing with headroom | add harder or fresher cases ([phase 6](6-gate-in-ci.md#keep-the-set-hard)) | `calibrated-agent-evaluation ingest`, `generate --run` | no, process |

!!! note "If you lead the team"
    The process rows are the ones that slip. Put the quarterly re-check on a calendar with an
    owner, and make "re-ran the check" part of the definition of done for any change to the
    agent's prompt, model or tools.
