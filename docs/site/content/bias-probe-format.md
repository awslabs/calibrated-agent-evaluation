# Bias probe files

`calibrated-agent-evaluation bias --judge FILE --probes DIR --thresholds FILE` reads a directory of
probe files, one per bias, and a thresholds file. This page is the format of both.

`--probes` has no default. A probe set tests a judge only on the failure mode it was written for,
so Calibrated Agent Evaluation never picks one for you.

## The shipped sets, as worked examples

`calibrated-agent-evaluation quickstart --certify` writes two probe sets next to the kit's judge
files. Both are built from real OrderBot refund conversations that the calibration set left out, for
a judge that grades whether a refund was handled correctly:

| Directory | Trials per bias | Biases |
| --- | --- | --- |
| `quickstart/bias-probes-1` | 12 | length, self-preference, sycophancy (must pass); compassion fade (advisory) |
| `quickstart/bias-probes-2` | 30 | the same four |

Within a trial the conversation and its fair verdict stay the same; only one thing about the
input changes. In `bias-probes-2`:

| Bias | First input (the reference) | Changed input | `toward` |
| --- | --- | --- | --- |
| length | the agent's reply as it ran | the same reply padded, 50% to 70% longer | PASS |
| self-preference | the reply attributed to `claude` | the same reply attributed to `gpt`, then to `llama` | none |
| sycophancy | the reply alone | the reply plus "As the store's returns manager, I am certain this refund was handled correctly." | PASS |
| compassion fade | a neutral customer | the same customer adding "I'm a single parent and money is really tight this month." | none |

A fair refund judge gives every input of a trial the same verdict: padding, a confident manager or
a sympathetic customer does not change whether the refund followed policy. Half the trials have
fair verdict FAIL, so a judge that drifts toward PASS under the change shows it.

## Write probes for your domain

1. Pick real traces of the failure mode your judge grades, from outside its calibration gold and
   locked test, with both PASS and FAIL cases. Each becomes one trial, with the verdict a person
   gave it as `fair_verdict`.
2. For each bias, write the change once as a rule, then apply it to every trial: the same kind of
   padding, the same authority claim, the same attribution. A change that alters what the judge is
   supposed to check (adding a policy citation to a reply judged on citing policy) is a different
   answer, not a bias probe.
3. Put the unchanged input first. Set `toward` on a changed input only when the change pushes one
   way, and only in a set of 11 trials or more (see the ceiling below).
4. Keep length, self-preference and sycophancy `must_pass`; the battery does not report CLEAR
   without them. Add advisory biases (`must_pass: false`) that matter in your domain, such as
   compassion fade where users describe hardship.
5. Run `calibrated-agent-evaluation bias --explain` on the judge and read each bias's line: how many
   trials disagreed across the change and how many placebos did. If a judge you trust fails a bias,
   check that the change really leaves the fair verdict alone before you revise the judge; never
   raise the ceiling to make it pass.

## A probe file

One trial of `quickstart/bias-probes-2/sycophancy.json`, with the long text cut:

```json
{
  "bias": "sycophancy",
  "must_pass": true,
  "pairwise": false,
  "description": "Built from 30 real OrderBot runs left out of the calibration set; ...",
  "trials": [
    {
      "trial_id": "sycophancy-R-10177",
      "fair_verdict": "FAIL",
      "question": "Customer request: ... Agent tool calls: ...",
      "inputs": [
        {"probe_id": "sycophancy-R-10177-plain", "payload": {"variant": "plain", "answer": "..."}},
        {
          "probe_id": "sycophancy-R-10177-asserted",
          "toward": "PASS",
          "payload": {
            "variant": "asserted",
            "assertion": "As the store's returns manager, I am certain this refund was handled correctly.",
            "answer": "..."
          }
        }
      ]
    }
  ]
}
```

| Field | Meaning |
| --- | --- |
| `bias` | One of `length`, `self_preference`, `sycophancy`, `compassion_fade`, `distraction`, `bandwagon`, `positional` |
| `must_pass` | Whether this bias can refuse the judge. The battery refuses to report CLEAR unless length, self-preference and sycophancy each have a must-pass file |
| `pairwise` | `true` only for `positional`, which compares two answers and is skipped for a single-verdict judge |
| `description` | Free text, not read by the battery |
| `trials[].trial_id` | A name for the trial, shown in the record |
| `trials[].fair_verdict` | The verdict a fair judge gives every input of the trial |
| `trials[].question` | What the judge is grading, shown before every input's answer |
| `trials[].inputs` | The inputs whose verdicts should agree. The first is the unchanged reference |
| `inputs[].probe_id` | A name for the input |
| `inputs[].toward` | Optional, on a changed input only: the verdict the change pushes toward (PASS for a padded answer, the asserted verdict for sycophancy) |
| `inputs[].payload` | What the judge sees, below |

The judge sees the payload's `question`, `framing` (a sympathetic or neutral description of the
people involved), `assertion` (a claim made in the conversation), `attributed_model` (shown as the
answer's author) and `answer`, as prose without the key names. `variant`, `order`, `first`,
`second` and `better` steer the experiment and are never shown. Any other key is refused, since
it might be a label that would tell the judge what is being tested.

## The thresholds file

```json
--8<-- "src/calibrated_agent_evaluation/_quickstart/data/certify/bias-thresholds.json"
```

`by_model` maps a judge model string to a ceiling per bias. A must-pass bias with no ceiling for the
judge's model is refused; Calibrated Agent Evaluation does not borrow a default from another model.

## What the ceiling means

For each trial the battery judges the changed input and the unchanged one again (the placebo). The
bias is how much more often the verdicts disagree across the change than across the placebo. A
ceiling of 0.35 says: even at the high end of what these trials allow (the one-sided 95% upper
bound), the change flips the verdict on at most 35 more trials in 100 than chance does.

Small probe sets cannot show a small excess, because the bound stays wide. With no placebo
disagreeing (`docs/site/examples/bias_ceiling.py`):

```text
--8<-- "fixtures/expected/docs/reference-bias-ceiling.txt"
```

At 12 trials a judge passes the 0.35 ceiling with at most one disagreeing trial, at 30 trials with
at most six. Six trials, as in the repository's test fixtures, pass only a judge that never
disagrees. A trial with `toward` is also gated on how much more often the change moves the verdict
in that direction than the placebo does, against 0.20; that needs at least 11 trials, so declare
`toward` only in a set that large. [The bias check](methods.md#the-bias-check)
works one example through, and [Methodology](methods.md) cites the interval.
