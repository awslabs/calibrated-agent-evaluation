# 3. Look and label

With the `build-annotation-ui` skill, ask:

> Open the review surface for round 1 of my_agent's hunted traces.

The skill checks the store has traces and the `ui` extra is installed, then prints the command for
you to run with your own rater id. It never labels a trace itself. Without the skill:

```bash
calibrated-agent-evaluation label --frame discovery --cell label_round=1
```

What you'll see:

<div class="wrap-output" markdown>

```text
--8<-- "fixtures/expected/docs/tutorial-label-command.txt"
```

</div>

Run the printed command with your id in place of `RATER_ID`, and open the URL:

```bash
calibrated-agent-evaluation label --serve --rater alice --frame discovery --cell label_round=1
```

Read each trace from the top, write what the agent did and whether that was right, and press
<kbd>1</kbd> for PASS or <kbd>2</kbd> for FAIL. A FAIL needs a critique. Write the critique the
way you would explain the verdict to the second rater: name the rule from the
[expected-behavior spec](1-define-behavior.md) the run broke and the facts in the trace that show
it. Phase 4 clusters these critiques into failure modes, so a critique that only says "wrong" is
lost there.

??? example "A FAIL label from the bundled kit"

    <div class="wrap-output" markdown>

    ```text
    --8<-- "fixtures/expected/docs/tutorial-labeled-trace.txt"
    ```

    </div>

    The kit's labels come from a rule that applies the store's written return policy
    (`synthetic-rule-oracle`), so a large sample could ship in the package. Yours come from people.
    The critique names the policy, the facts that decided it (52 days since delivery, a 15-day
    window) and the tool result the agent trusted.

[Review traces in the UI](../review-traces.md) covers every hotkey, the guards against
accidental labels, and the crew attribution fields.

A session resumes where you stopped. `--limit 50` caps a session, and `--relabel` brings back
traces you already labeled; when you relabel a trace, your newest label is the one that counts.
Every label is stored with `kind="human"` and your rater id; the time you spent on each trace goes
to `review-sessions.jsonl` under the store root.

## Name the mode as you label

On the first round there is no codebook yet: label PASS or FAIL with a critique, and let phase 4
name the modes. From the second round on, serve the form with the codebook version phase 4
ratified, and every FAIL must carry one of its modes:

```bash
calibrated-agent-evaluation label --serve --rater alice --frame discovery --cell label_round=2 --codebook-version 1
```

The form then shows a mode picker between the critique and the verdict buttons (<kbd>M</kbd> jumps
to it). When no mode fits, press <kbd>N</kbd> to propose a new one with a name and a note. The label
is saved with the proposed mode, the proposal is flagged for the codebook owner, and from the next
trace on the picker offers it, marked "(proposed)", in every session on the same store and codebook
version. Phase 4 adopts or merges each proposal with `calibrated-agent-evaluation cluster --ratify`. Every label records
the codebook version it was written under.

## Labels come from people

Only labels a person records count as gold. A model can propose a label through the library's
pre-fill hook, and the person accepts or overrides it in the surface, but `calibrated-agent-evaluation label` offers no
suggestions. `calibrated-agent-evaluation calibrate` counts only `kind="human"` labels as gold, so model verdicts never
stand in for a person's.

## Double-code the calibration gold

The domain expert labels every round. For the round a judge will be calibrated on (phase 5), the
second rater labels the same traces, so the gold is double-coded and no extra traces are needed:

```bash
calibrated-agent-evaluation label --serve --rater bob --frame discovery --cell label_round=2 --codebook-version 1
```

Then check that the two of you agree before any judge is certified on these labels:

```bash
calibrated-agent-evaluation label --agreement --raters alice,bob --cell label_round=2 --codebook-version 1
```

What you'll see, on a store where alice and bob share 40 traces:

<div class="wrap-output" markdown>

```text
--8<-- "fixtures/expected/docs/guide-label-agreement.txt"
```

</div>

??? example "The numbers behind the verdict (`--explain`)"

    <div class="wrap-output" markdown>

    ```text
    --8<-- "fixtures/expected/docs/guide-label-agreement-explain.txt"
    ```

    </div>

The check reports Krippendorff's alpha with its interval (and, with `--explain`, Cohen's kappa
and agreement per class) and one of four verdicts. It exits 0 only on the first:

| Verdict | When | What to do |
| --- | --- | --- |
| usable | enough traces are double-coded and the 95% lower bound of alpha is at least 0.80 | go on to phase 4 |
| top up overlap | too few traces are double-coded, or alpha is at least 0.80 but its lower bound is not | have the second rater label more of the round, as the output says, and run the check again |
| revise codebook | alpha is between 0.667 and 0.80 | the definitions are ambiguous; sharpen the spec or the codebook, and relabel the traces you disagreed on |
| codebook broken | alpha is below 0.667 | the definitions are not working; rewrite them before labeling more |

Below the minimum overlap the gate answers "top up overlap" unless alpha is low enough that the
codebook is plainly the problem. The minimum is 130 shared traces, and
[Review traces in the UI](../review-traces.md#how-many-traces-to-double-code) says how it is
worked out.

`--axis outcome` (the default) checks agreement on PASS and FAIL; `--axis failure_mode` checks
agreement on which mode a failure is, once phase 4 has named the modes. Pooled agreement can hide a
rare class the raters never agree on, which is why `--explain` prints the per-class figures.
Agreement is measured within one codebook version, so labels written before a codebook change are
not pooled with labels written after it. Every check is stored under `<root>/records/agreement/`.

`calibrated-agent-evaluation calibrate` runs the same check over the gold it is about to use: gold from two or more
raters whose agreement is not usable is refused, and gold from a single rater is certified with a
note that agreement was not measured.

!!! note "If you lead the team"
    Do not skip the second rater to save time. Without double-coded gold, a judge's certificate
    says it agrees with one person, and nothing shows that person's labels mean what the spec says.

[Checking a judge](../../concepts/how-the-numbers-work/checking-a-judge.md#when-two-people-label)
explains the agreement thresholds.

Next: [4. Name failure modes](4-name-failure-modes.md).
