# 2. Read and label traces

A team that skips this step writes a judge from what it expects the agent to get wrong. OrderBot's
team expected rude replies and wrong order numbers. When they read their traces, the failure that
mattered was a different one: OrderBot refunded orders the policy forbids, because the policy tool
said yes and the agent believed it. No judge written beforehand would have looked for that.

Reading traces is also where the team gets the answers a judge will be checked against later. A
person reads each trace, marks it PASS or FAIL, and writes why. Every later step measures against
these labels.

## Open the review page

=== "CLI"

    ```bash
    calibrated-agent-evaluation label --frame discovery --cell label_round=1
    ```

    <div class="wrap-output" markdown>

    ```text
    --8<-- "fixtures/expected/docs/tutorial-label-command.txt"
    ```

    </div>

    Run the printed command with your own id, then open the URL:

    ```bash
    calibrated-agent-evaluation label --serve --rater alice --frame discovery --cell label_round=1
    ```

=== "With Claude Code"

    > Open the review page for round 1 of OrderBot's hunted traces.

    The `build-annotation-ui` skill checks that the store has traces and the `ui` extra is
    installed, then prints the command for you to run with your rater id. You do the labeling.

Read each trace from the top, press <kbd>1</kbd> for PASS or <kbd>2</kbd> for FAIL, and write the
reason. A FAIL needs one. Write it the way you would explain the verdict to a colleague: name the
rule in the [behavior spec](define.md#the-behavior-spec) the run broke, and the facts in the trace
that show it. Step 3 groups failures by these reasons, so a reason that only says "wrong" is lost
there.

??? example "One of OrderBot's labeled failures"

    <div class="wrap-output" markdown>

    ```text
    --8<-- "fixtures/expected/docs/tutorial-labeled-trace.txt"
    ```

    </div>

    The reason names the policy, the facts that decided it (52 days since delivery, a 15-day
    window), and the tool result the agent trusted. This label was written by the rule that stands
    in for a reviewer in the sample set; yours come from people.

A session picks up where you stopped. `--limit 50` caps a session, and `--relabel` brings back
traces you already labeled; your newest label on a trace is the one that counts. [Review traces in
the UI](../how-to/review-ui.md) has every hotkey.

## Only people write labels

A label counts as ground truth only when a person recorded it. `calibrated-agent-evaluation label`
offers no model suggestions, and [step 4](judge.md) checks judges against human labels only. If a
model wrote the answers, checking a judge against them would measure how much two models agree.

## Name the failure as you label

Round 1 is for discovery, and there is no list of failure modes yet: label PASS or FAIL with a
reason, and let [step 3](cluster.md) name the modes. Round 2 is the set the judge is checked on in
[step 4](judge.md), labeled after the judge's prompt is final. Serve it with the list step 3 agreed,
and every FAIL picks one of its modes:

```bash
calibrated-agent-evaluation label --serve --rater alice --frame discovery --cell label_round=2 --codebook-version 1
```

When no mode fits, press <kbd>N</kbd> to propose a new one. The proposal is saved with the label,
flagged for whoever owns the list, and offered to everyone labeling the same store from the next
trace on.

## Two people, the same traces

Remember the two teammates from [step 0](define.md) who disagreed about the 52-day refund. If only
one of them labels, a judge checked against those labels agrees with one person's reading of the
spec, and nothing shows that reading is the team's. So for round 2, a second person labels the
same traces:

```bash
calibrated-agent-evaluation label --serve --rater bob --frame discovery --cell label_round=2 --codebook-version 1
```

Then measure how often the two of you agree:

```bash
calibrated-agent-evaluation label --agreement --raters alice,bob --codebook-version 1
```

<div class="wrap-output" markdown>

```text
--8<-- "fixtures/expected/docs/guide-label-agreement.txt"
```

</div>

Add `--cell label_round=2` to check one round. The score in the output, `alpha`, runs from 1 when
the two people always agree down to 0 when they agree no more often than chance would. Here alice
and bob agree well on the 40 traces they share, but 40 is too few to be sure the agreement holds,
so the check asks for 90 more. It gives one of four answers:

| Answer | What it means | What to do |
| --- | --- | --- |
| usable | enough shared traces, and the two agree well even on a cautious reading | go on to step 4 |
| top up overlap | too few shared traces (fewer than 130), or agreement that looks good but isn't certain yet | the second person labels more of the round, as the output says |
| revise codebook | the two agree too rarely for their labels to check a judge | sharpen the spec or the modes, and relabel the traces you disagreed on |
| codebook broken | the two agree little more than chance | rewrite the definitions before labeling more |

The command exits 0 only on "usable". `calibrated-agent-evaluation calibrate` runs the same check
on the labels it is about to use: labels from two people who don't agree well enough are turned
away, and labels from one person are accepted with a note that agreement was never measured.
[Methods](../methods.md#how-many-traces-to-double-code) has the agreement measure and the
thresholds behind each answer.

## What you have now

Labeled traces, each FAIL with a written reason, and for the judge's test round, labels from two
people with a measure of how well they agree.

Next: [3. Name what goes wrong](cluster.md).
