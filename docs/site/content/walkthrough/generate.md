# 1. Make test inputs

The OrderBot team has two piles of traces. One is the escalations and complaints support forwarded
last month: almost all of them are failures, which is why someone forwarded them. The other is a
random day of live traffic. If the two piles get mixed, the escalations make OrderBot look several
times worse than it is, and nobody can say by how much.

This step gets traces into the store and marks each one with why it was collected. Labeling and
naming failures (steps 2 and 3) want the hunted pile. The failure rate (steps 5 and 7) is measured
on the random pile only.

| Mark | What goes in it | Used for |
| --- | --- | --- |
| `discovery` | traces picked to find failures: the frame's inputs, escalations, complaints, edge cases | labeling, naming failures, checking the judge |
| `production-representative` | a random draw from real traffic, nothing hand-picked | the failure rate and the CI gate |

`calibrated-agent-evaluation run` leaves `discovery` traces out of the rate and says how many it
left out. Both kinds can sit in one store.

## Bring the traces you have

=== "CLI"

    ```bash
    calibrated-agent-evaluation ingest --dry-run traces/
    ```

    <div class="wrap-output" markdown>

    ```text
    --8<-- "fixtures/expected/docs/calibrated-agent-evaluation-ingest-dry-run.txt"
    ```

    </div>

    `--dry-run` reports what would be stored and what would be turned away, and writes nothing.
    Run it on every new source first, then again without `--dry-run`.

=== "With Claude Code"

    > Ingest the traces in traces/ for OrderBot.

    The `run-error-analysis` skill picks the route that matches what you have (trace files, an
    agent it can run in process, or an endpoint you are authorized to test) and runs the matching
    command. Ask it for a dry run first if you want one.

The output above is a folder of OpenTelemetry exports. Seven files convert. Two of those carry only
the input and the final answer, so their traces can be graded on the outcome but cannot show which
step went wrong. Two files would overwrite each other in the store, so neither is written until
they get distinct ids, and one file is in a format the reader doesn't recognize. When a file is
turned away, fix the export or pass `--convention`; don't edit the trace to make it fit.

Traces arrive marked `discovery` unless you pass `--frame production-representative`. For the
random pile, let the toolkit do the drawing:

```bash
calibrated-agent-evaluation ingest traces/ --sample 200 --seed 7
```

That ingests a seeded random sample of 200 traces and marks it `production-representative`. The
same seed over the same files draws the same sample, so the rate can be reproduced.
[Step 7](production.md) runs this on a schedule.

[Connect your agent](../integrate/index.md) covers Langfuse, in-process capture, routing an agent
through the toolkit, and the other formats.

## Or generate inputs from the grid

When the team has few real traces for a situation, they turn the frame from
[step 0](define.md) into inputs:

=== "CLI"

    ```bash
    calibrated-agent-evaluation generate --frame frame.json --out inputs.jsonl
    ```

    <div class="wrap-output" markdown>

    ```text
    --8<-- "fixtures/expected/docs/walkthrough-generate-template.txt"
    ```

    </div>

=== "With Claude Code"

    > Generate inputs for every row of frame.json and run OrderBot on them.

    The `run-error-analysis` skill runs `calibrated-agent-evaluation generate`, then
    `generate --run` against the agent. You give it the agent's `module:callable`.

Each input carries the grid cell it came from. The offline template writes a placeholder request
per row, which is enough to check the plumbing. `--model` writes realistic customer messages with a
model on Bedrock instead (the `bedrock` extra).

`--run` sends each input through the agent and stores the resulting traces, marked with the row's
cell and `discovery`:

```bash
calibrated-agent-evaluation generate --run --frame frame.json \
  --executor strands --agent corpus.agents.orderbot_strands:build
```

`--multi-turn` holds a conversation per row, with a simulated customer writing the replies
([Route your agent through](../integrate/route-your-agent.md)).

## Mark each labeling round

The team labels in rounds. Round 1 is for discovery ([step 2](label.md)). Round 2 is the set
[step 4](judge.md) checks the judge on, and each later version of the judge needs a fresh round of
its own. Give each batch of hunted traces a cell value that names its round, for example
`label_round=1`, so later commands can pick one round with `--cell label_round=1`. The judge kit
from `quickstart --certify` skips discovery: its two rounds are the test sets for its two judge
versions.

## If you can't see inside the agent

When the agent is someone else's endpoint, `calibrated-agent-evaluation probe --endpoint URL
--authorized` collects its replies. Those traces show only what went in and what came out, so you
can measure and gate the outcome but cannot say which step failed. [Probe a black-box
endpoint](../integrate/probe-black-box.md) covers it.

## What you have now

OrderBot's traces in one store, each marked `discovery` or `production-representative`, with hunted
traces tagged by grid cell and labeling round.

Next: [2. Read and label traces](label.md).
