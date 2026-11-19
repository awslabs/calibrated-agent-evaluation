# 1. Define behavior

Before anyone labels a trace, write down what the agent is supposed to do. This phase produces two
of the three artifacts everything later is checked against. The third, the failure-mode codebook,
comes out of error analysis in [phase 4](4-name-failure-modes.md).

| Artifact | Who writes it | Format | Read by |
| --- | --- | --- | --- |
| Expected-behavior spec | domain expert, with the agent owner | a markdown page in your repo | the raters, the codebook, the judge prompts |
| Input taxonomy | agent owner, with the domain expert | a JSON object for `calibrated-agent-evaluation define` | `calibrated-agent-evaluation define`, `generate`, `cluster` |
| Failure-mode codebook | domain expert | named modes, ratified in phase 4 | labels, judges, the gate |

## The expected-behavior spec

The spec is a page your team keeps next to the agent's code. Calibrated Agent Evaluation does not
read it, and it is not one of its file formats. Its job is to give the two raters the same
definition of a correct run, and later to supply the pass and fail conditions in each judge prompt.

If you already have a design doc, a BRD or a PRD for the agent, extract these sections from it
instead of writing them anew. Most of the content exists there already; the work is making each
rule specific enough that two people reading the same trace reach the same verdict.

```markdown
# my_agent: expected behavior

## Scope
- Handles: <the requests the agent is meant to resolve>
- Hands off: <requests it must route to a person or another system, and how>
- Out of scope: <requests it must decline>

## Policies
<The business rules a correct answer follows, stated so a rater can check them
against the trace. Quote the source document and its version.>
- <rule>: <the condition, with the numbers, e.g. "refundable if delivered no
  more than return_window_days ago">

## Tool rules
- <tool name>: when to call it, which arguments must come from where, and
  whether the agent may act on its result without checking it.

## Must never
<Each line is a candidate failure mode for the codebook.>
- <e.g. issue a refund the policy forbids, whatever a tool reports>
- <e.g. state an amount that does not match the order>

## Examples
| Trace id | Verdict | Why |
| --- | --- | --- |
| <a real trace> | PASS | <which rules it follows> |
| <a real trace> | FAIL | <which rule it breaks> |
```

The spec feeds the rest of the guide in three places:

- The must-never list seeds the codebook. Raters start from it when they name failures in phase 3,
  and phase 4 keeps, merges or renames the modes the traces show.
- The policies and tool rules become the pass and fail conditions of each judge prompt in phase 5.
- The examples are for raters. A trace you quote in a judge prompt must stay out of that judge's
  calibration labels, because a judge checked on its own prompt examples is checked on the answers
  it was given.

!!! note "If you lead the team"
    A spec change after labeling has started changes what the labels mean. Version the spec, and
    treat a change to a policy or a must-never line as a codebook change
    ([phase 8](8-production.md#when-to-re-certify) says what that triggers).

## The input taxonomy

The taxonomy is the grid of situations the agent handles: a few dimensions, each with a few values.
Take the dimensions from the spec's scope and policies (the request type, the customer, the
channel, the policy branch). Write it as a JSON object of dimension to values:

```json
{
  "feature": ["refunds", "exchanges", "order_status"],
  "persona": ["new_customer", "returning", "reseller"],
  "channel": ["chat", "email"]
}
```

`calibrated-agent-evaluation define` turns it into a discovery frame, a covering array in which
every pair of values appears at least once, so a small set of inputs reaches every combination of
two dimensions:

```bash
calibrated-agent-evaluation define --taxonomy taxonomy.json --out frame.json
```

What you'll see, for the taxonomy above (`docs/site/examples/guides/taxonomy.json`):

<div class="wrap-output" markdown>

```text
--8<-- "fixtures/expected/docs/guide-define-frame.txt"
```

</div>

Nine rows cover every pair of values from three dimensions whose full grid has 18 cells. `--strength
3` covers every combination of three dimensions instead, at the cost of more rows. The frame is the
input to `calibrated-agent-evaluation generate` in [phase 2](2-get-traces-in.md);
`calibrated-agent-evaluation cluster` reads the same grid in phase 4 to list the combinations where
no failure has been labeled yet.

With the `run-error-analysis` skill, ask:

> Draft a taxonomy for my_agent from docs/my_agent-behavior.md and build the frame.

The skill writes the taxonomy file with you and runs `calibrated-agent-evaluation define`. It asks
you for the dimensions and does not invent them.

[The stage map](../../concepts/stage-map.md#define-decide-what-youre-going-to-evaluate) explains
the discovery and production-representative frames.

Next: [2. Get traces in](2-get-traces-in.md).
