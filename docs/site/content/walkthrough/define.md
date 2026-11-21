# 0. Decide what good means

Ask two people on the OrderBot team whether this reply is correct: the customer's keyboard was
delivered 52 days ago, the return window is 15 days, the policy tool said "refund allowed", and
OrderBot refunded it. One says yes, because the agent followed its tool. The other says no, because
the policy forbids it. If they never settle that, their labels disagree on exactly the cases that
cost money, and any judge checked against those labels inherits the disagreement.

So before anyone labels a trace, the team writes down what a correct run is. This step produces two
things: a behavior spec that people read, and a grid of situations that the toolkit reads.

## The behavior spec

The spec is a Markdown page the team keeps next to the agent's code. The toolkit does not read it.
It gives the people who label the same definition of a correct run, and later it supplies the pass
and fail conditions of each judge prompt. If you already have a design doc or a product spec for
the agent, pull these sections out of it; the work is making each rule specific enough that two
people reading the same trace reach the same verdict.

```markdown
# OrderBot: expected behavior

## Scope
- Handles: refund requests, exchanges, order status
- Hands off: chargebacks and fraud reports, to a person
- Out of scope: product recommendations

## Policies
- Final-sale items are never refundable.
- Any other item is refundable if it was delivered no more than return_window_days ago.
- An item reported damaged, broken, defective or not working is refundable up to 90 days
  after delivery, even outside its return window.

## Tool rules
- check_refund_policy: call it before any refund. Check its verdict against the dates in
  lookup_order; do not act on it alone.

## Must never
- issue a refund the policy forbids, whatever a tool reports
- state an amount that does not match the order

## Examples
| Trace id | Verdict | Why |
| --- | --- | --- |
| Q-10000 | FAIL | refunded 52 days after delivery against a 15-day window |
```

Each part has a later use:

- Every line under "Must never" is a candidate failure mode. People start from this list when they
  name failures in [step 3](cluster.md).
- The policies and tool rules become the pass and fail conditions of each judge prompt in
  [step 4](judge.md).
- The examples are for the people labeling. Keep any trace you quote in a judge prompt out of the
  labels you check that judge against; a judge tested on the examples it was shown has been given
  the answers.

!!! warning "Changing the spec changes what the labels mean"
    Version the spec. Once labeling has started, a change to a policy or a "Must never" line is a
    change to the failure modes, and [step 7](production.md) says what that triggers.

## The grid of situations

A handful of test inputs written by one engineer covers the cases that engineer thought of. The grid
makes the team list the situations OrderBot handles, so the gaps show. Take a few dimensions from
the spec (the request type, the kind of customer, the channel), each with a few values, and write
them as JSON:

```json
--8<-- "docs/site/examples/guides/taxonomy.json"
```

=== "CLI"

    ```bash
    calibrated-agent-evaluation define --taxonomy taxonomy.json --out frame.json
    ```

    <div class="wrap-output" markdown>

    ```text
    --8<-- "fixtures/expected/docs/guide-define-frame.txt"
    ```

    </div>

=== "With Claude Code"

    > Draft a taxonomy for OrderBot from docs/orderbot-behavior.md and build the frame.

    The `run-error-analysis` skill writes the taxonomy file with you and runs
    `calibrated-agent-evaluation define`. You supply the dimensions.

The full grid has 18 combinations. Nine rows are enough to put every pair of values (refunds by
email, a reseller on chat, and so on) in front of the agent at least once. `--strength 3` covers
every combination of three dimensions instead, at the cost of more rows.

The file is called a frame, and every row in it is marked `discovery`: these are inputs chosen to
find failures, not a sample of real traffic. [Step 1](generate.md) turns the rows into inputs, and
[step 3](cluster.md) reads the same grid back to list the combinations where nobody has seen a
failure yet.

## What you have now

A written behavior spec, and the grid of situations to test saved as `frame.json`.

Next: [1. Make test inputs](generate.md).
