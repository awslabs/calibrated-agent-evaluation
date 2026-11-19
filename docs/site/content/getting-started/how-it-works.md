# How it works

Calibrated Agent Evaluation turns your agent's runs into numbers you can defend. This page walks
through the pieces in five pictures: where traces come from, the order of the work, how a run
becomes a verdict, the checks a judge passes before it scores anything, and the checks before a
failure rate is reported.

## The pieces

--8<-- "docs/site/content/includes/diagrams/three-planes.md"

Your agent's runs arrive as trace files, from a Langfuse project, from a run the library drives
through Inspect or Strands, or from a black-box endpoint it probes. Whatever the source, it converts
each run into one shape, a `Trace`, and refuses a conversion that would lose content instead of
dropping it quietly. Traces live in a store: a local SQLite file by default, or Langfuse. The
evaluation stages read and write that store and never touch your agent's framework directly, so
switching frameworks doesn't change how you evaluate.

## The order of the work

--8<-- "docs/site/content/includes/diagrams/loop.md"

You look at failures before you score anything. Decide what to test, label a sample of runs by
hand, and group the failures into named kinds. Then check a judge against your labels and only
let it score at scale once it passes. Last, gate a change on the judge's verdicts and check that a
fix really worked. The dashed arrows are the loops you'll take most: new kinds of failure send you
back to labeling, and a judge whose prompt or model changed has to be checked again before it
scores. If you don't own the agent, `calibrated-agent-evaluation probe` collects its runs first.
[The stage map](../concepts/stage-map.md) covers each stage.

## From a run to a verdict

--8<-- "docs/site/content/includes/diagrams/agent-to-verdict.md"

Each trace is read in the convention it was written in. If the conversion would drop content, it
is refused and nothing is written. If the full tree of steps comes through, the result can say
which step failed; if only the input and output survive, only the outcome is graded. Your labels
and the judge's verdicts attach to the same traces, which is how the library measures how often
the judge agrees with you, and how a run ends with a gate result and a corrected failure rate.

## The checks before a judge scores

--8<-- "docs/site/content/includes/diagrams/trust-gates.md"

A judge has to pass three checks before it writes a single verdict: its file is well formed, it
agrees with your labels closely enough on both failures and passes, and its verdict doesn't move
when an answer is padded, attributed to another author or pushed back on. Each refusal names what
to collect or fix next.

## The checks before a failure rate

--8<-- "docs/site/content/includes/diagrams/rate-gates.md"

After the judge scores, a corrected failure rate is reported only when the evidence supports
one: enough labels for the accuracy the judge showed, a large enough random sample of scored
traces, and a judge that tells failures from passes clearly. Short of that, you get the range, or a
note saying what to collect.
[How the numbers work](../concepts/how-the-numbers-work/index.md) explains the thresholds.

## Next

- [Quickstart](quickstart.md): run the first half on the bundled sample corpus.
- [Choose your path](../integrate/index.md): get your own agent's traces in.
- [From first trace to production](../guides/lifecycle/index.md): your own agent through every
  stage, from the first traces to a gated pull request and a scheduled production check.
