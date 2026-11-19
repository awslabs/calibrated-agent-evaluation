# How the numbers work

You don't need this section to use Calibrated Agent Evaluation. Every command prints what happened
and what to do next in plain terms. Read on when you want to know why it asked for 55 labels instead
of 20, why it reported 6% when your judge said 15%, or why it refused to give you a number at all.

Each page starts from a small example with real numbers and builds up to the method. Formulas come
last, and you can skip them.

1. [Why a judge's raw rate is wrong](raw-rate.md): a judge that is right 90% of the time can report
   a failure rate more than twice the real one.
2. [Checking a judge against your labels](checking-a-judge.md): how many human labels it takes to
   trust a judge, and why the plan picks a size that more labels cannot undo.
3. [Correcting the rate](correcting-the-rate.md): undoing the judge's known mistakes, and the one
   rule about which traces you may use.
4. [Why the range matters](the-range.md): what "between 0% and 13%" promises, how often that promise
   was measured to hold, and where it does not.
5. [When the library refuses, and why](refusals.md): every refusal, what triggers it, and what fixes it.
   This includes the judge bias check.
6. [The CI gate](ci-gate.md): how many traces a failure mode needs before a build can pass.
7. [Did the fix work?](did-the-fix-work.md): before/after comparisons, and the crew fix that only
   moved the failure.
8. [Error clusters and when to stop labeling](error-clusters.md): how the library decides you have seen
   enough failures.

The checks a judge passes before its numbers reach you:

--8<-- "docs/site/content/includes/diagrams/trust-gates.md"

And the checks before a failure rate is reported:

--8<-- "docs/site/content/includes/diagrams/rate-gates.md"

## For reviewers

The methods, the code that implements each one, the primary sources, the coverage evidence and the
known limits are on a separate page: [Reviewer notes](../reviewer-notes.md). It is written for a
reader checking the claims, not for a reader learning them.
