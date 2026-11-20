# Docs writing style

<!-- writing-standards: ignore-file (this page quotes the patterns it bans) -->

The site has two jobs. It has to convince an engineer who ships an agent, and the lead who decides
whether the team adopts this, that an LLM judge needs checking before anyone trusts its numbers.
Then it has to show them how. It argues with what happens to a team, and it teaches with one
running example. This file is the standard for every page under `content/`. When in doubt, cut a
sentence.

## The running example

Every page that teaches uses the same team: a customer-support team whose agent, OrderBot,
answers order questions and issues refunds. OrderBot and ClaimsCrew are the reference agents in
`corpus/agents/`. The data is synthetic, and pages say so where a number appears for the first
time. Do not invent a second team.

The team runs into four problems over the walkthrough, and each one is told once, on the page
where it belongs:

1. The judge misses a failure: runs that refund an item outside policy pass (step 4).
2. The judge over-flags: on a mostly healthy agent its false alarms outnumber the real failures,
   and the team spends a sprint on failures that aren't there (step 4).
3. A generic score hides the failure: "helpfulness" goes up while the named failures stay flat
   (step 3).
4. A fix moves failures instead of removing them (step 6).

Show a problem on OrderBot's own captured output when the sample can show it. When it can't, use
another reference sample (ClaimsCrew, the synthetic rate sample, the AgentCore audit) and name it.
Never present another sample's numbers as OrderBot's.

## Rules

1. **Argue with outcomes.** When a page needs to say why a step matters, say what happens to a
   team that skips it, shown on OrderBot or on a cited public incident. Do not argue with
   adjectives ("rigorous", "honest", "principled") or with the name of a statistical method.

2. **No statistics outside Methods.** Pages outside Reference do not name a statistical method
   (Rogan–Gladen, Wilson, Jeffreys, Good–Turing, McNemar, Holm, Krippendorff) or use TPR, TNR,
   kappa or "confidence interval". Say what the number means in the reader's terms: "the judge
   catches 9 of every 10 real failures", "a cautious estimate", "somewhere between 2% and 9%".
   Link to [Methods](methods.md) once, at the end of a section, for readers who want the method.
   Every method the toolkit uses has its primary reference there, and a method new to the docs gets
   its entry in the same change. Cite only references that resolve (a DOI, an arXiv id, or a
   publisher record); never write one from memory.

3. **Example before prose.** Lead a section with the trace, the command or the output. Explain
   after, briefly.

4. **Show real output, captured.** After a command, show its actual output, then one line on what
   it means. Every output block is a golden: register the command in `docs_examples.py`, capture it
   with `just docs-goldens`, and include the file from `fixtures/expected/docs/`. Never paste or
   hand-write output into a page.

5. **Say what the tool does, not what it refuses to do.** When the toolkit declines to give a
   number, say what it prints and what to collect next. Do not write defensive guarantees
   ("never silently", "rather than quietly", "can never look like").

6. **One hook per idea, used once.** A number, a story or an analogy that carries one page does not
   come back on another. Link to it instead.

7. **No aphorisms, mic-drops or slogans.** Delete sentences whose job is to make the reader feel
   something: "That's the whole point.", "Zero breaks is not a clean bill." State the fact and move
   on. The hero line "Check your LLM judge before you trust its numbers." appears on the home page
   and in the README only.

8. **Bold is for defined terms, not emphasis.** Bold a term the first time it is defined. Do not
   open a paragraph or list item with a bold run-in label (`**Goal:**`). When a page needs
   scannable steps, give each a `###` heading. Material card grids are the exception.

9. **No em dashes (—).** Use a comma, a colon, parentheses, or two sentences. En dashes (–) stay in
   two-surname method names: Rogan–Gladen, Good–Turing.

10. **Active voice, second person, present tense.** "Run `calibrated-agent-evaluation judge`", not
    "the judge is run". Name the actor.

11. **Avoid stock cadences.** No "not X but Y" or "X, not Y" reversals used for rhythm, no triads
    added for rhythm, no "Here's the thing", no closing summary of what the page just said.

12. **Admonitions carry signal.** `tip` for shortcuts, `warning` for a mistake that changes a
    decision, `example` for a worked case, `failure` for an error the reader may hit, and
    collapsible `???` for long output and asides. At most a few per page.

13. **Plain names for internal ideas.** Do not use the codebase's internal jargon on the site
    ("planes", "lanes", "atoms" outside their reference pages).

## Where things go

- **Why** argues. It is the only page whose main job is persuasion.
- **Walkthrough** teaches the loop on OrderBot, one step per page. Each step opens with the problem
  it prevents, shows the CLI and Claude Code ways to do it in tabs, and ends with what you now have.
  The reasoning a step needs lives on that step, in plain language.
- **Connect your agent** and **How-to** are imperative and terse. They link to the walkthrough step
  that explains why.
- **Reference** holds the CLI and API, formats, the glossary, and Methods, where every formula and
  citation lives.

## Before / after

> Calibrated Agent Evaluation corrects the raw rate with Rogan–Gladen, using the judge's TPR and TNR.

becomes

> The judge flagged 148 of 1,000 replies. It also flags some good replies, at a rate you measured
> against your own labels, so the toolkit takes those out: about 60 replies really failed.

---

> A label that was not recorded never looks recorded.

becomes

> An unsaved label shows as blank in the export.
