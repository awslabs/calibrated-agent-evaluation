# Docs writing style

<!-- writing-standards: ignore-file (this page quotes the patterns it bans) -->

The voice of the Calibrated Agent Evaluation docs is a well-maintained open-source Python project (think the READMEs and
docs of httpx, pydantic, uv, FastAPI): plain, example-first, and calm. State what a thing does, then
show it. Let the output and the method names carry the credibility. Do not sell.

This file is the standard for every page under `content/`. When in doubt, cut a sentence.

## Rules

1. **Example before prose.** Lead a section with the command, the code, or the output. Explain after,
   briefly. Do not write a hype sentence before a code block.

2. **One hook per idea, used once.** Do not repeat a framing device across pages: a hook that opens
   one page (for example the 14.8% versus 6.0% panel on the home page) does not open another.
   Repetition reads as generated copy.

3. **No aphorisms, mic-drops or slogans.** Delete sentences whose job is to make the reader feel
   something: "That's the whole point.", "A refused number is more honest than a confident wrong
   one.", "Calibrated Agent Evaluation makes you look before you measure." State the fact and move on. The one exception
   is the tagline "Mark the doubtful. Measure the rest.", which appears only in the home page hero
   and the README hero and nowhere else.

4. **Bold is for defined terms, not emphasis.** Bold a term of art the first time it is defined
   (**true-positive rate**, **discovery frame**). Do not bold for stress (`**one** trace`,
   `**zero** copyleft`), and do not open a paragraph or a list item with a bold run-in label
   (`**Goal:**`, `**Pitfall.**`). When a page needs scannable steps or pitfalls, give each a `###`
   heading. If a page has more than a handful of bold spans, it is over-bolded. Material card grids
   (a bold link title per card) are the exception.

5. **No em-dashes (—).** They read as an AI tell. Use a comma, a colon, parentheses, or two
   sentences. Keep en-dashes (–) only in two-surname method names: Rogan–Gladen, Good–Turing.

6. **Active voice, second person, present tense.** "Run `calibrated-agent-evaluation judge`", not "the judge is run".
   Name the actor.

7. **Cite the method by name.** Rogan–Gladen, Good–Turing coverage, Wilson lower bound,
   Jeffreys interval, Holm, BY-FDR. For a statistics tool, the named method is the credibility
   signal.
   Link each to its concept page or API symbol where it first appears. A name is not a source:
   every method a page relies on must have its primary reference in
   `content/concepts/methodology.md`, and a method new to the docs gets its entry there in the same
   change. Cite only references that resolve (a DOI, an arXiv id, or a publisher record); never
   write one from memory.

8. **Keep explanation out of how-tos.** Guides are imperative and terse and link to Concepts for the
   "why". Concepts is where the reasoning and alternatives live.

9. **Show real output, captured.** After a command, show its actual output block, then one line on
   what it means. Every output block is a golden: register the command in `docs_examples.py`,
   capture it with `just docs-goldens`, and include the file from `fixtures/expected/docs/`. Never
   paste or hand-write output into a page; a pasted block drifts and no check catches it.

10. **Admonition vocabulary.** Use a small, consistent set: `tip` for shortcuts, `warning` for
    statistical foot-guns, `example` for worked cases, `failure` for an error a reader may hit,
    `note` for an aside to one role (`!!! note "If you lead the team"`), and collapsible `???` for
    troubleshooting, long output and asides. Do not decorate; each callout should carry signal.

11. **Plain names for internal ideas.** Do not use the codebase's internal jargon on the site. Say
    "the local store and Langfuse", not "both worlds".

## Before / after

> Most eval tooling stops at "the LLM judge said 0.82." Calibrated Agent Evaluation bakes in the statistics the popular
> playbook skips.

becomes

> Calibrated Agent Evaluation takes the statistical steps that raw LLM-judge workflows usually skip.

---

> A crew is not a special case — it's a `Trace` with a deeper span tree and handoff spans. <!-- writing-standards: ignore -->

becomes

> A crew is a `Trace` with a deeper span tree, one span per agent step.
