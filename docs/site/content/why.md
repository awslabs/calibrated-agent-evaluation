# Why check your judge

In April 2025 OpenAI shipped a GPT-4o update that flattered users and agreed with them too readily,
and rolled it back within days. In its account of what went wrong, OpenAI wrote that its offline
evaluations "generally looked good" and that the small number of A/B test users "liked it". Some
expert testers had said the model "felt" slightly off. But, in OpenAI's words, "we also didn't have
specific deployment evaluations tracking sycophancy."[^openai]

The people closest to the model noticed something, and the measurements said everything was fine,
because none of them looked for that failure.

Smaller teams hit the same gap with support agents. Air Canada's chatbot told a customer he could
claim a bereavement fare after his trip, which the airline's policy did not allow, and a tribunal
made the airline pay.[^aircanada] Cursor's support bot told users a login rule existed that didn't,
and users cancelled before the company corrected it.[^cursor] New York City's MyCity bot told
business owners they could take a cut of workers' tips and refuse housing vouchers, both
illegal.[^mycity] In each case the agent stated a policy that wasn't real, in replies that someone who knew the
policy could have read before launch.

## Your judge is untested code

Most teams that measure an agent at all do it with another model: an LLM judge that reads each run
and says pass or fail. The judge's verdicts feed a dashboard and, increasingly, a CI gate. A judge
is a prompt and a model, which makes it code, and in most teams it is the only code in the release
path that nobody has tested.

An untested judge goes wrong in ways the dashboard can't show:

- It misses the failure you care about. When failures are rare, a judge that passes everything
  agrees with your reviewers almost all the time. Hamel Husain's example: if 5% of runs fail, a
  judge that always says pass has 95% agreement and catches none of them.[^hamel-judge]
- It flags runs that are fine. Every judge makes some mistakes in both directions. On a mostly
  healthy agent, the good runs it wrongly flags can outnumber the real failures, so the dashboard
  can report double the true failure rate.
- It drifts when you touch it. Edit the judge's prompt or move it to a newer model and its verdicts
  change. The dashboard moves, and nobody can say whether the agent changed or the judge did.
- It has tastes. Judges tend to prefer longer answers and answers that sound like their own
  writing, and they back down when a user insists. An agent tuned against such a judge learns to
  pad its replies and agree with customers.

The check that catches all four is the one you would give a new hire before letting their reviews
gate production: label a sample of runs yourself, grade the same runs with the judge, and compare.

## Look before you measure

A judge can only check for failures someone has named. The common shortcut is a generic score such
as "helpfulness from 1 to 5", which a team can improve for months while the specific thing that
hurts its users stays flat.

Other people made this case first.
Hamel Husain calls error analysis, reading real traces and writing down what went wrong, the most
valuable activity in AI development.[^hamel-field] Shreya Shankar and colleagues found that people
can't write good grading criteria before they have graded outputs, because reading outputs is how
the criteria form.[^shankar] Eugene Yan's advice is the same: look at the data first, and don't
expect a judge to fix a process that skips that step.[^yan]

This toolkit builds on that practice and turns the advice into checks that run on every build:

- Error analysis comes before scoring. Failures are named from human labels on real traces, and the
  toolkit tells you when new labels have stopped turning up new kinds of failure.
- A judge is checked against your team's labels before it scores anything. One that misses too
  many real failures, or flags too many good runs, is not allowed to produce verdicts, so it can't
  gate a build. When the labels are too few to decide, it says how many more of each kind to
  collect.
- The check is pinned to the judge's model and prompt. Change either and the judge has to pass
  again.
- The judge is tested for the tastes above: padding a reply, writing in its own style, and a user
  pushing back must not change its verdict.
- The CI gate decides each failure mode separately, so one bad mode can't hide behind a good
  average, and it needs enough traces to be sure before it passes a mode.
- The failure rate on the dashboard has the judge's measured mistakes taken out, with a range that
  says how sure the number is.

## What happens to one team

The [walkthrough](walkthrough/index.md) follows a support team whose agent, OrderBot, answers
order questions and issues refunds. OrderBot ships with the toolkit, so you can run every step yourself. Each problem below is shown on
OrderBot where its data can show it, and on another sample that ships with the toolkit where it
can't.

![Four ways a judge's number misleads: a missed failure, over-flagging, a generic score, and a fix that only works on the eval set](assets/images/four-ways-light.svg#only-light){ .figure }
![Four ways a judge's number misleads: a missed failure, over-flagging, a generic score, and a fix that only works on the eval set](assets/images/four-ways-dark.svg#only-dark){ .figure }

1. The judge passes runs where OrderBot refunds an item outside the return window, because a long,
   confident reply sways it. [Step 4](walkthrough/judge.md) shows the check that caught OrderBot's
   first judge.
2. The judge flags more good replies than there are real failures, and the team spends a sprint
   on replies that were fine. [Step 4](walkthrough/judge.md) shows it on a synthetic sample where
   the true rate is known.
3. "Helpfulness" improves while the failures that matter stay flat, because nobody named them.
   [Step 3](walkthrough/cluster.md) shows a helpfulness score rising on a sample SQL agent.
4. A prompt fix cuts one failure mode, and the failures turn up under another.
   [Step 6](walkthrough/improve.md) shows it on the ClaimsCrew sample and compares a fix against
   every failure mode, not only the one it targeted.

## Where this fits

Keep the tools you have. Tracing and observability tools such as Langfuse, Arize Phoenix and
OpenTelemetry collectors record what your agent did. Eval runners and metric libraries such as
Inspect, Ragas and DeepEval run judges and metrics over it. This toolkit reads traces from those
tools and adds the step most of them leave to you: deciding whether a judge agrees with your team
well enough to trust, and refusing to use it until it does. When we read their docs in October 2026, Langfuse and
Ragas could show how often a judge agrees with human labels, and neither turned that into an accept
or refuse decision sized by how many labels you have.

It is the wrong tool when you need tracing or production dashboards (use Langfuse, Phoenix or
AgentCore Observability), or ready-made benchmarks (Inspect ships hundreds).

## If you decide whether to ship

A gated build passes or fails, and when it can't decide, the message says what to collect.

| Result | What it means | What your team does |
| --- | --- | --- |
| Pass | Every failure mode clears your limit, even on a cautious reading of the traces you have. | Ship. |
| Fail | At least one failure mode doesn't clear it. The result names the mode. A mode with too few traces fails too, because a handful of clean runs can't show it clears the limit. | Fix the mode, or score more runs if it only needs more traces. |
| Refused | There is nothing to gate: no verdicts from a judge that passed its check, or none on randomly drawn runs. | Run the step the message names first. |

The cost is mostly people's time. Before a judge can score, someone who knows what a good reply
looks like labels a sample of its runs: about 55 failures and 55 passes for a judge you expect to be
right about 9 times in 10. Quoting the corrected failure rate needs more, up to 162 of each, and at
least 150 randomly drawn runs for the judge to score. When the sample judge kit was recorded on
Bedrock, its judge made 944 model calls for $2.11, about $2.24 per 1,000. Everything the toolkit
computes runs in seconds on a laptop. A judge that passes its check is reused until its model or
prompt changes.

## When this is overkill

- You can't get anyone to label runs. Without human labels no judge can be checked, and nothing on
  this site applies.
- Your checks are deterministic. If "did it call the refund tool with the right amount" is decided
  by code, score it with code (`calibrated-agent-evaluation score`). You still need error analysis
  to know what to check, but there is no judge to put on probation.
- Nobody uses the agent yet. Before real traffic, reading a handful of runs teaches you more than
  any measurement.
- Failures are frequent and obvious. If half the runs are visibly broken, fix them first. The
  check earns its keep once the remaining failures are rare enough to hide.

[^openai]: OpenAI, "Expanding on what we missed with sycophancy", May 2025.
    <https://openai.com/index/expanding-on-sycophancy/>
[^aircanada]: The Guardian, "Air Canada ordered to pay customer who was misled by airline's
    chatbot", 16 February 2024.
    <https://www.theguardian.com/world/2024/feb/16/air-canada-chatbot-lawsuit>
[^cursor]: The Register, "Cursor AI support bot hallucinated its own company policy", 18 April
    2025. <https://www.theregister.com/2025/04/18/cursor_ai_support_bot_lies/>
[^mycity]: The Markup, "NYC's AI Chatbot Tells Businesses to Break the Law", 29 March 2024.
    <https://themarkup.org/news/2024/03/29/nycs-ai-chatbot-tells-businesses-to-break-the-law>
[^hamel-judge]: Hamel Husain, "Using LLM-as-a-Judge For Evaluation: A Complete Guide".
    <https://hamel.dev/blog/posts/llm-judge/>
[^hamel-field]: Hamel Husain, "A Field Guide to Rapidly Improving AI Products".
    <https://hamel.dev/blog/posts/field-guide/>
[^shankar]: Shankar et al., "Who Validates the Validators? Aligning LLM-Assisted Evaluation of LLM
    Outputs with Human Preferences", UIST 2024. <https://arxiv.org/abs/2404.12272>
[^yan]: Eugene Yan, "An LLM-as-Judge Won't Save The Product, Fixing Your Process Will".
    <https://eugeneyan.com/writing/eval-process/>
