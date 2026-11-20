<!-- Single source for the site's hover tooltips and the glossary page (gen_glossary.py).
Each entry is one abbr line; the tooltip shows its text, so keep it plain. A "see:" comment after an
entry links the glossary row to the page that explains it. "group:" lines split the glossary. A
"glossary-only:" comment is a glossary row with no tooltip, for words too common to underline
everywhere. -->

<!-- group: Everyday terms -->
<!-- glossary-only: trace: The record of one agent run: its input, its output, and the steps (spans) in between. -->
<!-- see: trace-format.md -->
*[span]: One agent step inside a trace: what it received, the tool calls it made and what it handed on.
*[spans]: One agent step inside a trace: what it received, the tool calls it made and what it handed on.
<!-- see: trace-format.md -->
*[crew]: Several agents working on one task and handing work to each other. Its trace has one span per agent step, linked by parent.
*[crews]: Several agents working on one task and handing work to each other. Its trace has one span per agent step, linked by parent.
<!-- see: how-to/evaluate-a-crew.md -->
*[LLM judge]: A model prompted to grade an agent's trace as pass or fail. Calibrated Agent Evaluation checks it against gold labels before trusting its verdicts.
*[LLM judges]: A model prompted to grade an agent's trace as pass or fail. Calibrated Agent Evaluation checks it against gold labels before trusting its verdicts.
<!-- see: walkthrough/judge.md -->
<!-- glossary-only: label: A pass or fail verdict on a trace, usually with a short critique. A gold label is one a human recorded. -->
<!-- see: how-to/review-ui.md -->
*[failure mode]: A named kind of failure, such as misreading an amount. Labels are grouped into failure modes, and rates are reported per mode.
*[failure modes]: A named kind of failure, such as misreading an amount. Labels are grouped into failure modes, and rates are reported per mode.
<!-- see: walkthrough/cluster.md -->
*[hunted]: Picked on purpose to find failures, so the set holds far more of them than production does. Used to check a judge, never to measure a rate.
<!-- see: walkthrough/judge.md -->
*[certify]: Check a judge against a locked test set of gold labels. It passes only if it catches enough of the real failures and passes enough of the good runs, judged cautiously for how many labels you have.
<!-- see: walkthrough/judge.md -->
*[CI gate]: The pass or fail calibrated-agent-evaluation run --ci gives your build. Every failure mode has to clear the threshold on a cautious reading of the traces you have, or the build fails and names the mode.
<!-- see: walkthrough/gate.md -->
*[sampling frame]: Which population a trace was drawn from: discovery (picked to find failures) or production-representative (picked to match real traffic). Rates are only measured on production-representative traces.
*[sampling frames]: Which population a trace was drawn from: discovery (picked to find failures) or production-representative (picked to match real traffic). Rates are only measured on production-representative traces.
<!-- see: walkthrough/index.md -->
<!-- glossary-only: discovery: Traces picked to find failures, covering many combinations of situations. Never used to measure a failure rate. -->
<!-- see: walkthrough/index.md -->
*[production-representative]: Traces picked to match real traffic. The only kind a failure rate is measured on.
<!-- see: walkthrough/judge.md -->
*[taxonomy cell]: One combination of the situations an agent handles, for example feature, persona and scenario. Each trace records the cell it exercises.
*[taxonomy cells]: One combination of the situations an agent handles, for example feature, persona and scenario. Each trace records the cell it exercises.
<!-- see: walkthrough/index.md -->
*[in-cell]: A trace inside the taxonomy cell a command was limited to with --cell.
<!-- see: walkthrough/index.md -->
*[blind spot]: A taxonomy cell with no observed failure yet. Either the agent is safe there or nobody has tested it.
*[blind spots]: A taxonomy cell with no observed failure yet. Either the agent is safe there or nobody has tested it.
<!-- see: walkthrough/cluster.md -->
*[owned]: A trace from an agent you instrument. Its step-by-step spans let Calibrated Agent Evaluation point at the step that caused a failure.
<!-- see: trace-format.md -->
*[probed]: A trace from an endpoint you can only call: input and output, no internal steps, so analysis stays at the outcome.
<!-- see: integrate/probe-black-box.md -->
*[capability map]: What probing saw an endpoint do, such as the tools its replies mentioned. Always a lower bound, never a full list.
<!-- see: integrate/probe-black-box.md -->
*[gold label]: A pass or fail with a written critique, recorded by a human. Judges and rates are checked against gold labels; a model may pre-fill one but never replaces it.
*[gold labels]: A pass or fail with a written critique, recorded by a human. Judges and rates are checked against gold labels; a model may pre-fill one but never replaces it.
<!-- see: walkthrough/judge.md -->
*[locked test set]: The gold-labeled traces a judge is certified on. Each one can certify a judge once.
<!-- see: walkthrough/judge.md -->
*[certified]: The judge agreed with your gold labels well enough, on both failures and passes, to be trusted to score.
*[CERTIFIED]: The judge agreed with your gold labels well enough, on both failures and passes, to be trusted to score.
<!-- see: walkthrough/judge.md -->
*[rate sample]: A random sample of production traces, scored by the judge, whose flag rate gets corrected. Never the gold-labeled set.
<!-- see: walkthrough/judge.md -->
*[culprit]: In a multi-agent trace, the first step that produced a wrong output from the input it was given.
<!-- see: how-to/evaluate-a-crew.md -->
*[manifestation]: The step where a crew's failure becomes visible, often downstream of the culprit: a payout agent paying a claim that intake misread.
<!-- see: how-to/evaluate-a-crew.md -->
*[relocation]: A fix that lowers one agent's failures while the crew's overall outcome does not improve: the failure moved instead of going away. calibrated-agent-evaluation improve rejects it.
<!-- see: walkthrough/improve.md -->
*[cassette]: A file of recorded model calls. Replaying it reruns an agent or judge with no network and no credentials.
<!-- see: integrate/route-your-agent.md -->
<!-- glossary-only: convention: The attribute naming an agent's OpenTelemetry spans follow, such as GenAI, OpenInference or OpenLLMetry. Ingest detects it and maps it to one trace shape. -->
<!-- see: integrate/gallery/index.md -->

<!-- group: Statistical terms -->
*[TPR]: True-positive rate: of the traces that really failed, the share the judge flagged.
<!-- see: walkthrough/judge.md -->
*[TNR]: True-negative rate: of the traces that really passed, the share the judge left alone.
<!-- see: walkthrough/judge.md -->
*[Wilson bound]: The lowest value a rate could plausibly have, given how many traces it was measured on. Gates read this bound, so a small sample cannot pass on a lucky streak.
<!-- see: walkthrough/judge.md -->
*[Rogan–Gladen]: The correction that undoes a judge's known false alarms and misses to turn its flag rate into an estimate of the true failure rate.
*[Rogan-Gladen]: The correction that undoes a judge's known false alarms and misses to turn its flag rate into an estimate of the true failure rate.
<!-- see: walkthrough/judge.md -->
*[Youden's J]: TPR + TNR - 1, how much better than a coin the judge is. Below 0.5 the corrected rate is too unstable to report as a single number, so only its range is given.
<!-- see: walkthrough/judge.md -->
<!-- glossary-only: coverage: How often a range contains the true value across repeated studies. A 95% range should contain it 95 times in 100. Good–Turing coverage, used when clustering, is a different quantity: the chance a new failure falls in a mode already seen. -->
<!-- see: methods.md -->
*[coverage floor]: The fewest gold labels per class at which the corrected-rate range was measured to hold up, for a judge of a given accuracy: 35 at 0.90, 162 at 0.99. Certifying can need more: a 0.90 judge needs 55 per class to certify, so the plan for it is 55. A stronger rerun found a few gaps above the floor.
<!-- see: how-to/troubleshooting.md -->
*[rate-safe]: The judge was certified on enough gold labels, for the accuracy it showed, to be given a corrected rate.
<!-- see: how-to/troubleshooting.md -->
*[rate-unsafe]: The judge is certified to score but was checked on too few gold labels, for the accuracy it showed, to be given a corrected rate.
<!-- see: how-to/troubleshooting.md -->
*[intersection-union]: A gate that passes only when every check passes, such as the CI gate across failure modes. It needs no correction for running many checks.
<!-- see: walkthrough/gate.md -->
*[CALM]: The set of judge biases Calibrated Agent Evaluation checks: length, self-preference and sycophancy must pass; distraction, bandwagon and compassion fade are recorded; positional applies to judges that compare two answers.
<!-- see: how-to/troubleshooting.md -->
*[placebo]: In the bias check, the unchanged input judged again, to measure how often the judge flips its verdict when nothing changed.
<!-- see: how-to/troubleshooting.md -->
