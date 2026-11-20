---
hide:
  - navigation
  - toc
---

# Calibrated Agent Evaluation

<div class="hero-tagline" markdown>
Error analysis and judge calibration for AI agents.<br>Mark the doubtful. Measure the rest.
</div>

<p class="hero-lead">
Most teams grade their AI agent with another AI model, called an LLM judge. Calibrated Agent Evaluation checks that judge
against answers from people on your team, then reports how often your agent really fails. When the
evidence is too thin to say, it tells you so, and how much more to collect.
</p>

## What a judge's number hides

![1,000 sample traces, marked by what the judge said and what was true](assets/charts/trace-grid-light.svg#only-light){ .trace-grid }
![1,000 sample traces, marked by what the judge said and what was true](assets/charts/trace-grid-dark.svg#only-dark){ .trace-grid }

--8<-- "fixtures/expected/docs/home-value-panel.md"

[See it step by step](why/index.md).

??? example "The program behind this picture"

    ```text
    --8<-- "fixtures/expected/docs/example-rate-correction.txt"
    ```

    The sample is synthetic, so the true rate is known. The picture and this output are rebuilt from
    `docs/site/examples/rate_correction.py` on every docs build.

## Start where you are

<div class="grid cards audience" markdown>

-   **[New to agent evals](why/primer.md)**

    Five pictures: what a trace, a judge and a label are, and why the judge needs checking.

-   **[Deciding whether to ship](why/leaders.md)**

    What the toolkit tells you, what it costs your team, and how to read its answer.

-   **[Building an agent](getting-started/quickstart.md)**

    Run it on the sample corpus in a few minutes, then [connect your own traces](integrate/index.md).

-   **[Checking the math](concepts/methodology.md)**

    The methods, the published work behind each one, and the coverage studies.

</div>

## Three possible answers

<div class="grid cards outcomes" markdown>

-   **Pass**

    Every failure mode is under your limit, even at the cautious end of its range.

-   **Fail**

    At least one failure mode is over your limit. The verdict names it.

-   **Not enough evidence**

    The judge isn't checked yet, or there are too few traces. The verdict says what to collect.

</div>

[Every refusal and its reason](concepts/how-the-numbers-work/refusals.md).

## How it works

<div class="steps" markdown>

<div class="step" markdown>
<span class="step__n">1</span>
**Bring your traces**

The recorded steps of each agent run, from OpenTelemetry, Langfuse, Inspect or Strands.
</div>

<div class="step" markdown>
<span class="step__n">2</span>
**Label a sample**

People on your team mark a sample of runs pass or fail in the review UI: about 55 failures and
55 passes for a judge you expect to be right 90% of the time.
</div>

<div class="step" markdown>
<span class="step__n">3</span>
**Check the judge**

Calibrated Agent Evaluation measures the judge against those labels, and tests whether longer answers or pushback sway it.
</div>

<div class="step" markdown>
<span class="step__n">4</span>
**Report or gate**

A corrected failure rate with its range, and a pass, fail or refusal for your CI build.
</div>

</div>

[How a trace becomes a verdict](getting-started/how-it-works.md).

## Try it

From a checkout of the repository, with no API keys and no network
([Installation](getting-started/installation.md) has the clone command):

```bash
uv sync
uv run calibrated-agent-evaluation quickstart
```

??? example "What it prints"

    ```text
    --8<-- "fixtures/expected/docs/calibrated-agent-evaluation-quickstart.txt"
    ```

The [Quickstart](getting-started/quickstart.md) continues from here.

## Works with what you have

It reads traces from [OpenTelemetry GenAI spans](integrate/gallery/otel-genai.md),
[Langfuse](integrate/langfuse.md), [OpenInference](integrate/gallery/openinference.md),
[OpenLLMetry](integrate/gallery/openllmetry.md), [Inspect AI](integrate/gallery/inspect.md),
[Strands Agents](integrate/gallery/strands.md) and plain [datasets](integrate/gallery/datasets.md). It can also sample an agent you only reach over
HTTP. [Choose your path](integrate/index.md).

---

Distribution: `calibrated-agent-evaluation` · CLI: `calibrated-agent-evaluation` ·
import name: `calibrated_agent_evaluation` · License: Apache-2.0 ·
Source: [awslabs/calibrated-agent-evaluation](https://github.com/awslabs/calibrated-agent-evaluation)
