---
hide:
  - navigation
  - toc
---

# Calibrated Agent Evaluation

<div class="hero" markdown>

<p class="hero-title">Check your LLM judge before you <mark>trust its numbers</mark>.</p>

<p class="hero-lead">
Most teams grade their AI agent with another model, an LLM judge, and ship on what it says. This
toolkit has people on your team label a sample of runs, measures the judge against those labels, and
lets the judge score the runs that gate your CI build only after it passes. Then it tells you how often your agent
really fails.
</p>

<div class="cta" markdown>
[Why check your judge](why.md){ .md-button .md-button--primary }
[Walk through it](walkthrough/index.md){ .md-button }
</div>

</div>

## A judge is a new hire on review duty

<div class="analogy" markdown>

You would not let a new hire's code reviews gate production on their first day. You would read a
few dozen of their calls next to your own first, see where they go easy and where they nitpick, and
sign them off once they agree with you often enough. An LLM judge needs the same treatment, and most
teams skip it.

</div>

![A judge's calls compared with a person's on a sample, then cleared to review alone](assets/images/judge-probation-light.svg#only-light){ .figure }
![A judge's calls compared with a person's on a sample, then cleared to review alone](assets/images/judge-probation-dark.svg#only-dark){ .figure }

This toolkit runs that probation. Your team labels a sample of the agent's runs, the judge grades
the same runs, and the two are compared. A judge that misses too many real failures, or flags too
many good runs, cannot score anything, so its verdicts never reach the build's gate. Changing the
judge's model or prompt starts its probation over.

## What a judge's number hides

![1,000 sample traces, marked by what the judge said and what was true](assets/charts/trace-grid-light.svg#only-light){ .trace-grid }
![1,000 sample traces, marked by what the judge said and what was true](assets/charts/trace-grid-dark.svg#only-dark){ .trace-grid }

--8<-- "fixtures/expected/docs/home-value-panel.md"

??? example "The program behind this picture"

    ```text
    --8<-- "fixtures/expected/docs/example-rate-correction.txt"
    ```

    The sample is synthetic, so the true rate is known. The picture and this output are rebuilt from
    `docs/site/examples/rate_correction.py` on every docs build.

## Four ways a judge's number misleads a team

<div class="grid cards burns" markdown>

-   **It misses a failure**

    The judge passes runs where the agent issues a refund the policy doesn't allow. The dashboard
    is green, and the money is gone.
    [Step 4](walkthrough/judge.md)

-   **It flags failures that aren't there**

    The judge wrongly flags a few good replies in every hundred. When real failures are rare, those
    outnumber them, and the team spends a sprint fixing replies that were fine.
    [Step 4](walkthrough/judge.md)

-   **A generic score hides the real problem**

    "Helpfulness" goes up release after release while the refund-policy mistakes stay where they
    were, because nobody named them.
    [Step 3](walkthrough/cluster.md)

-   **A fix only works on the test set**

    The failure rate drops on the traces the team tuned against. In production, the same failures
    show up under a different name.
    [Step 6](walkthrough/improve.md)

</div>

## How it works

<div class="steps" markdown>

<div class="step" markdown>
<span class="step__n">1</span>
**Read your traces**

Bring the recorded runs of your agent from OpenTelemetry, Langfuse, Inspect, Strands or a file, and
read them before deciding what to measure.
</div>

<div class="step" markdown>
<span class="step__n">2</span>
**Label and name failures**

People on your team mark runs pass or fail in a review UI, and the toolkit groups the failures into
named failure modes.
</div>

<div class="step" markdown>
<span class="step__n">3</span>
**Put a judge on probation**

The judge grades the labeled runs. It may score new runs only after it agrees with your team often
enough and passes a bias check: padding a reply, writing in the judge's own style, or a customer
insisting they are right must not change its verdict.
</div>

<div class="step" markdown>
<span class="step__n">4</span>
**Gate and report**

A certified judge scores new runs. The CI gate passes or fails each failure mode, and the report
gives the failure rate with the judge's mistakes taken out.
</div>

</div>

The [walkthrough](walkthrough/index.md) follows one support team through every step.

## Try it

No API keys and no network. The sample corpus ships with the package:

```bash
pip install calibrated-agent-evaluation
calibrated-agent-evaluation quickstart
```

??? example "What it prints"

    ```text
    --8<-- "fixtures/expected/docs/calibrated-agent-evaluation-quickstart.txt"
    ```

[Install](how-to/install.md) lists the optional extras.

## Works with what you have

It reads traces from [OpenTelemetry GenAI spans](integrate/gallery/otel-genai.md),
[Langfuse](integrate/langfuse.md), [OpenInference](integrate/gallery/openinference.md),
[OpenLLMetry](integrate/gallery/openllmetry.md), [Inspect AI](integrate/gallery/inspect.md),
[Strands Agents](integrate/gallery/strands.md), [Bedrock AgentCore](integrate/agentcore.md) and
plain [datasets](integrate/gallery/datasets.md). It can also sample an agent you only reach over
HTTP. [Connect your agent](integrate/index.md).

---

Package and CLI: `calibrated-agent-evaluation` · import name: `calibrated_agent_evaluation` ·
License: Apache-2.0 ·
Source: [awslabs/calibrated-agent-evaluation](https://github.com/awslabs/calibrated-agent-evaluation)
