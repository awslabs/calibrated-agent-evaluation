# Guides

Task-oriented how-tos for putting Calibrated Agent Evaluation to work. Each one assumes you've run
the [Quickstart](../getting-started/quickstart.md).

!!! tip "Wiring Calibrated Agent Evaluation into your own app?"
    Start with **[Integrate](../integrate/index.md)**: the end-to-end on-ramps from your agentic
    app (ingest existing traces, route your agent through, or probe a black box) to a gated result.
    These guides then go deep on individual tasks.

<div class="grid cards" markdown>

- **[From first trace to production](lifecycle/index.md)**: take your own agent through every stage, from its first traces to a gated pull request and a scheduled production check.
- **[Calibrate a judge](calibrate-a-judge.md)**: measure a judge against human labels and decide whether it's trustworthy enough to score with.
- **[Gate a PR in CI](gate-in-ci.md)**: fail a pull request when a failure-mode rate regresses.
- **[Review traces in the UI](review-traces.md)**: the fast hotkey labeling surface.
- **[Evaluate a multi-agent crew](evaluate-a-crew.md)**: attribution across agents and handoffs.
- **[Rank subjective qualities](rank-subjective-qualities.md)**: pairwise preferences and Bradley–Terry ratings, tracked and never gated.
- **[Run fully offline](offline.md)**: the air-gapped path with no cloud and no API keys.
- **[Troubleshooting](troubleshooting.md)**: what each refusal means and what to do about it.

</div>

For shorter single-job pages, see [Recipes](../recipes/index.md). For what each stage does, see
the [stage map](../concepts/stage-map.md); for the exact flags of a command, the
[CLI reference](../cli/index.md).
