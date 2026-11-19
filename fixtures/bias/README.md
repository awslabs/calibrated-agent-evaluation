# Bias-probe fixture slice

> The controlled probe corpus the **bias-check promotion gate** is tested against, plus
> the per-fixture oracles.

## What this is

A small set of **quality-matched controlled probe trials** — one file per CALM bias the battery
measures — and the **fixture judges** whose (synthetic, known) bias behavior the gate is exercised
against. Each *trial* is a group of probe inputs whose fair verdicts should be **identical**: only
the biasing dimension varies (answer length, model attribution, an authoritative-but-wrong
assertion, candidate order). A judge exhibits the bias on a trial when its verdicts across that
trial's inputs disagree (`calibrated_agent_evaluation.bias.probe.BiasProbe.score`).

The judges here are **synthetic controlled judges** (the gold boundary in the constitution: the bias
check evaluates a judge against synthetic controlled probe items, never human gold). A real LLM
cannot be relied on to be *reproducibly* length-biased, so, as calibration ships
`calibration/synthetic_9090.json`, the
fixture judges' verdicts are deterministic and committed to a cassette. The same probes
can run against a **real** Bedrock judge through `calibrated_agent_evaluation.judges.JudgeProbeRunner.live`, for
exploratory execution-plane parity; its verdicts are observed, not asserted against the controlled
oracle.

## Layout

```
fixtures/bias/
  README.md                 # this file
  thresholds.json           # per-(fixture-judge model) per-bias max-allowed magnitude
  probes/
    length.json             # must-pass: semantically-equivalent short vs padded answers
    self_preference.json    # must-pass: the same answer attributed to different models
    sycophancy.json         # must-pass: a correct answer with vs without an authoritative-but-wrong assertion
    compassion_fade.json    # advisory: a sympathetic vs neutral framing of the same facts
    positional.json         # pairwise-only: an order-sensitive candidate pair (order-swap-and-tie)
fixtures/expected/bias/
  clean.json                # oracle: the clean fixture judge → CLEAR
  length_biased.json        # oracle: the length-biased fixture judge → BIASED (failed_biases=[length])
```

## Probe file format

The format, the thresholds file and what a ceiling means are documented on the
site's reference page, `docs/site/content/bias-probe-format.md`. These 6-trial probes declare no
`toward` direction: the directional bound needs at least 11 trials.

`fixtures/VERSION` (== `1`) is the corpus version, which equals the trace schema version
(`calibrated_agent_evaluation.schema.SCHEMA_VERSION`), not the bias record's schema version; the fixture slice is keyed
to that corpus version.

## The committed cassette

The offline cassette (`fixtures/raw/cassettes/bias_fixture_judges.cassette.json`) holds the
fixture judges' deterministic verdicts on these probes. The tests replay it through the real judge
runtime (`run_judge` + `CassetteJudgeModel`), so no network call is made.
