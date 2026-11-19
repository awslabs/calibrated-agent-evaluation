# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this project
adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.0] - 2026-11-24

The first public release.

### Added

- A Python library and the `calibrated-agent-evaluation` CLI that take agent traces through one
  pipeline: probe, define, generate, label, cluster, calibrate, bias, judge, run and improve.
- Trace ingestion from OpenTelemetry (OTLP JSON and protobuf, with the OpenInference and
  OpenLLMetry conventions), Inspect AI logs, Strands agents, Langfuse, JSONL and Parquet datasets,
  into a local SQLite/CSV store or an optional Langfuse backend.
- Judge calibration against human gold labels. A judge is certified only when the 95% lower bounds
  on its true-positive and true-negative rates clear the gate, and each gold set is used once.
- Failure rates corrected for judge error (Rogan–Gladen) with an interval, a CI gate on each
  failure mode's pass-rate lower bound, input-drift reports and multiple-comparison correction.
- A CALM bias battery for judges, Good–Turing saturation checks, failure-mode clustering and
  sample-size planners.
- Before-and-after evaluation of a fix against a pre-registered target, with McNemar tests and a
  Holm-corrected guard panel.
- A FastHTML review UI for human labeling, an inter-rater agreement gate, and pairwise preference
  ranking for subjective qualities that never gates.
- An opt-in Amazon Bedrock AgentCore plug-in: account checks, trace ingest, a Runtime executor,
  evaluator judges, online evaluation, dataset publishing and A/B test audits.
- A Claude Code skills plugin with five skills that drive the CLI.
- `calibrated-agent-evaluation quickstart`, which runs a bundled sample corpus end to end, offline
  and with no keys.

[0.1.0]: https://github.com/awslabs/calibrated-agent-evaluation/releases/tag/v0.1.0
