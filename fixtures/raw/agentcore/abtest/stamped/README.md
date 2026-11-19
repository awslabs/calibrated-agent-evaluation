# Synthetic stamped traces for the A/B arm read

Each file is one read window, as a trace source delivers it: a list of `SourceTrace` documents,
each an OTLP-JSON trace under the source id it arrived as. They are built from the table in
the A/B audit spec's stamped-trace table by `tests/_arms_fixture.py`, and a
test keeps them equal to it. The `config.bundle_version` and `config.source` span attributes are
the fixture agent's, passed as operator flags, never built into Calibrated Agent Evaluation. Account `111122223333` is
AWS's documentation account; every id is made up. The windows are `main`, `half-stamped`, `repeated-source-id`, `unstamped`,
`other-experiment-only`, `second-window`, `changed-stamp`, `grown` and `shrunk`. A redacted capture
of a live A/B test's rows is in `../live/`; these synthetic windows stay for the planted cases. The windows over 8 KB (`main`, `second-window`, `changed-stamp`, `grown`, `shrunk`) are stored gzipped as `<window>.json.gz`.
