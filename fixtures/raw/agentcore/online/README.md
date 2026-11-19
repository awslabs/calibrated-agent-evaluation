# AgentCore online judge replay fixtures

Recorded by `scripts/record_agentcore_online.py`. They replay through a fake span reader and
botocore `Stubber` with no AWS call, and the AgentCore ingest and span-reader tests read their
traces as a real decoded capture.

- `traces.json.gz`: 200 OrderBot runtime traces as the span reader decoded them (span records,
  content records, session ids).
- `results.json.gz`: the online config's result record for each of those traces.
- `evaluator.json`: the attached evaluator's `GetEvaluator` document.
- `record.json`: the online record `calibrated-agent-evaluation online attach` stored, with the calibration and bias
  records the judge was attached with.

These are live recordings, read-only, from one window of an online config in an AWS test account
on 2026-11-06: the certified quickstart v2 judge attached at 100% sampling to a deployed
OrderBot, sent the quickstart's 200 production-style refund requests, one session each. The
verdicts (163 PASS, 37 FAIL), the prompts, the agent's replies and tool calls, and the calibration
and bias statistics are as recorded.

Before writing, the recorder replaces the account-id field of every ARN (the 12 digits between
its fourth and fifth colon) with `111122223333`, and every access key id the spans carry with
`REDACTED-ACCESS-KEY-ID`, and the AWS-internal host of the cell that served each invocation
(`http.server_name`, `http.url`) with `agentcore-runtime.invalid`. Nothing else is changed: session ids, span durations and request ids can
hold 12-digit runs too and are kept as recorded. It refuses to write a file that still holds the
recorded account's id anywhere, or an access key id. The placeholder account changes the
evaluator's ARN, and so its fingerprint, so the online, calibration and bias records are re-pinned
to the fingerprint the scrubbed evaluator gives. The
files still name the test account's runtime (`calibrated_agent_evaluation_orderbot-…`), evaluator and config ids and the
runtime session ids; none of them grants access without the account's credentials.

## Names rewritten after recording

The names in these recordings that carried the project's earlier working name (resource names,
session ids, span attribute keys and scope names) were rewritten in place by
`scripts/rewrite_recordings.py`; see "Names rewritten after recording" in `fixtures/README.md`.
Nothing else in them changed.
