# Redacted capture of a live A/B test

Captured on 2026-11-03 from the integration account's A/B test: two arms, `C` and `T1`, of one
configuration bundle on a test runtime, routed 50/50 by an HTTP Gateway, with 20 requests sent
through the Gateway between 06:16:44Z and 06:18:22Z, one runtime session each.

- `abtest_live.json`: `GetABTest` once AgentCore had posted its results (still `RUNNING`).
- `bundle_version_c.json`, `bundle_version_t1.json`: the two arms' `GetConfigurationBundleVersion`
  responses. Every system prompt is replaced by a canary placeholder; the arms still differ in the
  one key they differ in live, `sql_generator_system_prompt`.
- `insights_spans.json.gz`, `insights_content.json.gz`: the Logs Insights reader's span rows
  (`aws/spans`) and content rows (the runtime's `otel-rt-logs`), in the order the live queries
  returned them. `omni_spans.json.gz`, `omni_content.json.gz`: the Omni reader's rows, same order.

Redaction: the account is `111122223333`; the runtime, Gateway, bundle, A/B test, online
evaluation and role names, and both bundle version ids, are the fixture placeholders, the same in
every file. Span rows keep only the span ids, names, kinds, times, the arm stamps, `session.id`,
`http.route`, the fixture agent's `config.bundle_version` and `config.source`, and
`gen_ai.operation.name`; their resource keeps `service.name` and `aws.log.group.names`. Content
rows keep only the `gen_ai` event records, with their bodies replaced by a canary placeholder.
Trace ids and runtime session ids are the live ones; they name nothing in the account.

What the capture shows: every span carries both arm stamps; node spans carry the trace's own id
as `session.id`, the `POST /invocations` server span the runtime session; the runtime session is
one the runtime assigned, not the session id the client sent to the Gateway. The agent writes its
message content as span attributes (`gen_ai.input.messages`), not as the `otel-rt-logs` records
the ingest re-homes, so the ingest delivers every trace as missing its content.
`fixtures/expected/audit/arms.json.gz` is written from these rows by `make_arms.py`.
