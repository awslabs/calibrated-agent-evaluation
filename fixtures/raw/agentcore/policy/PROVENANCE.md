# Provenance: Gateway Policy decision spans

## Recorded (real)

Every `<trace ID>.spans.json.gz` named by a 32-hex ID starting `6ae` is the list of span records a
real AgentCore Gateway with a Policy engine wrote to CloudWatch Logs `aws/spans` for one probe of
a recorded run on 2026-10-30, as Logs Insights returned them: each record's `@message`, in the
order returned, duplicates included (the span store holds two or three records of a span), one
record per line, gzipped so the recordings stay small in a review (`gzip -dc <file>` reads one).
Only identifiers were replaced, wherever they appeared:

| Recorded | Replaced by |
|---|---|
| the account ID, in ARNs and `aws.account.id` | `111122223333`, AWS's documentation account |
| the Gateway ID suffix | `abcdef1234` (`calibrated-agent-evaluation-f040-spike-abcdef1234`) |
| the Policy engine ID suffix | `abcdef1234` (`calibrated_agent_evaluation_f040_spike-abcdef1234`) |
| the forbid policy's ID suffix | `abcdef1234` (`calibrated_agent_evaluation_f040_forbid_delete-abcdef1234`) |
| the permit policy's ID suffix | `abcdef5678` (`calibrated_agent_evaluation_f040_permit_rows-abcdef5678`) |
| the Gateway target ID | `ABCDEFGHIJ` |

The request IDs (`aws.request.id`, `target.downstream_request_id`) are kept: they are random per
request and name no account or resource.

The run called the Gateway's MCP endpoint directly, with no agent: three probes per kind in each
phase, `read_rows` with a small limit (the "A" probes), `read_rows` with a limit above what the
permit policy allows (the "B" probes) and `delete_rows` (the "C" probes), plus one `tools/list`
call per phase.

| Battery in `feeds.json` | Gateway | Policies |
|---|---|---|
| `phase0` | no Policy engine attached | none |
| `phase1a` | engine attached, log-only | none |
| `phase1b` | engine attached, log-only | a permit on `read_rows` up to limit 100, a forbid on `delete_rows` |
| `phase2` | engine attached, enforcing | the same two |
| `phase3` | engine attached, enforcing | the same two, the forbid policy itself in log-only mode |

`fixtures/expected/probe/policy_decisions.json` gives, per recorded trace, the decisions the run's
own probe log expected and observed for it (tool, decision, mode, the determining policy and any
flipping policy), not anything the extractor computed. A log-only gateway decides asynchronously
and records no per-call decision, so its traces have none.

## Derived from the recordings

- `silent`, `partial` and `planted-agent-trial` in `feeds.json` re-use recorded traces: `silent`
  feeds the `delete` case the phase-0 traces, and `partial` gives it two phase-2 traces and one
  phase-0 trace.
- `planted.json` names three traces built from recorded ones, each under a made-up trace ID
  starting `f0400000`: `planted-no-tool` is the first phase-2 delete trace with the Gateway span's
  `tool.name` removed; `planted-no-selector` is the same trace with the decision span's
  `rpc.method` removed; `planted-agent-trial` joins the phase-2 `tools/list` spans and the first
  phase-2 bulk-read spans under one trace ID, the shape of one agent trial that lists its tools
  and then calls one.

## Hand-written

- The prompts in `cases.json` and `cases_bare_names.json`: the run sent tool calls, not agent
  prompts, so the prompts only label the cases. The expectations (`read_rows` allowed,
  `delete_rows` denied) are the policies' intent.
- `cases_bare_names.json` names the tools without the Gateway target prefix, the wrong form, to
  exercise the refusal for a case file whose tools match no decision.

## Names rewritten after recording

The names in these recordings that carried the project's earlier working name (resource names,
session ids, span attribute keys and scope names) were rewritten in place by
`scripts/rewrite_recordings.py`; see "Names rewritten after recording" in `fixtures/README.md`.
Nothing else in them changed.
