# AgentCore judge replay fixtures

Written offline by `scripts/record_agentcore_evaluate.py`, with no AWS call, for the demo's
`agentcore-judge` stage, which replays them through botocore `Stubber`.

- `get_evaluator.json`: a `GetEvaluator` document for `Builtin.Correctness`.
- `evaluate.json`: for ORD-1002 and CLM-4471, the `Evaluate` request the serializer builds and the
  response.
- `goal-success-evaluator.json`: the live `GetEvaluator` document for `Builtin.GoalSuccessRate`,
  read on 2026-11-05. The built-in's ARN carries no account, and the document holds
  no account, runtime or caller id, so it is committed as returned.
- `price_list.json`: the Price List `GetProducts` request for us-east-1 and its entries.

These are synthetic recordings of live values. Each trace's value and label, the evaluator's type,
level and rating scale, and the Price List rates were recorded live on 2026-11-03 in an AWS
test account; each explanation is abridged from the live run's. The evaluator's instructions, rung
definitions, model id and timestamps are placeholders. No live response is committed, so the files
hold no account, runtime or caller id; the recorder refuses to write one.

## Names rewritten after recording

The names in these recordings that carried the project's earlier working name (resource names,
session ids, span attribute keys and scope names) were rewritten in place by
`scripts/rewrite_recordings.py`; see "Names rewritten after recording" in `fixtures/README.md`.
Nothing else in them changed.
