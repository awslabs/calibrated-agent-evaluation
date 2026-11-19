# AgentCore recommendation fixtures

Hand-built `GetRecommendation` and `GetOnlineEvaluationConfig` responses for the recommendation
audit's tests, written to the installed botocore models (1.43.106) and validated by `Stubber` when
the tests run. The account is AWS's documentation account, `111122223333`, and every id is made
up. Each inline, recommended or explanation text carries the canary string `F036-CANARY-PROMPT`,
so a test can show that no text reaches a stored record.

- `rec_system_prompt.json`: a completed system-prompt recommendation on bundle version `...0001`
  with result version `...0002`, whose agent traces are 10 inline session spans. Their trace ids
  are the H2 replay set's development root (`replay/development.json`).
- `rec_spans_missing.json`: the same with one more span trace id the development root lacks.
- `rec_completed_while_in_progress.json`: a second read of the first, taken while it ran.
- `rec_inline_text.json`, `rec_failed.json` (a result holding only an error code and message),
  `rec_failed_no_result.json`, `rec_tool_bundle.json` (two tools on a bundle, each with its own
  JSON path, recommended text and explanation) and `rec_tool_inline.json` (two inline tools).
- `rec_batch.json`, `rec_cloudwatch.json` and `rec_online.json`: the first recommendation on a
  batch evaluation, a CloudWatch Logs window and an online evaluation; `rec_cloudwatch_extra_group`
  adds a log group the window read does not cover, `rec_cloudwatch_bad_arn` a group ARN with no
  `:log-group:` part.
- `online_eval_*.json`: the online evaluation configuration the online recommendation names, with
  a 5-minute session timeout, with no session configuration, with a log-group name prefix, and
  updated one minute after the widened window starts (`late-update`).

`replay/` holds the H2 replay set (see `../make_replay_roots.py`). `live/` holds a redacted
capture of the integration account's recommendation, taken on 2026-11-03: an inline-text system
prompt read from a CloudWatch Logs window. Its texts are replaced by canary placeholders and its
ids by the fixture ones. The synthetic files above are to be replaced by redacted captures where
a live recommendation of that shape exists; the planted negative cases stay synthetic.
