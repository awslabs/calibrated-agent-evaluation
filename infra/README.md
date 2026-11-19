# Infrastructure templates

## `agentcore-live-test-role.yaml`

The IAM role the opt-in live AgentCore tests run as (`just test-agentcore-live`). It grants the
read-only calls `calibrated-agent-evaluation doctor --agentcore` makes, on the named runtime's ARN where the action
supports one and on `*` (pinned to the stack's region) where it doesn't. A feature that needs more
calls adds statements here; `tests/test_live_test_role_template.py` fails on a wildcard action or
on `*` for an action that takes resource ARNs.

It also grants the runtime executor's three calls on one named runtime endpoint (the qualifier the
live tests invoke): `bedrock-agentcore:InvokeAgentRuntime`, which is billed,
`bedrock-agentcore:StopRuntimeSession` and `bedrock-agentcore:GetAgentRuntimeEndpoint`. AWS checks
these actions against both the runtime ARN and the endpoint ARN
(`runtime/<runtime-id>/runtime-endpoint/<endpoint-name>`), so each statement names both. The
invoke and stop grants make this role more than read-only.

For the AgentCore span reader, part of the trace ingest, the role grants the CloudWatch Logs reads
the reader's Logs Insights path makes for a runtime's traces: `logs:StartQuery`,
`logs:GetQueryResults` and `logs:FilterLogEvents` on the shared `aws/spans` group and on the named
endpoint's own group (`/aws/bedrock-agentcore/runtimes/<runtime-id>-<endpoint-name>`). The reader
uses `logs:FilterLogEvents` to tell a lagging Logs Insights index from a group with no content, and
the AWS Service Authorization Reference also lists it against the `StartQuery` and `GetQueryResults`
operations, so it is granted on both groups; it reads nothing the queries cannot. Each group is
named as is and with `:*`, which covers only that group's own streams. `aws/spans` holds the spans
of every runtime in the account and region, and IAM cannot narrow a grant to one runtime's spans, so
the role can read all of them. `logs:StopQuery` takes no resource ARN and the reader never calls it,
so it is not granted, and neither is `logs:Unmask`, which would reveal data a log group's data
protection policy masks. The reader's score reads, which query the evaluation result groups under
`/aws/bedrock-agentcore/evaluations/`, are not granted either. Logs Insights bills each query by the
data it scans, and the billed-call budget (`CALIBRATED_AGENT_EVALUATION_AGENTCORE_MAX_BILLED_CALLS`) does not count these
queries.

For the reader's CloudWatch Omni path (`use_omni=True`), the role grants
`cloudwatch:StartTelemetryQuerySession`, `cloudwatch:StartTelemetryQuery`,
`cloudwatch:GetTelemetryQueryResults` and `cloudwatch:StopTelemetryQuerySession`, plus the
`cloudwatch:GetMetricData`, `cloudwatch:GetRecords` and `cloudwatch:ListMetrics` reads that
`StartTelemetryQuery` authorizes. The SDK client is `cloudwatchomni`, but the IAM actions use the
`cloudwatch` prefix. None of these actions takes a resource ARN the reader knows in advance, so they
are granted on `*`, pinned to the stack's region; they read telemetry and write nothing.

For the AgentCore judge's live test it grants `bedrock-agentcore:Evaluate`, which is billed, and
`bedrock-agentcore:GetEvaluator` on the built-in `Builtin.Correctness` evaluator only. A built-in
evaluator is AWS-owned, and `Evaluate` returns its ARN with no region or account, but IAM was seen
to check it in the caller's own region and account (`evaluator/Builtin.Correctness` under the
stack's region and account), so the grant names that form. It also grants `pricing:GetProducts`,
which reads public prices for the judge's cost estimate, on `*` pinned to `us-east-1`, the Price
List endpoint the judge calls.

For the dataset live test it grants the dataset calls on `dataset/calibrated_agent_evaluation_live_*` only, in the
stack's account and region: AgentCore generates a dataset's id (its name plus a random suffix)
when it is created, so the test names its datasets `calibrated_agent_evaluation_live_<random>` and the grant covers
those. `bedrock-agentcore:CreateDataset` takes no resource ARN, so it has its own region-pinned
statement on `*`. The test deletes every dataset it creates.

For the online judge's live tests it grants `GetEvaluator`, `DeleteEvaluator` and `Evaluate` on
`evaluator/calibrated_agent_evaluation_live_*`, and `GetOnlineEvaluationConfig` and `DeleteOnlineEvaluationConfig` on
`online-evaluation-config/calibrated_agent_evaluation_live_*`: AgentCore generates each id as the name the test gives
plus a suffix. `CreateEvaluator` and `CreateOnlineEvaluationConfig` take no resource ARN, so they
have their own region-pinned statement on `*`. It reads the result log groups
`/aws/bedrock-agentcore/evaluations/results/calibrated_agent_evaluation_live_*`, may pass only the role named by
`EvaluationRoleArn` and only to AgentCore, and
may invoke only the judge model (`JudgeInferenceProfileId` and `JudgeFoundationModelId`, which default to the quickstart judge's Claude Haiku 4.5; `CreateEvaluator` checks that the caller may invoke the evaluator's model), and may publish metrics to the `CalibratedAgentEvaluation/Judges` namespace only. It grants no alarm, dashboard or SNS
call; the operator sets any alarm on those metrics. The tests delete every evaluator and online
config they create.

`EvaluationRoleArn` is the role AgentCore assumes to run an online config, the one the tests read
from `CALIBRATED_AGENT_EVALUATION_AGENTCORE_EVAL_ROLE_ARN`. The account owner creates it outside this stack: it trusts
`bedrock-agentcore.amazonaws.com` for this account's evaluators and online configs, and may read the
span and runtime log groups, write under `/aws/bedrock-agentcore/evaluations/`, and invoke the judge's
model (see the [AgentCore Evaluations prerequisites](https://docs.aws.amazon.com/bedrock-agentcore/latest/devguide/evaluations-prerequisites.html)).

The account owner deploys it with their own credentials:

```sh
just lint-infra
aws cloudformation deploy \
  --stack-name calibrated-agent-evaluation-agentcore-live-test \
  --template-file infra/agentcore-live-test-role.yaml \
  --capabilities CAPABILITY_NAMED_IAM \
  --region us-west-2 \
  --parameter-overrides \
    TrustedPrincipalArn=arn:aws:iam::<account>:role/<your-sso-role> \
    AgentRuntimeId=<runtime-id> \
    AgentRuntimeEndpointName=<endpoint-name> \
    EvaluationRoleArn=arn:aws:iam::<account>:role/<evaluation-role>
```

Then set `CALIBRATED_AGENT_EVALUATION_AGENTCORE_LIVE_ROLE_ARN` to the stack's `LiveTestRoleArn` output and point
`AWS_PROFILE` at a profile that assumes it.

### Redeploying after the online judge's statements were added

The online judge's statements add the `EvaluationRoleArn` parameter, which has no default. To add
them, the account owner reruns the deploy command above with
`EvaluationRoleArn=arn:aws:iam::<account>:role/<evaluation-role>` added. The update changes only
the role's inline policy, description and parameters; the role ARN stays the same.

### Redeploying after the runtime executor's statements were added

A stack deployed before the runtime executor's statements has no `AgentRuntimeEndpointName`
parameter, and the parameter has no default. To add the new grants, the account owner reruns the
deploy command above with `AgentRuntimeEndpointName=<endpoint-name>` added, naming the endpoint the
live tests invoke (for example `DEFAULT` or `live`). The update changes only the role's inline
policy and description; the role ARN stays the same, so `CALIBRATED_AGENT_EVALUATION_AGENTCORE_LIVE_ROLE_ARN` does not
change. To review the change first, add `--no-execute-changeset`, which creates the change set
without executing it, and inspect it before executing it.

### Redeploying after the span reader's statement was added

The span reader's statement uses the `AgentRuntimeId` and `AgentRuntimeEndpointName` parameters
the stack already has, so the account owner reruns the deploy command above with the same
parameters. On a stack deployed from the template as it was just before this statement, the update
adds the statement to the role's inline policy and rewords the stack's description and the role's
own to name these reads and the dataset grants already on the role. Nothing else on the role
changes: its ARN, its trust policy and its other grants stay as they were. As before,
`--no-execute-changeset` creates the change set for review without executing it.

### GitHub Actions

The role does not trust GitHub's OIDC provider, so `.github/workflows/integration-aws.yml` can't
assume it yet and the live tests run only from a maintainer's machine. To enable the workflow, the
account owner adds a `token.actions.githubusercontent.com` OIDC provider to the account and a trust
statement allowing `sts:AssumeRoleWithWebIdentity` with `aud` equal to `sts.amazonaws.com` and
`sub` equal to `repo:awslabs/calibrated-agent-evaluation:environment:aws-integration`, then sets
the `AWS_INTEGRATION_ROLE_ARN` variable on that environment.
