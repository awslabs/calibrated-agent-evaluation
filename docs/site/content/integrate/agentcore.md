# Amazon Bedrock AgentCore

Calibrated Agent Evaluation can run an agent deployed to AgentCore Runtime, probe one, check that an
account is set up for AgentCore, run an agent's generated code in the managed Code Interpreter to
grade it, run the `strands-agents-evals` failure detector over traces you already have, publish the
failures your reviewers caught as AgentCore evaluation datasets, and record what an A/B test claimed
and which arm each trace ran in; each needs the `agentcore` extra. It also reads a runtime's traces
for a time window with `calibrated-agent-evaluation ingest --source agentcore`. This page says what
works today and what is planned.

## What works today

### Run a deployed agent

```bash
uv sync --inexact --extra sampling --extra stats --extra agentcore
```

```text
--8<-- "fixtures/expected/docs/integrate-route-agentcore.txt"
```

`calibrated-agent-evaluation generate --run --executor agentcore` sends each generated input to the
runtime and stores one trace per input, carrying the `discovery` frame and the input's combination
of dimensions. The command above replays the repository's recorded run of a synthetic runtime, so it
needs no account. The flags, the confirmation step and the cassette are in [Route your
agent](route-your-agent.md#your-agent-on-agentcore-runtime).

Each live run goes like this:

1. Check the runtime. Before the first invocation, read-only calls read the endpoint's live
   runtime version (or the runtime's latest when the endpoint names none), its authorizer and its
   protocol. A runtime with a custom JWT authorizer only accepts OAuth, and one that serves MCP,
   A2A or AG-UI rather than HTTP is not invoked; either stops the run before anything billed is sent.
2. Invoke once. The executor makes the run's session ID (`calibrated-agent-evaluation-` and 32
   hex characters) and W3C trace context, and sends them with the `traceparent` and `baggage`
   headers, so the agent's spans can be found by ID. The AWS SDK sends the request once; the
   executor itself retries only a busy session or throttling, in the same session, up to three
   attempts.
3. Read the answer. The response is read as an event stream or one JSON document. The final
   text joins, in order, every string value and, for each object, the first string under
   `response`, `content`, `message`, `text`, `output` or `answer` (then `data` in an event
   stream). An error event, an empty body, or an answer under another key stops the run with a
   message that names the session and leaves out the agent's text.
4. Read the spans. The span reader returns the run's span tree, every span keyed to the
   run's trace ID.
5. Stop the session. Once an invocation was sent, the session is stopped whether the run
   succeeded or failed. A stop that fails is logged, and tried once more when the executor
   closes.

Each run, whatever its outcome, adds one record to the run manifest: the trace and session IDs, the
runtime ARN, qualifier and version (and whether the version came from the endpoint), the
configuration bundle, the timing, the attempts, the response status and how the session was stopped.
The manifest is the pair `agentcore_run_record.json`, which is read back, and
`agentcore_run_record.csv`, one record per run, each file written atomically under a lock on the log
directory. `calibrated-agent-evaluation dump` and `calibrated-agent-evaluation load` do not carry
it.

Each stored trace also carries its run's session ID as `session_id`, the same value as its run
record. `calibrated-agent-evaluation ingest --source agentcore` fills `session_id` from the
`session.id` its spans carry, and a trace whose spans carry none gets none. A mixed-session trace
re-keyed with `--rekey-sessions` gets the runtime session it was re-keyed to, or none when the
reader found no runtime session. A trace ingested before the library stored sessions has no
`session_id`, and pulling the same window again into that store is refused as a different stored
trace. `--overwrite` is refused with `--source agentcore`, so to pick up the sessions, ingest the
window into a fresh store (a new `--root`, or a new Langfuse project).

Live runs read each run's spans with the AgentCore per-run span reader
(`calibrated_agent_evaluation.agentcore.AgentCoreSpanReader`), which finds the run by its trace ID
and waits for its telemetry to settle; the window ingest above is a separate reader for a time
window. `calibrated-agent-evaluation generate` builds the reader for a run that can call AWS; from
Python, pass the executor a span reader as `span_reader=`, such as `AgentCoreSpanReader(clients)`.
The executor's own calls need `bedrock-agentcore:GetAgentRuntime` on the runtime, and
`GetAgentRuntimeEndpoint`, `InvokeAgentRuntime` and `StopRuntimeSession` on the runtime and the
endpoint, as `infra/agentcore-live-test-role.yaml` in the repository grants them.

### Check an account is ready

Install the extra and run the check:

```sh
uv sync --inexact --extra agentcore
calibrated-agent-evaluation doctor --agentcore --region us-west-2 --runtime nl_sql_agent --omni-domain my-domain
```

It checks that AgentCore has an endpoint in the region, that Transaction Search is on, that the
default X-Ray indexing rule covers 100% of spans, that the `aws/spans` and runtime log groups exist,
that the CloudWatch Omni domain is active, and that the runtime sets
`AWS_GENAI_CONTENT_EXTRACTION_OPT_OUT=true` so prompts and responses stay on its spans. Each check
prints pass, fail or unknown with the next step; `--explain` adds what it observed. Every call is
read-only: the command makes no other call and changes no setting, including the ones it reports. It exits 0 only when every check passes. `--runtime` and `--omni-domain` are
optional; without them those checks report unknown. The credentials need the read-only actions the
checks call: `bedrock-agentcore:ListEvaluators`, `ListAgentRuntimes` and `GetAgentRuntime`,
`xray:GetTraceSegmentDestination` and `GetIndexingRules`, `logs:DescribeLogGroups` and
`cloudwatch:ListDomains`. The role in `infra/agentcore-live-test-role.yaml` in the repository grants
them, and grants the other live tests more, none of which the doctor calls; the template names and
explains each of those grants. Among them are invoking one runtime endpoint and stopping its
sessions, creating datasets, and editing, versioning and deleting
`calibrated_agent_evaluation_live_*` ones, so the role is not read-only.

### Draft failure categories with the failure detector

The extra installs `strands-agents-evals`, whose failure detector reads one trace and names what
went wrong from a fixed list of 33 categories in 10 groups. Three names on
`calibrated_agent_evaluation.agentcore` use it, from Python only (there is no CLI command):

- `failure_vocabulary()` reads that list as typed entries, plus `other`;
- `FailureDetector` runs the detector on one normalized, owned trace and returns a `Detection`
  whose failures point at the normalized trace's spans and tool calls;
- `DetectorDraftNamer` names draft failure-mode clusters from the same list, for `build_draft`.

```python
from calibrated_agent_evaluation.agentcore import (
    DetectorDraftNamer,
    FailureDetector,
    detector_bedrock_model,
    failure_vocabulary,
)
from calibrated_agent_evaluation.cluster import build_draft, hierarchical_cluster, ratify

vocabulary = failure_vocabulary()  # 34 entries: the 33 categories, then "other"

# Any strands Model works; pass model_id= if its config has none. This one builds its
# Bedrock client on the first live call.
model = detector_bedrock_model("global.anthropic.claude-sonnet-4-6")

detector = FailureDetector(model, max_live_calls=3)
detection = detector.detect(trace)  # one normalized, owned trace
for failure in detection.failures:
    print(failure.span_id, failure.tool_call_index, failure.category, failure.confidence)

# critiques: your FAIL critiques; embeddings: one vector per critique from your embedder.
assignments = hierarchical_cluster(embeddings)
namer = DetectorDraftNamer(model, vocabulary, max_live_calls=1)
draft = build_draft(critiques, embeddings, assignments, namer=namer)
codebook = ratify(draft, ratified_by="you", descriptions=vocabulary.descriptions())
```

The billed path is opt-in and capped. Each detection and each naming is one model call, billed to
the account behind the model you pass. `max_live_calls` is 0 unless you raise it, and a live call
past the cap raises `DetectorCallLimitExceeded` before it is sent. With `cassette_path=` the calls
are recorded once and then replayed without calling the model, and replays do not count against
the cap; a replay through `detector_bedrock_model` builds no client and needs no credentials.
Replay mode with no `cassette_path` raises `DetectorConfigError` when the detector or namer is
built, because every call would reach the model. Every `Detection` and every naming result
records the detector version, the prompt template and its hash, and the model id and the config
its call was made with.

Neither step writes a label. A draft name becomes a failure mode only when you ratify it, and
`descriptions=` attaches each category's definition to the mode that carries its name.

What to expect when a trace or an answer does not fit:

- A probed trace, or one with no spans, raises `SessionConversionError` before any model call.
- A trace too large for one model call raises `DetectorSessionTooLarge`. The detector would
  otherwise retry it in chunks with more billed calls; the plug-in stops instead.
- An answer that cannot be read exactly, or that names an unknown category or span, raises
  `DetectorOutputError`. A `Detection` with no failures means the model listed none, or that each
  one it listed fell below your `confidence_threshold` (counted in `below_threshold`). An unreadable
  answer raises `DetectorOutputError` instead.
- Traces where any span lacks a time (hand-written fixtures, Inspect logs, datasets) are sent with
  placeholder times on every span, which keep only the span order, and the `Detection` says so. A
  timeout found on such a trace rests on the messages alone.

Confidence is the model's own word (`low`, `medium` or `high`) and has not been checked against
labels. Read a detection as a draft for a reviewer. It is not a judge and does not gate a release.

### Run the agent's code locally

An AgentCore agent is usually a Strands, LangGraph or other framework agent packaged for the
Runtime. Run the same code in a local process and use the [Strands recipe](gallery/strands.md),
[Capture in process](capture-and-ingest.md), or an OpenTelemetry export with
[ingest](ingest-existing-traces.md). You get owned traces with every step.

### Score generated code in the managed Code Interpreter

`calibrated-agent-evaluation score --scorer code` grades an agent's generated code by running it
against the task's tests. By default it runs in the local stdlib sandbox; with `--sandbox agentcore`
it runs on the Bedrock AgentCore Code Interpreter instead. The local sandbox is best-effort and is
not a security boundary: it strips the environment and blocks sockets with Python-level patches
that code can undo, and it cannot stop writes outside its temporary directory. Use
`--sandbox agentcore` for code written by a model you do not trust. The verdict is the same on
either backend:

```sh
uv sync --inexact --extra agentcore
calibrated-agent-evaluation score --scorer code --mode wrong_code --expected tests.jsonl \
  --sandbox agentcore --region us-west-2
```

The candidate code is read from each trace's output; the tests come from the `--expected` file (one
`{"trace_id": ..., "expected": {"tests": ...}}` per line). Each scored trace is one billed
`InvokeCodeInterpreter` call, so the run is bounded: a billed-call budget (`--max-billed-calls`,
else `CALIBRATED_AGENT_EVALUATION_AGENTCORE_MAX_BILLED_CALLS`, else one call per trace) refuses the
call past the cap before it is sent, and `--role-arn` requires the run to execute as a named IAM
role. Without the `agentcore` extra, asking for the backend prints the command that installs it. SQL
scoring (`--scorer sql`) always runs locally on SQLite, so it does not take `--sandbox`. The
verdicts are `kind="code"` labels that `calibrated-agent-evaluation run --verdicts scorer` gates,
with no calibration.

### Probe a deployed agent

```sh
uv sync --inexact --extra agentcore
# RUNTIME_ARN is the runtime's own ARN: arn:aws:bedrock-agentcore:<region>:<account>:runtime/<id>
calibrated-agent-evaluation probe --executor agentcore --runtime-arn "$RUNTIME_ARN" --payload-format agentcore_sse \
  --authorized --yes --prompt "What can you help me with?"
```

This sends each `--prompt`, or the probe's two built-in prompts when none is given, through the same
SigV4 call path as `calibrated-agent-evaluation generate --run --executor agentcore`, signed with
the credentials of `--profile` (default `AWS_PROFILE`). It reads no spans, so it needs no span
reader and runs live today, and each reply is stored as a probed, outcome-only trace: the prompt as
input, the final text as output, no spans. Use it when you may invoke an agent on AgentCore Runtime
but cannot read its spans.

- Each prompt runs in a session of its own, stopped once the reply is read, whatever the outcome.
  Its trace ID holds the run's W3C trace ID, so probing again adds traces and never replaces
  earlier ones.
- Both `--authorized` and `--yes` are required. Without `--authorized` the probe is refused before
  its plan is printed. With it, the plan (the runtime ARN, the qualifier, the configuration bundle
  when one is named, and how many billed invocations) is printed before any billed call, and
  nothing is sent without `--yes`.
- The executor's read-only preflight runs before the first invocation, so a runtime that only
  accepts OAuth, or one that doesn't serve HTTP, is refused before anything billed is sent.
- `--payload-format` is required and never guessed; `custom` is not accepted.
- The first prompt that fails stops the probe, and none of the replies are stored. Every prompt's
  run still adds a record to the run manifest `generate` uses, under `<root>/records/agentcore/` or
  `--log-dir`.
- There is no cassette, so `CALIBRATED_AGENT_EVALUATION_CASSETTE_MODE=replay` is refused.
  `CALIBRATED_AGENT_EVALUATION_AGENTCORE_MAX_BILLED_CALLS` caps the invocations, retries included.
- `--endpoint`, the `--auth-*` flags, `--timeout` (pass `--read-timeout`, default 300 seconds) and
  the `--battery` flags are refused with `--executor agentcore`.

The probe makes the same four calls as the executor, so it needs the same permissions. An endpoint
you call over HTTP with a bearer token, an API key or raw headers still goes through the URL probe
(`--endpoint`), which speaks the AgentCore request shapes and reads server-sent event replies. Both
are in [Probe a black-box endpoint](probe-black-box.md).

### Audit an A/B test or a recommendation

AgentCore Optimization's A/B tests report each arm's average score and the vendor's own judgment
of whether the difference is real. The audit records and prints those as returned, labeled as a
vendor claim, and uses none of them in a verdict or gate.

```sh
uv sync --inexact --extra agentcore
calibrated-agent-evaluation audit abtest --ab-test-id "$AB_TEST_ID" --region us-west-2
calibrated-agent-evaluation audit arms --ab-test-id "$AB_TEST_ID" --runtime my_agent \
  --since 2026-11-02T00:00:00+00:00 --until 2026-11-02T06:00:00+00:00 --rekey-sessions
```

`abtest` reads the test with `GetABTest` and each arm's configuration bundle version, and stores
a claim record: the test's state, each arm, and every evaluator's figures as AgentCore returned
them. Bundle components and prompts are kept only as SHA-256 digests and lengths, per component
and per top-level key, so the record shows which key the arms differ in, or that the test is an
A/A comparison, without storing any prompt. A test that is still running is marked provisional.

`arms` reads the window's traces through the same source as `calibrated-agent-evaluation ingest
--source agentcore` (pass the same `--omni` and `--rekey-sessions` flags you ingested with) and
reads each trace's arm from the experiment ARN and variant name the Gateway stamps on its spans. A
trace is assigned to its arm, or set aside with the reason: no stamp, half a stamp (one of the two
attributes), two arms, another experiment, a runtime session split across both arms, or a bundle
condition you named with `--applied-bundle-attribute` or `--applied-bundle-require`. It prints, per
arm, the traces and runtime sessions beside AgentCore's sample sizes, and how many assigned traces
your store does not hold. Each read is kept under an id built from what was read and how, and a
stored read never changes.

If the agent writes its messages as span attributes rather than the `otel-rt-logs` records the
ingest reads, the ingest refuses its traces for missing content; `arms` still reads each one's arm
from its spans and counts it as having come back without content.

### Re-test the change with the library's guards, and print both verdicts

Replay one held-out set of questions under both arms' bundle versions, each question under the same
trace id in both stores so the paired test can match them. `AgentCoreRuntimeExecutor` does this from
Python: name the arm's bundle version on the `RuntimeTarget`, give each `AgentCoreRuntimeTask` its
question's `trace_id`, and pass `log_dir=<root>/records/agentcore`. Its run manifest records which
bundle each run used, and `retest` reads it from there. `calibrated-agent-evaluation generate --run
--executor agentcore` gives every run a new trace id, so its stores can't be paired. Then judge or
label both stores, register the target with `calibrated-agent-evaluation improve --register` before
the treatment replay runs, and decide the change with `calibrated-agent-evaluation improve`'s paired
test and guard panel:

--8<-- "fixtures/expected/docs/integrate-agentcore-audit-retest.txt"

Before the before/after guard runs, `retest` checks that every replayed trace ran under its arm's bundle version,
that both stores ask the same prompt per trace id, that no confirmation question is one the change
was developed on (compared by prompt text, whatever the input key), that the slice has not decided
another change, and, for judge verdicts, that one certified, unbiased judge scored each mode in
both stores. It stores the guard's improve record and an audit record of what the verdict rests on.
`report` prints that verdict beside AgentCore's claim, from stored records only:

--8<-- "fixtures/expected/docs/integrate-agentcore-audit-report.txt"

`calibrated-agent-evaluation audit recommendation --recommendation-id ID` records a recommendation
the same way, and `calibrated-agent-evaluation audit retest METRIC --recommendation-id ID` re-tests
its result bundle version against its input version. The development set comes from the
recommendation's own traces: the session spans' trace ids in `--development-root`, a batch
evaluation's store, or the CloudWatch Logs or online evaluation window, read on the Logs Insights
path. A recommendation on inline text, or one whose target was registered after it started, is
refused.

Every AWS call goes through a read-only allowlist: the A/B test, recommendation, bundle and online
evaluation reads and the ingest's trace reads. The role in `infra/agentcore-live-test-role.yaml`
grants the four audit reads, each on one exact ARN, when their template parameters are set.

### Turn caught failures into AgentCore datasets

When a reviewer marks a trace FAIL, the case can become a regression test that AgentCore's
evaluators rerun against every new version of your agent. `calibrated-agent-evaluation dataset`
builds those test scenarios from your human labels and publishes them as an AgentCore evaluation
dataset, one immutable version per publish.

A scenario takes its input from the trace and its expected answer from the label's `expected`.
Labels don't say which tools the agent should have called, or which plain-language rules the
answer must meet, so you write those once per failure mode in an answer key. Start from the one
`answers` writes, which lists the failure modes on your human FAIL labels:

```text
--8<-- "fixtures/expected/docs/integrate-agentcore-dataset-answers.txt"
```

Fill in a mode's `expected_trajectory` (the tool names, in order) and `assertions`. A `by_trace`
entry overrides them for one trace:

```json
{
  "by_failure_mode": {
    "trusted_broken_policy_tool": {
      "assertions": ["Agent denies a refund outside the return window"],
      "expected_trajectory": ["lookup_order", "check_refund_policy"]
    }
  },
  "by_trace": {}
}
```

Then build the slice. Only trace-level labels count, and a trace whose raters disagree, or record
different expected answers, is left out and named:

```text
--8<-- "fixtures/expected/docs/integrate-agentcore-dataset-build.txt"
```

AgentCore's `Builtin.Correctness` evaluator checks the expected answer, its trajectory evaluators
check the tool order, and `Builtin.GoalSuccessRate` checks the assertions. A scenario with no
assertions is still published, and the goal check judges it without ground truth.

Publishing needs the extra and credentials for the account:

```sh
uv sync --inexact --extra agentcore
calibrated-agent-evaluation dataset publish slice.json --name orderbot_regressions --region us-west-2
calibrated-agent-evaluation dataset publish slice.json --name orderbot_regressions --region us-west-2 --yes
```

Without `--yes`, `publish` only prints its plan: the dataset, the region, the schema, how many
scenarios have assertions and a tool order, and whether it creates a dataset or the next version
of one. With `--yes` it creates the dataset (or replaces the draft of the one the slice was
published to), waits until AgentCore reports it active, freezes it as a numbered version, and
writes that version's ARN and number into the slice file. A version never changes after that, so
a result can always name the exact test cases it used. Rebuilding with `--force` keeps the slice's
version, so the next publish creates the version after it; publishing an unchanged slice creates
nothing.

`calibrated-agent-evaluation dataset pull --dataset-id ID --version N --out FILE` reads a published
version back into a slice file. It refuses the editable draft, a version that does not exist, a
schema other than `AGENTCORE_EVALUATION_PREDEFINED_V1` and `AGENTCORE_EVALUATION_SIMULATED_V1`, and
any field it can't map, rather than dropping it.

The credentials need `bedrock-agentcore:CreateDataset` and `ListDatasets` (which take no resource
ARN), and `GetDataset`, `ListDatasetVersions`, `ListDatasetExamples`, `AddDatasetExamples`,
`DeleteDatasetExamples` and `CreateDatasetVersion` on the dataset. Dataset evaluation is in public
preview in AgentCore.

### Certify an AgentCore judge and watch it online

A Calibrated Agent Evaluation prompt judge can run in AgentCore as an LLM-as-judge evaluator, be
certified there against your gold labels, and then score a deployed agent's live sessions. The
commands, in order:

```bash
calibrated-agent-evaluation online create-evaluator --judge judge.json --yes      # writes judge.agentcore.json
calibrated-agent-evaluation calibrate --judge judge.agentcore.json --yes           # against your gold, as any judge
calibrated-agent-evaluation bias --judge judge.agentcore.json --probes DIR --thresholds FILE --yes
calibrated-agent-evaluation online attach --judge judge.agentcore.json --name NAME --sampling 100 \
    --runtime-arn RUNTIME_ARN --role-arn EVALUATION_ROLE_ARN --yes
calibrated-agent-evaluation online read --judge judge.agentcore.json               # the verdicts, back in your store
calibrated-agent-evaluation online publish --judge judge.agentcore.json            # the corrected pass rate
calibrated-agent-evaluation online cost
```

`--runtime-arn` and `--role-arn` default to `CALIBRATED_AGENT_EVALUATION_AGENTCORE_RUNTIME_ARN` and
`CALIBRATED_AGENT_EVALUATION_AGENTCORE_EVAL_ROLE_ARN`; the evaluation role is the one AgentCore assumes to run the
config, and `infra/README.md` describes the live-test setup. Each command that writes to AWS or makes billed calls shows the request or the estimate and sends
nothing until you add `--yes`. `attach` refuses a judge without a certifying calibration record
and a `CLEAR` bias battery. A judge's version is a digest of the evaluator's configuration, so
attaching it, which makes AgentCore lock the evaluator and move its `updatedAt`, leaves the
calibration in force, while any edit to the instructions, scale or model voids it. Once an
enabled online config has used an evaluator, AgentCore refuses to edit it for good, even after
the config is paused or deleted, so to change an attached judge create a new evaluator from the revised prompt and certify
it; `read` and `publish` still check each attached evaluator's configuration against the one it
was certified with, and report one that changed or was deleted.

`read` reads the config's results from its own log group and stores each as the judge's verdict.
It leaves out the sessions the judge's own calibration and bias runs sent, stores nothing for a
result AgentCore could not score (it counts them by error type), and says how many sessions went
idle long enough ago to have a result and have none. It always checks each judge for staleness
against the calibration it was attached with.

`publish` corrects the online failure rate with the judge's measured errors and publishes three
metrics to the `CalibratedAgentEvaluation/Judges` namespace: `CorrectedPassRate`,
`CorrectedPassRateLower95` and `CorrectedPassRateUpper95`, with `JudgeId`, `EvaluatorFingerprint`
and `GoldSource` dimensions. It uses only the verdicts a `read` recorded, so verdicts the same judge
gave on demand (`calibrated-agent-evaluation judge`) are never mixed in. It needs at least 150
online verdicts and publishes nothing below that; the raw pass rate is never published, since
AgentCore's own evaluation metrics carry it. `GoldSource` is where the calibration's gold came from
(`synthetic-rule` for the quickstart, `human` for your own labels). `calibrated-agent-evaluation
calibrate` records it with the calibration, so `read`, `publish` and `cost` name the same source
whichever store you run them against, and the alarm sees one series. For a calibration made before
the library recorded it, they read it from the gold in the current store and say so; with none of
that gold there, `publish` refuses rather than publish `unknown`, so publish from the store you
calibrated in. Nothing here creates an alarm. To be told when quality may have fallen, set one
yourself on `CorrectedPassRateLower95` with a threshold just below the value you accept, as AWS
describes in [Create a CloudWatch alarm based on a static
threshold](https://docs.aws.amazon.com/AmazonCloudWatch/latest/monitoring/ConsoleAlarms.html).

`cost` counts the evaluations it can prove (the calibration and bias calls the attached judge's
runs logged, and the config's scored results) and prices them per evaluation. It is a floor: the
evaluator's own model usage is billed on top and is not in it, and a run refused part-way logged
no calls.

#### Checking online traffic for drift

`calibrated-agent-evaluation online read` checks drift only when you ask, with the same two flags as
`calibrated-agent-evaluation run`:

```bash
calibrated-agent-evaluation online read --judge judge.agentcore.json \
    --drift-baseline BASELINE_STORE --drift-feature input-chars
```

The baseline is a local store holding an earlier random sample of the same agent's traffic, for
example the store an earlier `calibrated-agent-evaluation online read` wrote, or a sampled
`calibrated-agent-evaluation ingest`. A store of hand-picked or gold traces is the wrong baseline:
drift compares two random samples, and comparing a picked set with live traffic flags drift every
time. Pick the feature that would move if the traffic changed in a way that matters to the judge:
`input-chars` for longer or shorter requests, `turns` for more or fewer agent steps, `tool-calls`
for more or fewer tool calls. Each side needs at least 50 traces; below that the read says drift was
not measured and how many traces each side has. A flagged drift also flags the judge as possibly
stale. Drift is not measured on a config with filters, whose sessions are not a random sample.
Without the flags the read says drift was not measured.

### Run simulated conversations and judge them

Some failures only show up over several turns: the agent gives in when the customer pushes back, or
forgets what they said two messages ago. `calibrated-agent-evaluation dataset run` holds one
simulated conversation per scenario of a simulated slice. A simulated customer opens with the
scenario's input and pursues its goal, and your agent answers each turn. The slice can be
hand-written or pulled from AgentCore with `calibrated-agent-evaluation dataset pull`:

```json
{
  "schema_version": "1",
  "form": "simulated",
  "scenarios": [
    {
      "form": "simulated",
      "scenario_id": "ORD-1001-sim",
      "actor_profile": {
        "context": "You bought a countertop blender (order A-7781) 51 days ago. The return window is 30 days.",
        "goal": "Get a refund for order A-7781.",
        "traits": {"tone": "polite"}
      },
      "input": "I want a refund for order A-7781, the blender just isn't for me.",
      "max_turns": 4,
      "assertions": ["Agent denies a refund outside the return window"]
    }
  ]
}
```

Run it against a deployed agent, and judge each conversation with a session-level evaluator:

```sh
uv sync --inexact --extra agentcore
calibrated-agent-evaluation dataset run slice.json --executor agentcore --runtime-arn "$RUNTIME_ARN" \
  --payload-format agentcore_sse --judge goal-success.json
calibrated-agent-evaluation dataset run slice.json --executor agentcore --runtime-arn "$RUNTIME_ARN" \
  --payload-format agentcore_sse --judge goal-success.json --yes
```

`goal-success.json` is a judge file whose `level` is `session`:

```json
{
  "backend": "agentcore",
  "judge_id": "agentcore-goal-success",
  "evaluator_id": "Builtin.GoalSuccessRate",
  "failure_mode": "goal_not_met",
  "level": "session"
}
```

Without `--yes`, the run only prints its plan: the scenarios, the most agent turns it can send,
and the most `Evaluate` calls. With `--yes`, it holds every conversation, then stores each as one
trace: its transcript, the goal, the assertions, why it stopped, and every turn's spans. A slice
that was published also names its dataset id and version on each trace. Then it sends each
conversation to the evaluator as one session, each turn as its own trace and the scenario's
assertions as the session's ground truth. It prints the estimate and one verdict per scenario,
and stores each verdict as an advisory `llm-judge` label on the conversation's trace. A scenario
with no assertions is judged on its goal alone, and the output names it.

- `--executor strands --agent module:callable` runs a local Strands agent instead of a deployed
  one. Each turn reruns the agent over the whole history, so its model tokens grow with the turns.
- `--simulator strands|agentcore` picks who plays the customer, as on `generate --multi-turn`.
- Nothing is stored unless every conversation finishes. A conversation that fails names the
  scenario, the turn and whether the customer or the agent failed, and its runtime session is
  still stopped.
- `CALIBRATED_AGENT_EVALUATION_AGENTCORE_MAX_BILLED_CALLS` caps the run. It is checked before anything is sent,
  against every scenario running to `max_turns` plus one `Evaluate` per conversation.
- With `--cassette PATH`, the agent's turns are recorded in `PATH` and the customer's turns
  beside it, in `<stem>.actor.json`. With `CALIBRATED_AGENT_EVALUATION_CASSETTE_MODE=replay`, the whole run replays
  offline, the simulated customer included; a recording that no longer matches the agent's
  replies is refused. Judging is a live call, so `--judge` is refused in replay.

The credentials need what `generate --run --executor agentcore` needs, plus
`bedrock-agentcore:Evaluate` and `GetEvaluator` on the session evaluator. A `TOOL_CALL` evaluator
is still refused.

## Troubleshooting

The messages below are specific to AgentCore. The rest are on
[Troubleshooting](../how-to/troubleshooting.md).

### AgentCore datasets

| You see | Why | Do |
| --- | --- | --- |
| `there is nothing to build` | no trace in scope has a trace-level human FAIL that every rater agrees on | label failures first (`calibrated-agent-evaluation label --serve`); with `--failure-mode`, check the name against `calibrated-agent-evaluation dataset answers` |
| `left out ... their raters disagree` or `recorded different expected outputs` | two raters' newest labels on a trace conflict | adjudicate the trace, then build again |
| `gives different expected tool orders for failure modes that share a trace` | a trace has two failure modes whose answer-key entries name different tool orders | set that trace's `expected_trajectory` under `by_trace` |
| `Dataset name ... is not allowed` | AgentCore names allow 1 to 48 letters, digits and underscores, starting with a letter | use a name like `orderbot_regressions` (no hyphens) |
| `A dataset named ... already exists ... and this slice was not published to it` | the name belongs to another dataset | choose another `--name`, or `calibrated-agent-evaluation dataset pull` that dataset to start from it |
| `more than the 1000 examples AgentCore accepts in one request` | AgentCore takes at most 1000 inline examples per call | split the slice |
| `Version 'DRAFT' can't be pulled` | only a published version is frozen; the draft can still change | pull a version number, the one `publish` printed |
| `... not found for datasetId=...` | the dataset or that version does not exist, or was deleted | check the id and version with `aws bedrock-agentcore-control list-dataset-versions` |
| `Dataset schema 'THIRD_PARTY_EVALUATION_V1' is not supported` | that schema has no documented shape (the CloudWatch console makes it when traces are saved to a dataset) | pull a predefined or simulated dataset |
| `has field(s) ... that ... does not define; refusing rather than dropping them` | an example holds a field the codec can't map | remove the field in AgentCore, or keep it under `metadata` |
| `draft may now differ from its last version` | a publish stopped after it changed the draft | publish again; published versions are never changed |
| `was created but holds no version yet` | a first publish stopped after creating the dataset | publish again with the same `--name`; the new dataset is picked up and versioned |

??? failure "Output"

    ```text
    --8<-- "fixtures/expected/docs/ts-dataset-publish-no-credentials.txt"

    --8<-- "fixtures/expected/docs/ts-dataset-pull-draft.txt"
    ```


### AgentCore A/B and recommendation audit

| You see | Why | Do |
| --- | --- | --- |
| `refused bedrock-agentcore:<operation>: this command only makes the calls it lists` | `calibrated-agent-evaluation audit` reads only an A/B test, its bundle versions and the trace reads of `calibrated-agent-evaluation ingest --source agentcore`; any other call is refused before it is sent | nothing to fix: the audit never writes to AWS |
| `GetABTest for ... names both a bundle version and a target` (or `neither`) | an arm must name exactly one configuration; the model allows both or neither | fix the A/B test's variant in AgentCore, then read it again |
| `no claim record is stored for A/B test ...` | `calibrated-agent-evaluation audit arms` works from a recorded claim | run `calibrated-agent-evaluation audit abtest --ab-test-id <id>` first |
| `... claim records are stored for A/B test ... Name one with --claim-record` | every `calibrated-agent-evaluation audit abtest` read stores a new record | pass `--claim-record` with one of the ids listed |
| `no span in ... carries both ...routing_experiment_arn and ...routing_experiment_variant_name` | no trace in the window was routed by the A/B test: only agents built on `BedrockAgentCoreApp` and invoked through the test's Gateway carry the stamps | check the window and the runtime, and that traffic went through the Gateway; when the message names traces that came back unread, read again with the flags you ingested with (often `--rekey-sessions`) |
| `no trace in ... is stamped with the audited test` | the window's stamps name another experiment | check `--ab-test-id`; the message lists the experiments it saw |
| a trace refused with `carrying one arm attribute without the other` | the request's baggage named the experiment or the variant but not both, so its arm can't be read safely | send both baggage keys through the Gateway; the rest of the read is kept |
| a trace refused with `is stamped with ... experiment(s) ... and ... variant(s)` | one trace's spans name two arms | look up the trace; it is left out of every count |
| traces refused with `runtime session ... has traces in C and T1` | one runtime session's traces sit in both arms | the session's arm is ambiguous; nothing to fix on this side |
| an assignment refused with `traces arrived under source id` | two traces share the id the source delivered them under (with `--rekey-sessions`, two turns of one runtime session) | read a narrower window, or without `--rekey-sessions` if the store was ingested without it |
| a trace refused naming `--applied-bundle-attribute` or `--applied-bundle-require` values | the trace's spans don't carry the bundle version or value you required | check the agent applied the arm's bundle; drop the flag to count traces regardless |
| `arms read ... is already stored and this re-read differs` | a stored read never changes, and this one added, dropped or reassigned a trace | read another window (it is kept beside the first) |
| `... trace(s) came back without their message content` in an arms read | the agent writes its messages as span attributes, not as the `otel-rt-logs` records the ingest reads, so the ingest refuses those traces | the arms are still read from the spans' stamps; the traces can't be replayed from this window |
| `calibrated-agent-evaluation audit retest ... needs --development-root` | a re-test must show its held-out slice was not what the change was developed on | pass the store of the development traces; a session-span recommendation names them by trace id |
| `--development-root is not taken for recommendation ...` | a CloudWatch Logs or online-evaluation recommendation names its own window, which the audit reads | drop the flag |
| `AgentCoreDevelopmentInputMissing` | the development store holds no trace, or a trace the recommendation read is not in it (a store ingested with `--rekey-sessions` keys traces by session id) | pass the store that holds those trace ids, ingested without re-keying |
| `AgentCoreDevelopmentSourceUnread` | the development read did not cover a log group the recommendation names, or the source names a group by prefix | read the window with every group readable; a prefix source can't be checked |
| `AgentCoreDevelopmentSourceChanged` | the online evaluation was updated after the widened development window began, so its traces may have followed another configuration | re-test against a recommendation started after the last change |
| `AgentCoreReplayProvenanceError` | a replayed trace has no run provenance, names another bundle version, asks a different prompt in the two stores, or has no single prompt text | replay both arms from one input set with an executor that records the bundle each run used |
| `ConfirmationContaminationError` from `calibrated-agent-evaluation audit retest` | a confirmation trace asks a question the change was developed on, whatever its input key | replay a slice held out from the development traces |
| `AgentCoreSliceReused` | the slice already decided another change (by prompt digest, or a `calibrated-agent-evaluation improve --evaluate` with the same slice id) | replay a fresh held-out slice |
| `AgentCoreUncertifiedJudge` | a judged mode's rater has no current `CERTIFIED` calibration record or `CLEAR` bias record for its fingerprint, names no `judge@fingerprint`, or two raters judge one mode across the stores | run `calibrated-agent-evaluation calibrate` and `calibrated-agent-evaluation bias` for that judge, or re-judge both stores with one judge |
| `AgentCoreNotReplayable` | the arm routes to a Gateway target, or the recommendation failed, is not complete, had inline input or names no result bundle | nothing to replay: re-test a bundle-version change |
| `the pre-registration of ... was frozen at ..., not before the recommendation was created` | the target was registered after AgentCore started the recommendation | register the target before starting the next recommendation |
| `several audit records are stored for ... Name one with --audit-id` | each re-test on a fresh slice stores its own audit record | pass `--audit-id` with one of the ids listed |
| `audit ... rests on claim-record (or arms-read) ..., not ...` | a stored re-test prints the claim and arms read it was decided on | drop the flag, or name the record's own id |


### Running an agent on AgentCore Runtime

These come from `calibrated-agent-evaluation generate --run --executor agentcore`
([Route your agent](route-your-agent.md#your-agent-on-agentcore-runtime)). Each stops the
command before any AWS call.

| You see | Why | Do |
| --- | --- | --- |
| `--executor agentcore needs the [agentcore] extra` | the extra isn't installed | run the printed command |
| `--executor agentcore needs --runtime-arn ARN` or `--payload-format NAME` | both are required; the body format is never guessed, since each guess would be a billed call | pass the runtime's ARN and the format its agent reads |
| `... is not an AgentCore Runtime ARN` | the ARN names a runtime endpoint or another resource | pass the runtime's own ARN, `arn:aws:bedrock-agentcore:<region>:<account>:runtime/<id>` |
| `--config-bundle-arn is empty, so nothing was run`, or `--config-bundle-version` | the flag was given an empty value, often `"$VAR"` with `VAR` unset, which names no bundle, so the agent would run without one | pass the bundle's ARN and version, or drop both flags |
| `--runtime-arn only applies to --executor agentcore` | an AgentCore flag came with another executor, or with none | add `--run --executor agentcore`, or drop the flag |
| `--runtime-arn is read only when --run sends the inputs` | `--executor agentcore` and an AgentCore flag came without `--run`, so nothing would be sent and the flag would be ignored | add `--run`, or drop the flag |
| `--agent doesn't apply to --executor agentcore` | the agent under test is the deployed runtime; `--agent-model` is refused too | drop the flag |
| `replay needs a cassette` | `CALIBRATED_AGENT_EVALUATION_CASSETTE_MODE` is `replay` and no `--cassette` was given | pass `--cassette PATH`, or set the mode to `auto` or `record` to call AWS |
| `... rows, and --executor agentcore runs discovery inputs only` | the frame holds rows of another frame | pass a discovery frame with `--frame FILE` (`calibrated-agent-evaluation define --out FILE` writes one) |
| `... is not a valid cassette mode` | `CALIBRATED_AGENT_EVALUATION_CASSETTE_MODE` is not `record`, `replay` or `auto` | set it to one of those, or unset it |
| `read_timeout_seconds=... is not a positive, finite number of seconds` | `--read-timeout` is 0 or less, `nan`, `inf`, or longer than a socket can wait | pass how long to wait for each read of the response, such as `300` |
| `Langfuse reachability check failed` | with `--backend langfuse`, a run that can call AWS checks the trace store before its plan, and Langfuse can't be reached or refuses the keys | start Langfuse (`just langfuse-up` for the local stack) or fix `LANGFUSE_*`, then rerun; nothing was sent |
| `this run can make the billed AWS calls listed above, so it needs --yes` | in any mode but `replay`, the run can call AWS | read the printed plan, then rerun with `--yes`, or replay a cassette |
| `CALIBRATED_AGENT_EVALUATION_AGENTCORE_MAX_BILLED_CALLS=... is not a non-negative whole number` | the billed-call cap is not a whole number | set it to the most billed calls to allow, or unset it |

Once runs have started, a run that is refused or fails stops the command with
`agent run failed: RuntimeInvocationError: ...`. The message names what stopped it, such as a
cassette miss, a cassette recorded for another runtime or qualifier, a runtime that only accepts
OAuth or doesn't serve HTTP, the billed-call cap, an AWS error code, or an error event in the
agent's answer, and the runtime session when one was opened. The command also prints the path of
the run manifest, `agentcore_run_record.json` (with `agentcore_run_record.csv` beside it), where
the run's record carries its `error_code`. When a run's record could not be added to the manifest,
for example because a file stands where `--log-dir` points, that line says the manifest holds no
record of the run and why. `could not close the invocation clients (...)` after every run answered
means the traces are already stored; only connections were left open, and they are released when
the process exits.

??? failure "Output"

    ```text
    --8<-- "fixtures/expected/docs/ts-missing-agentcore-generate.txt"

    --8<-- "fixtures/expected/docs/ts-agentcore-no-payload-format.txt"

    --8<-- "fixtures/expected/docs/ts-agentcore-flag-without-executor.txt"

    --8<-- "fixtures/expected/docs/ts-agentcore-replay-no-cassette.txt"

    --8<-- "fixtures/expected/docs/integrate-route-agentcore-plan.txt"

    ```


### Probing an agent on AgentCore Runtime

These come from `calibrated-agent-evaluation probe --executor agentcore`; the full flow is in
[Probe a black-box endpoint](probe-black-box.md). Each stops the command before any AWS
call, and the ones `generate` shares read as in the table above.

| You see | Why | Do |
| --- | --- | --- |
| `--runtime-arn only applies to --executor agentcore` | an AgentCore flag, such as `--yes` or `--log-dir`, came without `--executor agentcore` | add `--executor agentcore`, or drop the flag |
| `--endpoint doesn't apply to --executor agentcore` | the runtime ARN names the target | drop `--endpoint`, or drop `--executor agentcore` to probe a URL |
| `--auth-bearer-env doesn't apply to --executor agentcore`, or another `--auth-*` flag | calls are signed with SigV4 from `--profile`, and OAuth-only runtimes are not supported yet | drop the flag; pass `--profile NAME` when `AWS_PROFILE` is not the one to use |
| `--timeout doesn't apply to --executor agentcore` | `--timeout` is the URL probe's per-request timeout | pass `--read-timeout SECONDS` instead (default 300) |
| `--battery doesn't apply to --executor agentcore`, or `--trials`, `--allow-use-restricted` or `--grader-model` | the benchmark battery runs against a URL endpoint only | drop the battery flags, or run the battery with `--endpoint` |
| `--executor agentcore needs --runtime-arn ARN` or `--payload-format NAME` | both are required, and the probe never asks for them on a terminal | pass the runtime's ARN and the format its agent reads |
| `--prompt is empty, so nothing was sent`, or `1 of the 2 --prompt values is empty` | an empty prompt, often `--prompt "$VAR"` with `VAR` unset, would be a billed invocation that asks the agent nothing; it is refused before the plan, so no prompt before it is billed | give each `--prompt` the text to send |
| `--config-bundle-arn is empty, so nothing was run`, or `--config-bundle-version` | the flag was given an empty value, which names no bundle, so the agent would run without one | pass the bundle's ARN and version, or drop both flags |
| `Probing an endpoint needs your confirmation that you're allowed to`; under `--explain`, `... requires confirmed authorization` | `--authorized` was not given; it is checked before the plan is printed | pass `--authorized` only if you are allowed to test the runtime |
| `... is not a valid cassette mode` | `CALIBRATED_AGENT_EVALUATION_CASSETTE_MODE` is not `record`, `replay` or `auto` | set it to one of those, or unset it |
| `CALIBRATED_AGENT_EVALUATION_CASSETTE_MODE is replay, but probe has no replay; nothing was sent` | the probe keeps no cassette, so every AgentCore probe is live | unset the variable (`auto` and `record` are accepted and ignored) |
| `--executor agentcore needs the [agentcore] extra` | the extra isn't installed | run the printed command |
| `... is not an AgentCore Runtime ARN` | the ARN names a runtime endpoint or another resource | pass the runtime's own ARN, `arn:aws:bedrock-agentcore:<region>:<account>:runtime/<id>` |
| `CALIBRATED_AGENT_EVALUATION_AGENTCORE_MAX_BILLED_CALLS=... is not a non-negative whole number` | the billed-call cap is not a whole number | set it to the most invocations to allow, or unset it |
| `Langfuse reachability check failed` | with `--backend langfuse`, the probe checks the trace store before its plan, and Langfuse can't be reached or refuses the keys | start Langfuse (`just langfuse-up` for the local stack) or fix `LANGFUSE_*`, then rerun; nothing was sent |
| `nothing was sent: this probe makes the billed AWS calls listed above, so it needs --yes` | every prompt is a billed invocation | read the printed plan, then rerun with `--yes` |
| `read_timeout_seconds=... is not a positive, finite number of seconds` | `--read-timeout` is 0 or less, `nan`, `inf`, or longer than a socket can wait | pass how long to wait for each read of the reply, such as `300` |
| `--policy-cases applies only to --executor agentcore` | the policy battery reads an AgentCore Gateway's decisions | add `--executor agentcore` and `--runtime-arn`, or drop `--policy-cases` |
| `--prompt doesn't apply with --policy-cases` | the case file supplies every prompt, one per case and trial | drop `--prompt`, or move the prompt into a case |
| `calibrated-agent-evaluation probe: probe record ... is stored; ...` before a trace-store error | the probe record was written, then storing a probed trace or a `policy-engine` label failed | fix the trace store and run again; the record and the run manifest keep what ran |
| `the policy case file ... is invalid: ...` | a decision other than `allow` or `deny`, an empty `expect`, fewer than one trial, a repeated case ID, or no control case | fix the file as the message says |
| `PolicyDecisionsMissing: no Policy decision on a tool call was read for any of the ... trial(s)` | no trial's trace held a decision: no engine is attached, the gateway is in log-only mode, the agent called no tool or none it could, or trace context did not reach the Gateway | attach the engine in enforcing mode, check the agent's tool calls go through the Gateway with trace context, and run again |
| `control case '...' got no Policy decision in any of its ... trial(s)` | the control case's calls never reached the engine | check the control case's prompt makes the agent call a tool through the Gateway |
| `PolicyCaseError: the decisions named the tools ..., and none of them is a tool a case lists` | the case file names tools in another form than the Gateway, usually without the `<target>___` prefix | write each tool as the decisions name it |
| `PolicyModeMixed: case '...' got decisions under both enforce and log_only gateway modes` | the gateway's mode changed during the run | run the case again with the gateway in one mode |
| `PolicySpanMalformed: span ... holds a Policy decision, but ...` | a decision span could not be read whole, so it is refused rather than counted or dropped | report the span; it may be a new Gateway span format |
| `AgentCoreSettlingLag: the spans of ... trial(s) were still arriving at the ...s bound` | the span index had not caught up with the log store within the bound the plan printed (300 s plus 10 s per trial) | run again, since the index's lag varies from run to run; the trials' run records are in the run manifest |

Once the prompts have started, the first one that is refused or fails stops the probe with
`RuntimeInvocationError: ...`, and none of the replies are stored, not even those of the prompts
before it. The message names what stopped it, such as a runtime that only accepts OAuth or doesn't
serve HTTP (found by the read-only preflight before the first invocation), the billed-call cap, an
AWS error code, an error event in the reply, or a reply with no text, which usually means the wrong
`--payload-format`. It also says whether the request was never sent or the agent may have run, and
names the runtime session when one was opened. The command then prints the path of the run
manifest, `agentcore_run_record.json`, which holds a record of each prompt's run, the failed one
with its `error_code`; when a record could not be added, that line says which and why. Fix the
cause and probe again: the new traces never replace earlier ones. `could not close the invocation
clients (...)` after every prompt answered means the replies are already stored; only connections
were left open, and they are released when the process exits.

## What is planned

The integration is a set of optional plug-ins under the `agentcore` extra, in the package
`calibrated_agent_evaluation.agentcore`. The library's own path stays the default wherever both
offer the same capability. The extra and its package exist now: they back
`calibrated-agent-evaluation doctor --agentcore`, `calibrated-agent-evaluation ingest --source
agentcore`, `calibrated-agent-evaluation generate --run --executor agentcore` with its per-run span
reader, `calibrated-agent-evaluation probe --executor agentcore`, `calibrated-agent-evaluation
dataset publish`, `pull` and `run` with session-level judging, the failure detector above,
`calibrated-agent-evaluation audit` and the online judge commands (`calibrated-agent-evaluation
online`). OAuth-only runtimes are not built yet. Without the extra, asking for an AgentCore module
or command prints the command that installs it.

### OAuth-only runtimes

Run the executor against runtimes with a custom JWT authorizer, which the AWS SDK cannot call.
