# Troubleshooting

A refusal exits 1 and says what to do; a usage mistake exits 2; a traceback is a bug to report.
`--explain` (or `CALIBRATED_AGENT_EVALUATION_EXPLAIN=1`) shows the statistics and record paths
behind a result. Library errors mostly derive from
`calibrated_agent_evaluation.schema.CalibratedAgentEvaluationDataError`, the CLI's from
`calibrated_agent_evaluation.cli.errors.CliError`.

## Installation and extras

| You see | Why | Do |
| --- | --- | --- |
| an extra is not installed | the command needs a third-party package | run the command it prints, such as `uv sync --inexact --extra stats`; `calibrated-agent-evaluation doctor` lists all ([Installation](../getting-started/installation.md)) |
| `calibrated-agent-evaluation capture needs the OTel SDK and OTLP encoder` | the `strands` extra carries them, for any agent | `uv sync --inexact --extra strands` (OTLP-JSON `ingest` needs no extra) |
| `This needs the [agentcore] extra` | an AgentCore command or module was used without its extra | run the printed command, such as `uv sync --inexact --extra agentcore` |
| `No AWS credentials were found for this command` | `calibrated-agent-evaluation doctor --agentcore`, `calibrated-agent-evaluation dataset publish` or `pull` found no usable profile or keys | set `AWS_PROFILE`, or pass `--profile` ([AgentCore](../integrate/gallery/agentcore.md)) |

??? failure "Output"

    ```text
    --8<-- "fixtures/expected/docs/ts-missing-stats.txt"

    --8<-- "fixtures/expected/docs/ts-missing-ui.txt"

    --8<-- "fixtures/expected/docs/ts-missing-langfuse.txt"

    --8<-- "fixtures/expected/docs/ts-missing-agentcore.txt"

    --8<-- "fixtures/expected/docs/ts-agentcore-no-credentials.txt"
    ```

## AgentCore datasets

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

## AgentCore A/B and recommendation audit

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

## Getting traces in

| You see | Why | Do |
| --- | --- | --- |
| `cannot load agent '...'` | the current directory isn't on the import path | `PYTHONPATH=. calibrated-agent-evaluation capture ...`, or install your agent's package |
| `captured 0 spans` | the agent emitted no OpenTelemetry spans | enable instrumentation; for another process, `calibrated-agent-evaluation ingest` its OTLP-JSON ([Capture](../integrate/capture-and-ingest.md)) |
| `the store already holds trace '...'` | a capture never replaces a stored trace on re-run | new `--trace-id`, omit it for the unique OTel id, or `--overwrite` |
| `[REFUSED] ... resolve to trace id ...` | two files in a batch share a trace id | rename one so the name prefixes differ, or set `calibrated_agent_evaluation.trace_id` |
| `[skipped] ... could not detect` | no attributes the ingest recognizes | pass `--convention`, or add GenAI attributes ([gallery](../integrate/gallery/index.md)) |
| `degraded to ownership='probed'` | input and output but no agent or tool steps | stored outcome-only (still labelable) unless `--require-owned` |

??? failure "Output"

    ```text
    --8<-- "fixtures/expected/docs/ts-agent-not-found.txt"

    --8<-- "fixtures/expected/docs/ts-no-spans.txt"
    ```

    The third command is the refusal:

    ```text
    --8<-- "fixtures/expected/docs/integrate-capture.txt"

    --8<-- "fixtures/expected/docs/calibrated-agent-evaluation-ingest-dry-run.txt"

    --8<-- "fixtures/expected/docs/integrate-ingest-require-owned.txt"
    ```

## Running an agent on AgentCore Runtime

These come from `calibrated-agent-evaluation generate --run --executor agentcore`
([Route your agent](../integrate/route-your-agent.md#your-agent-on-agentcore-runtime)). Each is
refused before any AWS call.

| You see | Why | Do |
| --- | --- | --- |
| `--executor agentcore needs the [agentcore] extra` | the extra isn't installed | run the printed command, such as `uv sync --inexact --extra agentcore` |
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

## Probing an agent on AgentCore Runtime

These come from `calibrated-agent-evaluation probe --executor agentcore`; the full flow is in
[Probe a black-box endpoint](../integrate/probe-black-box.md). Each is refused before any AWS call,
and the ones `generate` shares read as in the table above.

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
| `--executor agentcore needs the [agentcore] extra` | the extra isn't installed | run the printed command, such as `uv sync --inexact --extra agentcore` |
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

## The trace store

| You see | Why | Do |
| --- | --- | --- |
| `the langfuse backend needs LANGFUSE_PUBLIC_KEY / LANGFUSE_SECRET_KEY` | no credentials | export both and `LANGFUSE_HOST` (per project) |
| `Langfuse ... failed: ConnectError` | nothing answers at `LANGFUSE_HOST` | `just langfuse-up` for the local stack; `calibrated-agent-evaluation doctor` checks reachability |
| `StaleReadError` | async writes unreadable after `CALIBRATED_AGENT_EVALUATION_READ_WAIT_SECONDS` (30) | rerun later, or raise the wait |
| `no trace with trace_id '...'` | wrong store; the default is `$CALIBRATED_AGENT_EVALUATION_LOCAL_ROOT`, then `./.calibrated-agent-evaluation` | pass `--root` or run from the right directory |
| `SchemaVersionError` on `calibrated-agent-evaluation load` | records never convert silently between versions | load with the version that wrote the dump ([Versioning](../versioning.md)) |
| `holds dump files but no dump-manifest.json` on `load` (`IncompleteDumpError`) | the dump was interrupted or is still being written, since the manifest is written last | dump the source store again |
| `differs from DIR/dump-manifest.json` or `is missing` on `load` (`DumpMismatchError`) | a file was edited after the dump, or another dump overwrote part of this one | dump the source store again; to commit a dump, keep git from rewriting its line endings ([Gate a PR in CI](gate-in-ci.md#seed-the-jobs-store)) |
| `file(s) its dump-manifest.json does not list` on `load` (`UnlistedDumpFileError`) | a `.csv` or `.json` file in the directory was not written by that dump | remove the files it names, or dump into an empty directory |

??? failure "Output"

    ```text
    --8<-- "fixtures/expected/docs/integrate-langfuse-no-keys.txt"

    --8<-- "fixtures/expected/docs/ts-langfuse-unreachable.txt"

    --8<-- "fixtures/expected/docs/ts-unknown-trace.txt"
    ```

    The dump here was edited to look older:

    ```text
    --8<-- "fixtures/expected/docs/ts-schema-version.txt"
    ```

## Calibration and judging

| You see | Why | Do |
| --- | --- | --- |
| `no calibration, no score` | a judge scores only once certified and bias-checked in this store (library: `UncalibratedScoreError`) | `calibrate`, `bias`, then `judge` ([guide](calibrate-a-judge.md)) |
| not enough labels to trust this judge | too few labels to tell | label the per-class count it names |
| labels already used to test a judge for this mode | a set tests a mode once, whatever the result or judge file name; overlapping sets count too | label a fresh set ([Re-certify](../recipes/recertify-after-prompt-change.md)) |
| bias check failed | a must-pass bias may exceed its threshold | revise the prompt, then re-certify on fresh labels |
| `no recorded verdict (cassette miss)` | the prompt was never recorded; replay never calls live | record these traces with `--cassette-mode record` (it calls the live model), or narrow with `--cell` |
| `model failed after 3 attempts` | live Bedrock call without a valid AWS session | fix credentials, or pass `--cassette` |

??? failure "Output"

    ```text
    --8<-- "fixtures/expected/docs/ts-no-calibration.txt"

    --8<-- "fixtures/expected/docs/guide-calibrate-not-enough.txt"

    --8<-- "fixtures/expected/docs/guide-reuse-gold.txt"

    --8<-- "fixtures/expected/docs/guide-certify-v1-bias-explain.txt"

    --8<-- "fixtures/expected/docs/ts-cassette-miss.txt"

    --8<-- "fixtures/expected/docs/ts-live-model-down.txt"
    ```

## Rates and gating

| You see | Why | Do |
| --- | --- | --- |
| rate not estimated (too few scored traces) | under 150 production-sample traces, the range misses too often | `calibrated-agent-evaluation judge --frame production-representative` on more |
| rate not estimated (judge needs more labels) | certified, but on too few labels for a rate | label what `--explain` names, or use `--rate-safe` |
| `no judge verdicts with a failure mode ... to gate` | empty input; no source swap | `calibrated-agent-evaluation judge`, or `--verdicts human` / `scorer` |
| usage error, exit 2 | bad flag or argument | `calibrated-agent-evaluation <command> --help`, [CLI reference](../cli/index.md) |

In both rate-not-estimated cases the judge still scores and the gate still sets the exit code.

??? failure "Output"

    ```text
    --8<-- "fixtures/expected/docs/guide-rate-too-few.txt"

    --8<-- "fixtures/expected/docs/ts-nothing-to-gate.txt"

    --8<-- "fixtures/expected/docs/ts-usage-error.txt"
    ```
