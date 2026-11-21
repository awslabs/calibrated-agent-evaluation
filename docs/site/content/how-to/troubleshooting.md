# Troubleshooting

When a command can't give you an answer, it exits 1 and says what to collect or change. A usage
mistake exits 2. A Python traceback is a bug; please
[report it](https://github.com/awslabs/calibrated-agent-evaluation/issues). Add `--explain` (or set
`CALIBRATED_AGENT_EVALUATION_EXPLAIN=1`) to see the numbers and record paths behind a result.
In the library, errors derive from `calibrated_agent_evaluation.schema.CalibratedAgentEvaluationDataError`;
the CLI's derive from `calibrated_agent_evaluation.cli.errors.CliError`.

AgentCore messages are on [Bedrock AgentCore](../integrate/agentcore.md#troubleshooting).

## Messages you may see and what to do

| Message | What triggered it | What to do |
| ------- | ----------------- | ---------- |
| Judge not certified | On your labels, the judge's cautious catch rate for failures, or for passes, is below 0.80 | Label more traces of the class that is short (the message says how many), or improve the judge |
| Few-shot pool used as a test set | The traces you checked the judge on are the examples in its prompt | Check it on a separate set |
| Labels already used to test a judge | This set already decided one judge for this mode | Label a fresh set of traces the judge hasn't been checked on |
| Rate measured on the calibration set | The rate sample is the set the judge was checked on | Score a random sample of production traffic |
| Rate sample too small | Fewer than 150 traces in the rate sample | Score more production traces (the message says how many) |
| Not enough labels for a corrected rate | The judge is certified, but on too few labels for its accuracy to give a reliable range | Label more, or plan with `--rate-safe` next time |
| Judge too weak to correct | The judge's catch rates for failures and passes add up to less than 1.5 | Use the range the report still prints, or improve the judge |
| Flag rate outside what the judge can produce | The judge flagged fewer traces than its own false-alarm rate, or more than it catches | Check the rate sample; the range is still reported |
| Judge biased | The judge changes its verdict when something irrelevant about the input changes, such as a longer answer | Fix the judge prompt, then check it again on fresh labels |

[Methods](../methods.md) gives the rule behind each threshold.

## Installation and extras

| You see | Why | Do |
| --- | --- | --- |
| an extra is not installed | the command needs a third-party package | run the command it prints; `calibrated-agent-evaluation doctor` lists them all ([Install](install.md)) |
| `calibrated-agent-evaluation capture needs the OTel SDK and OTLP encoder` | the `strands` extra carries them, for any agent | install the `strands` extra (OTLP-JSON `ingest` needs no extra) |
| `This needs the [agentcore] extra` | an AgentCore command or module was used without its extra | run the printed command |
| `No AWS credentials were found for this command` | `calibrated-agent-evaluation doctor --agentcore`, `calibrated-agent-evaluation dataset publish` or `pull` found no usable profile or keys | set `AWS_PROFILE`, or pass `--profile` ([AgentCore](../integrate/agentcore.md)) |

??? failure "Output"

    ```text
    --8<-- "fixtures/expected/docs/ts-missing-stats.txt"

    --8<-- "fixtures/expected/docs/ts-missing-ui.txt"

    --8<-- "fixtures/expected/docs/ts-missing-langfuse.txt"

    --8<-- "fixtures/expected/docs/ts-missing-agentcore.txt"

    --8<-- "fixtures/expected/docs/ts-agentcore-no-credentials.txt"
    ```


## Getting traces in

| You see | Why | Do |
| --- | --- | --- |
| `cannot load agent '...'` | the current directory isn't on the import path | `PYTHONPATH=. calibrated-agent-evaluation capture ...`, or install your agent's package |
| `captured 0 spans` | the agent emitted no OpenTelemetry spans | enable instrumentation; for another process, `calibrated-agent-evaluation ingest` its OTLP-JSON ([Capture](../integrate/capture-and-ingest.md)) |
| `the store already holds trace '...'` | a re-run capture keeps the stored trace | new `--trace-id`, omit it for the unique OTel id, or `--overwrite` |
| `[REFUSED] ... resolve to trace id ...` | two files in a batch share a trace id | rename one so the name prefixes differ, or set `calibrated_agent_evaluation.trace_id` |
| `[skipped] ... could not detect` | no attributes the ingest recognizes | pass `--convention`, or add GenAI attributes ([gallery](../integrate/gallery/index.md)) |
| `degraded to ownership='probed'` | input and output but no agent or tool steps | stored outcome-only (still labelable) unless `--require-owned` |

??? failure "Output"

    ```text
    --8<-- "fixtures/expected/docs/ts-agent-not-found.txt"

    --8<-- "fixtures/expected/docs/ts-no-spans.txt"
    ```

    The third command stops on the missing agent and tool steps:

    ```text
    --8<-- "fixtures/expected/docs/integrate-capture.txt"

    --8<-- "fixtures/expected/docs/calibrated-agent-evaluation-ingest-dry-run.txt"

    --8<-- "fixtures/expected/docs/integrate-ingest-require-owned.txt"
    ```


## The trace store

| You see | Why | Do |
| --- | --- | --- |
| `the langfuse backend needs LANGFUSE_PUBLIC_KEY / LANGFUSE_SECRET_KEY` | no credentials | export both and `LANGFUSE_HOST` (per project) |
| `Langfuse ... failed: ConnectError` | nothing answers at `LANGFUSE_HOST` | `just langfuse-up` for the local stack; `calibrated-agent-evaluation doctor` checks reachability |
| `StaleReadError` | async writes unreadable after `CALIBRATED_AGENT_EVALUATION_READ_WAIT_SECONDS` (30) | rerun later, or raise the wait |
| `no trace with trace_id '...'` | wrong store; the default is `$CALIBRATED_AGENT_EVALUATION_LOCAL_ROOT`, then `./.calibrated-agent-evaluation` | pass `--root` or run from the right directory |
| `SchemaVersionError` on `calibrated-agent-evaluation load` | the dump was written by another schema version, and records are not converted between versions | load with the version that wrote the dump ([Versioning](../versioning.md)) |
| `holds dump files but no dump-manifest.json` on `load` (`IncompleteDumpError`) | the dump was interrupted or is still being written, since the manifest is written last | dump the source store again |
| `differs from DIR/dump-manifest.json` or `is missing` on `load` (`DumpMismatchError`) | a file was edited after the dump, or another dump overwrote part of this one | dump the source store again; to commit a dump, keep git from rewriting its line endings ([Configure the CI job](ci-config.md#seed-the-jobs-store)) |
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
| `no calibration, no score` | a judge scores only once it is certified and has passed its bias check in this store (library: `UncalibratedScoreError`) | `calibrate`, `bias`, then `judge` ([Put a judge on probation](../walkthrough/judge.md)) |
| not enough labels to trust this judge | too few labels to tell | label the per-class count it names |
| labels already used to test a judge for this mode | a set tests a mode once, whatever the result or judge file name; overlapping sets count too | label a fresh set ([Watch production](../walkthrough/production.md)) |
| bias check failed | the judge was swayed by a change that should not matter more often than its threshold allows | revise the prompt, then check it again on fresh labels |
| `no recorded verdict (cassette miss)` | the prompt was never recorded, and replay does not call the live model | record these traces with `--cassette-mode record` (it calls the live model), or narrow with `--cell` |
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
| rate not estimated (too few scored traces) | under 150 randomly sampled traces, the range is wrong too often | `calibrated-agent-evaluation judge --frame production-representative` on more |
| rate not estimated (judge needs more labels) | certified, but on too few labels for a rate | label what `--explain` names, or use `--rate-safe` |
| `no judge verdicts with a failure mode ... to gate` | there are no verdicts of the source you asked for in scope | `calibrated-agent-evaluation judge`, or `--verdicts human` / `scorer` |
| usage error, exit 2 | bad flag or argument | `calibrated-agent-evaluation <command> --help`, [CLI reference](../cli/index.md) |

In both rate-not-estimated cases the judge still scores and the gate still sets the exit code.

??? failure "Output"

    ```text
    --8<-- "fixtures/expected/docs/guide-rate-too-few.txt"

    --8<-- "fixtures/expected/docs/ts-nothing-to-gate.txt"

    --8<-- "fixtures/expected/docs/ts-usage-error.txt"
    ```
