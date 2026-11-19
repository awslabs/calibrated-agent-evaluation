# Integration gallery

One short recipe per trace source. Each one shows the command or code, the real output it
produced on a trace from the repository's fixtures, what you get, and its limits.

<div class="grid cards" markdown>

-   **[OpenTelemetry GenAI spans](otel-genai.md)**

    ---

    Any agent that emits the OpenTelemetry GenAI semantic conventions, exported as OTLP-JSON.

-   **[OpenInference (Arize Phoenix)](openinference.md)**

    ---

    Spans from an OpenInference instrumentor, the format Phoenix collects.

-   **[OpenLLMetry (Traceloop)](openllmetry.md)**

    ---

    Spans from Traceloop's OpenLLMetry instrumentations.

-   **[Strands Agents](strands.md)**

    ---

    Ingest a Strands OTel export, or let Calibrated Agent Evaluation drive the topology.

-   **[Inspect AI](inspect.md)**

    ---

    `.eval` logs from Inspect runs, one trace per sample.

-   **[Datasets: JSONL, parquet, HuggingFace](datasets.md)**

    ---

    A table of captured traces, one trace per record.

-   **[Langfuse](../langfuse.md)**

    ---

    Read every trace in a Langfuse project, or use Langfuse as the store.

-   **[Black-box HTTP and WebSocket](../probe-black-box.md)**

    ---

    An agent you can only call: prompts in, replies out, no steps.

-   **[Amazon Bedrock AgentCore](agentcore.md)**

    ---

    Run an agent deployed to AgentCore Runtime with `calibrated-agent-evaluation generate --run
    --executor agentcore`, one trace per input, and publish the failures your reviewers caught as
    AgentCore evaluation datasets with `calibrated-agent-evaluation dataset publish` (`pull` reads a
    version back), and read a runtime's traces for a time window with `calibrated-agent-evaluation
    ingest --source agentcore`. Live runs read each run's spans with the per-run span reader,
    `AgentCoreSpanReader`, and `--cassette` records runs once and replays them with no AWS
    credentials. `calibrated-agent-evaluation probe --executor agentcore` reads no spans; it probes
    a deployed agent live and stores each reply as a probed trace.

</div>

Not listed? If your framework emits OpenTelemetry spans, or has an OpenLLMetry or OpenInference
instrumentation you can turn on, start with [OpenTelemetry GenAI spans](otel-genai.md): the same
command detects the convention.
