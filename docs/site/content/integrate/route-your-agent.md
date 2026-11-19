# Route your agent through Calibrated Agent Evaluation

On this path Calibrated Agent Evaluation runs your agent and records each run as a trace. You hand
it a function that builds your agent, or name an agent you have deployed to Amazon Bedrock AgentCore
Runtime. Use it when you want the library to drive the agent over inputs it generates, rather than
evaluate runs you captured elsewhere. The traces are your agent's real behavior: the library
supplies the inputs and the recording, never the outputs.

Three kinds of agent are supported:

| Your agent is | Executor | You provide | Extra |
| ------------- | -------- | ----------- | ----- |
| A Strands `Agent`, `Graph` or `Swarm` | `StrandsExecutor` | `build(model)` returning the topology | `strands` |
| An Inspect AI task | `InspectExecutor` | the task, with your agent as its solver | `inspect` |
| Deployed to AgentCore Runtime | `AgentCoreRuntimeExecutor` | the runtime ARN and the request body its agent reads | `agentcore` |

If your agent is on none of these, use [Capture in process](capture-and-ingest.md) or export its
spans and [ingest the files](ingest-existing-traces.md).

## Replay a recorded run first

Executors record every model call to a cassette file and can replay it later, so a run needs no
model and no credentials. That is also how you keep CI offline. This replays the repository's
ClaimsCrew, a Strands graph with a supervisor and four specialists, from its committed cassette:

```python
--8<-- "docs/site/examples/integrate/route_strands_replay.py:replay"
```

```text
--8<-- "fixtures/expected/docs/integrate-route-strands.txt"
```

The full program is `docs/site/examples/integrate/route_strands_replay.py`. The Inspect equivalent
replays the OrderBot task:

```python
--8<-- "docs/site/examples/integrate/route_inspect_replay.py:replay"
```

```text
--8<-- "fixtures/expected/docs/integrate-route-inspect.txt"
```

`cassette_mode` is `Mode.RECORD`, `Mode.REPLAY` or `Mode.AUTO`. `AUTO`, the default, replays a call
the cassette holds and records one it does not. The `CALIBRATED_AGENT_EVALUATION_CASSETTE_MODE` environment variable
sets the default, so `CALIBRATED_AGENT_EVALUATION_CASSETTE_MODE=replay` in CI makes any unrecorded call fail instead of
reaching a model.

## Your Strands agent

Write a function that takes the model and returns your topology. The executor passes in the model
so it can wrap it for recording and replay; you only assemble the agents. This is a two-agent
graph:

```python
# needs the strands extra; the first run calls Bedrock and records the cassette
from calibrated_agent_evaluation.execute import Mode, StrandsExecutor, StrandsTask


def build_claims_crew(model):
    from strands import Agent
    from strands.multiagent import GraphBuilder

    intake = Agent(model=model, name="intake", system_prompt="Extract the claim fields.")
    checker = Agent(model=model, name="checker", system_prompt="Apply the policy exclusions.")
    builder = GraphBuilder()
    builder.add_node(intake, "intake")
    builder.add_node(checker, "checker")
    builder.add_edge("intake", "checker")
    builder.set_entry_point("intake")
    return builder.build()


with StrandsExecutor(
    model_id="global.anthropic.claude-sonnet-4-6",   # the default
    region="us-west-2",                              # the default
    cassette_path="cassettes/claims.json",
    cassette_mode=Mode.RECORD,
    timeout_seconds=60.0,
) as executor:
    traces = executor.run(
        StrandsTask(
            build=build_claims_crew,
            input="Claim 4471: water damage, policyholder away in Florida last month.",
            taxonomy_cell={"line": "property", "peril": "water"},
        )
    )
```

A live run goes through Bedrock, so set your AWS credentials the usual way (for example
`AWS_PROFILE`). Commit the cassette and later runs replay it. `taxonomy_cell` is copied onto the
trace, so failures can later be broken down by line and peril.

The default `global.` model ID is a cross-Region inference profile, so Bedrock may process the
agent's prompts in other Regions. If you have data-residency requirements, pass an in-Region model
ID or a geographic profile; see
[Bedrock model IDs and Regions](../getting-started/installation.md#bedrock-model-ids-and-regions).

The executor sends `temperature=0.0` by default. The newest Anthropic models reject an explicit
temperature, so pass `temperature=None` to send none.

## Your Inspect task

Your agent is the task's solver. `InspectExecutor` takes an Inspect model string:

```python
# needs the inspect extra
from inspect_ai import Task
from inspect_ai.dataset import Sample

from calibrated_agent_evaluation.execute import InspectExecutor

task = Task(dataset=[Sample(input="I want a refund for order 1001.")], solver=my_solver())

with InspectExecutor(model="bedrock/global.anthropic.claude-sonnet-4-6") as executor:
    traces = executor.run(task)
```

For recording and replay, use a `cassette/<name>` model with `cassette_path`, as in the replay
example above. `cassette_inner_model` names the real model a recording calls.

## Generate the inputs too: calibrated-agent-evaluation generate --run

`calibrated-agent-evaluation generate` builds a discovery frame from your input dimensions, a
covering array in which every pair of dimension values appears, writes one input per row, and with
`--run` sends each through your agent and stores the traces:

```bash
uv sync --inexact --extra sampling --extra stats --extra strands
calibrated-agent-evaluation generate --run --taxonomy taxonomy.json \
  --executor strands --agent mypkg.agents:build_claims_crew
```

```bash
uv sync --inexact --extra sampling --extra stats --extra inspect
calibrated-agent-evaluation generate --run --taxonomy taxonomy.json \
  --executor inspect --agent mypkg.agents:my_solver \
  --agent-model bedrock/global.anthropic.claude-sonnet-4-6
```

- `--taxonomy FILE` is a JSON object of dimension to values, such as
  `{"line": ["property", "auto"], "peril": ["water", "fire"]}`. Without it the dimensions come
  from the stored traces' taxonomy cells, and a store with none is refused. `--frame FILE` takes a
  frame written by `calibrated-agent-evaluation define --out` instead.
- `--agent module:callable` is your agent: a topology builder for Strands, a solver for Inspect
  (the generated inputs are wrapped into the task). It is required with `--run`, except with
  `--executor agentcore`, where the agent is the deployed runtime
  ([below](#your-agent-on-agentcore-runtime)).
- `--agent-model` is required for Inspect: the model the agent runs on, either
  `bedrock/<model-id>` or `cassette/<name>` together with `--cassette`. Strands uses its own
  default model.
- `--cassette PATH` records and replays the agent's model calls.
- `--model <id>` writes the inputs with an LLM (needs the `bedrock` extra). Without it, inputs come
  from an offline template.

The traces `generate --run` stores carry the `discovery` frame: they were made to find failures,
so they never feed a production rate estimate.

From Python, the same pieces are `calibrated_agent_evaluation.synth.generate_and_run` with an
executor and a task builder from `calibrated_agent_evaluation.execute`:
`StrandsTaskBuilder(build_topology=...)` or `InspectTaskBuilder(solver=...)`, or
`AgentCoreTaskBuilder()` from `calibrated_agent_evaluation.agentcore`. It refuses to run without an
executor, since the outputs must come from your agent.

## Your agent on AgentCore Runtime

`--executor agentcore` sends each generated input to an agent deployed to Amazon Bedrock
AgentCore Runtime and stores one trace per input. This replays the repository's recorded run of a
synthetic runtime, so it needs no AWS account:

```bash
uv sync --inexact --extra sampling --extra stats --extra agentcore
```

```text
--8<-- "fixtures/expected/docs/integrate-route-agentcore.txt"
```

The frame file holds one combination of input dimensions, that of the ClaimsCrew claim CLM-4471, and
the cassette holds the recorded answer to the input generated for it. Each run, replayed or live,
adds one record to the run manifest: the session ID, the runtime version, the qualifier, the
configuration bundle, the timing, the attempts and how the session was stopped, keyed by the trace
ID. The manifest is a pair of files, `agentcore_run_record.json` (the one the library reads back)
and `agentcore_run_record.csv`, with one record per run. Each file is rewritten atomically while it
holds a lock on their directory, so two runs writing at once both keep their record. The pair goes
to `<root>/records/agentcore/` unless you pass `--log-dir`. `calibrated-agent-evaluation dump` and
`calibrated-agent-evaluation load` do not carry that directory, so copy it yourself to keep the
provenance.

In a live run, the executor makes a new runtime session ID and W3C trace context for each input, invokes
the runtime with them, reads the final answer from the response, reads the run's span tree, and
stops the session. Two options are required:

- `--runtime-arn ARN` is the runtime, by its own ARN (not an endpoint's). Every AWS call goes to
  the ARN's region.
- `--payload-format NAME` is the request body the agent reads: `generic_message`
  (`{"message": ...}`), `openai` (`{"messages": [...]}`), `agentcore_sse`
  (`{"prompt": ..., "session_id": ...}`) or `nested_agentcore`. It is never guessed, since each
  guess would be a billed call.

The others are optional:

- `--qualifier NAME` is the endpoint to invoke (default `DEFAULT`).
- `--config-bundle-arn ARN` and `--config-bundle-version VERSION`, given together, name a
  configuration bundle for the agent to run with. They are sent in the `baggage` header, named in
  the plan, and kept in the run manifest and the cassette. An empty value for either is refused,
  since the agent would otherwise run without the bundle.
- `--read-timeout SECONDS` is how long an invocation waits for each read of the response, not
  for the whole answer (default 300). It must be a positive, finite number of seconds.
- `--profile NAME` is the AWS profile (default `AWS_PROFILE`).
- `--cassette PATH` records runs into this file and replays them from it.
- `--log-dir DIR` is where the run manifest goes instead.
- `--yes` goes ahead with the billed calls.

These flags are refused with another executor or without `--run`, and `--agent` and
`--agent-model` are refused with this one. The frame must hold discovery rows only.

A run that can call AWS, in any cassette mode but `replay`, first checks
`CALIBRATED_AGENT_EVALUATION_AGENTCORE_MAX_BILLED_CALLS` and that a span reader is installed, then that the trace store
can take the traces (with `--backend langfuse`, that the host answers and accepts the keys), since
they are stored only once the last input has run. It then prints what it will call and stops unless
you pass `--yes`:

```text
--8<-- "fixtures/expected/docs/integrate-route-agentcore-plan.txt"
```

Before the first invocation, read-only calls check the runtime. One that only accepts OAuth (it
has a custom JWT authorizer) or that does not serve HTTP (MCP, A2A or AG-UI) is refused before
anything billed is sent; the executor signs its calls with SigV4. A busy session or throttling is retried
in the same session, up to three attempts. Any other failure, an error event in the agent's
response included, stops the command at that input, and the traces of the inputs before it are
not stored: with `--cassette` they stay in the cassette, without it only their records in the
run manifest are kept. `CALIBRATED_AGENT_EVALUATION_AGENTCORE_MAX_BILLED_CALLS=N` caps the billed calls a command
makes, retries included. It does not count CloudWatch Logs Insights queries (`logs:StartQuery`),
which a span reader may make and which Logs Insights bills by the data each one scans.

Live runs read each run's span tree from AgentCore Observability with the per-run span reader,
`calibrated_agent_evaluation.agentcore.AgentCoreSpanReader`, which `calibrated-agent-evaluation
generate` builds from the same clients as the executor. It finds the run by its trace ID, waits for
the telemetry to settle, and returns the whole tree or refuses the run; a run whose trace context
was not sampled records no spans, so it is refused, naming that cause. The window ingest,
`calibrated-agent-evaluation ingest --source agentcore`, is a separate reader for a time window.

From Python, build an `AgentCoreRuntimeExecutor` from a `RuntimeTarget` and run
`AgentCoreRuntimeTask`s, or pass it to `generate_and_run` with `AgentCoreTaskBuilder()`. Its
`span_reader` takes any object that implements `RunSpanReader`, its `run_records` holds the
`RunRecord` of every run it finished, and its `manifest_errors` says why any of those records could
not be added to the run manifest; the [`calibrated_agent_evaluation.agentcore`
reference](../reference/calibrated_agent_evaluation/agentcore/index.md) describes all of them. The
executor builds no AWS client until its first live call, so a replay builds none and needs no
credentials.

## Hold a conversation instead of one shot: --multi-turn

Many failures only appear on the third turn. With `--multi-turn`, each generated input opens a
conversation: a simulated person writes the user turns, your agent answers them, and the
conversation is stored as one trace.

```bash
uv sync --inexact --extra sampling --extra stats --extra strands --extra synth
calibrated-agent-evaluation generate --run --taxonomy taxonomy.json --executor strands --multi-turn --max-turns 4 \
  --agent mypkg.agents:build_support_agent
```

Every agent turn is a fresh run of your agent over the conversation so far, which is what a
multi-turn agent gets in production. The conversation is stored as one trace that keeps every
turn's spans, in turn order and prefixed `t1:`, `t2:` and so on. Its input is the whole transcript,
with the goal and why the conversation stopped (`goal_completed`, `actor_stopped` or `max_turns`),
so a labeler or a judge reads the conversation rather than its opening line.

- `--max-turns N` stops the conversation after N agent turns (default 6). The simulated person can
  end it earlier, when it decides its goal is met.
- `--persona FILE` is a JSON object with `traits`, `context` and `actor_goal`. Without it, each
  input gets a persona built from its own taxonomy cell.
- `--simulator strands|agentcore` picks who plays the customer: the `strands-agents-evals` actor
  (the default), or AgentCore's actor simulator (the `agentcore` extra). `--simulator-model`
  sets its model.
- `--multi-turn` works with `--executor strands` and `--executor agentcore`. On an AgentCore
  Runtime agent, every turn of a conversation is sent on one runtime session, so the deployed
  agent keeps its own memory of the earlier turns, and the session is stopped once the
  conversation ends. The Inspect route evaluates one input per sample, so it has no turns.

```json
{
  "traits": {"tone": "impatient", "expertise": "none"},
  "context": "A customer on their third call about the same order.",
  "actor_goal": "Get the refund issued today, or a clear reason why not."
}
```

The simulated person is `strands_evals`'s `ActorSimulator` (the `synth` extra), and it writes only
the user turns. The library never writes an agent turn: each one comes back from your agent through the
executor, and a turn that produces no text is refused rather than recorded as an empty reply.

## Next

- [Look and label](../guides/lifecycle/3-look-and-label.md) takes stored traces through labeling,
  judging and the gate.
- The [`calibrated_agent_evaluation.execute`
  reference](../reference/calibrated_agent_evaluation/execute/index.md) lists every executor
  parameter, and the [`calibrated_agent_evaluation.agentcore`
  reference](../reference/calibrated_agent_evaluation/agentcore/index.md) the AgentCore Runtime
  executor's.
