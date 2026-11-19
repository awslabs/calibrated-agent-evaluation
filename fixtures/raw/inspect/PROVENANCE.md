# Provenance — Inspect `.eval` fixtures

> Repo rule: fixture traces are genuine vendor output, never hand-shaped. This file records where
> each log came from so that is auditable.

## `ORD-1001.eval`, `ORD-1002.eval`, `CLM-4471.eval` — minted by a real eval run

Produced by `corpus/agents/orderbot_inspect.py` via `corpus/generate/record.py`: a genuine
`inspect_ai.eval()` over a real `Task` with real `@tool`s, solved by the built-in `react` agent. The
model runs through the `cassette/...` provider, which recorded once against live Bedrock and replays
deterministically offline, so the `ModelEvent` / `ToolEvent` transcript is authentic.

Each is a **one-row** dataset, so each holds exactly one sample with `id=1`.

## `multisample-3.eval` — three real samples, one container

Written by `generate_multisample.py` (run it to regenerate). Needed by the fix for multi-sample logs
losing samples, which had to prove `normalize_all` returns one `Trace` per sample: no multi-sample log existed, because
every log above is a one-row Task.

**It is assembled, not freshly minted, and that is a deliberate limit.** A genuinely new 3-row eval
cannot be recorded offline: the cassette provider keys replay on the request, and a combined `Task`
carries the union of all three scenarios' tools, so every sample's `toolConfig` differs from what was
recorded and every request misses the cassette. Minting one needs a live Bedrock run.

What the script does and does not touch:

- **Sample content is carried over verbatim** — the real recorded `events`, `output`, `metadata`
  (including each sample's own `trace_id` and `taxonomy_cell`). This is the same method and the same
  justification as `fixtures/raw/otel/generate_convention_fixtures.py`: the container is transport,
  the samples are the genuine transcripts.
- **`sample.id` is rewritten** to the scenario name. Every source log has `id=1`, so all three would
  collide — the zip writer warns, and an id-keyed read loses two of the three. A genuine 3-row dataset
  has distinct ids; each sample takes the value its own `metadata["trace_id"]` already carries.
- **`eval.dataset.samples` / `sample_ids` are rewritten.** That is the index
  `read_eval_log_samples` fetches by, not decoration — left stale it said `samples=1, sample_ids=[1]`
  and a reader raised `IndexError`. Verified by reading the written file back.
- **`results` and `reductions` are dropped.** They describe the original single-sample run and would
  be a false claim about this container.

So this fixture is sound for what it is used for — proving per-sample normalization, trace-id
derivation, and the multi-sample refusal — and is **not** evidence about how a real 3-row eval scores
or aggregates. Nothing asserts that against it.
