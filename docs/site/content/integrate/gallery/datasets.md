# Datasets: JSONL, parquet, HuggingFace

You need a table of captured traces with one session per record and its spans in a column.
JSONL needs no extra, parquet needs `datasets`, and a HuggingFace repo needs `hf` and `datasets`.

## Ingest

```bash
calibrated-agent-evaluation ingest captures/week-30.jsonl
calibrated-agent-evaluation ingest captures/slice.parquet
calibrated-agent-evaluation ingest --format hf Exgentic/agent-llm-traces
```

On the repository's frozen slice of the
[`Exgentic/agent-llm-traces`](https://huggingface.co/datasets/Exgentic/agent-llm-traces)
dataset, as JSONL and as parquet. The slice holds three SWE-bench sessions, and the JSONL file
repeats the first of them:

```text
--8<-- "fixtures/expected/docs/integrate-ingest-jsonl.txt"
```

```text
--8<-- "fixtures/expected/docs/integrate-ingest-parquet.txt"
```

Each record's spans go through the same convention detection as an OTLP file. Here they use
`gen_ai/messages`.

## Your columns

The defaults match the Exgentic layout: spans in `spans`, the session id in `session_id`, and
`harness` and `benchmark` copied onto each trace's taxonomy cell. A dataset with other names maps
them from Python:

```python
from pathlib import Path

from calibrated_agent_evaluation.normalize.datasets import DatasetMapping, JSONLDataset, load_traces

mapping = DatasetMapping(
    spans_column="otel_spans",
    session_id_column="conversation_id",
    taxonomy_columns=("product", "locale"),
)
for trace in load_traces(JSONLDataset(path=Path("captures/week-30.jsonl"), mapping=mapping)):
    ...
```

`ParquetDataset` and `HFDataset` take the same `mapping`.

## What you get

One owned trace per record, tagged with the `discovery` frame. A benchmark slice was collected to
find failures, not sampled from your production traffic, so the toolkit estimates no production
rate from it.

## Limits

- A dataset's own `success` or `score` column may be copied onto the taxonomy cell as a field to
  filter by, but it does not become a label. Labels come from your reviewers.
- The third session in the slice is a code-action agent whose tool use is written into its reply
  text, so it comes in with 0 tool calls.
- `--format hf` downloads every parquet shard of the repo before it reads the first record. For
  `Exgentic/agent-llm-traces` that is 39 shards, about 984 MB.
- The frozen slice comes from `Exgentic/agent-llm-traces` at a pinned revision, under
  CDLA-Permissive-2.0. `fixtures/raw/hf/README.md` names the revision and the sessions.
