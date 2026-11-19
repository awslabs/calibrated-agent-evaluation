# `fixtures/raw/hf/`: frozen HuggingFace agent-trace slice (dataset ingestion)

A small frozen slice of a published agent-trace dataset. The JSONL, parquet and HuggingFace
readers use it as their offline, deterministic ingest fixture. Every row is real captured output,
never hand-shaped (see the corpus rule in the parent `fixtures/README.md`).

## Source and license

- Dataset: [`Exgentic/agent-llm-traces`](https://huggingface.co/datasets/Exgentic/agent-llm-traces)
  (v1), revision `70036b93a04e61b0ea2706a68b962f4f26774587`.
- License: [CDLA-Permissive-2.0](https://cdla.dev/permissive-2-0/), as declared on the dataset
  card and in the HuggingFace API metadata for that revision.
- Only SWE-bench sessions are taken. AppWorld sessions are excluded: they carry bearer-token-like
  strings, and AppWorld's own license asks for its data to be redistributed only in encrypted form.

## Files

| File | Size | Contents |
|---|---|---|
| `swebench-slice.parquet` | 102 KB | 3 whole sessions, one per row group. The parquet loader reads this. |
| `renamed-columns-slice.parquet` | 103 KB | The same 3 sessions with `spans` renamed to `otel_events` and `session_id` to `trace_key`. A default ingest skips every row and names the missing `spans` column, so a `DatasetMapping` is needed. |
| `swebench-session.jsonl` | 145 KB | 1 whole session, repeated from the parquet slice, as one JSON object per line. It exercises the JSONL loader and the cross-source session id clash. |
| `MANIFEST.json` | | Source, revision, shard sha256s, and the selected session ids with their shard and row. |
| `make_slice.py` | | Rebuilds every file above from the pinned revision. |

No row was trimmed or edited. The blank-line and malformed-line JSONL cases are built inside the
tests, not committed here.

## Sessions

| session_id | shard | harness | model | spans | exercises |
|---|---|---|---|---|---|
| `e3bfdef4aca3_c0427304` | `train-00012` row 3 | `claude_code` | `azure/DeepSeek-V3.2` | 2 | structured `tool_call` and `tool_call_response` parts paired by `id` (`TodoWrite`, `mcp__environment__bash`) |
| `c0d1d84c7898_35f42bfd` | `train-00019` row 38 | `tool_calling` | `azure/Kimi-K2.5` | 12 | a longer structured tool-call session (`bash` 8 times, then `finish`) |
| `140646846ae7_8d10c195` | `train-00019` row 13 | `smolagents_code` | `azure/Kimi-K2.5` | 25 | a code-action harness: tool use is written into the assistant text, so no `tool_call` parts exist |

Each was the smallest session of its harness in those two shards.

## Shape notes for ingestion

- v1 rows hold `harness`, `benchmark`, `models`, `max_tokens`, `total_tokens`, `session_id`,
  `spans` and `collected_at`. The default `DatasetMapping` reads them unchanged.
- v1 spans have no `parent_span_id`. Each session is a flat list of `chat` spans emitted by LiteLLM,
  with no `invoke_agent` or `execute_tool` boundary spans, so the messages adapter collapses it to a
  one-span `owned` Trace.
- v1 has no outcome columns (`success`, `score`). Ingestion would not turn them into a `Label`
  anyway: a benchmark's self-reported pass is not a gold label.

## Regenerating

```bash
uv run --extra datasets --extra hf python fixtures/raw/hf/make_slice.py
```

The script downloads `data/train-00012-of-00039.parquet` (41 MB) and
`data/train-00019-of-00039.parquet` (92 MB) at the pinned revision, refuses either one if its
sha256 differs from the LFS oid recorded in the script, selects the sessions by id and rewrites the
files and `MANIFEST.json`. Pass `--shard-dir DIR` to keep or reuse the downloaded shards. Output is
byte-identical across runs for a given pyarrow version. After regenerating, rebuild the
`gen_ai/messages` OTLP fixture with `fixtures/raw/otel/generate_convention_fixtures.py`, which
reads `swebench-slice.parquet`.
