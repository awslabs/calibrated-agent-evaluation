# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Rebuild the frozen dataset fixtures in this directory from a pinned upstream revision.

Downloads only the two shards that hold the selected sessions, checks each against the sha256
the HuggingFace API publishes for that revision, picks rows by explicit ``session_id`` and writes
them back unchanged. Rows are never trimmed or edited, so every fixture is genuine captured output.

    uv run --extra datasets --extra hf python fixtures/raw/hf/make_slice.py
    uv run --extra datasets --extra hf python fixtures/raw/hf/make_slice.py --shard-dir /tmp/shards

``--shard-dir`` reuses shards downloaded earlier; a file whose sha256 does not match is refused.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import tempfile
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq
from huggingface_hub import hf_hub_download

REPO_ID = "Exgentic/agent-llm-traces"
REVISION = "70036b93a04e61b0ea2706a68b962f4f26774587"
LICENSE = "CDLA-Permissive-2.0"

# sha256 of each shard at REVISION, as published in the HF tree API's LFS oid.
SHARDS: dict[str, str] = {
    "data/train-00012-of-00039.parquet": (
        "b21531f9446828298e26e07d5718e49290a6ecf80ba14b1fc6cb0601f62cca7c"
    ),
    "data/train-00019-of-00039.parquet": (
        "1273811490529293ff9e825efb4232b451e32511c46934826a7a9d830f6d60bf"
    ),
}

# (shard, session_id, what the session exercises). Order here is row order in the outputs.
SESSIONS: list[tuple[str, str, str]] = [
    (
        "data/train-00012-of-00039.parquet",
        "e3bfdef4aca3_c0427304",
        "structured tool_call parts paired by id (TodoWrite, mcp__environment__bash)",
    ),
    (
        "data/train-00019-of-00039.parquet",
        "c0d1d84c7898_35f42bfd",
        "structured tool_call parts over a longer session (bash x8, finish)",
    ),
    (
        "data/train-00019-of-00039.parquet",
        "140646846ae7_8d10c195",
        "code-action harness: tool use embedded in assistant text, no tool_call parts",
    ),
]

# The JSONL twin repeats one parquet session so the cross-source id clash has a real case.
JSONL_SESSION = "e3bfdef4aca3_c0427304"

RENAMED_COLUMNS = {"spans": "otel_events", "session_id": "trace_key"}

_HERE = Path(__file__).resolve().parent


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--shard-dir", type=Path, help="directory holding or receiving shards")
    parser.add_argument("--out-dir", type=Path, default=_HERE)
    args = parser.parse_args()

    shard_dir = args.shard_dir or Path(tempfile.mkdtemp(prefix="exgentic-shards-"))
    shard_dir.mkdir(parents=True, exist_ok=True)
    shard_paths = {name: _fetch(name, shard_dir) for name in SHARDS}

    rows: list[dict[str, Any]] = []
    provenance: list[dict[str, Any]] = []
    for shard, session_id, exercises in SESSIONS:
        table = pq.read_table(shard_paths[shard])
        ids = table.column("session_id").to_pylist()
        matches = [i for i, sid in enumerate(ids) if sid == session_id]
        if len(matches) != 1:
            raise SystemExit(f"{session_id}: expected one row in {shard}, found {len(matches)}")
        (index,) = matches
        row = table.slice(index, 1).to_pylist()[0]
        rows.append(row)
        provenance.append(
            {
                "session_id": session_id,
                "shard": shard,
                "row": index,
                "harness": row["harness"],
                "benchmark": row["benchmark"],
                "models": row["models"],
                "spans": len(row["spans"]),
                "exercises": exercises,
            }
        )

    schema = pq.read_schema(shard_paths[SESSIONS[0][0]]).remove_metadata()
    table = pa.Table.from_pylist(rows, schema=schema)
    out = args.out_dir
    _write_parquet(table, out / "swebench-slice.parquet")
    _write_parquet(
        table.rename_columns([RENAMED_COLUMNS.get(n, n) for n in table.column_names]),
        out / "renamed-columns-slice.parquet",
    )
    (jsonl_row,) = [r for r in rows if r["session_id"] == JSONL_SESSION]
    (out / "swebench-session.jsonl").write_text(
        json.dumps(jsonl_row, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    manifest = {
        "source": f"https://huggingface.co/datasets/{REPO_ID}",
        "revision": REVISION,
        "license": LICENSE,
        "selection": "whole sessions picked by session_id from swebench shards; no row edited",
        "shards": SHARDS,
        "files": {
            "swebench-slice.parquet": [s[1] for s in SESSIONS],
            "renamed-columns-slice.parquet": [s[1] for s in SESSIONS],
            "swebench-session.jsonl": [JSONL_SESSION],
        },
        "renamed_columns": RENAMED_COLUMNS,
        "sessions": provenance,
        "regenerate": "uv run --extra datasets --extra hf python fixtures/raw/hf/make_slice.py",
    }
    (out / "MANIFEST.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    for name in ("swebench-slice.parquet", "renamed-columns-slice.parquet"):
        print(f"{name}: {(out / name).stat().st_size} B")
    print(f"swebench-session.jsonl: {(out / 'swebench-session.jsonl').stat().st_size} B")


def _fetch(name: str, shard_dir: Path) -> Path:
    local = shard_dir / name
    if not local.exists():
        hf_hub_download(
            repo_id=REPO_ID,
            filename=name,
            repo_type="dataset",
            revision=REVISION,
            local_dir=shard_dir,
        )
    digest = _sha256(local)
    if digest != SHARDS[name]:
        raise SystemExit(f"{local}: sha256 {digest} does not match {SHARDS[name]} at {REVISION}")
    return local


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _write_parquet(table: pa.Table, path: Path) -> None:
    # One row per row group, so the reader's walk across row groups runs on real data.
    pq.write_table(table, path, row_group_size=1, compression="zstd")


if __name__ == "__main__":
    main()
