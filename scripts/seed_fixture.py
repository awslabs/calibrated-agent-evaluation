#!/usr/bin/env python3
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Seed the shared fixture corpus into a backend (the data-plane acceptance harness).

Loads ``fixtures/normalized/*.json`` + ``fixtures/labels/gold.json`` and ``put``-s them into the
chosen backend, so the manual-acceptance checklist and the both-worlds parity check (the local
store and Langfuse return the same results) have a seeded store to run against. The data plane
ships this recipe because no earlier stage provides one.

Usage (via the justfile):
    just seed-fixture local      # seed the offline LocalBackend (default)
    just seed-fixture langfuse   # seed self-hosted Langfuse (needs `just langfuse-up` + env)

Or directly:
    uv run python scripts/seed_fixture.py --backend local [--root .calibrated-agent-evaluation]
    uv run python scripts/seed_fixture.py --backend langfuse
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

from calibrated_agent_evaluation.backends.base import TraceStoreBackend
from calibrated_agent_evaluation.backends.local import LocalBackend
from calibrated_agent_evaluation.schema.serde import label_from_dict, trace_from_dict
from calibrated_agent_evaluation.schema.types import Label, Trace

_REPO_ROOT = Path(__file__).resolve().parent.parent
_FIXTURES = _REPO_ROOT / "fixtures"


def load_corpus() -> tuple[list[Trace], list[Label]]:
    """Load every normalized trace and every gold label from the committed corpus."""
    traces = [
        trace_from_dict(json.loads(p.read_text(encoding="utf-8")))
        for p in sorted((_FIXTURES / "normalized").glob("*.json"))
    ]
    labels = [
        label_from_dict(item)
        for item in json.loads((_FIXTURES / "labels" / "gold.json").read_text(encoding="utf-8"))
    ]
    return traces, labels


def _local_backend(root: str) -> LocalBackend:
    return LocalBackend(root=root)


def _langfuse_backend() -> TraceStoreBackend:
    from calibrated_agent_evaluation.backends.langfuse import LangfuseBackend

    host = os.environ.get("LANGFUSE_HOST", "http://localhost:3000")
    public_key = os.environ.get("LANGFUSE_PUBLIC_KEY")
    secret_key = os.environ.get("LANGFUSE_SECRET_KEY")
    if not public_key or not secret_key:
        raise SystemExit(
            "LANGFUSE_PUBLIC_KEY / LANGFUSE_SECRET_KEY must be set (see docker/langfuse/.env.example)"  # noqa: E501
        )
    return LangfuseBackend.from_credentials(public_key, secret_key, host)


def seed(backend: TraceStoreBackend) -> tuple[int, int]:
    """Seed the corpus into ``backend``. Returns (n_traces, n_labels) written."""
    traces, labels = load_corpus()
    for trace in traces:
        backend.put_trace(trace)
    for label in labels:
        backend.put_score(label)
    return len(traces), len(labels)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Seed the shared fixture corpus into a backend.")
    parser.add_argument("--backend", choices=["local", "langfuse"], default="local")
    parser.add_argument(
        "--root", default=".calibrated-agent-evaluation", help="LocalBackend store root"
    )
    args = parser.parse_args(argv)

    backend: TraceStoreBackend = (
        _local_backend(args.root) if args.backend == "local" else _langfuse_backend()
    )
    n_traces, n_labels = seed(backend)
    target = args.root if args.backend == "local" else os.environ.get("LANGFUSE_HOST")
    print(f"Seeded {n_traces} traces + {n_labels} labels into {args.backend} backend ({target}).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
