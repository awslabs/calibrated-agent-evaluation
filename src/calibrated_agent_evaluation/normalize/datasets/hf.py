# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""HuggingFace-hub dataset resolver.

Resolves a dataset repo id to its local parquet shard files (via the ``[hf]`` extra's
``huggingface_hub``), then streams each through the shared :mod:`parquet` reader; so a remote
dataset normalizes through exactly the same path as a local ``ParquetDataset``.
``huggingface_hub`` is imported lazily: a missing ``[hf]`` extra surfaces as a typed
:class:`~calibrated_agent_evaluation.normalize.errors.NormalizationError` naming the installable
string, and any hub/OS resolve error is wrapped typed and chained; never a raw exception.

Network use is confined here and is opt-in: only an ``HFDataset`` source reaches it.
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path
from typing import TYPE_CHECKING, Any

from calibrated_agent_evaluation._install import install_hint
from calibrated_agent_evaluation.normalize.datasets import parquet
from calibrated_agent_evaluation.normalize.errors import NormalizationError

if TYPE_CHECKING:
    from calibrated_agent_evaluation.normalize.sources import HFDataset


def iter_rows(source: HFDataset) -> Iterator[dict[str, Any]]:
    """Resolve the repo's parquet shards and yield one record dict per row."""
    shard_paths = _resolve_shards(source.repo_id, source.split)
    for shard in shard_paths:
        yield from parquet.iter_rows(shard)


def _resolve_shards(repo_id: str, split: str) -> list[Path]:
    """Download the dataset repo's parquet shards for ``split`` to local paths."""
    hub = _import_hf_hub()
    fs = hub.HfFileSystem()
    pattern = f"datasets/{repo_id}/**/*.parquet"
    try:
        remote = fs.glob(pattern)
    except Exception as exc:  # network / auth / not-found; wrap typed
        raise NormalizationError(
            f"could not list parquet shards for HuggingFace dataset {repo_id!r}: {exc}"
        ) from exc
    if not remote:
        raise NormalizationError(
            f"no parquet shards found for HuggingFace dataset {repo_id!r} (looked for {pattern!r})"
        )
    # Prefer the requested split's shards when the repo lays them out by split; else take all.
    split_shards = [r for r in remote if f"/{split}" in r or f"{split}-" in Path(r).name]
    chosen = split_shards or remote
    local: list[Path] = []
    for r in sorted(chosen):
        # HfFileSystem path: "datasets/<repo>/<...>/file.parquet" → filename relative to the repo.
        filename = r.split(f"datasets/{repo_id}/", 1)[-1]
        try:
            path = hub.hf_hub_download(repo_id=repo_id, filename=filename, repo_type="dataset")
        except Exception as exc:  # download / OS error; wrap typed
            raise NormalizationError(
                f"could not download shard {filename!r} of HuggingFace dataset {repo_id!r}: {exc}"
            ) from exc
        local.append(Path(path))
    return local


def _import_hf_hub() -> Any:
    """Import ``huggingface_hub`` or raise a typed error naming the missing extra."""
    try:
        import huggingface_hub
    except ImportError as exc:
        raise NormalizationError(
            "resolving a HuggingFace dataset requires the optional extra; install it with "
            f"`{install_hint('hf')}`"
        ) from exc
    return huggingface_hub
