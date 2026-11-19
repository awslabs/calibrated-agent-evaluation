# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Missing optional extras raise a typed error naming the extra.

`pyarrow`/`huggingface_hub` are dev-installed, so simulate their absence by making the import fail
inside the reader; the loader must surface a typed NormalizationError naming the installable extra,
never let a bare ImportError escape.
"""

from __future__ import annotations

import builtins
from pathlib import Path

import pytest

from calibrated_agent_evaluation._install import install_hint
from calibrated_agent_evaluation.normalize.datasets import load_traces
from calibrated_agent_evaluation.normalize.errors import NormalizationError
from calibrated_agent_evaluation.normalize.sources import HFDataset, ParquetDataset

_REPO = Path(__file__).resolve().parents[2]
_SLICE = _REPO / "fixtures" / "raw" / "hf" / "swebench-slice.parquet"


def _block_import(monkeypatch: pytest.MonkeyPatch, blocked_prefix: str) -> None:
    """Make `import <blocked_prefix...>` raise ImportError, as if the extra were not installed."""
    real_import = builtins.__import__

    def _fake_import(name: str, *args: object, **kwargs: object) -> object:
        if name == blocked_prefix or name.startswith(blocked_prefix + "."):
            raise ImportError(f"No module named {name!r} (simulated missing extra)")
        return real_import(name, *args, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(builtins, "__import__", _fake_import)


def test_missing_datasets_extra_raises_named_error(monkeypatch: pytest.MonkeyPatch) -> None:
    _block_import(monkeypatch, "pyarrow")
    with pytest.raises(NormalizationError) as exc:
        list(load_traces(ParquetDataset(_SLICE)))
    assert install_hint("datasets") in str(exc.value)


def test_missing_hf_extra_raises_named_error(monkeypatch: pytest.MonkeyPatch) -> None:
    _block_import(monkeypatch, "huggingface_hub")
    with pytest.raises(NormalizationError) as exc:
        list(load_traces(HFDataset("Exgentic/agent-llm-traces"), limit=1))
    assert install_hint("hf") in str(exc.value)
