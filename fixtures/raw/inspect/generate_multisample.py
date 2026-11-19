# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Assemble the multi-sample `.eval` fixture from the three real single-sample logs.

The fix for multi-sample logs losing samples needs a >=3-sample log to prove `normalize_all`
returns one Trace per sample. No such log existed: every committed `.eval` is a one-row Task.

Minting a genuinely new 3-row eval is not possible offline. The cassette provider keys its replay on
the request, and a combined Task carries the union of all three scenarios' tools, so every sample's
`toolConfig` differs from what was recorded and every request misses the cassette. It would need a
live Bedrock run.

So this **relocates real samples into one container**, changing no sample content. It is the same
method (and the same justification) as `fixtures/raw/otel/generate_convention_fixtures.py`: the
envelope is transport, the samples are the genuine recorded transcripts — real `ModelEvent` /
`ToolEvent` sequences from real Bedrock runs, carried over verbatim with their own metadata,
`trace_id` and `taxonomy_cell` intact.

Run: `uv run python fixtures/raw/inspect/generate_multisample.py`
"""

from __future__ import annotations

from pathlib import Path

from inspect_ai.log import read_eval_log, write_eval_log

_HERE = Path(__file__).resolve().parent
_SOURCES = ("ORD-1001.eval", "ORD-1002.eval", "CLM-4471.eval")
_OUT = _HERE / "multisample-3.eval"


def main() -> int:
    logs = [read_eval_log(str(_HERE / name), resolve_attachments=True) for name in _SOURCES]

    combined = logs[0].model_copy(deep=True)
    samples = []
    for name, log in zip(_SOURCES, logs, strict=True):
        if log.samples is None:
            raise SystemExit(f"{name}: no samples to carry over")
        for sample in log.samples:
            # Each source Task had a ONE-row dataset, so every sample is `id=1` and all three would
            # collide (the zip writer warns, and an id-keyed read loses two of the three). A genuine
            # 3-row dataset has distinct ids, so each sample takes its own scenario id -- the value
            # its own `metadata["trace_id"]` already carries. Nothing else is touched.
            sample.id = name.removesuffix(".eval")
            samples.append(sample)
    combined.samples = samples
    combined.location = str(_OUT)
    # `eval.dataset` is the INDEX `read_eval_log_samples` fetches by, not decoration: it still said
    # `samples=1, sample_ids=[1]` from the first source log, so a reader asked the log for sample
    # id 1 and got IndexError. Verified against inspect_ai 0.3.x by reading the written file back.
    combined.eval.dataset.samples = len(samples)
    combined.eval.dataset.sample_ids = [str(s.id) for s in samples]
    # `results`/`stats` describe the original single-sample run and would be a claim about this
    # container that is not true, so they are dropped rather than carried over misleadingly.
    combined.results = None
    combined.reductions = None

    write_eval_log(combined, str(_OUT))
    ids = [str(s.id) for s in samples]
    print(f"wrote {_OUT.name} with {len(samples)} samples: {', '.join(ids)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
