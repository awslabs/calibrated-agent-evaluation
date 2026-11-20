# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""``calibrated-agent-evaluation quickstart``: one command, zero setup, a real result.

The fastest path from a fresh install to seeing the tool work. It seeds the bundled
sample corpus (six real agent traces + their gold labels; see
:mod:`calibrated_agent_evaluation._quickstart`) into a local, offline trace store, then reports what
a first error-analysis pass reveals: how many traces, how many failed, and the named failure modes
found. No credentials, no network, no external files; the offline local backend needs no extras, so
this works on a bare install.

It reuses the shipped surfaces (the bundled loader + the backend ``put``/``query`` API); it computes
no statistic itself. The closing lines point at the next steps (the review UI, the pipeline stages),
so the command doubles as a guided on-ramp.

Six gold labels cannot certify a judge, so ``--certify`` seeds a second, larger bundled set
instead: real OrderBot runs with synthetic gold in two hunted rounds, each sized to the rate-safe
calibration plan, and a judge kit (two versions of one judge, their model's recorded replies, a
bias probe set per round) with which ``calibrated-agent-evaluation calibrate``,
``calibrated-agent-evaluation bias``, ``calibrated-agent-evaluation judge`` and
``calibrated-agent-evaluation run`` run offline. The first version certifies and is refused for
bias; the second is the one prompt revision that result prompted, scored on the second round only.
Each gate's decision is the recorded judge's own: nothing is arranged for it to pass.
"""

from __future__ import annotations

import argparse
import shlex
import tempfile
from collections import Counter
from pathlib import Path

from calibrated_agent_evaluation._install import install_hint
from calibrated_agent_evaluation.cli.dispatch import EXIT_OK, StageOutcome
from calibrated_agent_evaluation.cli.errors import CliError


def quickstart_handler(args: argparse.Namespace) -> StageOutcome:
    """Seed the bundled corpus into a local store and report the first-pass failure modes.

    ``--root`` (or ``CALIBRATED_AGENT_EVALUATION_LOCAL_ROOT``) chooses the local store directory;
    when neither is set a temp dir is used so there is nothing to clean up. A chosen directory that
    already holds data is refused unless ``--force``, so sample traces and gold never mix into a
    real store unannounced. Otherwise exits 0: quickstart is a demonstration, not a gate.
    """
    from calibrated_agent_evaluation._quickstart import load_sample_corpus
    from calibrated_agent_evaluation.backends.local import LocalBackend
    from calibrated_agent_evaluation.cli.backends import explicit_local_root

    # The same root every other stage reads, so `calibrated-agent-evaluation --root D quickstart`
    # seeds a store that `calibrated-agent-evaluation --root D cluster` then sees. Without one, a
    # temp dir: nothing to clean up.
    chosen = explicit_local_root(args)
    force = bool(getattr(args, "force", False))
    if chosen is not None and not force:
        _refuse_existing(Path(chosen), certify=bool(getattr(args, "certify", False)))
    merged = chosen is not None and force and _holds_data(Path(chosen))
    root = (
        str(chosen)
        if chosen is not None
        else tempfile.mkdtemp(prefix="calibrated-agent-evaluation-quickstart-")
    )
    if getattr(args, "certify", False):
        return _certify(Path(root), merged=merged)
    traces, labels = load_sample_corpus()
    with LocalBackend(root=root) as backend:
        for trace in traces:
            backend.put_trace(trace)
        for label in labels:
            backend.put_score(label)

    # Trace-level outcome: a trace fails if any of its gold labels is a FAIL. Each failure mode
    # counts the distinct traces with a FAIL label for it, as `cluster` does, so two raters labeling
    # one trace count once. This is plain aggregation over the gold labels; the real calibrated
    # judging lives in `calibrated-agent-evaluation judge` (needs [stats]); quickstart just shows
    # the shape.
    failing = {lb.trace_id for lb in labels if lb.outcome == "FAIL"}
    failing_pairs = dict.fromkeys(
        (lb.failure_mode, lb.trace_id) for lb in labels if lb.outcome == "FAIL" and lb.failure_mode
    )
    modes = Counter(mode for mode, _trace_id in failing_pairs)

    n_traces = len(traces)
    n_fail = len(failing)
    mode_lines = "\n".join(f"    - {mode}: {count}" for mode, count in modes.most_common())

    # The sample lives in a temp root unless --root was given, so each next step must name it.
    here = f"calibrated-agent-evaluation --root {shlex.quote(root)}"
    summary = (
        "calibrated-agent-evaluation quickstart: a first error-analysis pass on the bundled "
        "sample corpus\n"
        "\n"
        f"Seeded {n_traces} agent traces + {len(labels)} gold labels into an offline local store\n"
        f"  ({root})\n"
        f"{_MERGED_NOTE if merged else ''}"
        "\n"
        f"First-pass result: {n_fail} of {n_traces} traces have a labeled failure.\n"
        "Failure modes found:\n"
        f"{mode_lines}\n"
        "\n"
        "Those counts come from the gold labels that ship with the corpus: nothing has been\n"
        "judged yet. Next steps:\n"
        "  - calibrated-agent-evaluation doctor\n"
        "    see which optional extras are installed\n"
        f"  - {install_hint('ui')}, then\n"
        f"    {here} label --serve --rater YOUR_ID\n"
        "    review traces in the UI\n"
        f"  - {install_hint('stats')}\n"
        "    judge calibration and corrected failure rates\n"
        f"  - {here} cluster\n"
        "    check failure-mode saturation and coverage\n"
        f"  - {here} run --verdicts human --ci\n"
        "    the CI gate over the gold labels\n"
        "  - calibrated-agent-evaluation calibrate, calibrated-agent-evaluation bias, "
        "calibrated-agent-evaluation judge\n"
        "    certify a judge, then score with it\n"
        "  - calibrated-agent-evaluation quickstart --certify\n"
        "    a set big enough to certify one, offline"
    )
    return StageOutcome(stage="quickstart", exit_code=EXIT_OK, summary=summary)


_MERGED_NOTE = (
    "  The store already held data (--force): the sample is mixed into it, so every later stage\n"
    "  counts the sample's traces and gold labels alongside yours.\n"
)


def _holds_data(root: Path) -> bool:
    return root.is_dir() and any(root.iterdir())


def _refuse_existing(root: Path, *, certify: bool) -> None:
    """Refuse to seed sample data into a directory that already holds something."""
    if root.exists() and not root.is_dir():
        raise CliError(f"{root} is a file, not a store directory; pass --root a directory")
    if not _holds_data(root):
        return
    what = (
        "real agent traces with SYNTHETIC gold labels and a judge kit"
        if certify
        else "sample traces and their gold labels"
    )
    raise CliError(
        f"{root} already holds data. quickstart seeds {what}, which would mix with what is "
        "there: every later stage would count them as part of your store.\n"
        "Next: pass --root an empty or new directory, or leave --root and "
        "CALIBRATED_AGENT_EVALUATION_LOCAL_ROOT "
        "unset for a fresh temporary store. Pass --force to add the sample to this store anyway."
    )


def _certify(root: Path, *, merged: bool = False) -> StageOutcome:
    """``quickstart --certify``: seed the certify set and write the judge kit next to it.

    The set is real OrderBot runs with synthetic gold (see
    :mod:`calibrated_agent_evaluation._quickstart`), sized so a judge can certify and its corrected
    rate is not refused for too little gold. The output names the commands that take it the rest of
    the way, each replaying the judge's recorded replies so nothing needs credentials.
    """
    from calibrated_agent_evaluation._quickstart import (
        ROUND_KEY,
        certify_manifest,
        documented_commands,
        load_certify_set,
        write_certify_kit,
    )
    from calibrated_agent_evaluation.backends.local import LocalBackend

    traces, labels = load_certify_set()
    manifest = certify_manifest()
    with LocalBackend(root=root) as backend:
        for trace in traces:
            backend.put_trace(trace)
        for label in labels:
            backend.put_score(label)
    kit = write_certify_kit(root)

    by_id = {t.trace_id: t for t in traces}
    mode = next(lb.failure_mode for lb in labels if lb.outcome == "FAIL")
    n_prod = sum(1 for t in traces if t.frame == "production-representative")
    gold_lines = []
    for rnd in sorted(manifest["rounds"]):
        mine = [lb for lb in labels if by_id[lb.trace_id].taxonomy_cell.get(ROUND_KEY) == rnd]
        n_fail = sum(1 for lb in mine if lb.outcome == "FAIL")
        gold_lines.append(
            f"  round {rnd} ({ROUND_KEY}={rnd}): {len(mine)} hunted traces (discovery frame) "
            f"carry gold, {n_fail} FAIL for {mode} and {len(mine) - n_fail} PASS"
        )
    command = f"calibrated-agent-evaluation --root {shlex.quote(str(root))}"
    calibrate_v1, bias_v1, calibrate_v2, bias_v2, judge_v2, run_v2 = (
        f"  {command} {shlex.join(argv)}" for argv in documented_commands(root)
    )
    summary = "\n".join(
        [
            "calibrated-agent-evaluation quickstart --certify: the judge-trust chain on a set big "
            "enough to certify",
            "",
            f"Seeded {len(traces)} OrderBot refund traces into an offline local store",
            f"  ({root})",
            *([_MERGED_NOTE.rstrip("\n")] if merged else []),
            *gold_lines,
            f"  {n_prod} production-representative traces carry none; the rate is measured on them",
            "The traces are real agent runs. The gold is SYNTHETIC: a rule applied the store's "
            "written return policy to each order, in place of a human reviewer:",
            f"  {manifest['policy']}",
            "",
            f"Wrote the judge kit to {kit}: two versions of one judge, their model's recorded "
            "replies, and a bias probe set per round.",
            "Run these in order; each gate decides whether the next step may run. They replay "
            "the recorded replies, so none needs credentials.",
            "1. Certify judge v1 on round 1's gold, then run its bias battery:",
            calibrate_v1,
            bias_v1,
            "2. v2 is the one prompt revision v1's bias result prompted. A locked test is "
            "single-use, so v2 is certified and probed on round 2 only:",
            calibrate_v2,
            bias_v2,
            "3. If v2 passes both gates, score the production sample and run the CI gate:",
            judge_v2,
            run_v2,
        ]
    )
    return StageOutcome(stage="quickstart", exit_code=EXIT_OK, summary=summary)
