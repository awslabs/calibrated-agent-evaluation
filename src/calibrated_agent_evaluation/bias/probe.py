# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

r"""Bias probes + the injected judge-run port.

The judge-run seam is an injected :class:`ProbeRunner` port typed only against normalized types --
no vendor SDK is imported here. The shipped concrete runner is
:class:`calibrated_agent_evaluation.judges.JudgeProbeRunner`; ``bias/`` never imports it.

A :class:`BiasProbe` is a set of controlled **trials**. Each trial is a tuple of
:class:`ProbeInput`\s whose fair verdicts should be *identical* -- only the biasing dimension
varies (length, model attribution, an authoritative assertion, candidate order). The judge exhibits
the bias on a trial when its verdicts across that trial's inputs disagree. For the pairwise
positional probe, the disagreement is neutralized by order-swap-and-tie and counted.

A judge that samples its verdicts disagrees with itself on some inputs even when nothing varies, so
a disagreement is not by itself evidence of bias. Each trial therefore has a placebo: its first
input judged again, once per further input, so the placebo makes as many calls as the trial and
varies nothing. A fair judge's verdicts do not depend on the biasing dimension, so its trial
disagrees exactly as often as its placebo; the battery gates the excess.

A disagreement counts the same whichever way it goes, so a judge that is noisy on the plain input
and always lenient on the perturbed one can disagree no more than its placebo. A perturbed input
may therefore declare the verdict its bias pushes toward (``toward``). The trial then also counts a
directional move (the first input's verdict is not ``toward`` and the perturbed input's is) against
the same move from its placebo repeat, and the battery gates that excess too.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Protocol

from calibrated_agent_evaluation._model import replace
from calibrated_agent_evaluation.bias.mitigate import TIE, order_swap_and_tie
from calibrated_agent_evaluation.bias.record import BiasName


@dataclass(frozen=True)
class ProbeInput:
    """One judge invocation's controlled content (vendor-free)."""

    bias: BiasName
    probe_id: str  # references the committed fixtures/bias/ item
    payload: Mapping[str, object]
    repeat: int = 0  # 0 for the trial's own input; 1 and up for its placebo repeats
    toward: str | None = None  # the verdict this input's bias pushes toward; None: no direction


@dataclass(frozen=True)
class ProbeVerdict:
    """The judge's decision on one probe input, as a canonical label.

    ``label`` is ``"PASS"``/``"FAIL"`` for a single-verdict judge, or the chosen candidate id for a
    pairwise judge. The probe compares labels across a trial's inputs; a fair judge returns the same
    label for inputs that differ only in the biasing dimension.
    """

    label: str
    detail: str = ""


class ProbeRunner(Protocol):
    """The port :func:`calibrated_agent_evaluation.bias.battery.run_battery` depends on.

    The shipped adapter, :class:`calibrated_agent_evaluation.judges.JudgeProbeRunner`, drives the
    judge runtime over an injected model backend and wraps a failure as
    :class:`~calibrated_agent_evaluation.bias.errors.ProbeExecutionError`.
    ``bias/`` never imports it -- dependency injection keeps the methodology plane vendor-free.
    ``run`` raises ``ProbeExecutionError`` on an execution failure.
    """

    def run(self, judge_model: str, prompt_version: str, probe_input: ProbeInput) -> ProbeVerdict:
        """Judge one probe input with the given model and prompt version.

        Args:
            judge_model: The model id the judge runs on.
            prompt_version: The judge prompt version being checked.
            probe_input: The probe trace and its perturbation.

        Returns:
            The judge's verdict on that input.
        """
        ...


@dataclass(frozen=True)
class BiasProbe:
    r"""A controlled probe for one CALM bias.

    ``trials`` is a tuple of trials; each trial is a tuple of :class:`ProbeInput`\s whose fair
    verdicts should agree. ``is_must_pass`` marks whether an exceedance can drive the overall
    verdict to ``BIASED``. ``pairwise`` marks the positional probe, which uses
    order-swap-and-tie instead of plain disagreement.
    """

    bias: BiasName
    is_must_pass: bool
    trials: tuple[tuple[ProbeInput, ...], ...]
    pairwise: bool = field(default=False)

    def score(self, runner: ProbeRunner, judge_model: str, prompt_version: str) -> ProbeTally:
        """Run every trial and its placebo through ``runner`` and tally the disagreements.

        A trial disagrees when the judge's verdicts across its inputs differ. For the pairwise
        probe, a disagreement between the two orderings is resolved as a tie (order-swap-and-tie)
        and counted in ``tie_count`` too. The placebo reuses the trial's own verdict on its first
        input and adds ``len(trial) - 1`` repeats of that input, the only extra calls. Any
        ``ProbeExecutionError`` from ``runner`` propagates (no partial scoring).
        """
        biased = placebo = both = ties = 0
        directed = d_trial = d_placebo = d_both = 0
        for trial in self.trials:
            labels = [
                runner.run(judge_model, prompt_version, probe_input).label for probe_input in trial
            ]
            repeats = [
                runner.run(judge_model, prompt_version, replace(trial[0], repeat=i)).label
                for i in range(1, len(trial))
            ]
            if self.pairwise and len(labels) == 2:
                trial_flip = order_swap_and_tie(labels[0], labels[1]) == TIE
                ties += trial_flip
            else:
                trial_flip = len(set(labels)) > 1
            placebo_flip = len({labels[0], *repeats}) > 1
            biased += trial_flip
            placebo += placebo_flip
            both += trial_flip and placebo_flip
            pointed = [(j, inp.toward) for j, inp in enumerate(trial) if j and inp.toward]
            if pointed:
                trial_move = any(labels[0] != t and labels[j] == t for j, t in pointed)
                placebo_move = any(labels[0] != t and repeats[j - 1] == t for j, t in pointed)
                directed += 1
                d_trial += trial_move
                d_placebo += placebo_move
                d_both += trial_move and placebo_move
        return ProbeTally(
            n=len(self.trials),
            biased_count=biased,
            placebo_count=placebo,
            both_count=both,
            tie_count=ties,
            direction_n=directed,
            direction_count=d_trial,
            direction_placebo_count=d_placebo,
            direction_both_count=d_both,
        )


@dataclass(frozen=True)
class ProbeTally:
    """A probe's disagreement counts over its ``n`` trials and their placebos.

    ``biased_count`` trials disagreed, ``placebo_count`` placebos disagreed, and ``both_count``
    is the number of trials where both did: the paired table the battery's bound is read from.
    The ``direction_*`` counts are the same table for directional moves, over the
    ``direction_n`` trials that declare a direction.
    """

    n: int
    biased_count: int
    placebo_count: int
    both_count: int
    tie_count: int = 0
    direction_n: int = 0
    direction_count: int = 0
    direction_placebo_count: int = 0
    direction_both_count: int = 0
