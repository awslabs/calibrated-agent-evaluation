# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Order-swap-and-tie mitigation + positional probe scoring."""

from __future__ import annotations

from calibrated_agent_evaluation.bias import BiasProbe, ProbeInput, ProbeVerdict, order_swap_and_tie
from calibrated_agent_evaluation.bias.mitigate import TIE


def test_consistent_orderings_keep_the_winner_disagreement_becomes_tie() -> None:
    assert order_swap_and_tie("A", "A") == "A"  # order-independent pick stands
    assert order_swap_and_tie("A", "B") == TIE  # flipped with order -> neutralized as a tie


class _Runner:
    def __init__(self, labels: dict[str, str]) -> None:
        self._labels = labels

    def run(self, judge_model: str, prompt_version: str, probe_input: ProbeInput) -> ProbeVerdict:
        return ProbeVerdict(label=self._labels[probe_input.probe_id])


def test_positional_probe_counts_order_flips_as_ties() -> None:
    # trial 0 flips with order (A then B) -> tie; trial 1 is order-consistent -> no tie.
    probe = BiasProbe(
        bias="positional",
        is_must_pass=False,
        pairwise=True,
        trials=(
            (ProbeInput("positional", "p0-ab", {}), ProbeInput("positional", "p0-ba", {})),
            (ProbeInput("positional", "p1-ab", {}), ProbeInput("positional", "p1-ba", {})),
        ),
    )
    runner = _Runner({"p0-ab": "A", "p0-ba": "B", "p1-ab": "A", "p1-ba": "A"})
    tally = probe.score(runner, "test-model", "v1")
    assert tally.n == 2
    assert tally.biased_count == 1
    assert tally.tie_count == 1
    assert tally.placebo_count == 0  # a deterministic runner never disagrees with itself
