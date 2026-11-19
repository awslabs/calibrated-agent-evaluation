# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the Holm guard panel and crew slices.

Behavioural: guard_panel corrects the per-slice regression p-values with Holm at FWER 0.05, states
m, and names survivors; a forbidden fdr_bh method is refused; an empty panel is refused; crew_panel
derives per-agent + per-handoff slices and folds in the relocation check; an undeclared
handoff/agent or a non-workflow crew metric is refused.
"""

from __future__ import annotations

import pytest

from _helpers import TEST_INTERVAL_KIND
from calibrated_agent_evaluation.attribution import CrewDefinition, CrewOutcomes
from calibrated_agent_evaluation.attribution.types import ExpectedProcess
from calibrated_agent_evaluation.calibrate.errors import StatisticsInputError
from calibrated_agent_evaluation.calibrate.record import Interval
from calibrated_agent_evaluation.improve import (
    GuardPanelReport,
    GuardSlice,
    ImproveError,
    build_crew_slices,
    crew_panel,
    guard_panel,
)


def _iv(point: float) -> Interval:
    return Interval(
        point=point, lo=point - 0.05, hi=point + 0.05, refusal_reason=None, **TEST_INTERVAL_KIND
    )


def _slice(name: str, p: float) -> GuardSlice:
    return GuardSlice(
        name=name, kind="other", regression_p=p, rate_before=_iv(0.1), rate_after=_iv(0.1)
    )


def _crew() -> CrewDefinition:
    return CrewDefinition(
        agents=("intake", "policy-checker"),
        handoffs=("intake->policy-checker",),
        scenarios={
            "s1": ExpectedProcess(
                scenario="s1",
                expected_agents=("intake", "policy-checker"),
                terminal_decision="approve",
                payout_allowed=True,
                escalation_required=False,
            )
        },
        rules=(),
    )


def test_panel_states_method_and_m() -> None:
    """The panel states method='holm' and m = the slice count."""
    panel = guard_panel([_slice("a", 0.6), _slice("b", 0.9), _slice("c", 0.4)])
    assert panel.method == "holm"
    assert panel.m == 3


def test_holm_step_down_at_fwer_005() -> None:
    """Holm's step-down, hand-computed at alpha = 0.05 over four slices.

    Sorted p = 0.004, 0.03, 0.04, 0.5. Step 1: 0.004 * 4 = 0.016 <= 0.05, reject. Step 2:
    0.03 * 3 = 0.09 > 0.05, stop. Only the first slice regressed.
    """
    panel = guard_panel(
        [_slice("d", 0.5), _slice("a", 0.004), _slice("c", 0.04), _slice("b", 0.03)]
    )
    assert panel.regressed == ("a",)


def test_moderate_pvalues_no_longer_reject_the_fix() -> None:
    """P = 0.02, 0.03, 0.05 rejected all three under BY-FDR at q = 0.10; Holm rejects none.

    Holm adjusted p-values are 0.06, 0.06, 0.06 (0.02 * 3, then max(0.06, 0.03 * 2), then
    max(0.06, 0.05)), all above 0.05. BY at q = 0.10 compares the k-th smallest against
    k * 0.10 / (3 * 1.8333): 0.05 <= 0.0545 at k = 3, so all three were rejected and the fix with
    them.
    """
    panel = guard_panel([_slice("a", 0.02), _slice("b", 0.03), _slice("c", 0.05)])
    assert panel.regressed == ()


def test_partial_null_rejects_only_the_true_regression() -> None:
    """One slice truly regressed (p = 0.001) and three did not (0.02, 0.03, 0.04).

    Holm: 0.001 * 4 = 0.004, reject; 0.02 * 3 = 0.06, stop. BY at q = 0.10 rejected all four, three
    of them falsely, which for a gate that rejects on any survivor reads the same as one.
    """
    panel = guard_panel(
        [_slice("true", 0.001), _slice("n1", 0.02), _slice("n2", 0.03), _slice("n3", 0.04)]
    )
    assert panel.regressed == ("true",)


def test_partial_null_names_a_null_slice_only_below_alpha_over_four() -> None:
    """With one true regression, a null slice is named only when its p < 0.05 / 4.

    One slice truly regressed (p = 0) and four did not. Holm rejects the true slice at step one,
    then a null slice at step two only if the smallest null p is at most 0.05 / 4 = 0.0125. With
    independent uniform null p-values the family-wise error is therefore
    1 - (1 - 0.0125) ** 4 = 0.049070, under 0.05. (A Monte Carlo check is impractical here:
    statsmodels' multipletests runs gc.collect() on every call.)
    """
    nulls = (0.9, 0.7, 0.5)
    named = guard_panel(
        [_slice("true", 0.0), _slice("n0", 0.0124)]
        + [_slice(f"n{i + 1}", p) for i, p in enumerate(nulls)]
    )
    assert named.regressed == ("true", "n0")
    kept = guard_panel(
        [_slice("true", 0.0), _slice("n0", 0.0126)]
        + [_slice(f"n{i + 1}", p) for i, p in enumerate(nulls)]
    )
    assert kept.regressed == ("true",)
    fwer = 1 - (1 - 0.0125) ** 4
    assert fwer == pytest.approx(0.049070, abs=1e-6)


def test_stable_slices_yield_no_regression() -> None:
    """When no per-slice regression survives the correction, `regressed` is empty."""
    panel = guard_panel([_slice("a", 0.6), _slice("b", 0.9)])
    assert panel.regressed == ()


def test_surviving_regression_is_named() -> None:
    """A slice whose small regression p-value survives Holm is named in `regressed`."""
    # 0.0005 * 3 = 0.0015 <= 0.05 survives; 0.8 * 2 does not.
    panel = guard_panel([_slice("bad", 0.0005), _slice("b", 0.8), _slice("c", 0.9)])
    assert "bad" in panel.regressed


def test_empty_panel_refused() -> None:
    """A declared-but-empty guard panel is refused, not a vacuous 'no regression'."""
    with pytest.raises(ImproveError):
        guard_panel([])


def test_out_of_range_pvalue_refused_by_primitive() -> None:
    """An out-of-range regression p-value is refused by the multiplicity primitive."""
    with pytest.raises(StatisticsInputError):
        guard_panel([_slice("a", 1.5)])


def test_crew_slices_cover_other_agents_and_declared_handoffs() -> None:
    """build_crew_slices yields a slice per OTHER agent + per declared handoff."""
    crew = _crew()
    slices = build_crew_slices(
        crew,
        touched_agent="intake",
        agent_regression_p={"policy-checker": 0.7},
        agent_rates={"policy-checker": (_iv(0.1), _iv(0.1))},
        handoff_regression_p={"intake->policy-checker": 0.8},
        handoff_rates={"intake->policy-checker": (_iv(0.05), _iv(0.05))},
    )
    kinds = {(s.name, s.kind) for s in slices}
    assert ("policy-checker", "agent") in kinds  # the other agent (not the touched one)
    assert ("intake->policy-checker", "handoff") in kinds
    assert not any(s.name == "intake" for s in slices)  # the touched agent is excluded


def test_undeclared_handoff_refused() -> None:
    """A handoff naming a seam not in the crew definition is refused."""
    crew = _crew()
    with pytest.raises(ImproveError):
        build_crew_slices(
            crew,
            touched_agent="intake",
            agent_regression_p={},
            agent_rates={},
            handoff_regression_p={"intake->ghost": 0.5},
            handoff_rates={"intake->ghost": (_iv(0.1), _iv(0.1))},
        )


def test_missing_guarded_agent_key_raises_typed_error() -> None:
    """A non-touched agent absent from the supplied dicts raises ImproveError, not KeyError.

    Completeness guard (no silent failure): every non-touched crew agent + declared handoff must be
    supplied in both its regression-p and rates map, else the build would raise a raw KeyError off
    the public surface.
    """
    crew = _crew()
    with pytest.raises(ImproveError):
        # policy-checker (the non-touched agent) is omitted from both agent maps entirely.
        build_crew_slices(
            crew,
            touched_agent="intake",
            agent_regression_p={},
            agent_rates={},
            handoff_regression_p={"intake->policy-checker": 0.8},
            handoff_rates={"intake->policy-checker": (_iv(0.05), _iv(0.05))},
        )


def test_missing_handoff_rate_key_raises_typed_error() -> None:
    """A declared handoff missing from handoff_rates raises ImproveError, not KeyError."""
    crew = _crew()
    with pytest.raises(ImproveError):
        build_crew_slices(
            crew,
            touched_agent="intake",
            agent_regression_p={"policy-checker": 0.7},
            agent_rates={"policy-checker": (_iv(0.1), _iv(0.1))},
            handoff_regression_p={"intake->policy-checker": 0.8},
            handoff_rates={},  # declared handoff has no rates -> typed refusal, not KeyError
        )


def test_crew_panel_folds_in_relocation() -> None:
    """crew_panel attaches the relocation verdict (flat workflow relocates)."""
    crew = _crew()
    before = CrewOutcomes(
        workflow_failures=60, workflow_total=200, per_agent_failures={"intake": 60}
    )
    after = CrewOutcomes(
        workflow_failures=60, workflow_total=200, per_agent_failures={"intake": 20}
    )
    slices = [_slice("policy-checker", 0.8)]
    panel = crew_panel(crew, before, after, touched_agent="intake", slices=slices)
    assert panel.relocation is not None
    assert panel.relocation.verdict == "relocation"
    assert panel.relocation.relocated_from == "intake"


def test_crew_panel_refuses_unknown_touched_agent() -> None:
    """A touched_agent absent from the crew definition is refused."""
    crew = _crew()
    before = CrewOutcomes(workflow_failures=1, workflow_total=10, per_agent_failures={})
    after = CrewOutcomes(workflow_failures=1, workflow_total=10, per_agent_failures={})
    with pytest.raises(ImproveError):
        crew_panel(crew, before, after, touched_agent="ghost", slices=[_slice("x", 0.5)])


def test_panel_report_type_is_holm_only() -> None:
    """The GuardPanelReport method literal is holm (FDR methods are structurally excluded)."""
    panel: GuardPanelReport = guard_panel([_slice("a", 0.5)])
    assert panel.method == "holm"
