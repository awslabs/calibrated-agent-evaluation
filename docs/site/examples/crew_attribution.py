# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Who broke the claim? Attribute crew failures to the agent that caused them.

Reads three ClaimsCrew traces and their gold labels from the repo's fixtures. For each failed
claim, the walk-back starts at the span where the harm appeared (the manifestation) and checks the
culprit the reviewer recorded against the real span tree. CLM-4477 has two defensible culprits, so
it is recorded as disputed. Run from the repository root.

    uv run --extra stats python docs/site/examples/crew_attribution.py
"""

from __future__ import annotations

import json
from pathlib import Path

from calibrated_agent_evaluation.attribution import (
    CrewDefinition,
    ExpectedProcess,
    MandatedRule,
    build_crew_definition,
    validate_walkback,
)
from calibrated_agent_evaluation.schema.serde import label_from_dict, trace_from_dict

FIXTURES = Path("fixtures")


def claimscrew() -> CrewDefinition:
    """The crew's agents, handoffs and the expected process for a vacancy-exclusion claim."""
    vacancy = ExpectedProcess(
        scenario="S2_vacancy_exclusion",
        expected_agents=("supervisor", "intake", "policy-checker", "fraud-scorer", "payout"),
        terminal_decision="deny",
        payout_allowed=False,
        escalation_required=False,
    )
    return build_crew_definition(
        agents=("supervisor", "intake", "policy-checker", "fraud-scorer", "payout", "escalation"),
        handoffs=(
            "intake->policy-checker",
            "policy-checker->fraud-scorer",
            "fraud-scorer->payout",
            "fraud-scorer->escalation",
        ),
        scenarios={"S2_vacancy_exclusion": vacancy},
        rules=(
            MandatedRule("R2", "fraud flag raised", "escalation", deterministic=False),
            MandatedRule("R3", "occupancy narrative present", "intake"),
        ),
    )


crew = claimscrew()
gold = {
    (label.trace_id, label.span_id): label
    for label in map(label_from_dict, json.loads((FIXTURES / "labels" / "gold.json").read_text()))
}

for trace_id in ("CLM-4471", "CLM-4473", "CLM-4477"):
    trace = trace_from_dict(json.loads((FIXTURES / "normalized" / f"{trace_id}.json").read_text()))
    label = gold[(trace_id, None)]
    if label.culprit_span == "disputed":
        # The reviewer could not pick between the two agents, so both go forward as culprits.
        culprits: tuple[str, ...] = (f"{trace_id}/intake", f"{trace_id}/policy-checker")
    else:
        culprits = (label.culprit_span,) if label.culprit_span else ()
    attribution = validate_walkback(
        trace,
        crew,
        culprit_spans=culprits,
        manifestation_span=label.manifestation_span,
        handoff_implicated=label.handoff_implicated,
    )
    print(f"{trace_id} ({label.failure_mode}): {attribution.status}")
    print(f"  harm appeared at  {attribution.manifestation_span}")
    print(f"  caused by         {', '.join(attribution.culprit_spans)}")
    if label.handoff_implicated:
        print(f"  handoff           {label.handoff_implicated}")
