# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Subjective-outcome handling engine.

Public API re-exports; import from ``calibrated_agent_evaluation.subjective``.
"""

from calibrated_agent_evaluation.subjective.atomfile import atom_set_from_dict, load_atom_set
from calibrated_agent_evaluation.subjective.bradley_terry import bradley_terry
from calibrated_agent_evaluation.subjective.classify import (
    define_atoms,
    is_objective,
    is_subjective,
)
from calibrated_agent_evaluation.subjective.elo import compute_ratings
from calibrated_agent_evaluation.subjective.errors import (
    SubjectiveError,
    SubjectiveGateError,
    SubjectiveStoreError,
)
from calibrated_agent_evaluation.subjective.gate import validate_gateable
from calibrated_agent_evaluation.subjective.pairfile import read_preference_file
from calibrated_agent_evaluation.subjective.preference import record_preference
from calibrated_agent_evaluation.subjective.store import (
    SUBJECTIVE_SCHEMA_VERSION,
    PreferenceRecord,
    dump_preference_records,
    load_preference_records,
    preference_record,
)
from calibrated_agent_evaluation.subjective.types import (
    Atom,
    AtomSet,
    BradleyTerryRating,
    BradleyTerryRatings,
    EloRating,
    EloRatings,
    PreferencePair,
    SubjectiveDimension,
)

__all__ = [
    "SUBJECTIVE_SCHEMA_VERSION",
    "Atom",
    "AtomSet",
    "BradleyTerryRating",
    "BradleyTerryRatings",
    "EloRating",
    "EloRatings",
    "PreferencePair",
    "PreferenceRecord",
    "SubjectiveDimension",
    "SubjectiveError",
    "SubjectiveGateError",
    "SubjectiveStoreError",
    "atom_set_from_dict",
    "bradley_terry",
    "compute_ratings",
    "define_atoms",
    "dump_preference_records",
    "is_objective",
    "is_subjective",
    "load_atom_set",
    "load_preference_records",
    "preference_record",
    "read_preference_file",
    "record_preference",
    "validate_gateable",
]
