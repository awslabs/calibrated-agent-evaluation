# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Friction-free trace-review surface.

The library's flagship "look at the data" surface: a thin FastHTML app whose primary screen
is a single-trace inline-label view (not a metrics dashboard). It reads
normalized ``Trace``s and writes human ``Label``s *only* through the backend
interfaces (``TraceStoreBackend`` / ``LabelBackend``); no vendor SDK type appears in this
package. The surface exists to make human review fast enough that
error analysis stops being the step teams skip: hotkeys + optimistic auto-advance let a
competent labeler read + label 50 traces in under 30 minutes.

Public surface:
  * ``build_app``: the FastHTML application factory (``app.py``).
  * ``ReviewSession``: in-memory work-list / defer / undo / timing (``session.py``).
  * ``WorkScope`` / ``build_work_list``; which traces a session covers: resume per rater, plus
    trace / frame / cell / limit narrowing (``session.py``).
  * ``SessionLog``: the append-only JSONL record of verdicts and undos with per-trace timing
    (``session_log.py``).
  * ``build_label`` + ``PASS_DEFAULT_CRITIQUE``; critique-first label construction (``labeling``).
  * ``Prefill`` / ``Suggestion`` / ``NoPrefill``; the opt-in LLM pre-fill seam (``prefill.py``).
  * ``RendererRegistry``: the plugin-renderer seam keyed on trace metadata; construct with a
    default renderer, ``register(predicate, renderer)`` domain renderers, ``render(trace)`` guards
    a raising renderer back to the default (``renderers``).
  * ``inter_rater_agreement`` / ``alpha_gate`` / ``newest_per_rater_unit``; the gold-set
    agreement gate over double-coded human labels (``agreement``), and ``AgreementRecord`` /
    ``agreement_record_of``, what ``calibrated-agent-evaluation label --agreement`` stores
    (``record``).
  * ``CodebookFile`` / ``CodebookRecord`` / ``load_codebook_file``; the failure-mode list the
    review form offers, hand-written or ratified by ``calibrated-agent-evaluation cluster --ratify``
    (``record``).
  * ``plan_overlap`` / ``MINIMUM_OVERLAP``; how many double-coded traces the agreement gate
    needs before it can say ``usable`` (``overlap``).
  * ``ReviewError`` / ``RatificationError`` / ``LabelSubmitError`` / ``AgreementDataError``; the
    typed error family.
"""

from __future__ import annotations

from importlib import import_module
from typing import TYPE_CHECKING, Any

from calibrated_agent_evaluation.review.errors import (
    AgreementDataError,
    LabelSubmitError,
    RatificationError,
    ReviewError,
)
from calibrated_agent_evaluation.review.gate import (
    ALPHA_LOWER_BOUND_GATE,
    ALPHA_REVISE_FLOOR,
    MINIMUM_OVERLAP,
)
from calibrated_agent_evaluation.review.labeling import PASS_DEFAULT_CRITIQUE, build_label
from calibrated_agent_evaluation.review.prefill import NoPrefill, Prefill, Suggestion
from calibrated_agent_evaluation.review.record import (
    AgreementRecord,
    AgreementStat,
    CodebookFile,
    CodebookMode,
    CodebookRecord,
    agreement_record_of,
    load_codebook_file,
)
from calibrated_agent_evaluation.review.relabel import newest_per_rater_unit
from calibrated_agent_evaluation.review.renderers.base import RendererRegistry
from calibrated_agent_evaluation.review.session import (
    LabeledEntry,
    ReviewSession,
    WorkScope,
    build_work_list,
)
from calibrated_agent_evaluation.review.session_log import SessionLog

if TYPE_CHECKING:
    from calibrated_agent_evaluation.review.agreement import (
        InterRaterReport,
        alpha_gate,
        inter_rater_agreement,
    )
    from calibrated_agent_evaluation.review.overlap import OverlapPlan, plan_overlap

# The statistics behind these import numpy (through calibrated_agent_evaluation.stats); loading them
# on first use keeps `import calibrated_agent_evaluation.review` and reading the stored records
# working on an install without it.
_LAZY = {
    "InterRaterReport": "calibrated_agent_evaluation.review.agreement",
    "alpha_gate": "calibrated_agent_evaluation.review.agreement",
    "inter_rater_agreement": "calibrated_agent_evaluation.review.agreement",
    "OverlapPlan": "calibrated_agent_evaluation.review.overlap",
    "plan_overlap": "calibrated_agent_evaluation.review.overlap",
}


def __getattr__(name: str) -> Any:
    module = _LAZY.get(name)
    if module is None:
        raise AttributeError(
            f"module 'calibrated_agent_evaluation.review' has no attribute {name!r}"
        )
    return getattr(import_module(module), name)


__all__ = [
    "ALPHA_LOWER_BOUND_GATE",
    "ALPHA_REVISE_FLOOR",
    "MINIMUM_OVERLAP",
    "PASS_DEFAULT_CRITIQUE",
    "AgreementDataError",
    "AgreementRecord",
    "AgreementStat",
    "CodebookFile",
    "CodebookMode",
    "CodebookRecord",
    "InterRaterReport",
    "LabelSubmitError",
    "LabeledEntry",
    "NoPrefill",
    "OverlapPlan",
    "Prefill",
    "RatificationError",
    "RendererRegistry",
    "ReviewError",
    "ReviewSession",
    "SessionLog",
    "Suggestion",
    "WorkScope",
    "agreement_record_of",
    "alpha_gate",
    "build_label",
    "build_work_list",
    "inter_rater_agreement",
    "load_codebook_file",
    "newest_per_rater_unit",
    "plan_overlap",
]


def build_app(*args: Any, **kwargs: Any) -> Any:
    """Lazily import and return the FastHTML app factory (keeps the ``[ui]`` extra optional).

    Importing ``calibrated_agent_evaluation.review.app`` pulls in ``fasthtml`` (the ``[ui]`` extra);
    the rest of the package (labeling, session, renderers) has no such dependency, so callers doing
    pure label construction never need FastHTML installed. See
    ``calibrated_agent_evaluation.review.app.build_app``.
    """
    from calibrated_agent_evaluation.review.app import build_app as _build_app

    return _build_app(*args, **kwargs)
