# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Labeling form + completion view for the review surface.

The form is **critique-first**: the critique textarea is emitted before the
pass/fail verdict controls. For an owned crew trace it also exposes the span-attribution fields
(culprit / manifestation / handoff / span-level defect). A pre-fill
``Suggestion`` (if any) is shown above the critique with an explicit Accept control; the human
ratifies by accepting it or by overriding it (another verdict, an edited critique), never by a
verdict key alone. Produces FastHTML components; imported only with the ``[ui]`` extra.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from typing import Any
from urllib.parse import quote

from fasthtml.common import (
    H3,
    H4,
    A,
    Button,
    Div,
    Form,
    Hidden,
    Input,
    Label,
    Li,
    Option,
    P,
    Select,
    Span,
    Strong,
    Textarea,
    Ul,
)

from calibrated_agent_evaluation.review.prefill import Suggestion
from calibrated_agent_evaluation.review.record import CodebookMode
from calibrated_agent_evaluation.review.session import LabeledEntry
from calibrated_agent_evaluation.schema import Label as LabelRecord
from calibrated_agent_evaluation.schema import Trace


def _suggestion_banner(suggestion: Suggestion) -> Any:
    return Div(
        Strong("LLM suggestion (ratify before it counts): "),
        Span(f"{suggestion.outcome}", cls="hd-suggested-verdict"),
        P(suggestion.critique, cls="hd-suggested-critique"),
        Button("Accept suggestion (A)", type="button", cls="hd-accept"),
        Span(
            " Or override it: edit the critique, or pick the other verdict. A verdict key on the "
            "unedited suggestion does not accept it.",
            cls="hd-muted",
        ),
        cls="hd-suggestion",
    )


def _span_select(name: str, trace: Trace, chosen: str) -> Any:
    # A closed choice, not free text: a mistyped span id became a phantom per-agent failure mode in
    # the CI gate's ModeKey. The server re-checks membership on submit.
    options = [Option("(none)", value="", selected=not chosen)]
    options += [
        Option(span.span_id, value=span.span_id, selected=span.span_id == chosen)
        for span in trace.spans
    ]
    return Select(*options, id=name, name=name)


def _attribution_fields(trace: Trace, submitted: Mapping[str, str]) -> Any:
    return Div(
        H4("Crew attribution (FAIL only)"),
        Div(
            Label("Culprit span (first divergence):", fr="culprit_span"),
            _span_select("culprit_span", trace, submitted.get("culprit_span", "")),
        ),
        Div(
            Label("Manifestation span (where harm surfaced):", fr="manifestation_span"),
            _span_select("manifestation_span", trace, submitted.get("manifestation_span", "")),
        ),
        Div(
            Label("Handoff implicated:", fr="handoff_implicated"),
            Input(
                id="handoff_implicated",
                name="handoff_implicated",
                value=submitted.get("handoff_implicated", ""),
                placeholder="e.g. intake->policy-checker",
            ),
        ),
        Div(
            Label("Span-level defect (records a separate span-level FAIL):", fr="defect_span"),
            _span_select("defect_span", trace, submitted.get("defect_span", "")),
        ),
        cls="hd-attribution",
    )


def _failure_mode_field(
    modes: Sequence[CodebookMode], proposed: Sequence[CodebookMode], chosen: str
) -> Any:
    # A closed choice from the codebook plus the pending proposals; the server re-checks it.
    # Required on a FAIL, so hotkeys.js focuses it when a FAIL is asked for with nothing picked.
    options = [Option("(pick a failure mode)", value="", selected=not chosen)]
    options += [
        Option(m.name, value=m.name, selected=m.name == chosen, title=m.description or None)
        for m in modes
    ]
    options += [
        Option(
            f"{m.name} (proposed)",
            value=m.name,
            selected=m.name == chosen,
            title=m.description or None,
        )
        for m in proposed
    ]
    return Div(
        Label("Failure mode (required for FAIL; M to pick):", fr="failure_mode"),
        Select(*options, id="failure_mode", name="failure_mode"),
        Span("", cls="hd-failure-mode-hint", role="status"),
        cls="hd-failure-mode",
    )


def _new_mode_field(submitted: Mapping[str, str]) -> Any:
    # The escape when no listed mode fits: a name and a note, flagged for the codebook owner.
    # Hidden until N opens it (or a refused submission brought a proposal back).
    name = submitted.get("new_mode_name", "")
    return Div(
        Label("New failure mode (N): name", fr="new_mode_name"),
        Input(id="new_mode_name", name="new_mode_name", value=name, autocomplete="off"),
        Label("What it covers (required with a new mode):", fr="new_mode_note"),
        Input(id="new_mode_note", name="new_mode_note", value=submitted.get("new_mode_note", "")),
        Span("", cls="hd-new-mode-similar", role="status"),
        cls="hd-new-mode",
        hidden=not name,
    )


def _expected_field(value: str) -> Any:
    # The objective "correct output" for an objective FAIL (e.g. the action the agent
    # should have taken), entered as a JSON object. Optional and FAIL-only; subjective fails and
    # every PASS leave it blank. It NEVER substitutes for the critique.
    return Div(
        H4("Expected correct output (objective FAIL only, optional)"),
        Label("The correct output, as a JSON object:", fr="expected"),
        Textarea(
            value,
            id="expected",
            name="expected",
            rows="2",
            placeholder='e.g. {"action": "REFUND_DENIED", "amount_usd": 0}',
            cls="hd-expected-input",
        ),
        cls="hd-expected",
    )


def labeling_form(
    trace: Trace,
    suggestion: Suggestion | None = None,
    *,
    submitted: Mapping[str, str] | None = None,
    failure_modes: Sequence[CodebookMode] = (),
    proposed_modes: Sequence[CodebookMode] = (),
    can_propose: bool = False,
) -> Any:
    """Build the critique-first labeling form for ``trace``.

    ``submitted`` carries the values of a submission the server refused or failed to write, so
    the re-rendered form shows what the labeler typed instead of a blank form.
    It takes precedence over a suggestion's critique. ``failure_modes``, when non-empty, adds the
    failure-mode picker after the critique; a FAIL then needs one. ``proposed_modes`` are offered
    after them, marked as proposed, and ``can_propose`` adds the new-mode fields.
    """
    submitted = submitted or {}
    if "critique" in submitted:
        prefilled = submitted["critique"]
    else:
        prefilled = suggestion.critique if suggestion is not None else ""
    fields: list[Any] = []
    if suggestion is not None:
        fields.append(_suggestion_banner(suggestion))

    # Critique FIRST; before the verdict controls.
    fields.append(
        Div(
            Label("Critique: why / first failure (required for FAIL):", fr="critique"),
            Textarea(
                prefilled,
                id="critique",
                name="critique",
                rows="4",
                placeholder="What did the agent do, and why is it right or wrong?",
            ),
            cls="hd-critique",
        )
    )

    if failure_modes:
        chosen = submitted.get("failure_mode", "")
        fields.append(_failure_mode_field(failure_modes, proposed_modes, chosen))
        if can_propose:
            fields.append(_new_mode_field(submitted))

    # Verdict AFTER the critique.
    fields.append(
        Div(
            Strong("Verdict:"),
            Button("Pass (1)", name="outcome", value="PASS", type="submit", cls="hd-pass"),
            Button("Fail (2)", name="outcome", value="FAIL", type="submit", cls="hd-fail"),
            cls="hd-verdict",
        )
    )

    if len(trace.spans) > 1:  # crew trace; offer span attribution
        fields.append(_attribution_fields(trace, submitted))

    # Objective correct-answer capture; applies to single-agent traces too, so it is
    # offered for every trace (not gated on the crew branch above).
    fields.append(_expected_field(submitted.get("expected", "")))

    fields.append(Hidden(name="trace_id", value=trace.trace_id))

    return Form(
        *fields,
        # No hx-post: hotkeys.js owns submission, so a button click or an Enter in a field goes
        # through the same repeat, in-flight and dwell guards as the verdict keys.
        id="label-form",
        cls="hd-label-form",
        # what the Accept action submits (the server re-checks it against its own suggestion)
        data_suggested_outcome=suggestion.outcome if suggestion is not None else None,
        data_modes_required="1" if failure_modes else None,
        # The names a new mode is checked against for near-duplicates as it is typed.
        data_mode_names=json.dumps([m.name for m in (*failure_modes, *proposed_modes)])
        if can_propose
        else None,
    )


#: Every binding hotkeys.js handles, in the order the legend lists them.
HOTKEYS: tuple[tuple[str, str], ...] = (
    ("1", "Pass"),
    ("2", "Fail (needs a critique, and a failure mode when the codebook lists them)"),
    ("M", "pick the failure mode (when the codebook lists them)"),
    ("N", "propose a new failure mode when none fits (name and note, for the codebook owner)"),
    ("Ctrl/Cmd+Enter", "submit the verdict asked for, from inside the critique box"),
    ("Esc", "leave a text field so the keys work again"),
    ("A", "accept the LLM suggestion as shown (only when one is offered)"),
    ("D", "defer: move this trace to the end"),
    ("U", "undo the last verdict"),
    ("\u2190 / \u2192", "previous / next trace, no verdict"),
    ("H", "history: the traces labeled this session"),
    ("?", "show or hide this list"),
)


def _clock(seconds: float) -> str:
    whole = int(seconds)
    return f"{whole // 60}:{whole % 60:02d}"


def progress_bar(position: int, total: int, elapsed_seconds: float) -> Any:
    """``Trace n of m``, the session's on-trace time, and the legend hint, above every trace.

    The pacing signal for the 50-in-30-minutes target. hotkeys.js keeps the clock running between
    renders from ``data-elapsed-seconds``; the rendered text is the value at render time.
    """
    return Div(
        Span(f"Trace {position} of {total}", cls="hd-position"),
        Span(
            f"{_clock(elapsed_seconds)} on traces",
            cls="hd-elapsed",
            data_elapsed_seconds=f"{elapsed_seconds:.1f}",
        ),
        Span("? for keys", cls="hd-muted"),
        cls="hd-progress",
    )


def hotkey_legend() -> Any:
    """The full hotkey list, hidden until ``?`` toggles it."""
    return Div(
        H4("Keys"),
        *[P(Strong(keys), f" {action}") for keys, action in HOTKEYS],
        id="hd-legend",
        cls="hd-legend",
        hidden=True,
    )


def completion_view(summary: dict[str, float | int]) -> Any:
    """The end-of-work-list state: shown instead of an empty trace view."""
    return Div(
        H3("All traces labeled 🎉"),
        P(
            f"Labeled {summary['labeled_count']} traces in "
            f"{float(summary['elapsed_seconds']):.1f}s of on-trace time."
        ),
        Span("Nothing left in the work-list.", cls="hd-muted"),
        # No id: the page's own #review-root wraps this, and a second one made the id ambiguous.
        cls="hd-complete",
    )


def _swap_link(href: str, text: str) -> Any:
    # htmx swaps the target into #review-root; hotkeys.js re-primes after every htmx swap.
    return A(text, href=href, hx_get=href, hx_target="#review-root", hx_swap="innerHTML")


def history_view(entries: Sequence[LabeledEntry]) -> Any:
    """The traces labeled this session, newest first, each linking to a read-only view.

    ``/nav`` only steps through the pending queue, so before this a trace labeled a few back was
    reachable only by undoing everything after it.
    """
    rows = [
        Li(
            _swap_link(f"/trace/{quote(e.trace_id, safe='')}", e.trace_id),
            f" {e.label.outcome}",
            Span(f" ({e.seconds_on_trace:.0f}s on trace)", cls="hd-muted"),
        )
        for e in reversed(entries)
    ]
    return Div(
        H3("Labeled this session"),
        Ul(*rows) if rows else P("Nothing labeled yet.", cls="hd-muted"),
        _swap_link("/view", "Back to the queue"),
        cls="hd-history",
    )


def stored_labels_view(labels: Sequence[LabelRecord]) -> Any:
    """Read-only list of the labels the store holds for a trace that is no longer pending.

    To change a verdict, undo it (if it is the latest) and relabel; this view writes nothing.
    """
    rows = [
        Li(
            Strong(lbl.outcome),
            f" {lbl.span_id or 'trace-level'}",
            Span(f" by {lbl.rater_id or 'unattributed'} ({lbl.kind})", cls="hd-muted"),
            P(lbl.critique),
        )
        for lbl in labels
    ]
    return Div(
        H4("Stored labels (read-only)"),
        Ul(*rows) if rows else P("No stored labels.", cls="hd-muted"),
        _swap_link("/history", "Back to history"),
        " · ",
        _swap_link("/view", "Back to the queue"),
        cls="hd-stored-labels",
    )
