# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""The detector's failure vocabulary, read from the installed failure-detection prompt template.

The template the detector sends holds a markdown table of failure categories, one row per
category, grouped by a parent category that is set on a group's first row and left empty on the
rows that continue it. :func:`failure_vocabulary` reads that table at call time, so the
vocabulary is always the one the installed detector shows its model, and appends ``other``, the
category the template's guidelines tell the model to use when it is unsure. A table that has
changed shape is refused with
:class:`~calibrated_agent_evaluation.agentcore.errors.VocabularyParseError` naming the row, never
read into a vocabulary the detector's model was not shown.

Every read of the installed template goes through one place, so a release that drops, renames or
retypes a part of it Calibrated Agent Evaluation reads (the ``v0`` template, its ``CATEGORIES``
table, its guidelines, its system prompt, its ``build_prompt``) is refused with an
:class:`~calibrated_agent_evaluation.agentcore.errors.AgentCoreError` naming the installed version.
"""

from __future__ import annotations

import hashlib
import re
from collections.abc import Sequence
from importlib.metadata import PackageNotFoundError, version
from types import ModuleType
from typing import ClassVar

from pydantic import ConfigDict
from strands_evals.detectors.prompt_templates.failure_detection import get_template

from calibrated_agent_evaluation._model import CalibratedAgentEvaluationModel
from calibrated_agent_evaluation.agentcore.detectors.types import FailureVocabulary, VocabularyEntry
from calibrated_agent_evaluation.agentcore.errors import (
    AgentCoreError,
    DetectorError,
    VocabularyParseError,
)

__all__ = [
    "TEMPLATE_VERSION",
    "TemplateProvenance",
    "detector_prompt",
    "failure_vocabulary",
    "installed_detector",
    "parse_categories",
    "template_provenance",
]

TEMPLATE_VERSION = "v0"
"""The failure-detection template version the detector sends and the vocabulary is read from."""

_DETECTOR_DISTRIBUTION = "strands-agents-evals"
# The release whose template these readers were checked against.
_VERIFIED_VERSION = "1.6.0"
_OTHER_NAME = "other"
_OTHER_GUIDELINE_PREFIX = '- **Use "other"**: '
# The table's header row and the separator row under it come before the category rows.
_HEADER_ROWS = 2
_CELLS = 4
_ROW_FORM = "|parent|name|definition|examples|"
_NAME_INFIX = "-category-"
# A parent cell that is not empty is the group in bold: no asterisk inside, no space at either end.
_GROUP_CELL = re.compile(r"\*\*([^*\s](?:[^*]*[^*\s])?)\*\*")
# A refusal quotes the row cut to this many characters, marked with "...", to stay readable.
_EXCERPT_CHARS = 80


class TemplateProvenance(CalibratedAgentEvaluationModel):
    """Which detector and template a result came from.

    ``prompt_sha256`` is the SHA-256 of the prompt the detector sends for an empty session: the
    template's system prompt, a blank line, then the user prompt built around ``[]``.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")
    invalid_error: ClassVar[type[Exception]] = DetectorError

    detector_version: str
    prompt_template: str
    prompt_sha256: str


def parse_categories(text: str) -> tuple[VocabularyEntry, ...]:
    """Read the template's category table into one entry per row, in table order.

    The lines before the header row (the first line starting with ``|``), the header row and its
    separator are skipped; every line after them is a category row. Rows are numbered from 1,
    starting at the first category row, and a refusal names the row by that number. Each row
    ``|parent|name|definition|examples|`` gives one entry, its cells stripped. A parent cell
    ``**group**`` reads as ``group``; an empty one continues the group of the row above.
    Definition and example markup, such as ``<br>``, is kept as written.

    Args:
        text: The template's category table, with the prose that introduces it.

    Returns:
        The entries, without ``other``.

    Raises:
        VocabularyParseError: The text holds no category row; or a row does not have four cells,
            has a parent cell that is neither empty nor exactly ``**<group>**``, repeats the name
            of a row above it, or has a name that does not start with its parent group followed
            by ``-category-``. The error's ``check`` quotes the row, cut to 80 characters.
    """
    rows = _category_rows(text.splitlines())
    entries: list[VocabularyEntry] = []
    rows_by_name: dict[str, int] = {}
    parent: str | None = None
    for row, line in enumerate(rows, start=1):
        parent_cell, name, definition, examples = _cells(row, line)
        if parent_cell:
            parent = _group(row, line, parent_cell)
        _check_name(row, line, name, parent, rows_by_name)
        rows_by_name[name] = row
        entries.append(
            VocabularyEntry(name=name, parent=parent, definition=definition, examples=examples)
        )
    return tuple(entries)


def failure_vocabulary() -> FailureVocabulary:
    """The installed detector's failure categories in table order, then ``other``.

    The template is read on every call. The ``other`` entry has no parent and no examples; its
    definition is the text of the template's guideline line that starts
    ``- **Use "other"**:``.

    Returns:
        The vocabulary, with the detector version, template version and template hash.

    Raises:
        VocabularyParseError: The category table breaks a row check (see
            :func:`parse_categories`), or the guidelines have no line for ``other`` or one with
            no text after its start; the error then names the row ``other`` would take, one
            past the table's last.
        AgentCoreError: The installed detector package reports no version, or its template
            lacks a part the reader needs or holds one of another type.
    """
    template = _installed_template()
    categories = _text(template, "CATEGORIES")
    guidelines = _text(template, "DEFAULT_ANNOTATION_GUIDELINES")
    rows = parse_categories(categories)
    other = VocabularyEntry(
        name=_OTHER_NAME,
        parent=None,
        definition=_other_guideline(guidelines, row=len(rows) + 1),
        examples="",
    )
    provenance = _provenance_of(template)
    return FailureVocabulary(
        entries=(*rows, other),
        detector_version=provenance.detector_version,
        prompt_template=provenance.prompt_template,
        prompt_sha256=provenance.prompt_sha256,
    )


def template_provenance() -> TemplateProvenance:
    """The installed detector's version, its template version and the template's hash.

    Computed on every call, so the hash always describes the template as it is now.

    Returns:
        The template provenance.

    Raises:
        AgentCoreError: The installed detector package reports no version, or its template
            lacks a part the hash covers.
    """
    return _provenance_of(_installed_template())


def detector_prompt(session_json: str) -> tuple[str, str]:
    """The prompt the detector library sends for a serialized session, from the installed template.

    Args:
        session_json: The session serialized as the library serializes it.

    Returns:
        The user prompt, which the library checks against its context limit, and the full prompt
        it sends as one user message: the system prompt, a blank line, then the user prompt.

    Raises:
        AgentCoreError: The installed template lacks its system prompt or a ``build_prompt``
            that builds a prompt from ``session_json``.
    """
    template = _installed_template()
    user_prompt = _built_prompt(template, session_json)
    return user_prompt, _joined(template, user_prompt)


def installed_detector() -> str:
    """The installed detector package and its version, the way an error message names them.

    Returns:
        ``"strands-agents-evals <version>"``, or ``"strands-agents-evals (its version is not
        recorded)"`` when the package reports no version.
    """
    try:
        return f"{_DETECTOR_DISTRIBUTION} {version(_DETECTOR_DISTRIBUTION)}"
    except PackageNotFoundError:
        return f"{_DETECTOR_DISTRIBUTION} (its version is not recorded)"


def _provenance_of(template: ModuleType) -> TemplateProvenance:
    # The hash covers the prompt the detector sends for an empty session.
    digest = hashlib.sha256(_joined(template, _built_prompt(template, "[]")).encode()).hexdigest()
    return TemplateProvenance(
        detector_version=_detector_version(),
        prompt_template=TEMPLATE_VERSION,
        prompt_sha256=digest,
    )


def _joined(template: ModuleType, user_prompt: str) -> str:
    # The detector sends the system prompt and the user prompt as one message, joined this way.
    return f"{_text(template, 'SYSTEM_PROMPT')}\n\n{user_prompt}"


def _installed_template() -> ModuleType:
    try:
        template = get_template(TEMPLATE_VERSION)
    except ValueError as exc:
        raise _unusable(
            f"it has no failure-detection template {TEMPLATE_VERSION!r} ({type(exc).__name__})"
        ) from exc
    if not isinstance(template, ModuleType):
        raise _unusable(
            f"its failure-detection template {TEMPLATE_VERSION!r} is a "
            f"{type(template).__name__}, not a module"
        )
    return template


def _text(template: ModuleType, name: str) -> str:
    try:
        value = getattr(template, name)
    except AttributeError as exc:
        raise _unusable(
            f"its failure-detection template {TEMPLATE_VERSION!r} has no {name}"
        ) from exc
    if not isinstance(value, str):
        raise _unusable(
            f"its failure-detection template's {name} is {type(value).__name__}, not text"
        )
    return value


def _built_prompt(template: ModuleType, session_json: str) -> str:
    builder = getattr(template, "build_prompt", None)
    if not callable(builder):
        raise _unusable(
            f"its failure-detection template {TEMPLATE_VERSION!r} has no build_prompt function"
        )
    try:
        prompt = builder(session_json=session_json)
    except Exception as exc:
        raise _unusable(
            "its failure-detection template's build_prompt cannot build a prompt from a session "
            f"({type(exc).__name__})"
        ) from exc
    if not isinstance(prompt, str):
        raise _unusable(
            f"its failure-detection template's build_prompt returned {type(prompt).__name__}, "
            "not text"
        )
    return prompt


def _unusable(problem: str) -> AgentCoreError:
    return AgentCoreError(
        f"the installed {installed_detector()} cannot be read as the failure detector Calibrated "
        "Agent Evaluation "
        f"wraps: {problem}. Calibrated Agent Evaluation reads the template of "
        f"{_DETECTOR_DISTRIBUTION} "
        f"{_VERIFIED_VERSION}; install that version"
    )


def _detector_version() -> str:
    try:
        return version(_DETECTOR_DISTRIBUTION)
    except PackageNotFoundError as exc:
        raise AgentCoreError(
            f"the {_DETECTOR_DISTRIBUTION} package imports but reports no installed version, so "
            "a result could not say which detector produced it; reinstall the [agentcore] extra"
        ) from exc


def _other_guideline(guidelines: str, *, row: int) -> str:
    """The guideline for ``other``; ``row`` is the vocabulary row ``other`` takes, for a refusal."""
    check = (
        f"the other entry takes its definition from the guidelines line starting "
        f"{_OTHER_GUIDELINE_PREFIX.strip()!r}"
    )
    for line in guidelines.splitlines():
        if line.startswith(_OTHER_GUIDELINE_PREFIX):
            definition = line.removeprefix(_OTHER_GUIDELINE_PREFIX).strip()
            if not definition:
                raise VocabularyParseError(row, f"{check}; that line has no definition after it")
            return definition
    raise VocabularyParseError(row, f"{check}; the guidelines have no such line")


def _category_rows(lines: Sequence[str]) -> Sequence[str]:
    """The lines after the table's header and separator rows; refused when there are none."""
    header = next((index for index, line in enumerate(lines) if line.startswith("|")), None)
    rows = [] if header is None else lines[header + _HEADER_ROWS :]
    if not rows:
        raise VocabularyParseError(
            1, "a category row after the table's header and separator rows; the text holds none"
        )
    return rows


def _cells(row: int, line: str) -> list[str]:
    """The row's four cells, stripped; refused unless the row is ``|a|b|c|d|``."""
    stripped = line.strip()
    check = f"four cells, {_ROW_FORM}"
    if not stripped:
        raise VocabularyParseError(row, f"{check}; the row is blank")
    if not (stripped.startswith("|") and stripped.endswith("|")):
        raise VocabularyParseError(
            row, f"{check}; the row does not start and end with '|': {_excerpt(line)}"
        )
    cells = [cell.strip() for cell in stripped.split("|")[1:-1]]
    if len(cells) != _CELLS:
        raise VocabularyParseError(row, f"{check}; the row has {len(cells)}: {_excerpt(line)}")
    return cells


def _group(row: int, line: str, cell: str) -> str:
    """The group a non-empty parent cell names; refused unless the cell is ``**<group>**``."""
    match = _GROUP_CELL.fullmatch(cell)
    if match is None:
        raise VocabularyParseError(
            row,
            f"a parent cell is empty or exactly **<group>**; this one is {cell!r}: "
            f"{_excerpt(line)}",
        )
    return match.group(1)


def _check_name(
    row: int, line: str, name: str, parent: str | None, rows_by_name: dict[str, int]
) -> None:
    """Refuse a name already used by a row above, or one not prefixed by its parent group."""
    if name in rows_by_name:
        raise VocabularyParseError(
            row,
            f"each name is unique; {name!r} is also the name of row {rows_by_name[name]}: "
            f"{_excerpt(line)}",
        )
    check = f"each name starts with its parent group followed by {_NAME_INFIX!r}"
    if parent is None:
        raise VocabularyParseError(
            row,
            f"{check}; the parent cell is empty and no row above names a parent group: "
            f"{_excerpt(line)}",
        )
    prefix = f"{parent}{_NAME_INFIX}"
    if not name.startswith(prefix):
        raise VocabularyParseError(row, f"{check}, here {prefix!r}: {_excerpt(line)}")


def _excerpt(line: str) -> str:
    """The row as written, cut to ``_EXCERPT_CHARS`` characters with ``...`` marking a cut."""
    stripped = line.strip()
    if len(stripped) <= _EXCERPT_CHARS:
        return stripped
    return f"{stripped[:_EXCERPT_CHARS]}..."
