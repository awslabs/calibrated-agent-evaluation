# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""The failure vocabulary is read from the installed detector template's category table."""

from __future__ import annotations

import itertools
from collections import Counter
from collections.abc import Callable
from importlib.metadata import PackageNotFoundError

import pytest

pytest.importorskip("strands_evals")

from strands_evals.detectors.prompt_templates.failure_detection import failure_detection_v0

from calibrated_agent_evaluation._model import CalibratedAgentEvaluationModel
from calibrated_agent_evaluation.agentcore.detectors import vocabulary as vocabulary_module
from calibrated_agent_evaluation.agentcore.detectors.types import (
    DetectedFailure,
    Detection,
    DetectorProvenance,
    FailureVocabulary,
    NamingResult,
    VocabularyEntry,
)
from calibrated_agent_evaluation.agentcore.detectors.vocabulary import (
    failure_vocabulary,
    parse_categories,
)
from calibrated_agent_evaluation.agentcore.errors import (
    AgentCoreError,
    DetectorError,
    VocabularyParseError,
)

# The verified 1.6.0 template's groups, in table order, with their row counts.
_GROUPS = (
    ("execution-error", 9),
    ("task-instruction", 2),
    ("incorrect-actions", 4),
    ("context-handling-error", 1),
    ("hallucination", 6),
    ("repetitive-behavior", 3),
    ("orchestration-related-errors", 4),
    ("llm-output", 1),
    ("configuration-mismatch", 1),
    ("coding-use-case-specific-failure-types", 2),
)
_VERIFIED_SHA256 = "30d6baa2559a514b87e9477f58afefb977371cc413f5c841eaa3ad9798c809a9"
_OTHER_GUIDELINE = 'If uncertain about category, use "other" with explanation'
_OTHER_PREFIX = '- **Use "other"**: '

# A well-formed table of four data rows in two groups; each malformed case replaces one row.
_TABLE_HEAD = (
    "Use the specific category names in your annotations.\n"
    "\n"
    "|Parent Category|Category|Category Definition|Examples|\n"
    "|---|---|---|---|\n"
)
_ROWS: tuple[str, ...] = (
    '|**alpha**|alpha-category-one|Fails first|• "one"|',
    '||alpha-category-two|Fails second|• "two"|',
    "|**beta**|beta-category-three|Fails third||",
    '||beta-category-four|Fails fourth|• "four"|',
)
_LONG_ROW = "|beta|beta-category-three|" + "Fails third, at length. " * 6 + "||"


def test_the_installed_template_yields_33_categories_in_10_groups_then_other() -> None:
    vocabulary = failure_vocabulary()

    *rows, other = vocabulary.entries
    assert len(rows) == 33
    assert [group for group, _ in itertools.groupby(row.parent for row in rows)] == [
        group for group, _ in _GROUPS
    ]
    assert Counter(row.parent for row in rows) == dict(_GROUPS)
    assert all(row.parent and row.name.startswith(f"{row.parent}-category-") for row in rows)
    assert vocabulary.names()[0] == "execution-error-category-authentication"
    assert len(set(vocabulary.names())) == 34
    assert other.name == "other"
    assert other.parent is None
    assert [row.name for row in rows if not row.examples] == [
        "coding-use-case-specific-failure-types-category-edge-case-oversights",
        "coding-use-case-specific-failure-types-category-dependency-issues",
    ]
    assert vocabulary.detector_version == "1.6.0"
    assert vocabulary.prompt_template == "v0"
    assert vocabulary.prompt_sha256 == _VERIFIED_SHA256


def test_continuation_rows_inherit_the_parent_group() -> None:
    table = (
        "Use the specific category names in your annotations.\n"
        "\n"
        "|Parent Category|Category|Category Definition|Examples|\n"
        "|---|---|---|---|\n"
        '|**alpha**|alpha-category-one|Fails first<br><br> • a detail|• "one"|\n'
        '||alpha-category-two|Fails second|• "two"|\n'
        "|**beta**|beta-category-three|Fails third||\n"
        '||beta-category-four|Fails fourth<br>again|• "four"<br> • "more"|\n'
    )

    assert parse_categories(table) == (
        VocabularyEntry(
            name="alpha-category-one",
            parent="alpha",
            definition="Fails first<br><br> • a detail",
            examples='• "one"',
        ),
        VocabularyEntry(
            name="alpha-category-two",
            parent="alpha",
            definition="Fails second",
            examples='• "two"',
        ),
        VocabularyEntry(
            name="beta-category-three", parent="beta", definition="Fails third", examples=""
        ),
        VocabularyEntry(
            name="beta-category-four",
            parent="beta",
            definition="Fails fourth<br>again",
            examples='• "four"<br> • "more"',
        ),
    )


def test_the_other_entry_carries_the_templates_guideline(monkeypatch: pytest.MonkeyPatch) -> None:
    vocabulary = failure_vocabulary()

    assert vocabulary.entries[-1] == VocabularyEntry(
        name="other", parent=None, definition=_OTHER_GUIDELINE, examples=""
    )
    assert vocabulary.descriptions()["other"] == _OTHER_GUIDELINE

    changed = failure_detection_v0.DEFAULT_ANNOTATION_GUIDELINES.replace(
        _OTHER_GUIDELINE, "When no category fits, use other"
    )
    monkeypatch.setattr(failure_detection_v0, "DEFAULT_ANNOTATION_GUIDELINES", changed)

    assert failure_vocabulary().descriptions()["other"] == "When no category fits, use other"


def _parse_table_with_row(row: int, line: str) -> Callable[[pytest.MonkeyPatch], object]:
    """Parse the well-formed table with its row ``row`` (counted from 1) replaced by ``line``."""

    def parse(_: pytest.MonkeyPatch) -> object:
        rows = list(_ROWS)
        rows[row - 1] = line
        return parse_categories(_TABLE_HEAD + "\n".join(rows) + "\n")

    return parse


def _read_with_other_guideline(line: str) -> Callable[[pytest.MonkeyPatch], object]:
    """Read the installed vocabulary with the guidelines' ``other`` line replaced by ``line``."""

    def read(monkeypatch: pytest.MonkeyPatch) -> object:
        guidelines = failure_detection_v0.DEFAULT_ANNOTATION_GUIDELINES.splitlines()
        changed = [line if old.startswith(_OTHER_PREFIX) else old for old in guidelines]
        monkeypatch.setattr(
            failure_detection_v0, "DEFAULT_ANNOTATION_GUIDELINES", "\n".join(changed)
        )
        return failure_vocabulary()

    return read


@pytest.mark.parametrize(
    ("refuse", "row", "fragments"),
    [
        pytest.param(
            _parse_table_with_row(2, "||alpha-category-two|Fails second|"),
            2,
            ("four cells", "the row has 3", "||alpha-category-two|Fails second|"),
            id="a row with three cells",
        ),
        pytest.param(
            _parse_table_with_row(3, "|beta|beta-category-three|Fails third||"),
            3,
            ("**<group>**", "'beta'", "|beta|beta-category-three|Fails third||"),
            id="a parent cell without its bold markup",
        ),
        pytest.param(
            _parse_table_with_row(3, _LONG_ROW),
            3,
            ("**<group>**", _LONG_ROW[:80] + "..."),
            id="a long row, quoted cut",
        ),
        pytest.param(
            _parse_table_with_row(2, '||alpha-category-one|Fails second|• "two"|'),
            2,
            ("unique", "'alpha-category-one'", "row 1", "||alpha-category-one|Fails second|"),
            id="a repeated name",
        ),
        pytest.param(
            _parse_table_with_row(4, '||gamma-category-four|Fails fourth|• "four"|'),
            4,
            ("'beta-category-'", "||gamma-category-four|Fails fourth|"),
            id="a name without its parent prefix",
        ),
        pytest.param(
            _parse_table_with_row(1, '||alpha-category-one|Fails first|• "one"|'),
            1,
            ("no row above names a parent group", "||alpha-category-one|Fails first|"),
            id="a first row with no parent group",
        ),
        pytest.param(
            lambda _: parse_categories("The table was removed from the prompt.\n"),
            1,
            ("a category row", "holds none"),
            id="a text with no table",
        ),
        pytest.param(
            lambda _: parse_categories(_TABLE_HEAD),
            1,
            ("a category row", "holds none"),
            id="a table with no rows",
        ),
        pytest.param(
            _read_with_other_guideline(""),
            34,
            (_OTHER_PREFIX.strip(),),
            id="guidelines with no line for other",
        ),
        pytest.param(
            _read_with_other_guideline(_OTHER_PREFIX),
            34,
            (_OTHER_PREFIX.strip(), "no definition"),
            id="guidelines whose line for other is empty",
        ),
    ],
)
def test_a_malformed_category_table_is_refused_with_the_row_named(
    refuse: Callable[[pytest.MonkeyPatch], object],
    row: int,
    fragments: tuple[str, ...],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    with pytest.raises(VocabularyParseError) as refused:
        refuse(monkeypatch)

    assert refused.value.row == row
    assert [fragment for fragment in fragments if fragment not in refused.value.check] == []


def test_a_detector_package_reporting_no_version_is_refused(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def not_installed(distribution: str) -> str:
        raise PackageNotFoundError(distribution)

    monkeypatch.setattr(vocabulary_module, "version", not_installed)

    with pytest.raises(AgentCoreError, match="reports no installed version") as refused:
        failure_vocabulary()
    assert isinstance(refused.value.__cause__, PackageNotFoundError)


def _no_such_template(monkeypatch: pytest.MonkeyPatch) -> tuple[str, type[Exception] | None]:
    def unknown(template_version: str) -> object:
        raise ValueError(f"Unknown version: {template_version}")

    monkeypatch.setattr(vocabulary_module, "get_template", unknown)
    return "template 'v0'", ValueError


def _no_category_table(monkeypatch: pytest.MonkeyPatch) -> tuple[str, type[Exception] | None]:
    monkeypatch.delattr(failure_detection_v0, "CATEGORIES")
    return "CATEGORIES", AttributeError


def _a_category_table_that_is_not_text(
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[str, type[Exception] | None]:
    monkeypatch.setattr(failure_detection_v0, "CATEGORIES", ["execution-error-category-a"])
    return "CATEGORIES", None


def _a_prompt_builder_with_another_signature(
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[str, type[Exception] | None]:
    def build_prompt(session: str) -> str:
        return session

    monkeypatch.setattr(failure_detection_v0, "build_prompt", build_prompt)
    return "build_prompt", TypeError


@pytest.mark.parametrize(
    "change",
    [
        _no_such_template,
        _no_category_table,
        _a_category_table_that_is_not_text,
        _a_prompt_builder_with_another_signature,
    ],
    ids=["no v0 template", "no category table", "a table that is not text", "another builder"],
)
def test_a_template_the_reader_cannot_use_is_refused_naming_the_installed_version(
    monkeypatch: pytest.MonkeyPatch,
    change: Callable[[pytest.MonkeyPatch], tuple[str, type[Exception] | None]],
) -> None:
    what, cause_type = change(monkeypatch)

    with pytest.raises(AgentCoreError) as refused:
        failure_vocabulary()

    assert "strands-agents-evals 1.6.0" in str(refused.value)
    assert what in str(refused.value)
    cause = refused.value.__cause__
    assert (None if cause is None else type(cause)) is cause_type


@pytest.mark.parametrize(
    "model",
    [
        VocabularyEntry,
        FailureVocabulary,
        DetectorProvenance,
        DetectedFailure,
        Detection,
        NamingResult,
    ],
)
def test_an_invalid_detector_result_is_refused_with_the_detector_error(
    model: type[CalibratedAgentEvaluationModel],
) -> None:
    with pytest.raises(DetectorError, match=f"invalid {model.__name__}"):
        model.model_validate({})
