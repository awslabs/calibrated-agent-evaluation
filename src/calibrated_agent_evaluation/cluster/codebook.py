# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Codebook ratification and label tagging."""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping
from datetime import UTC, datetime

from calibrated_agent_evaluation._model import replace
from calibrated_agent_evaluation.cluster.errors import ClusteringError, UnratifiedCodebookError
from calibrated_agent_evaluation.cluster.types import Codebook, DraftCodebook, FailureMode
from calibrated_agent_evaluation.schema.types import Label


def ratify(
    draft: DraftCodebook,
    *,
    ratified_by: str,
    merges: dict[str, list[str]] | None = None,
    splits: dict[str, list[list[int]]] | None = None,
    renames: dict[str, str] | None = None,
    version: str = "1",
    parent_version: str | None = None,
    descriptions: Mapping[str, str] | None = None,
) -> Codebook:
    """Transform a DraftCodebook into a ratified Codebook.

    Raises ClusteringError if version <= parent_version (monotonicity).

    Args:
        draft: The draft codebook to ratify.
        ratified_by: Who ratified it.
        merges: New mode name to the draft modes merged into it.
        splits: Draft mode to the example index groups it splits into.
        renames: Draft mode name to its new name.
        version: The ratified codebook's version.
        parent_version: The version this one replaces.
        descriptions: Mode name to its definition, applied after merges and renames: a mode
            whose final name is a key gets that value as its description, and every other mode
            keeps the one it has (a rename keeps the old mode's). Keys that name no mode are
            unused, so a whole vocabulary can be passed. ``None`` attaches nothing.

    Returns:
        The ratified codebook.

    Raises:
        ClusteringError: ``version`` doesn't advance past ``parent_version``; ``splits`` is
            given; two draft clusters share a name; a merge or rename names an unknown mode;
            or a merge or rename targets a name that, when it is applied, a surviving mode
            holds which the operation does not consume (combine two modes with ``merges=``,
            which may list its own target).
    """
    if parent_version is not None and not _version_advances(version, parent_version):
        raise ClusteringError(
            f"Codebook version {version!r} must sort strictly after "
            f"parent version {parent_version!r} (monotonicity)"
        )

    if splits is not None:
        # Permanent: a split needs a new cut of the embeddings, which
        # clustering owns; ratify only records the human's call on clusters that already exist.
        raise ClusteringError(
            "ratify does not split clusters: re-run hierarchical_cluster with a lower threshold "
            "to cut them finer, then merge or rename"
        )

    # Two clusters under one name would collapse into the last of them and drop the others'
    # critiques, so the draft is refused instead.
    counts = Counter(cluster.name for cluster in draft.clusters)
    repeated = [name for name, count in counts.items() if count > 1]
    if repeated:
        first = repeated[0]
        second = f"{first} #2"
        raise ClusteringError(
            f"draft has more than one cluster named {', '.join(repr(n) for n in repeated)}; "
            f"ratifying would keep one and drop the others' critiques. Give each cluster its own "
            f"name (for example {second!r}) and, if they are one mode, combine them with "
            f"merges={{{first!r}: {[first, second]!r}}}"
        )

    # Start with all draft clusters as modes
    modes_by_name: dict[str, FailureMode] = {}
    for cluster in draft.clusters:
        modes_by_name[cluster.name] = FailureMode(
            name=cluster.name,
            example_critiques=cluster.member_critiques[:3],
        )

    # Apply merges: {new_name: [old_name_1, old_name_2]}
    if merges:
        for new_name, old_names in merges.items():
            # Writing the merged mode over a surviving mode the merge does not consume would
            # silently drop that mode and its critiques; a merge may list its own target.
            if new_name in modes_by_name and new_name not in old_names:
                raise ClusteringError(
                    f"merge into {new_name!r} would replace the mode {new_name!r}, which the "
                    f"merge does not list, and drop its critiques; to combine them, list it "
                    f"among the merged modes: merges={{{new_name!r}: {[new_name, *old_names]!r}}}"
                )
            examples: list[str] = []
            for old_name in old_names:
                if old_name not in modes_by_name:
                    raise ClusteringError(
                        f"merge references unknown mode {old_name!r}; "
                        f"available: {list(modes_by_name.keys())}"
                    )
                examples.extend(modes_by_name.pop(old_name).example_critiques)
            modes_by_name[new_name] = FailureMode(
                name=new_name,
                example_critiques=tuple(examples[:5]),
            )

    # Apply renames: {old_name: new_name}
    if renames:
        for old_name, new_name in renames.items():
            if old_name not in modes_by_name:
                raise ClusteringError(
                    f"rename references unknown mode {old_name!r}; "
                    f"available: {list(modes_by_name.keys())}"
                )
            # Renaming onto another surviving mode's name would replace that mode and drop its
            # critiques; combining two modes is a merge, so the rename is refused.
            if new_name != old_name and new_name in modes_by_name:
                raise ClusteringError(
                    f"rename of {old_name!r} to {new_name!r} would replace the mode "
                    f"{new_name!r} and drop its critiques; to combine the two modes, merge them "
                    f"instead: merges={{{new_name!r}: {[new_name, old_name]!r}}}"
                )
            old_mode = modes_by_name.pop(old_name)
            modes_by_name[new_name] = FailureMode(
                name=new_name,
                description=old_mode.description,
                example_critiques=old_mode.example_critiques,
            )

    modes = tuple(modes_by_name.values())
    if descriptions is not None:
        modes = tuple(
            replace(mode, description=descriptions[mode.name])
            if mode.name in descriptions
            else mode
            for mode in modes
        )

    return Codebook(
        version=version,
        modes=modes,
        ratified_by=ratified_by,
        ratified_at=datetime.now(tz=UTC),
        parent_version=parent_version,
    )


def _version_advances(version: str, parent_version: str) -> bool:
    """Return whether ``version`` sorts strictly after ``parent_version``.

    Compares NUMERICALLY when both parse as integers; so ``"9" -> "10"`` advances and
    ``"10" -> "9"`` regresses, which a pure lexicographic string compare gets backwards (``"10" <
    "9"``). Falls back to lexicographic when either is non-numeric (either scheme is permitted).
    """
    try:
        return int(version) > int(parent_version)
    except ValueError:
        return version > parent_version


def tag_label(label: Label, mode: str, codebook: Codebook | DraftCodebook) -> Label:
    """Tag a Label with a failure mode from a ratified Codebook.

    Raises UnratifiedCodebookError if codebook is a DraftCodebook.
    Raises ClusteringError if mode is not in the codebook.
    """
    if not isinstance(codebook, Codebook):
        raise UnratifiedCodebookError(
            "Codebook must be human-ratified before use; call ratify() first"
        )

    mode_names = [m.name for m in codebook.modes]
    if mode not in mode_names:
        raise ClusteringError(
            f"Mode {mode!r} not found in codebook version {codebook.version}; "
            f"available: {mode_names}"
        )

    return replace(label, failure_mode=mode, codebook_version=codebook.version)
