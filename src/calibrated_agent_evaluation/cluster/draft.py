# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Draft codebook assembly from cluster assignments."""

from __future__ import annotations

from collections import Counter

from calibrated_agent_evaluation.cluster.embed import DraftNamer
from calibrated_agent_evaluation.cluster.errors import ClusteringError
from calibrated_agent_evaluation.cluster.types import DraftCluster, DraftCodebook


def build_draft(
    critiques: list[str],
    embeddings: list[list[float]],  # noqa: ARG001
    assignments: list[int],
    *,
    namer: DraftNamer | None = None,
) -> DraftCodebook:
    """Assemble a DraftCodebook from cluster assignments.

    Groups critiques by assignment label, optionally asks the namer for names.

    Raises:
        ClusteringError: The namer returns a different number of names than there are
            clusters, or returns any name more than once.
    """
    groups: dict[int, list[int]] = {}
    for idx, label in enumerate(assignments):
        groups.setdefault(label, []).append(idx)

    # Assemble in a single canonical (sorted-by-label) order so the namer sees clusters in the
    # SAME order the DraftClusters are built; otherwise scipy's non-sorted labels would attach
    # each proposed name to the wrong cluster.
    sorted_groups = sorted(groups.items())
    cluster_critique_lists = [[critiques[i] for i in indices] for _label, indices in sorted_groups]

    if namer is not None:
        names = namer.name_clusters(cluster_critique_lists)
        # A namer (e.g. an LLM) that returns the wrong number of names would, under a lenient
        # zip, silently DROP the unnamed clusters and every critique in them; the exact
        # failure-mode-loss this module exists to prevent. Refuse loudly instead (no silent
        # truncation).
        if len(names) != len(sorted_groups):
            raise ClusteringError(
                f"namer returned {len(names)} name(s) for {len(sorted_groups)} cluster(s); "
                f"every cluster must be named or its critiques would be dropped"
            )
        # Two clusters under one name would collapse into a single mode at ratification and
        # lose the other cluster's critiques, so a repeated name is refused, not built.
        repeated = [name for name, count in Counter(names).items() if count > 1]
        if repeated:
            raise ClusteringError(
                f"namer gave more than one cluster the name(s) "
                f"{', '.join(repr(name) for name in repeated)}; every cluster needs its own "
                f"name or ratification would collapse them into one mode"
            )
    else:
        names = [f"mode_{i}" for i in range(len(sorted_groups))]

    clusters: list[DraftCluster] = []
    for (_label, indices), name in zip(sorted_groups, names, strict=True):
        clusters.append(
            DraftCluster(
                name=name,
                member_indices=tuple(indices),
                member_critiques=tuple(critiques[i] for i in indices),
            )
        )

    return DraftCodebook(clusters=tuple(clusters))
