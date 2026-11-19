# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Hierarchical agglomerative clustering of critique embeddings."""

from __future__ import annotations

from calibrated_agent_evaluation._install import install_hint
from calibrated_agent_evaluation.cluster.errors import ClusteringError


def hierarchical_cluster(
    embeddings: list[list[float]],
    *,
    threshold: float = 1.0,
) -> list[int]:
    """Cluster embeddings via Ward's linkage + distance-based cut.

    Returns a list of cluster assignments (0-indexed integers), one per embedding.
    Requires the ``[stats]`` extra (scipy).
    """
    if len(embeddings) < 2:
        raise ClusteringError(f"Clustering requires ≥2 FAIL critiques; got {len(embeddings)}")

    try:
        from scipy.cluster.hierarchy import fcluster, linkage
    except ImportError as exc:
        raise ClusteringError(
            f"scipy is required for hierarchical clustering; install with: {install_hint('stats')}"
        ) from exc

    import numpy as np

    matrix = np.array(embeddings, dtype=np.float64)
    z = linkage(matrix, method="ward")
    labels = fcluster(z, t=threshold, criterion="distance")
    # Convert from 1-indexed (scipy) to 0-indexed
    return [int(lbl) - 1 for lbl in labels]
