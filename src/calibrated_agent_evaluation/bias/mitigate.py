# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Runtime bias mitigations shipped alongside detection.

v1 ships **order-swap-and-tie** for pairwise judges: evaluate both orderings of a candidate pair
and, when the judge's pick flips with the order, resolve it as a tie rather than letting a
position-driven winner stand. (The optional multi-model PoLL jury is deferred -- out of scope.)
"""

from __future__ import annotations

TIE = "TIE"


def order_swap_and_tie(label_ab: str, label_ba: str) -> str:
    """Return the consistent winner across both orderings, or :data:`TIE` when they disagree.

    ``label_ab`` is the judge's pick with the candidates in one order; ``label_ba`` with the order
    swapped. A fair judge picks the same underlying candidate regardless of order, so a
    disagreement is an order effect -- neutralized as a tie.
    """
    return label_ab if label_ab == label_ba else TIE
