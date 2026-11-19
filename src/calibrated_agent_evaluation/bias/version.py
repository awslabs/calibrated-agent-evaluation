# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Schema version for bias-battery artifacts.

Isolated in its own module (like the enforcement ``version.py``) so ``errors``/``record`` can pin
the version without an import cycle.
"""

from __future__ import annotations

# Stamped on every persisted BiasBatteryRecord; load refuses on mismatch (no coercion).
# "2" added `provenance`; "3" gates the excess over a placebo (`placebo_count`, `both_count`), so
# a "2" record's magnitude measured something else and is refused, not reread; "4" adds the
# directional test (`direction_*`), so a "3" record never checked direction and is refused.
BIAS_SCHEMA_VERSION = "4"
