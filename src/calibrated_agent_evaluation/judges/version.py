# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Schema version for the judge-runtime artifacts (JudgeRunRecord, VerdictCache).

Separate from the trace-store ``SCHEMA_VERSION`` and the calibration ``CALIB_SCHEMA_VERSION``: the
judge-runtime artifacts (``run_record``, ``cache``) round-trip through their own CSV+JSON codecs,
so they carry their own version and refuse-on-mismatch, mirroring the calibration store
precedent. Bumped only when a stored judge-runtime artifact's shape changes.
"""

from __future__ import annotations

JUDGES_SCHEMA_VERSION = "1"
