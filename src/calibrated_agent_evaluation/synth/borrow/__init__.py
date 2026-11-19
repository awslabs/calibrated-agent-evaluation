# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Vendor-isolation seam for the synthesis engine.

The ONLY place under ``calibrated_agent_evaluation.synth`` permitted to import the ``strands_evals``
/ ``strands`` SDKs. The methodology-plane boundary test (``tests/test_boundary.py``) confines the
``_SYNTH_ROOTS`` = {``strands_evals``, ``strands``} to this sub-package; every other ``synth/``
module speaks the normalized schema only. Nothing here is imported at ``synth`` package-import time
(the SDK is a lazy, opt-in ``[synth]`` extra).
"""

from __future__ import annotations
