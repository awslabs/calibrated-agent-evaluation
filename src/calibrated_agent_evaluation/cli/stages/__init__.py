# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Per-stage CLI handlers for the ``calibrated-agent-evaluation`` pipeline.

One thin module per stage that needs more than a one-line dispatch entry (``run``, ``label``,
``generate``, ``dumpload``). Each handler reuses the feature module; it never
re-implements the methodology.
"""
