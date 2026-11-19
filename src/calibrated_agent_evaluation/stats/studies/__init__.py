# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""The studies that derive and check the shipped calibration constants.

``calibrated_agent_evaluation.calibrate.plan`` ships a coverage floor (``COVERAGE_FLOOR``) and a
minimum rate sample (``MIN_RATE_SAMPLE``), and the docs make claims about how the corrected rate
compares with a naive reading. These modules hold the computation behind each: the Monte Carlo
coverage study of ``rg_interval`` (:mod:`~calibrated_agent_evaluation.stats.studies.coverage`), the
floor and minimum derived from its scans
(:mod:`~calibrated_agent_evaluation.stats.studies.planner_floor`,
:mod:`~calibrated_agent_evaluation.stats.studies.rate_sample_minimum`) and the three-arm baseline
comparison (:mod:`~calibrated_agent_evaluation.stats.studies.baseline_arm`). Their output is
committed under ``fixtures/expected/coverage/``; the test suite re-derives the shipped values from
that output.

Importing this package needs the ``[stats]`` extra (numpy and scipy);
``calibrated_agent_evaluation.stats`` does not import it.
"""
