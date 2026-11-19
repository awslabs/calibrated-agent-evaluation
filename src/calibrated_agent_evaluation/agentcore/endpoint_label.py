# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""The name of one AgentCore Runtime endpoint, ``<runtime ARN>@<qualifier>``, with no AWS SDK.

A probe of a runtime names its target by this label: the authorization refusal names it, and the
endpoint fingerprint that every probed trace, its trace ID and the probe's finding carry is the
fingerprint of it. ``calibrated-agent-evaluation probe --executor agentcore`` refuses an
unauthorized probe before it checks for the ``[agentcore]`` extra, so this module imports nothing
the extra installs, and the runtime module takes its default qualifier from here.
"""

from __future__ import annotations

__all__ = ["DEFAULT_QUALIFIER", "endpoint_label"]

DEFAULT_QUALIFIER = "DEFAULT"
"""The endpoint an invocation goes to when the caller names none."""


def endpoint_label(runtime_arn: str, qualifier: str) -> str:
    """The label of a runtime's endpoint: ``<runtime ARN>@<qualifier>``.

    Args:
        runtime_arn: The runtime's ARN.
        qualifier: The endpoint invoked, such as :data:`DEFAULT_QUALIFIER`.

    Returns:
        The label.
    """
    return f"{runtime_arn}@{qualifier}"
