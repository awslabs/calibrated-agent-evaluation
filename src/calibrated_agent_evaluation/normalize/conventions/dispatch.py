# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Route a :class:`SessionSpans` to its convention adapter.

The single front-door the ``normalize()`` seam calls for a raw span list: it ensures the built-in
adapters are registered, resolves the convention (the declared ``semconv``, or :func:`detect` when
``"auto"``), and runs the registered adapter to the vendor-free ``Intermediate``. Keeping
this here (not in ``normalize/__init__``) preserves the dispatch shape and keeps the convention
layer self-contained. Importing this module imports the built-in adapters, which self-register.
"""

from __future__ import annotations

from calibrated_agent_evaluation._model import replace
from calibrated_agent_evaluation.normalize.conventions import (
    messages_adapter,  # noqa: F401 - imported for its registry.register_builtin side effect
    openinference_adapter,  # noqa: F401
    openllmetry_adapter,  # noqa: F401
    registry,
)
from calibrated_agent_evaluation.normalize.conventions.detect import detect
from calibrated_agent_evaluation.normalize.sources import SessionSpans
from calibrated_agent_evaluation.normalize.tree import Intermediate


def to_intermediate(session: SessionSpans) -> Intermediate:
    """Resolve the convention (declared or auto-detected) and run its adapter."""
    kind = detect(session.spans) if session.semconv == "auto" else session.semconv
    adapter = registry.get_adapter(kind)
    return replace(adapter(session), convention=kind)
