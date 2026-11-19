# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""The backend-abstraction interface.

``TraceStoreBackend`` and ``LabelBackend`` are :class:`typing.Protocol`s defined *solely*
against the normalized schema (``Trace`` / ``Span`` / ``Label``). No method signature here
exposes a Langfuse or Inspect (vendor) type; enforced by the introspection
test and the methodology-plane boundary check. Concrete backends
(``LocalBackend``, ``LangfuseBackend``) structurally satisfy these protocols.

Error contract (typed, never swallowed; "no silent failure"):
  * ``put_trace`` / ``put_score`` raise :class:`SchemaError` on an invalid artifact
    and persist nothing.
  * ``get_trace`` raises :class:`BackendError` when the id is absent or the backend is
    unreachable; it never returns ``None`` as a sentinel.
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any, ClassVar, Protocol, Self, runtime_checkable

from pydantic import ConfigDict, model_validator

from calibrated_agent_evaluation._model import CalibratedAgentEvaluationModel
from calibrated_agent_evaluation.schema.errors import BackendError
from calibrated_agent_evaluation.schema.types import Frame, Label, Ownership, Trace


@runtime_checkable
class TraceStoreBackend(Protocol):
    """Persist and retrieve normalized traces and their labels through one interface.

    The methodology plane is written once against this protocol and works regardless of
    which concrete backend is configured.
    """

    def put_trace(self, trace: Trace) -> None:
        """Persist a trace (and its span tree). Raises ``SchemaError`` on invalid input."""
        ...

    def get_trace(self, trace_id: str) -> Trace:
        """Return the trace for ``trace_id``, equal field-for-field to what was written.

        Raises ``BackendError`` if no such trace exists or the backend is unreachable.
        """
        ...

    def query(
        self,
        frame: Frame | None = None,
        cell: dict[str, str] | None = None,
        ownership: Ownership | None = None,
    ) -> list[Trace]:
        """Return the traces matching the given queryable dimensions.

        The locked query surface; ``frame`` / ``cell`` / ``ownership`` only; richer
        querying is out of scope. An empty match returns ``[]``.
        """
        ...

    def put_score(self, label: Label) -> None:
        """Persist a label. Raises ``SchemaError`` on invalid input."""
        ...

    def get_scores(self, trace_id: str, span_id: str | None = None) -> list[Label]:
        """Return the labels for a trace (optionally narrowed to one span)."""
        ...

    def retract_score(self, label: Label) -> None:
        """Retract a stored label, so no later ``get_scores`` returns it.

        ``label`` is matched by full content (``schema.serde.label_identity``), the same identity
        the read-side dedup uses, so exactly that label goes and a different verdict on the same
        target stays. Putting the same label again afterwards stores it again. Retracting a label
        that is not (yet) readable is not an error: an eventually-consistent store cannot tell
        "absent" from "still ingesting", and the retraction must win over a put that lands late.
        Raises ``BackendError`` if the backend is unreachable or the retraction is not written.
        """
        ...


@runtime_checkable
class LabelBackend(Protocol):
    """A human-labeling work-list source/sink, defined against normalized ``Label`` only.

    No implementation ships: this is the managed-labeling seam, deferred until a managed
    labeling backend is built. The review app reads its work list from a
    ``TraceStoreBackend`` and writes labels with ``put_score`` instead. The interface is fixed so a
    future labeling backend speaks the normalized schema, never a vendor labeling type.
    """

    def next_item(self) -> Trace | None:
        """Return the next trace awaiting a human label, or ``None`` when the work-list is empty."""
        ...

    def submit(self, label: Label) -> None:
        """Record a human label. Raises ``SchemaError`` on invalid input."""
        ...


class SourceTrace(CalibratedAgentEvaluationModel):
    """One trace read from a source store, in the form the store could give it.

    Exactly one of ``trace``, ``otlp`` and ``error`` is set:

    * ``trace``: the store holds a Calibrated Agent Evaluation-normalized trace and returned it
      losslessly;
    * ``otlp``: the store holds a trace something else wrote, mapped to a canonical OTLP-JSON
      document for the ingest path to detect and normalize like a file;
    * ``error``: this one trace could not be read, and why. The ingest path reports it as skipped
      and carries on, because one unreadable trace is no reason to drop the rest of the pull.

    Plain types only, so a vendor shape never crosses into the ingest plane.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")
    invalid_error: ClassVar[type[Exception]] = BackendError

    source_id: str
    trace: Trace | None = None
    otlp: dict[str, Any] | None = None
    error: str | None = None
    #: The session the source knows the trace by, stamped on the trace normalized from ``otlp``.
    #: Only with ``otlp``: a normalized ``trace`` already carries its own ``session_id``.
    session_id: str | None = None

    @model_validator(mode="after")
    def _exactly_one_form(self) -> Self:
        set_forms = [name for name in ("trace", "otlp", "error") if getattr(self, name) is not None]
        if len(set_forms) != 1:
            raise BackendError(
                f"source trace {self.source_id!r} must carry exactly one of trace, otlp or error; "
                f"got {', '.join(set_forms) or 'none'}"
            )
        if self.session_id is not None and not self.session_id.strip():
            # Refused here, as this one record's error, rather than when the trace is stored.
            raise BackendError(
                f"source trace {self.source_id!r} carries a blank session_id; leave it unset "
                "when the source recorded no session"
            )
        if self.session_id is not None and self.otlp is None:
            raise BackendError(
                f"source trace {self.source_id!r} carries a session_id with {set_forms[0]}; a "
                "session id is only stamped on a trace normalized from otlp"
            )
        return self


@runtime_checkable
class TraceSource(Protocol):
    """A store that can be pulled from, including traces it did not normalize itself."""

    def pull(self) -> Iterator[SourceTrace]:
        """Yield every trace in the store as a :class:`SourceTrace`.

        Raises ``BackendError`` before yielding anything if the store cannot be listed (unreachable
        or rejected credentials); a failure on one trace is a ``SourceTrace.error``, not a raise.
        """
        ...


class SkippedTrace(CalibratedAgentEvaluationModel):
    """A trace a whole-store read left out, and why."""

    model_config = ConfigDict(frozen=True, extra="forbid")
    invalid_error: ClassVar[type[Exception]] = BackendError

    trace_id: str
    reason: str


class TraceScan(CalibratedAgentEvaluationModel):
    """A whole-store read: the traces read, and every trace left out with its reason.

    ``skipped`` is how a read stays honest about what it did not return. A store shared with
    another client holds traces Calibrated Agent Evaluation did not write; a stage reads the rest
    and says how many it left out and why, instead of aborting on the first foreign trace or
    dropping it unseen.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")
    invalid_error: ClassVar[type[Exception]] = BackendError

    traces: list[Trace]
    skipped: tuple[SkippedTrace, ...] = ()


@runtime_checkable
class ScanningBackend(Protocol):
    """A store that can hold traces it cannot return as normalized ``Trace``s."""

    def scan(
        self,
        frame: Frame | None = None,
        cell: dict[str, str] | None = None,
        ownership: Ownership | None = None,
    ) -> TraceScan:
        """``query``, but a trace it cannot return is a :class:`SkippedTrace`, not a raise."""
        ...


def scan_traces(
    backend: TraceStoreBackend,
    *,
    frame: Frame | None = None,
    cell: dict[str, str] | None = None,
    ownership: Ownership | None = None,
) -> TraceScan:
    """Read every matching trace from ``backend``, reporting the ones it leaves out.

    A backend that can hold foreign traces (:class:`ScanningBackend`, e.g. ``LangfuseBackend``)
    scans; any other returns everything ``query`` does and skips nothing.
    """
    if isinstance(backend, ScanningBackend):
        return backend.scan(frame=frame, cell=cell, ownership=ownership)
    if frame is None and cell is None and ownership is None:
        return TraceScan(traces=backend.query())
    return TraceScan(traces=backend.query(frame=frame, cell=cell, ownership=ownership))
