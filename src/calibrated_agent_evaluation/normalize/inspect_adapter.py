# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Inspect ``.eval`` adapter.

Reads the log via ``read_eval_log_samples(path, resolve_attachments="full")`` (so message/tool
content is materialized, not left as ``attachment://`` placeholders) and builds the
span forest via Inspect's public ``event_tree(sample.events)`` (never the pre-flattened
``inspect_ai.analysis`` dataframes, which drop the parent/child tree). Surface verified against
the installed ``inspect-ai==0.3.242``: ``EventTreeSpan{id,parent_id,type,name,begin,end,children}``,
``ToolEvent{function,arguments,result,...}``.

Mapping policy: a leaf ``ToolEvent`` becomes a ``ToolCall`` on its enclosing agent; an
``agent``-type ``EventTreeSpan`` is an agent boundary → an :class:`AgentNode`. Inspect's built-in
``react`` agent is single-agent by construction, so an OrderBot/ClaimsCrew ``.eval`` normalizes to
a one-span tree whose root mirrors the trace I/O. The ``submit`` react control tool is
NOT a business tool call; its ``answer`` argument is the terminal output, so it is excluded from
``tool_calls`` and used as the trace/root output.

The only import of ``inspect_ai``; done lazily so importing
``calibrated_agent_evaluation.normalize`` needs no extra; a missing ``[inspect]`` surfaces as a
typed error naming the installable string.
"""

from __future__ import annotations

from typing import Any

from calibrated_agent_evaluation._install import install_hint
from calibrated_agent_evaluation.normalize import content as content_mod
from calibrated_agent_evaluation.normalize.errors import NormalizationError
from calibrated_agent_evaluation.normalize.sources import InspectEvalLog
from calibrated_agent_evaluation.normalize.tree import AgentNode, Intermediate, ToolInvocation

# Inspect's react agent emits a terminal `submit(answer=...)` control tool; not a business tool.
_REACT_SUBMIT = "submit"


def to_intermediate(source: InspectEvalLog) -> Intermediate:
    """Map an :class:`InspectEvalLog` source to the vendor-free intermediate: ONE sample's worth.

    Refuses a multi-sample log unless ``source.sample_id`` names which sample to read. It used to
    take the first sample and drain the rest away, so an eval over a dataset of N rows normalized to
    1 trace and every downstream rate, count and gate decision was computed from 1/N of the data,
    with nothing said. Use :func:`calibrated_agent_evaluation.normalize.normalize_all` for the whole
    log.
    """
    intermediates = to_intermediates(source)
    if len(intermediates) == 1:
        return intermediates[0]
    raise NormalizationError(
        f"Inspect .eval log {str(source.path)!r} holds {len(intermediates)} samples; normalize() "
        "returns one Trace, so it would evaluate the log at 1/N of its data. Use normalize_all() "
        "for one Trace per sample, or set InspectEvalLog(sample_id=...) to read one deliberately"
    )


def to_intermediates(source: InspectEvalLog) -> list[Intermediate]:
    """Map every sample in the log to its own intermediate: one per dataset row.

    A ``.eval`` holds one sample per dataset row, so this is the shape a real eval log has. Order
    follows the log.
    """
    read_eval_log_samples, event_tree, event_tree_span_cls = _load_inspect()
    samples, epoched = _read_samples(source, read_eval_log_samples)
    multi = len(samples) > 1
    return [
        _intermediate_for(
            source, sample, event_tree, event_tree_span_cls, multi=multi, epoched=epoched
        )
        for sample in samples
    ]


def _intermediate_for(
    source: InspectEvalLog,
    sample: Any,
    event_tree: Any,
    event_tree_span_cls: Any,
    *,
    multi: bool,
    epoched: bool,
) -> Intermediate:
    """Build one sample's intermediate.

    On a log run with ``epochs > 1`` every sample id appears once per epoch, so the id gets an
    ``#e{epoch}`` suffix; without it each epoch after the first resolved to the same trace id and
    either replaced the one before or was refused as a repeat.
    """
    metadata = dict(getattr(sample, "metadata", None) or {})
    sample_id = str(getattr(sample, "id", "eval"))
    # On a multi-sample log `source.trace_id` names the LOG, not a sample, so it cannot be the trace
    # id: `calibrated-agent-evaluation ingest` derives it from the file name, and using it would
    # give every sample in the file the same id and collapse them in the store. The sample's own id
    # disambiguates.
    if multi:
        trace_id = metadata.get("trace_id") or (
            f"{source.trace_id}/{sample_id}" if source.trace_id else sample_id
        )
    else:
        trace_id = source.trace_id or metadata.get("trace_id") or sample_id
    if epoched:
        trace_id = f"{trace_id}#e{_epoch(sample)}"
    taxonomy = dict(source.taxonomy_cell or metadata.get("taxonomy_cell") or {})

    trace_input = _sample_input(sample)
    tool_calls, terminal_answer = _tool_calls_and_answer(sample.events)
    trace_output = terminal_answer or content_mod.as_object(_completion(sample))

    agent_name = _agent_name(sample.events, event_tree, event_tree_span_cls)
    node = AgentNode(
        agent=agent_name,
        role="specialist",
        parent=None,
        input=dict(trace_input),
        output=dict(trace_output),
        tool_calls=tool_calls,
        span_id=f"{trace_id}/{agent_name}",
        order=0,
    )
    return Intermediate(
        trace_id=str(trace_id),
        frame=source.frame,
        taxonomy_cell=taxonomy,
        trace_input=trace_input,
        trace_output=trace_output,
        nodes=[node],
    )


def _load_inspect() -> tuple[Any, Any, Any]:
    """Import the verified inspect_ai surface lazily; a missing extra → typed error."""
    try:
        from inspect_ai.event import EventTreeSpan, event_tree
        from inspect_ai.log import read_eval_log_samples
    except ImportError as exc:
        raise NormalizationError(
            "reading an Inspect .eval log requires the optional extra; install it with "
            f"`{install_hint('inspect')}`"
        ) from exc
    return read_eval_log_samples, event_tree, EventTreeSpan


def _epoch(sample: Any) -> int:
    epoch = getattr(sample, "epoch", 1)
    return epoch if isinstance(epoch, int) else 1


def _read_samples(source: InspectEvalLog, read_eval_log_samples: Any) -> tuple[list[Any], bool]:
    """Read every sample, resolving attachments; wrap a read error typed.

    The generator is drained fully rather than after the first sample; draining used to be how
    samples 2..N were *discarded* (to close the file handle), which is the defect this replaces.
    """
    try:
        samples = list(read_eval_log_samples(str(source.path), resolve_attachments="full"))
    except (OSError, ValueError, KeyError, RuntimeError) as exc:
        raise NormalizationError(
            f"could not read Inspect .eval log {str(source.path)!r}: {exc}"
        ) from exc
    if not samples:
        raise NormalizationError(f"Inspect .eval log {str(source.path)!r} has no samples")
    epoched = any(_epoch(sample) > 1 for sample in samples)
    if source.sample_id is None:
        return samples, epoched
    wanted, _, epoch_text = source.sample_id.partition("#e")
    selected = [s for s in samples if str(getattr(s, "id", "")) == wanted]
    if epoch_text:
        selected = [s for s in selected if str(_epoch(s)) == epoch_text]
    if not selected:
        available = ", ".join(sorted({str(getattr(s, "id", "?")) for s in samples}))
        raise NormalizationError(
            f"Inspect .eval log {str(source.path)!r} has no sample with id "
            f"{source.sample_id!r}; available ids: {available}"
        )
    if len(selected) > 1:
        epochs = ", ".join(f"{wanted}#e{_epoch(s)}" for s in selected)
        raise NormalizationError(
            f"sample {wanted!r} in Inspect .eval log {str(source.path)!r} ran in "
            f"{len(selected)} epochs; select one with sample_id set to one of: {epochs}"
        )
    return selected, epoched


def _tool_calls_and_answer(events: Any) -> tuple[list[ToolInvocation], dict[str, Any] | None]:
    """Flatten leaf ToolEvents to ToolCalls; the react `submit` answer becomes the terminal output.

    Excludes the `submit` control tool from tool_calls (it is not a business call); its
    `answer` argument is returned as the terminal output object.
    """
    tools: list[ToolInvocation] = []
    answer: dict[str, Any] | None = None
    for event in events:
        if getattr(event, "event", None) != "tool":
            continue
        function = getattr(event, "function", None)
        if function == _REACT_SUBMIT:
            args = getattr(event, "arguments", None) or {}
            if isinstance(args, dict) and "answer" in args:
                answer = {"answer": args["answer"]}
            continue
        tools.append(
            ToolInvocation(
                name=str(function),
                args=content_mod.as_object(getattr(event, "arguments", None)),
                result=content_mod.as_object(getattr(event, "result", None)),
            )
        )
    return tools, answer


def _agent_name(events: Any, event_tree: Any, event_tree_span_cls: Any) -> str:
    """The agent name from the innermost ``agent``-type EventTreeSpan (default ``agent``)."""
    found: list[str] = []

    def walk(nodes: Any) -> None:
        for node in nodes:
            if isinstance(node, event_tree_span_cls):
                if getattr(node, "type", None) == "agent" and getattr(node, "name", None):
                    found.append(str(node.name))
                walk(node.children)

    walk(event_tree(events))
    return found[0] if found else "agent"


def _sample_input(sample: Any) -> dict[str, Any]:
    """Sample input as an object; a bare message string becomes ``{"message": ...}``."""
    raw = getattr(sample, "input", None)
    if isinstance(raw, str):
        return {"message": raw}
    return content_mod.as_object(raw)


def _completion(sample: Any) -> Any:
    output = getattr(sample, "output", None)
    return getattr(output, "completion", None)


def message_tool_call_counts(source: InspectEvalLog) -> list[int]:
    """Count, per sample in log order, the tool calls the model's messages requested.

    The loss gate's source-side measure for an Inspect log. The adapter builds tool calls from the
    sample's ``ToolEvent`` stream; the assistant messages' ``tool_calls`` are a separate record of
    the same requests, so comparing the two can catch tool calls the adapter failed to recover.
    The ``submit`` control tool is excluded here as it is from the recovered calls.
    """
    read_eval_log_samples, _event_tree, _span_cls = _load_inspect()
    samples, _epoched = _read_samples(source, read_eval_log_samples)
    return [
        sum(
            1
            for message in getattr(sample, "messages", None) or []
            if getattr(message, "role", None) == "assistant"
            for call in getattr(message, "tool_calls", None) or []
            if getattr(call, "function", None) != _REACT_SUBMIT
        )
        for sample in samples
    ]
