# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Run an `inspect_evals` benchmark against a probed endpoint: one sample, one trial.

The bridge the probe battery needs. A benchmark expects a model it can generate from; a probed
endpoint is someone else's agent behind HTTP. This module registers an Inspect model provider
(``probe-endpoint/...``) whose ``generate`` sends the sample's prompt through the probe
:class:`~calibrated_agent_evaluation.probe.adapter.EndpointAdapter` and returns the endpoint's reply
as the model output, so the benchmark's own solver and its own scorer run unchanged and the
benchmark decides what counts as a break.

One benchmark sample is one trial: the task runs once with ``limit=k``, Inspect scores each
sample, and each sample's score becomes one trial outcome. Running the whole benchmark k times
would be a different (and far more expensive) measurement.

Inspect is imported at module top level, so this module is lazy-imported by its callers and never
reached from ``calibrated_agent_evaluation.execute.__init__``. Written against
``inspect-ai==0.3.282`` and ``inspect_evals==0.14.4``; the verified surface is ``@modelapi``,
``ModelAPI.generate``, ``ModelOutput.from_content``,
``eval(task, model=, limit=, log_dir=, display=)`` and ``EvalLog.samples[*].scores[*].value``.
"""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from pathlib import Path
from typing import Any

from inspect_ai.model import (
    ChatMessage,
    GenerateConfig,
    ModelAPI,
    ModelOutput,
    modelapi,
)
from inspect_ai.tool import ToolChoice, ToolInfo

from calibrated_agent_evaluation.execute.errors import ExecutionError

__all__ = ["PROVIDER", "benchmark_trials", "configure"]

PROVIDER = "probe-endpoint"

# The sender the provider routes through. Inspect builds a model by name, so there is no argument
# to thread an adapter through; the same pattern the cassette provider uses. Write-once per run.
_SENDER: Callable[[str], str] | None = None


def configure(send: Callable[[str], str]) -> None:
    """Point the provider at the endpoint sender for the next benchmark run."""
    global _SENDER
    _SENDER = send


@modelapi(name=PROVIDER)
def probe_endpoint_provider() -> type[ModelAPI]:
    """Register the ``probe-endpoint/<name>`` provider (called by Inspect's model registry)."""
    return ProbeEndpointModelAPI


class ProbeEndpointModelAPI(ModelAPI):
    """An Inspect model whose completions come from the probed endpoint, not from a model API."""

    async def generate(
        self,
        input: list[ChatMessage],
        tools: list[ToolInfo],
        tool_choice: ToolChoice,  # noqa: ARG002 - the ModelAPI signature; an endpoint has no tools
        config: GenerateConfig,  # noqa: ARG002 - decoding is the endpoint owner's, not ours
    ) -> ModelOutput:
        """Send the conversation's last user message to the endpoint and return its reply."""
        if _SENDER is None:
            raise ExecutionError(
                "the probe-endpoint provider was used before configure(); call "
                "probe_bridge.configure(send) before running a benchmark"
            )
        if tools:
            raise ExecutionError(
                "this benchmark hands the model tools, and a probed endpoint takes a prompt and "
                "returns text; it cannot run through a black-box probe"
            )
        prompt = _prompt_of(input)
        if not prompt.strip():
            raise ExecutionError("the benchmark sample carried no prompt text to send")
        # The adapter is synchronous (stdlib urllib); run it off the event loop so Inspect's
        # concurrency is not blocked by the network wait.
        text = await asyncio.to_thread(_SENDER, prompt)
        return ModelOutput.from_content(model=self.model_name, content=text)


def _prompt_of(messages: list[ChatMessage]) -> str:
    """The text a probed endpoint is sent: the system preamble plus the last user message.

    An endpoint takes one prompt, so a benchmark's system message has to ride with the user turn
    or it would be dropped silently (XSTest's solver sets one).
    """
    system = [_text_of(m) for m in messages if m.role == "system"]
    users = [_text_of(m) for m in messages if m.role == "user"]
    parts = [*system, users[-1] if users else ""]
    return "\n\n".join(part for part in parts if part.strip())


def _text_of(message: ChatMessage) -> str:
    """One chat message's prose, whatever content shape it carries."""
    content = message.content
    if isinstance(content, str):
        return content
    texts = [getattr(block, "text", "") for block in content]
    return "\n".join(text for text in texts if isinstance(text, str) and text)


def benchmark_trials(
    identifier: str,
    send: Callable[[str], str],
    *,
    trials: int,
    log_dir: Path,
    grader_model: str | None = None,
    task: Any | None = None,
) -> list[bool]:
    """Run ``trials`` samples of a benchmark against the endpoint; one outcome per sample.

    Args:
        identifier: The `inspect_evals` benchmark to run (ignored when ``task`` is given).
        send: Sends one prompt to the endpoint and returns its reply.
        trials: How many benchmark samples to run; one sample is one trial.
        log_dir: Where Inspect writes the run's log.
        grader_model: The model the benchmark grades with, for the benchmarks that take one.
        task: A prepared Inspect task, instead of loading one by name (the test seam).

    Returns:
        One ``True`` per sample the benchmark's own scorer counted as a break.

    Raises:
        ExecutionError: The extra is absent, the benchmark could not be built or run, the run did
            not succeed, or a sample came back without a score.
    """
    from inspect_ai import eval as eval_fn

    from calibrated_agent_evaluation.probe.benchmarks import broke_from_score

    if trials < 1:
        raise ExecutionError(f"a benchmark run needs at least one trial, got {trials}")
    resolved_task = task if task is not None else _load_task(identifier, grader_model)
    configure(send)
    try:
        logs = eval_fn(
            resolved_task,
            model=f"{PROVIDER}/{identifier}",
            limit=trials,
            log_dir=str(log_dir),
            display="none",
        )
    except Exception as exc:
        raise ExecutionError(
            f"running benchmark {identifier!r} against the endpoint failed: "
            f"{type(exc).__name__}: {exc}"
        ) from exc

    outcomes: list[bool] = []
    for log in logs:
        if log.status != "success":
            raise ExecutionError(
                f"benchmark {identifier!r} did not succeed: status={log.status}, "
                f"error={getattr(log, 'error', None)}"
            )
        for sample in log.samples or []:
            outcomes.append(_broke(identifier, sample, broke_from_score))
    return outcomes


def _broke(identifier: str, sample: Any, broke_from_score: Callable[[str, Any], bool]) -> bool:
    """Read one scored sample's break verdict off the benchmark's own score."""
    scores = getattr(sample, "scores", None) or {}
    if not scores:
        raise ExecutionError(
            f"benchmark {identifier!r} returned sample {getattr(sample, 'id', '?')!r} with no "
            "score, so whether it broke is unknown; it is not counted as safe"
        )
    # One scorer per benchmark in the ones a text-only probe can run; take the first by name so
    # the choice is deterministic rather than dict-order dependent.
    name = sorted(scores)[0]
    try:
        return broke_from_score(identifier, scores[name].value)
    except Exception as exc:
        raise ExecutionError(
            f"benchmark {identifier!r} sample {getattr(sample, 'id', '?')!r}: {exc}"
        ) from exc


def _load_task(identifier: str, grader_model: str | None) -> Any:
    """Build the `inspect_evals` task, passing the grader model under the name it takes."""
    from calibrated_agent_evaluation.execute.benchmark_tasks import load_benchmark_task
    from calibrated_agent_evaluation.probe.benchmarks import grader_task_arg

    kwargs: dict[str, Any] = {}
    if grader_model is not None:
        arg = grader_task_arg(identifier)
        if arg is None:
            raise ExecutionError(
                f"benchmark {identifier!r} takes no grader model, so --grader-model has nowhere "
                "to go; drop it for this benchmark"
            )
        kwargs[arg] = grader_model
    return load_benchmark_task(identifier, **kwargs)
