# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Load `inspect_evals` benchmark tasks: the sole `inspect_evals` import.

Lives in the sanctioned execution plane (`execute/`) so that `probe/` imports no vendor SDK.
Each `inspect_evals` benchmark is a ``@task`` function of the
same name inside its submodule (verified against `inspect_evals==0.14.3`: e.g.
``inspect_evals.xstest.xstest`` -> ``inspect_ai.Task``). This loader resolves a module name to its
task, so `probe/benchmarks.py` can name benchmarks by string and never touch the vendor package.

`inspect_evals` is imported lazily so `import calibrated_agent_evaluation.execute` stays cheap and
does not require the ``[probe]`` extra; a missing package raises a clear :class:`ExecutionError` at
call time.
"""

from __future__ import annotations

from typing import Any

from calibrated_agent_evaluation._install import install_hint
from calibrated_agent_evaluation.execute.errors import ExecutionError


def load_benchmark_task(module_name: str, **task_kwargs: Any) -> Any:
    """Return the `inspect_ai.Task` for the `inspect_evals` benchmark ``module_name``.

    ``module_name`` is the submodule/task name (e.g. ``"xstest"``, ``"agentdojo"``). Raises
    :class:`ExecutionError` if the ``[probe]`` extra is absent or the benchmark is not present in
    the installed `inspect_evals` (the caller surfaces substitutions before reaching here).
    """
    try:
        module = __import__(f"inspect_evals.{module_name}", fromlist=[module_name])
    except ImportError as exc:
        raise ExecutionError(
            f"could not import inspect_evals.{module_name}; install the probing extra "
            f"({install_hint('probe')}) or check the benchmark name. ({exc})"
        ) from exc
    task_fn = getattr(module, module_name, None)
    if task_fn is None:
        raise ExecutionError(
            f"inspect_evals.{module_name} has no task callable {module_name!r} "
            "(surface changed vs the verified inspect_evals==0.14.3)"
        )
    try:
        return task_fn(**task_kwargs)
    except Exception as exc:
        raise ExecutionError(
            f"building inspect_evals benchmark {module_name!r} failed: {type(exc).__name__}: {exc}"
        ) from exc
