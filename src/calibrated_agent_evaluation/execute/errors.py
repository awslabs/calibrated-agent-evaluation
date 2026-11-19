# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Typed error family for the execution plane.

Every failure that crosses the ``execute/`` boundary surfaces as an ``ExecutionError``; never a
raw ``boto3.ClientError``, Inspect exception, ``json.JSONDecodeError``, or ``OSError``. Internal
errors (``CassetteError``, ``CassetteMiss``) are caught at the ``InspectExecutor`` boundary and
wrapped here.
"""

from __future__ import annotations


class ExecutionError(Exception):
    """A typed error from the execution plane.

    Wraps vendor/SDK exceptions via chaining (``raise ExecutionError(...) from exc``). The message
    carries diagnostic detail (file paths, eval status, fingerprint); the chained ``__cause__``
    preserves the original traceback for debugging.
    """


class SandboxError(ExecutionError):
    """Sandboxed execution could not be carried out.

    Raised when the process could not start, a remote sandbox was unreachable, or its response was
    malformed. A subclass so an existing ``except ExecutionError`` boundary still catches it. This
    is an *infrastructure* failure, distinct from a program that ran and failed its tests (that is a
    PASS/FAIL verdict, not an error); the execution scorers re-wrap it as ``ScorerError`` at their
    boundary and never return a Label for it.
    """


class JudgeConfigurationError(ExecutionError):
    """The judge model was called in a way that cannot work, whatever the model does.

    A subclass so an existing ``except ExecutionError`` boundary still catches it, but named so a
    caller that distinguishes transient from permanent can. ``judges/runtime._judge_one`` retries a
    bare ``ExecutionError`` as transient, so a permanent configuration fault raised as one burned
    the whole retry budget and then reported itself as "model failed after N attempts" -- sending
    the reader to look at the model instead of at the call site.
    """
