# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""The base for the library's pydantic models, and a validated copy-with-changes.

Every record and wire type subclasses :class:`CalibratedAgentEvaluationModel`, which turns a
pydantic ``ValidationError`` raised while building an instance into the model family's own typed
error (``invalid_error``), so a caller catches ``CalibrationError`` or ``RunError`` and never a raw
pydantic exception.
"""

from __future__ import annotations

import dataclasses
from typing import Any, ClassVar, Self, TypeVar

from pydantic import (
    BaseModel,
    ConfigDict,
    ModelWrapValidatorHandler,
    ValidationError,
    model_validator,
)

from calibrated_agent_evaluation._errors import CalibratedAgentEvaluationDataError

__all__ = ["CalibratedAgentEvaluationModel", "plain_error", "replace"]

T = TypeVar("T")


def plain_error(exc: ValidationError) -> str:
    """Summarize a validation failure as ``field: problem`` pairs, without pydantic's URLs."""
    parts = []
    for err in exc.errors(include_url=False):
        where = ".".join(str(p) for p in err["loc"]) or "value"
        parts.append(f"{where}: {err['msg']}")
    return "; ".join(parts)


class CalibratedAgentEvaluationModel(BaseModel):
    """A pydantic model whose invalid input raises the family's typed error.

    Subclasses set ``invalid_error`` to their package's base error and declare their own
    ``model_config`` (frozen, ``extra="forbid"``).
    """

    # Building every model's schema at import costs each CLI command ~70 ms before it does
    # anything; deferring builds a schema the first time its model is used.
    model_config = ConfigDict(defer_build=True)

    invalid_error: ClassVar[type[Exception]] = CalibratedAgentEvaluationDataError

    @model_validator(mode="wrap")
    @classmethod
    def _raise_typed(cls, data: Any, handler: ModelWrapValidatorHandler[Self]) -> Self:
        try:
            return handler(data)
        except ValidationError as exc:
            raise cls.invalid_error(f"invalid {cls.__name__}: {plain_error(exc)}") from exc


def replace(obj: T, /, **changes: Any) -> T:
    """Return a copy of ``obj`` with ``changes`` applied, validated like a fresh construction.

    Unlike ``BaseModel.model_copy(update=...)``, the copy goes through the model's validators, so
    a change that breaks an invariant raises exactly as constructing it would.

    Args:
        obj: A pydantic model or a dataclass instance.
        **changes: Field values to replace.

    Returns:
        The new instance.

    Raises:
        TypeError: ``obj`` is neither a pydantic model nor a dataclass instance.
    """
    if isinstance(obj, BaseModel):
        fields = {name: getattr(obj, name) for name in type(obj).model_fields}
        return type(obj)(**{**fields, **changes})
    if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
        return dataclasses.replace(obj, **changes)
    raise TypeError(f"replace() needs a pydantic model or a dataclass, got {type(obj).__name__}")
