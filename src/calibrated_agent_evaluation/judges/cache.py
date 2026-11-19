# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""VerdictCache: skip re-computing a verdict already produced by the same judge version.

Result caching means re-running a judge over already-scored traces doesn't re-pay
for the same verdict. This is a calibrated-agent-evaluation-owned, round-trippable artifact; NOT
Inspect's ``generate(cache=…)`` (whose key is the model+input and whose store lives in Inspect's
dir, so it isn't reproducible from our artifacts) and NOT the ``Cassette`` (which caches the raw
model call, not the mapped verdict, and is the offline-*test* double).

The key is ``(judge_id, model, prompt_version, fingerprint, trace_id, span_id)`` plus the repeat
number for a repeat judgment. Because the judge's version and content are IN the key, a
re-versioned judge simply never matches a stale entry: invalidation is implicit, no separate pass.
A repeat (``JudgeTarget.sample`` above 0) is a fresh model draw, so it is its own entry. The value
is a full ``Label``, serialized via the shared ``schema.serde`` codec so it round-trips losslessly.
"""

from __future__ import annotations

import json
from pathlib import Path

from calibrated_agent_evaluation._recordstore import atomic_write_text
from calibrated_agent_evaluation.judges.errors import JudgeRuntimeError
from calibrated_agent_evaluation.judges.executable import ExecutableJudge
from calibrated_agent_evaluation.judges.version import JUDGES_SCHEMA_VERSION
from calibrated_agent_evaluation.schema.errors import SchemaError
from calibrated_agent_evaluation.schema.serde import label_from_dict, to_dict
from calibrated_agent_evaluation.schema.types import Label

_JSON_NAME = "verdict_cache.json"
# Field separator for the flat string key. "\x1f" (ASCII Unit Separator) cannot occur in a
# judge_id/model/prompt_version/trace_id/span_id, so it disambiguates the parts unambiguously.
_KEY_SEP = "\x1f"


def _key(judge: ExecutableJudge, trace_id: str, span_id: str | None, sample: int = 0) -> str:
    """The cache key: the frozen judge version and content fingerprint + the scored target.

    The repeat number joins the key only for a repeat, so a first judgment keeps the key a cache
    written before repeats existed used.
    """
    parts = [judge.judge_id, judge.model, judge.prompt_version, judge.fingerprint]
    parts += [trace_id, span_id or ""]
    if sample:
        parts.append(str(sample))
    return _KEY_SEP.join(parts)


class VerdictCache:
    """An in-memory, round-trippable map from ``(judge version, target)`` to a cached ``Label``.

    Not a context manager: it owns no connection/handle (pure in-memory dict + explicit
    ``dump``/``load``), so the conventions' context-manager rule does not apply.
    """

    def __init__(self) -> None:
        self._entries: dict[str, Label] = {}

    def get(
        self, judge: ExecutableJudge, trace_id: str, span_id: str | None, sample: int = 0
    ) -> Label | None:
        """Return the cached ``Label`` for this judge version + target, or ``None`` on a miss.

        Returns ``None`` when the judge's ``model``/``prompt_version``/content differ from any
        stored entry, so a re-versioned judge never serves a stale verdict, and for a repeat
        number not judged yet.
        """
        return self._entries.get(_key(judge, trace_id, span_id, sample))

    def put(self, judge: ExecutableJudge, label: Label, sample: int = 0) -> None:
        """Store ``label`` keyed on this judge version, its (trace_id, span_id) and repeat."""
        self._entries[_key(judge, label.trace_id, label.span_id, sample)] = label

    def __len__(self) -> int:
        return len(self._entries)

    def dump(self, path: str | Path) -> None:
        """Write the cache to ``verdict_cache.json`` under ``path`` (lossless round-trip).

        Raises :class:`JudgeRuntimeError` on any filesystem failure (no raw ``OSError`` escapes).
        """
        out = Path(path)
        payload = {
            "schema_version": JUDGES_SCHEMA_VERSION,
            "entries": [
                {"key": key, "label": to_dict(label)} for key, label in self._entries.items()
            ],
        }
        try:
            out.mkdir(parents=True, exist_ok=True)
            atomic_write_text(out / _JSON_NAME, json.dumps(payload, indent=2, sort_keys=True))
        except OSError as exc:
            raise JudgeRuntimeError(f"failed to write verdict cache to {out}: {exc}") from exc

    @classmethod
    def load(cls, path: str | Path) -> VerdictCache:
        """Read a cache back from ``path``; an absent file yields an empty cache.

        Version-checks before returning: a stored ``schema_version`` != running raises
        :class:`JudgeRuntimeError` (no coercion). Wraps filesystem/parse/malformed failures.
        """
        src = Path(path)
        file = src / _JSON_NAME
        cache = cls()
        if not file.exists():
            return cache
        try:
            payload = json.loads(file.read_text(encoding="utf-8"))
        except OSError as exc:
            raise JudgeRuntimeError(f"failed to read verdict cache from {src}: {exc}") from exc
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise JudgeRuntimeError(
                f"verdict cache file in {src} is not valid JSON: {exc}"
            ) from exc
        if not isinstance(payload, dict):
            raise JudgeRuntimeError(
                f"verdict cache file in {src} must hold one JSON object, got "
                f"{type(payload).__name__}; refusing to load it"
            )

        stored = payload.get("schema_version", JUDGES_SCHEMA_VERSION)
        if stored != JUDGES_SCHEMA_VERSION:
            raise JudgeRuntimeError(
                f"verdict cache schema_version {stored!r} != running {JUDGES_SCHEMA_VERSION!r}; "
                "refusing to load (no coercion -)"
            )
        try:
            entries: dict[str, Label] = {
                entry["key"]: label_from_dict(entry["label"])
                for entry in payload.get("entries", [])
            }
        except (KeyError, TypeError, AttributeError, SchemaError) as exc:
            raise JudgeRuntimeError(
                f"malformed verdict cache in {src} (missing/mistyped field {exc}); "
                "refusing to load a partial cache"
            ) from exc
        cache._entries = entries
        return cache
