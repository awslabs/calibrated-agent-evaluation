# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""The provenance stamp every persisted record carries (reconstructability).

A stored interval can only be recomputed bit-for-bit if the record says what produced it: the seed
and resample count of the bootstrap, the gate thresholds in force, the Calibrated Agent Evaluation
version, the git commit when run from a checkout, and the exact numpy / scipy / statsmodels
versions, because ``scipy.stats.bootstrap``'s stream is not stable across the ``pyproject.toml``
ranges.

:func:`capture_provenance` reads the environment once; :class:`ProvenanceStamp` is the frozen,
JSON-round-trippable value the records hold. It needs no optional extra, so every record module
can import it.
"""

from __future__ import annotations

import functools
import platform
import subprocess  # nosec B404 - fixed argv, used only to read the checkout's commit
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any, ClassVar, Self

from pydantic import ConfigDict, Field, model_validator

from calibrated_agent_evaluation._model import CalibratedAgentEvaluationModel
from calibrated_agent_evaluation.schema.errors import CalibratedAgentEvaluationDataError

# The numeric dependencies whose versions decide the digits of a stored statistic.
STATS_DISTRIBUTIONS: tuple[str, ...] = ("numpy", "scipy", "statsmodels")


class ProvenanceError(CalibratedAgentEvaluationDataError):
    """A provenance stamp is malformed (bad timestamp, wrong-typed field, missing key)."""


class ProvenanceStamp(CalibratedAgentEvaluationModel):
    """What produced a record's numbers.

    ``run_at`` is a timezone-aware UTC ISO-8601 timestamp. ``git_sha`` is ``None`` unless Calibrated
    Agent Evaluation is imported from a git checkout (an installed wheel has no commit to report);
    it ends in ``-dirty`` when the checkout had uncommitted changes to tracked files, since the
    commit alone did not produce those numbers. ``dep_versions``
    maps each of :data:`STATS_DISTRIBUTIONS`, and any further distribution the caller named, to
    its installed version, ``None`` when absent.
    ``seed`` / ``n_resamples`` are ``None`` for a record whose computation draws no random numbers.
    ``thresholds`` names every gate threshold applied (e.g. ``{"gate": 0.80}``).
    ``sampling`` names the sampling parameters sent to the judge model (e.g.
    ``{"temperature": 0.0}``); empty means none were sent and the provider's defaults applied.
    """

    model_config = ConfigDict(frozen=True, extra="forbid", strict=True)
    invalid_error: ClassVar[type[Exception]] = ProvenanceError

    run_at: str
    calibrated_agent_evaluation_version: str
    git_sha: str | None
    python_version: str
    dep_versions: dict[str, str | None]
    seed: int | None = None
    n_resamples: int | None = None
    thresholds: dict[str, float] = Field(default_factory=dict)
    sampling: dict[str, float] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _check_invariants(self) -> Self:
        self._post_init()
        return self

    def _post_init(self) -> None:
        try:
            parsed = datetime.fromisoformat(self.run_at)
        except (TypeError, ValueError) as exc:
            raise ProvenanceError(f"run_at {self.run_at!r} is not ISO-8601") from exc
        if parsed.utcoffset() is None:
            raise ProvenanceError(f"run_at {self.run_at!r} has no timezone; stamps are UTC-aware")
        for name, value in (("seed", self.seed), ("n_resamples", self.n_resamples)):
            if value is not None and (isinstance(value, bool) or not isinstance(value, int)):
                raise ProvenanceError(f"{name} must be an int or None, got {value!r}")
        if self.n_resamples is not None and self.n_resamples < 1:
            raise ProvenanceError(f"n_resamples must be positive, got {self.n_resamples}")
        for kind, values in (("threshold", self.thresholds), ("sampling", self.sampling)):
            for key, value in values.items():
                if isinstance(value, bool) or not isinstance(value, int | float):
                    raise ProvenanceError(f"{kind} {key!r} must be a number, got {value!r}")

    def to_dict(self) -> dict[str, Any]:
        """A plain-JSON dict (the store codecs embed it as one field)."""
        return {
            "run_at": self.run_at,
            "calibrated_agent_evaluation_version": self.calibrated_agent_evaluation_version,
            "git_sha": self.git_sha,
            "python_version": self.python_version,
            "dep_versions": dict(self.dep_versions),
            "seed": self.seed,
            "n_resamples": self.n_resamples,
            "thresholds": dict(self.thresholds),
            "sampling": dict(self.sampling),
        }

    @classmethod
    def from_dict(cls, doc: Mapping[str, Any]) -> ProvenanceStamp:
        """Rebuild a stamp from :meth:`to_dict` output; raises :class:`ProvenanceError` if bad."""
        try:
            return cls(
                run_at=doc["run_at"],
                calibrated_agent_evaluation_version=doc["calibrated_agent_evaluation_version"],
                git_sha=doc["git_sha"],
                python_version=doc["python_version"],
                dep_versions=dict(doc["dep_versions"]),
                seed=doc["seed"],
                n_resamples=doc["n_resamples"],
                thresholds={k: float(v) for k, v in dict(doc["thresholds"]).items()},
                sampling={k: float(v) for k, v in dict(doc.get("sampling", {})).items()},
            )
        except (KeyError, TypeError, ValueError, AttributeError) as exc:
            raise ProvenanceError(f"malformed provenance stamp: missing/mistyped {exc!r}") from exc


def _dep_versions(extra: Sequence[str]) -> dict[str, str | None]:
    out: dict[str, str | None] = {}
    for dist in (*STATS_DISTRIBUTIONS, *extra):
        try:
            out[dist] = version(dist)
        except PackageNotFoundError:
            out[dist] = None
    return out


def _calibrated_agent_evaluation_version() -> str:
    try:
        return version("calibrated-agent-evaluation")
    except PackageNotFoundError:
        return "0+unknown"


@functools.cache
def _checkout_git_sha() -> str | None:
    """The commit of the source checkout, or ``None`` for an installed wheel.

    Only a ``src/calibrated_agent_evaluation`` layout with a ``.git`` beside ``src`` counts, so a
    wheel installed into a virtualenv that happens to sit inside some other repository never reports
    that repository. Read once per process: a commit made while a process runs does not change the
    code it loaded.

    A tree with uncommitted changes to tracked files reports ``<sha>-dirty``: HEAD alone claimed a
    commit that did not produce the record, which undercuts recomputing it bit for bit. Untracked
    files are not counted; they are not code the package imported.
    """
    package_dir = Path(__file__).resolve().parent
    repo = package_dir.parent.parent
    if package_dir.parent.name != "src" or not (repo / ".git").exists():
        return None
    return _commit_of(repo)


def _commit_of(repo: Path) -> str | None:
    """HEAD of ``repo``, suffixed ``-dirty`` when tracked files have uncommitted changes."""
    sha = _git(repo, "rev-parse", "HEAD")
    if not sha:
        return None
    status = _git(repo, "status", "--porcelain", "--untracked-files=no")
    if status is None:
        return None  # whether the tree was clean is unknown, so is what produced the record
    return f"{sha}-dirty" if status else sha


def _git(repo: Path, *argv: str) -> str | None:
    """Run git in ``repo`` and return its stripped stdout; ``None`` when git is missing or fails."""
    try:
        out = subprocess.run(  # noqa: S603  # nosec B603 B607 - fixed argv, no shell, bounded
            ["git", "-C", str(repo), *argv],  # noqa: S607
            capture_output=True,
            text=True,
            check=True,
            timeout=5,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return out.stdout.strip()


def capture_provenance(
    *,
    seed: int | None = None,
    n_resamples: int | None = None,
    thresholds: Mapping[str, float] | None = None,
    sampling: Mapping[str, float] | None = None,
    now: datetime | None = None,
    extra_distributions: Sequence[str] = (),
) -> ProvenanceStamp:
    """Stamp the current environment for a computation with the given seed/resamples/thresholds.

    Args:
        seed: The random seed the computation used, or ``None`` when it draws no random numbers.
        n_resamples: The bootstrap resample count, or ``None`` when there was no bootstrap.
        thresholds: Every gate threshold applied, by name.
        sampling: The sampling parameters sent to the judge model, by name.
        now: The stamp's time, for tests; it must be timezone-aware. Defaults to the UTC clock.
        extra_distributions: Distributions whose installed versions also decide the record, such
            as the vendor package behind a detector's output. Each is recorded in
            ``dep_versions`` beside :data:`STATS_DISTRIBUTIONS`, ``None`` when it is not
            installed.

    Returns:
        The frozen stamp.

    Raises:
        ProvenanceError: ``now`` is naive.
    """
    when = now or datetime.now(tz=UTC)
    if when.utcoffset() is None:
        raise ProvenanceError("capture_provenance: now must be timezone-aware")
    return ProvenanceStamp(
        run_at=when.astimezone(UTC).isoformat(timespec="seconds"),
        calibrated_agent_evaluation_version=_calibrated_agent_evaluation_version(),
        git_sha=_checkout_git_sha(),
        python_version=platform.python_version(),
        dep_versions=_dep_versions(extra_distributions),
        seed=seed,
        n_resamples=n_resamples,
        thresholds=dict(thresholds or {}),
        sampling=dict(sampling or {}),
    )


# A fixed stamp for the header-only sentinel records the stores build to name CSV columns. Never
# attached to a record that holds numbers.
HEADER_ONLY_STAMP = ProvenanceStamp(
    run_at="1970-01-01T00:00:00+00:00",
    calibrated_agent_evaluation_version="",
    git_sha=None,
    python_version="",
    dep_versions={},
)
