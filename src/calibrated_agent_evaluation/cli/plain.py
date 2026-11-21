# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Plain-language summaries for the stages whose results are statistical.

Each function turns a stage's result into a few lines an engineer can act on: what happened, in
terms of the agent and its labels, and the next command. The statistical account (bounds, methods,
thresholds) stays in the stage's ``detail``, shown under ``--explain``. Percentages are rounded to
whole numbers on purpose; the exact values are in the detail and the JSON records.
"""

from __future__ import annotations

import re
import shlex
from collections.abc import Iterable, Mapping, Sequence
from typing import Any

from calibrated_agent_evaluation.cli.present import ErrorExplainer, PlainError

KNOWN_LIMIT = (
    "Known limit: below 150 scored traces, or when a class shows about 2 to 3 judge errors, the "
    "rate's range was measured to contain the true rate less often than 95% of the time."
)

# What each bias in the battery looks like from the outside.
_BIAS_PHRASES: Mapping[str, str] = {
    "length": "changes its verdict when answers get longer",
    "sycophancy": "goes along with claims asserted in the conversation",
    "self_preference": "favors answers written by its own model",
    "compassion_fade": "judges differently depending on how the people involved are described",
    "positional": "favors whichever answer comes first",
}


def skipped(scan: Any) -> list[str]:
    """One plain line per reason traces were left out of a stage's scan; empty when none were."""
    counts: dict[str, int] = {}
    for item in getattr(scan, "skipped", ()):
        counts[item.reason] = counts.get(item.reason, 0) + 1
    return [
        f"Left out {n} stored trace(s) ({reason}); --explain names them."
        for reason, n in counts.items()
    ]


def points(value: float) -> str:
    """An absolute difference between two rates, as percentage points: ``10 percentage points``."""
    shown = f"{value * 100:.1f}".rstrip("0").rstrip(".")
    return f"{shown} percentage point" + ("" if shown == "1" else "s")


def pct(value: float) -> str:
    """A rate as a percentage: whole numbers, with one decimal near 0% and 100%.

    ``11%``, but ``99.7%`` and ``0.4%``: rounding 0.997 to ``100%`` would claim a judge never
    missed, and rounding 0.004 to ``0%`` would claim it never happens.
    """
    whole = f"{value:.0%}"
    if value in (0.0, 1.0) or whole not in ("0%", "1%", "99%", "100%"):
        return whole
    for places in (1, 2):
        shown = f"{value:.{places}%}"
        if float(shown.rstrip("%")) not in (0.0, 100.0):
            return shown
    return f"{value:.3%}"


def rate_range(point: float | None, lo: float, hi: float) -> str:
    """``11% (likely between 6% and 15%)``, or the range alone when the point is refused."""
    if point is None:
        return f"somewhere between {pct(lo)} and {pct(hi)}"
    return f"{pct(point)} (likely between {pct(lo)} and {pct(hi)})"


def _more(n_have: int, n_need: int, cls: str) -> str | None:
    return f"{n_need - n_have} more {cls}" if n_have < n_need else None


def _short_by(n_fail: int, n_pass: int, per_class: int) -> list[str]:
    return [s for s in (_more(n_fail, per_class, "FAIL"), _more(n_pass, per_class, "PASS")) if s]


_SYNTHETIC_SOURCES: Mapping[str, str] = {"synthetic-rule-oracle": "from the sample policy rule"}


def gold_note(rater_ids: Sequence[str], noun: str = "gold label(s)") -> str:
    """``324 gold labels``, saying so plainly when the gold is synthetic rather than human."""
    n = len(rater_ids)
    synthetic = [r for r in rater_ids if r.startswith("synthetic")]
    base = f"{n} {noun}"
    if not synthetic:
        return base
    sources = sorted({_SYNTHETIC_SOURCES.get(r, f"rater {r!r}") for r in synthetic})
    if len(synthetic) == n:
        return f"{base} (synthetic, {', '.join(sources)})"
    return f"{base} ({len(synthetic)} of them synthetic, {', '.join(sources)})"


def labeled_note(rater_ids: Sequence[str], noun: str) -> str:
    """``7 human FAIL label(s)`` for human labels; synthetic ones are named as such instead."""
    if any(r.startswith("synthetic") for r in rater_ids):
        return gold_note(rater_ids, noun)
    return f"{len(rater_ids)} human {noun}"


#: Words a "Next:" command leaves for the reader to fill in, so they are never quoted.
_PLACEHOLDERS = frozenset({"DIR", "FILE"})


def command(*words: str, root: str | None = None, cassette: str | None = None) -> str:
    """A ``calibrated-agent-evaluation`` command line that pastes into a POSIX shell as is.

    ``root`` and ``cassette`` carry the store and the judge-reply cassette of the command that
    printed it, so the next step reads the same store and, offline, the same replies.
    """
    parts = ["calibrated-agent-evaluation"]
    if root:
        parts += ["--root", root]
    parts += words
    if cassette:
        parts += ["--cassette", cassette]
    return " ".join(w if w in _PLACEHOLDERS else shlex.quote(w) for w in parts)


def carried_flags(args: Any) -> dict[str, str | None]:
    """The ``--root`` and ``--cassette`` a printed next command repeats, from the command line."""
    root = getattr(args, "root", None)
    cassette = getattr(args, "cassette", None)
    return {"root": str(root) if root else None, "cassette": str(cassette) if cassette else None}


def calibration(
    record: Any,
    *,
    plan_per_class: int,
    judge_path: str,
    gold: str,
    root: str | None = None,
    cassette: str | None = None,
) -> list[str]:
    """The plain result of ``calibrate``; ``gold`` is :func:`gold_note`'s description."""
    caught, cleared = pct(record.tpr.point), pct(record.tnr.point)
    if record.result != "CERTIFIED":
        short = _short_by(record.n_pos, record.n_neg, plan_per_class)
        if short:
            return [
                f"Not enough labels yet to trust this judge (checked on {gold}).",
                f"Next: label {' and '.join(short)} traces, then run `calibrated-agent-evaluation "
                "calibrate` again.",
            ]
        return [
            f"Judge not certified on {gold}: it disagrees with them too often (it caught "
            f"{caught} of the failures and cleared {cleared} of the passes).",
            "Next: revise the judge prompt, then certify the new version on freshly labeled "
            "traces.",
        ]
    lines = [
        f"Judge certified on {gold}: it agrees with them well enough to trust "
        f"(it caught {caught} of the failures and cleared {cleared} of the passes)."
    ]
    if record.below_coverage_floor:
        short = _short_by(record.n_pos, record.n_neg, record.coverage_floor)
        more = f"label {' and '.join(short)} traces and recalibrate" if short else "recalibrate"
        lines.append(
            f"It can score traces, but not yet estimate a failure rate: to get a rate, {more}."
        )
    bias_cmd = command(
        *("bias", "--judge", judge_path, "--probes", "DIR", "--thresholds", "FILE"),
        root=root,
        cassette=cassette,
    )
    lines.append(f"Next: check it for bias with `{bias_cmd}`.")
    return lines


def bias(
    record: Any, *, judge_path: str, root: str | None = None, cassette: str | None = None
) -> list[str]:
    """The plain result of ``calibrated-agent-evaluation bias``."""
    if record.verdict == "CLEAR":
        must = [r for r in record.results if r.is_must_pass]
        ran = [r for r in must if r.status == "run"]
        skipped = [r.bias for r in must if r.status != "run"]
        if skipped:
            head = (
                f"Bias check passed: the judge's verdicts held steady on the {len(ran)} must-pass "
                f"check(s) that ran. {_join(skipped)} didn't run (not applicable to this judge)."
            )
        else:
            head = (
                "Bias check passed: the judge's verdicts held steady across every must-pass check."
            )
        return [
            head,
            "Next: score traces with "
            f"`{command('judge', '--judge', judge_path, root=root, cassette=cassette)}`.",
        ]
    by_bias = {r.bias: r for r in getattr(record, "results", ())}
    seen = [b for b in record.failed_biases if _observed(by_bias.get(b))]
    unseen = [b for b in record.failed_biases if not _observed(by_bias.get(b))]
    lines: list[str] = []
    if seen:
        phrases = [_BIAS_PHRASES.get(b, f"fails the {b} check") for b in seen]
        lines.append(f"Bias check failed: the judge {_join(phrases)}.")
    if unseen:
        lead = "It also" if seen else "Bias check failed: the judge showed no bias, but"
        lines.append(
            f"{lead} has too few {_join(unseen)} trials to rule the bias out "
            "(add trials to those probes and run `calibrated-agent-evaluation bias` again)."
        )
    if seen:
        lines.append(
            "Next: revise the judge prompt to address this, then run `calibrated-agent-evaluation "
            "calibrate` and "
            "`calibrated-agent-evaluation bias` on the new version (with fresh labels)."
        )
    return lines


def _observed(result: Any) -> bool:
    """Whether a failed bias showed any excess, as opposed to failing on too few trials.

    A result with no counts to read (``None``) is treated as observed, the stricter wording.
    """
    if result is None:
        return True
    return bool(result.magnitude > 0 or (result.direction_magnitude or 0) > 0)


def judge(
    *,
    n_fail: int,
    n_pass: int,
    n_traces: int,
    mode: str,
    judge_path: str,
    root: str | None = None,
) -> list[str]:
    """The plain result of ``judge``: verdicts written, and how many traces they cover."""
    n = n_fail + n_pass
    where = f"{n_traces} trace(s)" if n_traces == n else f"spans of {n_traces} trace(s)"
    return [
        f"Wrote {n} verdict(s) for {mode} on {where}: {n_fail} FAIL, {n_pass} PASS.",
        "Next: turn these into a failure rate with "
        f"`{command('run', '--judge', judge_path, root=root)}`.",
    ]


def run(
    gate: Any,
    trend: Sequence[Any],
    mode_name: Any,
    *,
    threshold: float,
    ci: bool,
    notes: Sequence[str] = (),
) -> list[str]:
    """The plain result of ``run``: the gate, each corrected rate, and what was left out.

    ``notes`` are the plain lines for every rate not estimated and every verdict left out, so a
    gate that ran on uncorrected verdicts says so without ``--explain``.
    """
    if gate.passed:
        lines = [f"CI gate passed: all {gate.n_modes} failure mode(s) clear the bar."]
    else:
        failing = ", ".join(mode_name(m) for m in gate.failing_modes)
        lines = [
            f"CI gate failed: {len(gate.failing_modes)} of {gate.n_modes} failure mode(s) don't "
            f"clear the bar yet: {failing}.",
        ]
    lines.append(
        f"  (The bar: at least {pct(threshold)} of each mode's traces must pass, even at the "
        "cautious end of what this many traces can show.)"
    )
    for tp in trend:
        cr = tp.corrected_rate
        name = mode_name(tp.mode)
        if cr.point is None:
            lines.append(f"{name} failure rate: not estimated ({_refusal(cr.refusal_reason)}).")
        else:
            lines.append(f"{name} failure rate: {rate_range(cr.point, cr.lo, cr.hi)}.")
    lines += notes
    if not gate.passed:
        lines.append(
            "Next: fix the failing modes, or label more traces where a mode has too few to show "
            "its pass rate."
        )
        if not ci:
            lines.append("(Without --ci this run is informational and exits 0.)")
    return lines


def improve(verdict: Any) -> list[str]:
    """The plain result of ``calibrated-agent-evaluation improve``."""
    need = points(verdict.prereg.success_threshold)
    if verdict.verdict == "accepted":
        return [
            f"Fix accepted: the failure rate dropped by at least the {need} you pre-registered, "
            "and nothing it was checked against got worse."
        ]
    relocation = getattr(verdict.panel, "relocation", None)
    if relocation is not None and relocation.verdict == "relocation":
        return [
            "Fix not accepted: the agent you changed fails less, but the crew as a whole doesn't: "
            "the failures moved to another step instead of going away.",
            "Next: check where the failures went before shipping the change.",
        ]
    if relocation is not None and relocation.verdict == "regressed":
        return [
            "Fix not accepted: the crew's overall failure rate got worse.",
            "Next: look at the crew's outcomes before shipping the change.",
        ]
    regressed = list(verdict.panel.regressed)
    if regressed:
        return [
            f"Fix not accepted: it made {_join(regressed)} worse.",
            "Next: look at those slices before shipping the change.",
        ]
    return [
        f"Fix not accepted: the evidence doesn't show the failure rate dropping by the {need} you "
        "pre-registered.",
        "Next: if the fix looks right, measure it on more traces; the threshold stays as "
        "registered.",
    ]


def improve_order_unchecked(why: str) -> str:
    """The summary line for an evaluation that could not check when the target was frozen."""
    return f"Not checked: whether the target was registered before the after run began ({why})."


def cluster(
    named: Mapping[str, int], saturation: Any, *, n_blind: int | None, n_cells: int | None
) -> list[str]:
    """The plain result of ``calibrated-agent-evaluation cluster``."""
    if not named:
        return ["No failure modes named yet: name and ratify modes on your FAIL labels first."]
    total = sum(named.values())
    modes = ", ".join(f"{m} ({n})" for m, n in sorted(named.items(), key=lambda kv: -kv[1]))
    lines = [f"Found {len(named)} failure mode(s) in {total} labeled failure(s): {modes}."]
    if saturation is not None and saturation.saturated:
        lines.append("You have probably seen most of the failure modes this agent has.")
    elif saturation is not None:
        lines.append(
            "There are probably failure modes you haven't seen yet: keep labeling failures."
        )
    if n_blind:
        lines.append(
            f"{n_blind} of {n_cells} scenario combination(s) have no labeled failures yet; "
            "--explain lists them."
        )
    return lines


def _refusal(reason: str | None) -> str:
    text = (reason or "").lower()
    if "youden" in text or "weak" in text:
        return "the judge isn't accurate enough to correct its rate"
    if "identifiable" in text:
        return "the judge's error pattern makes the rate impossible to pin down here"
    if "coverage floor" in text or "rate-unsafe" in text:
        return "the judge needs more labels before it can estimate a rate"
    if "rate sample" in text or "traces" in text:
        more = re.search(r"score (\d+) more", text)
        if more:
            return (
                f"too few scored traces; next, score {more.group(1)} more production trace(s) "
                "with `calibrated-agent-evaluation judge`"
            )
        return "too few scored traces"
    return "see --explain"


def _join(items: Iterable[str]) -> str:
    items = list(items)
    if len(items) <= 1:
        return "".join(items)
    return ", ".join(items[:-1]) + " and " + items[-1]


def _const(headline: str, next_step: str | None = None) -> ErrorExplainer:
    return lambda _exc: PlainError(headline, next_step)


def _missing_extra(exc: BaseException) -> PlainError:
    extras = getattr(exc, "extras", ())
    names = ", ".join(f"[{e}]" for e in extras) or "an optional extra"
    return PlainError(
        f"This needs the {names} extra, which isn't installed.", getattr(exc, "hint", None)
    )


def _rate_sample_too_small(exc: BaseException) -> PlainError:
    n, minimum = getattr(exc, "n", None), getattr(exc, "minimum", None)
    if n is None or minimum is None:
        return PlainError("Too few scored traces to estimate a failure rate.")
    return PlainError(
        f"Too few scored traces to estimate a failure rate: {n} scored, {minimum} needed.",
        f"score {minimum - n} more production trace(s) with `calibrated-agent-evaluation judge`, "
        "then run again.",
    )


def _judge_not_certified(exc: BaseException) -> PlainError:
    reason = getattr(exc, "reason", None)
    path = getattr(exc, "judge_path", "FILE")
    headline = "This judge isn't certified yet" + (f": {reason}." if reason else ".")
    calibrate_cmd = command("calibrate", "--judge", path, root=getattr(exc, "root", None))
    return PlainError(headline, f"run `{calibrate_cmd}` first.")


def _judge_not_admitted(exc: BaseException) -> PlainError:
    refusals = getattr(exc, "refusals", ())
    which = f": {'; '.join(refusals)}" if refusals else ""
    return PlainError(
        "These judge verdicts came from a judge that hasn't passed its check against your labels "
        f"and its bias check, so they can't gate the build{which}.",
        "calibrate and bias-check the judge, then score the traces with "
        "`calibrated-agent-evaluation judge`; or gate your human labels with --verdicts human.",
    )


def _gold_set_reused(exc: BaseException) -> PlainError:
    prior = getattr(exc, "prior_result", None)
    outcome = {"CERTIFIED": "it certified that judge", "REFUSED": "it refused that judge"}
    how = f" ({outcome[prior]})" if prior in outcome else ""
    overlap = getattr(exc, "n_overlap", None)
    which = "These labels" if not overlap else f"{overlap} of these labels"
    return PlainError(
        f"{which} were already used to test a judge for this failure mode{how}, and a set of "
        "labels tests a judge only once.",
        "label fresh traces for this judge: re-testing on labels a judge was already checked on "
        "hides overfitting.",
    )


def _said(next_step: str) -> ErrorExplainer:
    """Keep the error's own message as the headline, since it names the dataset or field."""

    def explain(exc: BaseException) -> PlainError:
        text = str(exc)
        return PlainError(text[:1].upper() + text[1:], next_step)

    return explain


def _dataset_input(exc: BaseException) -> PlainError:
    if "version" in str(exc):
        step = (
            "pull a published version number, such as the one "
            "`calibrated-agent-evaluation dataset publish` printed."
        )
    else:
        step = "pass a --name of letters, digits and underscores, such as orderbot_regressions."
    return _said(step)(exc)


def default_error_explainers() -> dict[str, ErrorExplainer]:
    """Plain phrasings for the refusals an engineer is most likely to hit."""
    return {
        "MissingExtraError": _missing_extra,
        "AwsCredentialsMissing": _const(
            "No AWS credentials were found for this command.",
            "set AWS_PROFILE to a profile that can reach the account, or pass --profile.",
        ),
        "DatasetInputError": _dataset_input,
        "DatasetVersionNotFound": _said(
            "list the dataset's versions (`aws bedrock-agentcore-control list-dataset-versions "
            "--dataset-id ID`) and pull one that exists."
        ),
        "DatasetSchemaUnsupported": _said(
            "pull a predefined or simulated dataset; this one can't be read without losing fields."
        ),
        "DatasetDecodeError": _said(
            "remove or rename the field in AgentCore (free-form data belongs under metadata), then "
            "pull again."
        ),
        "DatasetNameConflict": _said(
            "publish under another --name, or pull that dataset to start from it."
        ),
        "DatasetPublishError": _said(
            "fix the cause above and publish again; published versions are never changed."
        ),
        "RateSampleTooSmallError": _rate_sample_too_small,
        "RateUnsafeError": _const(
            "The judge is certified, but it was checked on too few labels to estimate a "
            "failure rate.",
            "label more traces per class (the number is in `calibrated-agent-evaluation calibrate "
            "--explain`) and "
            "recalibrate.",
        ),
        "UncalibratedJudgeError": _const(
            "This judge isn't certified: either it was never checked against your labels, or the "
            "check refused it.",
            "run `calibrated-agent-evaluation calibrate --judge FILE`; its output says what to "
            "fix.",
        ),
        "WeakJudgeError": _const(
            "The judge is too close to guessing for its failure rate to be corrected.",
            "revise the judge prompt and certify the new version.",
        ),
        "NonIdentifiableRateError": _const(
            "The judge's error pattern makes the true failure rate impossible to pin down from "
            "this sample.",
            "certify a more accurate judge, or label the sample by hand.",
        ),
        "JudgeNotCertifiedError": _judge_not_certified,
        "JudgeNotAdmittedError": _judge_not_admitted,
        "LikertNotAllowedError": _const(
            "The judge file scores on a Likert (1-5 style) scale, and Calibrated Agent Evaluation "
            "judges give one "
            "PASS/FAIL verdict with a critique.",
            "rewrite the scale as one binary question per failure mode, one judge file each.",
        ),
        "MultiAxisLikertError": _const(
            "The judge file scores several things on Likert (1-5 style) scales in one call.",
            "split it into one binary PASS/FAIL judge file per thing it checks.",
        ),
        "WeightedCollapseError": _const(
            "The judge file weights checklist items into a score, which brings back 1-5 style "
            "scoring under another name.",
            "drop the weights and write one binary judge file per item that decides the verdict.",
        ),
        "CritiqueOrderError": _const(
            "The judge file asks for the verdict before the critique, or leaves the critique out.",
            'list output_fields as ["critique", "verdict"], so the judge reasons before it '
            "decides.",
        ),
        "GoldSetReusedError": _gold_set_reused,
        "LockedTestReuseError": _const(
            "These labels were already used once to certify this judge.",
            "label fresh traces for the new version: re-testing on the same labels after a "
            "change hides overfitting.",
        ),
        "SampleMisuseError": _const(
            "The traces used to check the judge can't also measure the failure rate.",
            "score a separate random sample of production traces.",
        ),
        "FewShotEvaluationError": _const(
            "The judge's own prompt examples can't be used to test it.",
            "certify it on traces that aren't in its prompt.",
        ),
        "UnratifiedGoldError": _const(
            "A judge can only be checked against human labels, and this set has a label no "
            "human confirmed.",
            "have a person label or ratify every trace in the set.",
        ),
        "BiasedJudgeError": _const(
            "This judge failed its bias check, so it can't score traces.",
            "revise the judge prompt, then calibrate and run `calibrated-agent-evaluation bias` "
            "on the new version.",
        ),
        "MissingBiasBatteryError": _const(
            "This judge hasn't been checked for bias yet.",
            "run `calibrated-agent-evaluation bias --judge FILE --probes DIR --thresholds FILE`.",
        ),
        "StaleBiasBatteryError": _const(
            "The bias check on record is for a different version of this judge.",
            "run `calibrated-agent-evaluation bias` again for the current version.",
        ),
        "InfeasiblePlanError": _const(
            "No number of labels can certify a judge at the accuracy you expect.",
            "plan for a higher accuracy (`--expected-accuracy`) or improve the judge first.",
        ),
        "AuthorizationError": _const(
            "Probing an endpoint needs your confirmation that you're allowed to.",
            "pass --authorized only if you are allowed to probe this endpoint.",
        ),
    }
