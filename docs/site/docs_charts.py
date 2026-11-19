# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""The docs site's data charts, rendered to SVG from committed study data and the library itself.

Each chart is drawn twice, once per theme, so a page can show the right one with Material's
``#only-light`` / ``#only-dark`` image suffixes. Inputs are the committed series under
``docs/site/data/``, the coverage study under ``fixtures/expected/coverage/`` and the
synthetic 90/90 calibration fixture, so a chart moves only when its data does. Output is
deterministic: no timestamps in the metadata and a fixed SVG id salt.

Colors are the brand's state colors: verdigris for the corrected estimate, fail red for the naive
number, sepia for a refusal. Every series also carries a direct label.
"""

from __future__ import annotations

import functools
import io
import json
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import matplotlib as mpl

mpl.use("svg")  # before pyplot is imported, so no GUI backend is probed

import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.ticker import PercentFormatter

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
_DATA = REPO_ROOT / "docs" / "site" / "data"
_COVERAGE = REPO_ROOT / "fixtures" / "expected" / "coverage" / "rg_interval_coverage.json"
_SYNTHETIC = REPO_ROOT / "fixtures" / "calibration" / "synthetic_9090.json"
GATE = 0.80


@dataclass(frozen=True)
class Theme:
    name: str
    text: str
    muted: str
    grid: str
    reference: str
    band: str
    series: tuple[str, str, str]


# Series slots: the corrected estimate (verdigris), the naive or raw number (fail red), and a
# refused or insufficient state (sepia), following the brand's state colors.
LIGHT = Theme(
    "light",
    "#1e2440",
    "#4f5571",
    "#e4e0d6",
    "#8a8578",
    "#ece6da",
    ("#287a70", "#c0392b", "#776c59"),
)
DARK = Theme(
    "dark",
    "#f5f0e6",
    "#c4c0b6",
    "#343a55",
    "#8f93a6",
    "#2a3049",
    ("#5fb3a6", "#e06a5c", "#a89c86"),
)


def _read_dat(name: str) -> tuple[list[str], list[list[float]]]:
    lines = [ln for ln in (_DATA / name).read_text().splitlines() if not ln.startswith("#")]
    header, *rows = lines
    return header.split(), [[float(v) for v in row.split()] for row in rows]


def _column(name: str, column: str) -> tuple[list[float], list[float]]:
    header, rows = _read_dat(name)
    i = header.index(column)
    return [r[0] for r in rows], [r[i] for r in rows]


def _style(ax: Axes, theme: Theme) -> None:
    ax.set_facecolor("none")
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(theme.grid)
    ax.tick_params(colors=theme.muted, labelsize=9)
    ax.xaxis.label.set_color(theme.muted)
    ax.yaxis.label.set_color(theme.muted)
    ax.title.set_color(theme.text)
    ax.grid(color=theme.grid, linewidth=0.6)
    ax.set_axisbelow(True)


def _figure(theme: Theme, ncols: int) -> tuple[Figure, list[Axes]]:
    fig, axes = plt.subplots(1, ncols, figsize=(4.2 * ncols, 3.4), layout="constrained")
    fig.patch.set_alpha(0)
    axes_list = list(axes) if ncols > 1 else [axes]
    for ax in axes_list:
        _style(ax, theme)
    return fig, axes_list


def _gate_line(ax: Axes, theme: Theme, *, horizontal: bool = True) -> None:
    line = ax.axhline if horizontal else ax.axvline
    line(GATE, color=theme.reference, linewidth=1.2, linestyle=(0, (4, 3)))


def calibration_size(theme: Theme) -> Figure:
    """The gate's lower bound against labels per class, and the labels needed by accuracy."""
    # Both panels come from the library, with the count rounded down the way the planner rounds
    # it, so the guarantee line on the left is the 90% point on the right.
    import math

    from calibrated_agent_evaluation.calibrate import wilson_calibration_size
    from calibrated_agent_evaluation.stats import wilson_lower

    fig, (left, right) = _figure(theme, 2)
    honest = theme.series[0]

    n = list(range(5, 121))
    lb = [wilson_lower(math.floor(0.9 * k + 1e-9), k, alpha=0.10) for k in n]
    left.plot(n, lb, color=honest, linestyle="none", marker="o", markersize=3)
    _gate_line(left, theme)
    left.text(n[-1], GATE - 0.012, "gate 0.80", color=theme.muted, fontsize=9, ha="right", va="top")
    planned = wilson_calibration_size(0.90, GATE)
    first = next(k for k, b in zip(n, lb, strict=True) if b >= GATE)
    left.axvline(planned, color=theme.muted, linewidth=1.0)
    left.annotate(
        f"first clears at {first}",
        (first, lb[n.index(first)]),
        textcoords="offset points",
        xytext=(-12, 16),
        ha="right",
        color=theme.text,
        fontsize=9,
        arrowprops={"arrowstyle": "-", "color": theme.muted, "linewidth": 0.8},
    )
    left.text(
        planned + 2,
        0.53,
        f"from {planned} on,\nevery size clears",
        color=theme.text,
        fontsize=9,
        ha="left",
        va="bottom",
    )
    left.set(
        title="A judge right about 90% of the time",
        xlabel="gold labels per class",
        ylabel="agreement, at the cautious end",
        ylim=(0.5, 0.9),
    )

    accuracy = [round(0.99 - 0.01 * i, 2) for i in range(15)]
    needed = [wilson_calibration_size(a, GATE) for a in accuracy]
    right.plot(accuracy, needed, color=honest, linewidth=2, marker="o", markersize=4)
    for target in (0.95, 0.90, 0.85):
        at = needed[accuracy.index(target)]
        right.annotate(
            f"{at}",
            (target, at),
            textcoords="offset points",
            xytext=(0, 9),
            ha="center",
            color=theme.text,
            fontsize=10,
            fontweight="bold",
        )
    right.invert_xaxis()
    right.set(
        title="Labels needed to clear the gate",
        xlabel="expected judge accuracy",
        ylabel="gold labels per class",
        ylim=(0, max(needed) * 1.12),
    )
    return fig


@dataclass(frozen=True)
class _Judge:
    judge_id: str
    model: str
    prompt_version: str


@functools.cache  # a locked test certifies once per process, and both themes need the numbers
def _synthetic_rate() -> tuple[float, float, float, float]:
    """Raw flag rate and corrected rate (point, lo, hi) for the synthetic 90/90 judge."""
    from calibrated_agent_evaluation.calibrate import (
        CalibrationSet,
        InMemoryLockedTestLedger,
        LabelPair,
        certify,
        corrected_rate,
    )

    ledger = InMemoryLockedTestLedger()
    from calibrated_agent_evaluation.schema import Label, Outcome

    doc = json.loads(_SYNTHETIC.read_text())

    def pairs(verdicts: list[int], gold: Outcome) -> list[LabelPair]:
        def verdict(v: int) -> Outcome:
            return "FAIL" if v else "PASS"

        return [
            LabelPair(
                gold=Label(
                    trace_id=f"{gold}-{i}",
                    span_id=None,
                    kind="human",
                    outcome=gold,
                    critique="gold",
                ),
                verdict=Label(
                    trace_id=f"{gold}-{i}",
                    span_id=None,
                    kind="llm-judge",
                    outcome=verdict(v),
                    critique="v",
                ),
            )
            for i, v in enumerate(verdicts)
        ]

    hunted = doc["hunted_set"]
    locked = CalibrationSet(
        tuple(pairs(hunted["cal_pos"], "FAIL") + pairs(hunted["cal_neg"], "PASS")),
        split="locked_test",
        set_id=f"{hunted['set_id']}-docs-chart",
    )
    record = certify(_Judge(**doc["judge"]), locked, ledger=ledger)
    verdicts = doc["rate_sample"]["verdicts"]
    rate = corrected_rate(
        record,
        verdicts,
        rate_set_id=doc["rate_sample"]["set_id"],
        rate_trace_ids=[f"prod-{i}" for i in range(len(verdicts))],
    )
    interval = rate.corrected_rate
    if interval is None or interval.point is None:
        raise SystemExit("docs_charts: the synthetic 90/90 judge produced no corrected rate")
    return sum(verdicts) / len(verdicts), interval.point, interval.lo, interval.hi


def rate_correction(theme: Theme) -> Figure:
    """What a 90/90 judge reports against the true rate, and the corrected estimate."""
    fig, (left, right) = _figure(theme, 2)
    honest, naive = theme.series[0], theme.series[1]
    true_rate = json.loads(_SYNTHETIC.read_text())["params"]["true_prevalence"]

    p, reported = _column("prevalence.dat", "q90")
    left.plot(p, p, color=theme.reference, linewidth=1.2, linestyle=(0, (4, 3)))
    left.plot(p, reported, color=naive, linewidth=2)
    left.text(0.31, 0.27, "true rate", color=theme.muted, fontsize=9, va="top")
    left.text(0.02, 0.30, "what the judge reports", color=naive, fontsize=9, va="bottom")
    at = reported[p.index(true_rate)]
    left.plot([true_rate], [at], marker="o", color=naive, markersize=6)
    left.annotate(
        f"true {true_rate:.0%}, reported {at:.1%}",
        (true_rate, at),
        textcoords="offset points",
        xytext=(8, -2),
        color=theme.text,
        fontsize=9,
    )
    left.set(
        title="A 90/90 judge on a low failure rate",
        xlabel="true failure rate",
        ylabel="rate the judge reports",
        xlim=(0, 0.5),
        ylim=(0, 0.5),
    )

    raw, point, lo, hi = _synthetic_rate()
    rows = [("judge flag rate", raw, None, naive), ("corrected rate", point, (lo, hi), honest)]
    for y, (_, value, bounds, color) in enumerate(reversed(rows)):
        if bounds is not None:
            right.plot(bounds, [y, y], color=color, linewidth=2, solid_capstyle="round")
        right.plot([value], [y], marker="o", color=color, markersize=7)
        text = (
            f"{value:.1%}" if bounds is None else f"{value:.1%} [{bounds[0]:.1%}, {bounds[1]:.1%}]"
        )
        right.annotate(
            text,
            (value, y),
            textcoords="offset points",
            xytext=(0, 9),
            ha="center",
            color=theme.text,
        )
    right.axvline(true_rate, color=theme.reference, linewidth=1.2, linestyle=(0, (4, 3)))
    right.text(true_rate, 1.45, f" true rate {true_rate:.0%}", color=theme.muted, fontsize=9)
    right.set_yticks([0, 1], [label for label, *_ in reversed(rows)])
    right.set_xticks([0, 0.05, 0.10, 0.15, 0.20])
    right.set(
        title="1,000 sample traces",
        xlabel="failure rate, with its likely range",
        xlim=(0, 0.2),
        ylim=(-0.6, 1.6),
    )
    right.xaxis.set_major_formatter(PercentFormatter(1.0, decimals=0))
    right.grid(axis="y", visible=False)
    return fig


def interval_coverage(theme: Theme) -> Figure:
    """How often each interval contains the true rate, across true failure rates."""
    fig, (ax,) = _figure(theme, 1)
    fig.set_size_inches(6.4, 3.6)
    honest, naive = theme.series[0], theme.series[1]
    doc = json.loads(_COVERAGE.read_text())
    cells = sorted(
        (c for c in doc["cells"] if c["p_hat"] == 0.90 and c["n_eval"] == 1000),
        key=lambda c: c["prevalence"],
    )
    lo, hi = doc["acceptance_band"]
    ax.axhspan(lo, hi, color=theme.band, linewidth=0)
    ax.axhline(doc["nominal"], color=theme.reference, linewidth=1, linestyle=(0, (4, 3)))
    x = [c["prevalence"] for c in cells]
    for arm, color, label, y_text in (
        ("rg_interval", honest, "Calibrated Agent Evaluation corrected-rate interval", 0.86),
        ("naive_wilson", naive, "Wilson interval on the raw flag rate", 0.20),
    ):
        ax.plot(
            x, [c["arms"][arm]["coverage"] for c in cells], color=color, linewidth=2, marker="o"
        )
        ax.text(0.03, y_text, label, color=color, fontsize=9)
    ax.text(0.5, hi + 0.012, "nominal 95%", color=theme.muted, fontsize=9, ha="right")
    ax.set(
        title="Share of 2,000 simulated studies whose interval contains the true rate",
        xlabel="true failure rate",
        ylabel="coverage",
        ylim=(-0.03, 1.05),
        xlim=(0, 0.52),
    )
    ax.title.set_fontsize(10)
    return fig


CHARTS: dict[str, Callable[[Theme], Figure]] = {
    "calibration-size": calibration_size,
    "rate-correction": rate_correction,
    "interval-coverage": interval_coverage,
}


def trace_grid(theme: Theme) -> bytes:
    """The 1,000 synthetic traces as dots: what the judge flagged, and how much of it was real.

    The split comes from the fixture's stated judge accuracy and true rate. It must add up to the
    number of traces the judge actually flagged in the sample, or the build fails.
    """
    doc = json.loads(_SYNTHETIC.read_text())
    params, verdicts = doc["params"], doc["rate_sample"]["verdicts"]
    n = len(verdicts)
    failing = round(params["true_prevalence"] * n)
    caught = round(params["tpr"] * failing)
    false_alarms = round((1 - params["tnr"]) * (n - failing))
    if caught + false_alarms != sum(verdicts):
        raise ValueError(f"grid split {caught}+{false_alarms} != {sum(verdicts)} flagged")
    _, point, lo, hi = _synthetic_rate()

    _, naive, _ = theme.series
    neutral = theme.grid
    cols, cell, gap, pad = 50, 11, 3, 4
    step = cell + gap
    rows = -(-n // cols)
    kinds = (
        ["caught"] * caught
        + ["alarm"] * false_alarms
        + ["missed"] * (failing - caught)
        + ["fine"] * (n - failing - false_alarms)
    )
    dots = []
    for i, kind in enumerate(kinds):
        x, y = pad + (i % cols) * step, pad + (i // cols) * step
        if kind == "caught":
            dots.append(
                f'<rect x="{x}" y="{y}" width="{cell}" height="{cell}" rx="2" fill="{naive}"/>'
            )
        elif kind == "alarm":
            dots.append(
                f'<rect x="{x}" y="{y}" width="{cell}" height="{cell}" rx="2" fill="{neutral}"/>'
                f'<path d="M{x + 2} {y + cell - 2}L{x + cell - 2} {y + 2}" stroke="{naive}" '
                'stroke-width="2" stroke-linecap="round"/>'
            )
        elif kind == "missed":
            dots.append(
                f'<rect x="{x + 0.75}" y="{y + 0.75}" width="{cell - 1.5}" '
                f'height="{cell - 1.5}" rx="2" fill="none" stroke="{naive}" stroke-width="1.5"/>'
            )
        else:
            dots.append(
                f'<rect x="{x}" y="{y}" width="{cell}" height="{cell}" rx="2" fill="{neutral}"/>'
            )

    grid_w = pad * 2 + cols * step - gap
    grid_h = pad * 2 + rows * step - gap
    font = "font-family=\"'Helvetica Neue', Helvetica, Arial, sans-serif\""

    # The legend sits under the grid in two columns, so the picture stays narrow enough to read
    # on a phone.
    def item(x: float, y: float, swatch: str, head: str, note: str) -> str:
        return (
            f'<g transform="translate({x} {y})">{swatch}'
            f'<text x="24" y="12" font-size="16" font-weight="700" fill="{theme.text}">'
            f"{head}</text>"
            f'<text x="24" y="32" font-size="14" fill="{theme.muted}">{note}</text></g>'
        )

    col2 = grid_w / 2
    top = grid_h + 24
    legend = [
        item(
            pad,
            top,
            f'<rect width="15" height="15" rx="2" fill="{naive}"/>',
            f"{caught} real failures",
            "the judge caught these",
        ),
        item(
            col2,
            top,
            f'<rect width="15" height="15" rx="2" fill="{neutral}"/><path d="M3 12L12 3" '
            f'stroke="{naive}" stroke-width="2" stroke-linecap="round"/>',
            f"{false_alarms} false alarms",
            "fine traces the judge flagged",
        ),
        item(
            pad,
            top + 50,
            f'<rect x=".75" y=".75" width="13.5" height="13.5" rx="2" fill="none" '
            f'stroke="{naive}" stroke-width="1.5"/>',
            f"{failing - caught} missed failures",
            "real failures it passed",
        ),
        item(
            col2,
            top + 50,
            f'<rect width="15" height="15" rx="2" fill="{neutral}"/>',
            f"{n - failing - false_alarms} fine traces",
            "correctly passed",
        ),
    ]
    width = grid_w
    height = top + 50 + 40
    label = (
        f"{n} traces. The judge flagged {caught + false_alarms}: {caught} real failures and "
        f"{false_alarms} false alarms. It missed {failing - caught}. Calibrated Agent Evaluation "
        f"reports {point:.1%}, "
        f"likely between {lo:.1%} and {hi:.1%}."
    )
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" role="img" '
        f'aria-label="{label}" {font}>' + "".join(dots) + "".join(legend) + "</svg>\n"
    ).encode()


RAW_CHARTS: dict[str, Callable[[Theme], bytes]] = {"trace-grid": trace_grid}


def render_svg(fig: Figure) -> bytes:
    buffer = io.BytesIO()
    fig.savefig(buffer, format="svg", metadata={"Date": None, "Creator": None})
    plt.close(fig)
    return buffer.getvalue()


def render_all() -> dict[str, bytes]:
    """``{"<chart>-<theme>.svg": svg_bytes}`` for every chart in both themes."""
    with mpl.rc_context(
        {
            "svg.hashsalt": "calibrated-agent-evaluation-docs",
            "svg.fonttype": "none",
            "font.family": "sans-serif",
            "font.sans-serif": ["Helvetica Neue", "Helvetica", "Arial", "DejaVu Sans"],
            "axes.titlesize": 11,
            "axes.titleweight": "bold",
            "axes.labelsize": 9,
        }
    ):
        charts = {
            f"{name}-{theme.name}.svg": render_svg(draw(theme))
            for name, draw in CHARTS.items()
            for theme in (LIGHT, DARK)
        }
    charts.update(
        {
            f"{name}-{theme.name}.svg": draw(theme)
            for name, draw in RAW_CHARTS.items()
            for theme in (LIGHT, DARK)
        }
    )
    return charts
