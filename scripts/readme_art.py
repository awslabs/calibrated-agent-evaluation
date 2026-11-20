#!/usr/bin/env python3
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Render the README banner and pipeline diagram as light and dark SVGs.

GitHub serves README images through a sandboxed proxy that loads no web fonts, so every label is
converted to outlined paths here. The text is set in Inter (SIL OFL 1.1), shaped with HarfBuzz so
kerning matches the font. Neither tool is a project dependency:

    uv run --with fonttools --with uharfbuzz python scripts/readme_art.py \\
        --font-dir <inter>/extras/ttf
"""

from __future__ import annotations

import argparse
from functools import cache
from itertools import pairwise
from pathlib import Path

import uharfbuzz as hb
from fontTools.pens.svgPathPen import SVGPathPen
from fontTools.pens.transformPen import TransformPen
from fontTools.ttLib import TTFont
from pydantic import BaseModel, ConfigDict

OUT_DIR = Path("docs/site/content/assets/readme")


class Palette(BaseModel):
    model_config = ConfigDict(frozen=True)

    name: str
    background: str
    border: str
    text: str
    muted: str
    accent: str
    accent_soft: str
    fail: str
    chip: str


LIGHT = Palette(
    name="light",
    background="#FFFFFF",
    border="#D0D7DE",
    text="#1F2328",
    muted="#656D76",
    accent="#0B7A75",
    accent_soft="#E3F2F1",
    fail="#AFB8C1",
    chip="#F6F8FA",
)
DARK = Palette(
    name="dark",
    background="#0D1117",
    border="#30363D",
    text="#E6EDF3",
    muted="#8D96A0",
    accent="#3FC1B8",
    accent_soft="#12302F",
    fail="#484F58",
    chip="#161B22",
)


class Typesetter:
    """Turns a string into one SVG path in a given font, size and position."""

    def __init__(self, font_dir: Path) -> None:
        self._font_dir = font_dir

    @cache  # noqa: B019 - one Typesetter per run; the cache holds four fonts
    def _load(self, weight: str) -> tuple[TTFont, hb.Font]:
        path = self._font_dir / f"Inter-{weight}.ttf"
        blob = hb.Blob.from_file_path(str(path))
        return TTFont(path), hb.Font(hb.Face(blob))

    def width(self, text: str, size: float, weight: str) -> float:
        ttfont, hbfont = self._load(weight)
        scale = size / ttfont["head"].unitsPerEm
        _, positions = self._shape(hbfont, text)
        return sum(p.x_advance for p in positions) * scale

    @staticmethod
    def _shape(hbfont: hb.Font, text: str) -> tuple[list, list]:
        buf = hb.Buffer()
        buf.add_str(text)
        buf.guess_segment_properties()
        hb.shape(hbfont, buf, {"kern": True, "liga": True})
        return buf.glyph_infos, buf.glyph_positions

    def path(
        self,
        text: str,
        x: float,
        baseline: float,
        size: float,
        weight: str,
        fill: str,
        anchor: str = "start",
    ) -> str:
        ttfont, hbfont = self._load(weight)
        scale = size / ttfont["head"].unitsPerEm
        glyph_set = ttfont.getGlyphSet()
        infos, positions = self._shape(hbfont, text)
        total = sum(p.x_advance for p in positions) * scale
        cursor = x - {"start": 0.0, "middle": total / 2, "end": total}[anchor]
        pen = SVGPathPen(glyph_set, ntos=_num)
        for info, pos in zip(infos, positions, strict=True):
            name = ttfont.getGlyphName(info.codepoint)
            gx = cursor + pos.x_offset * scale
            gy = baseline - pos.y_offset * scale
            glyph_set[name].draw(TransformPen(pen, (scale, 0, 0, -scale, gx, gy)))
            cursor += pos.x_advance * scale
        return f'<path fill="{fill}" d="{pen.getCommands()}"/>'


def _num(v: float) -> str:
    text = f"{v:.1f}".rstrip("0").rstrip(".")
    return "0" if text == "-0" else text


def _svg(width: int, height: int, body: list[str], title: str) -> str:
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}" role="img" aria-label="{title}">'
        f"<title>{title}</title>{''.join(body)}</svg>\n"
    )


class Row(BaseModel):
    model_config = ConfigDict(frozen=True)

    label: str
    low: float
    point: float
    high: float


# Illustrative only: the shape of a gate report, not numbers from any run.
BANNER_ROWS = [
    Row(label="wrong tool", low=0.84, point=0.91, high=0.96),
    Row(label="bad hand-off", low=0.71, point=0.86, high=0.95),
    Row(label="hallucinated id", low=0.82, point=0.88, high=0.93),
    Row(label="ignored policy", low=0.62, point=0.74, high=0.84),
]
GATE = 0.80


def banner(ts: Typesetter, p: Palette) -> str:
    w, h = 1600, 400
    body = [
        f'<rect x="1" y="1" width="{w - 2}" height="{h - 2}" rx="24" '
        f'fill="{p.background}" stroke="{p.border}" stroke-width="2"/>',
        ts.path("Calibrated Agent", 96, 168, 76, "Bold", p.text),
        ts.path("Evaluation", 96, 256, 76, "Bold", p.accent),
        ts.path(
            "Error analysis and judge calibration for AI agents", 98, 318, 28, "Regular", p.muted
        ),
    ]

    plot_x0, plot_x1 = 1040, 1500
    lo, hi = 0.55, 1.0

    def sx(v: float) -> float:
        return plot_x0 + (v - lo) / (hi - lo) * (plot_x1 - plot_x0)

    top, step = 112, 52
    gx = sx(GATE)
    body.append(
        f'<line x1="{gx:.1f}" y1="{top - 40}" x2="{gx:.1f}" y2="{top + step * 3 + 30}" '
        f'stroke="{p.text}" stroke-width="2" stroke-dasharray="6 6"/>'
    )
    body.append(ts.path(f"gate {GATE:.2f}", gx, top - 52, 20, "Medium", p.muted, "middle"))
    for i, row in enumerate(BANNER_ROWS):
        y = top + i * step
        colour = p.accent if row.low >= GATE else p.fail
        body.append(ts.path(row.label, plot_x0 - 28, y + 7, 22, "Medium", p.muted, "end"))
        body.append(
            f'<line x1="{sx(row.low):.1f}" y1="{y}" x2="{sx(row.high):.1f}" y2="{y}" '
            f'stroke="{colour}" stroke-width="6" stroke-linecap="round"/>'
        )
        body.append(f'<circle cx="{sx(row.point):.1f}" cy="{y}" r="11" fill="{colour}"/>')
    legend_y = top + step * 3 + 74
    body.append(
        ts.path(
            "95% lower bound must clear the gate",
            (plot_x0 + plot_x1) / 2 - 40,
            legend_y,
            20,
            "Regular",
            p.muted,
            "middle",
        )
    )
    return _svg(w, h, body, "Calibrated Agent Evaluation")


class Stage(BaseModel):
    model_config = ConfigDict(frozen=True)

    name: str
    note: list[str] = []


class Group(BaseModel):
    model_config = ConfigDict(frozen=True)

    title: str
    stages: list[Stage]
    gate: bool = False


PIPELINE = [
    Group(
        title="Error analysis",
        stages=[
            Stage(name="probe"),
            Stage(name="define", note=["covering array"]),
            Stage(name="generate"),
            Stage(name="label", note=["human gold"]),
            Stage(name="cluster", note=["Good\u2013Turing"]),
        ],
    ),
    Group(
        title="Trust gates",
        gate=True,
        stages=[
            Stage(name="calibrate", note=["TPR and TNR", "lower bounds ≥ 0.80"]),
            Stage(name="bias", note=["CALM battery"]),
        ],
    ),
    Group(
        title="Scoring",
        stages=[Stage(name="judge"), Stage(name="run", note=["CI gate on the", "lower bound"])],
    ),
    Group(title="Fix", stages=[Stage(name="improve", note=["McNemar", "Holm guards"])]),
]


def pipeline(ts: Typesetter, p: Palette) -> str:
    w, h = 1600, 300
    chip_w, chip_h, gap, group_gap = 128, 60, 20, 36
    n = sum(len(g.stages) for g in PIPELINE)
    total = n * chip_w + (n - len(PIPELINE)) * gap + (len(PIPELINE) - 1) * group_gap
    x = (w - total) / 2
    chip_y = 120
    body = [
        f'<rect x="1" y="1" width="{w - 2}" height="{h - 2}" rx="24" '
        f'fill="{p.background}" stroke="{p.border}" stroke-width="2"/>'
    ]
    centres: list[float] = []
    for gi, group in enumerate(PIPELINE):
        g_start = x
        for si, stage in enumerate(group.stages):
            fill, stroke, colour = (
                (p.accent_soft, p.accent, p.accent) if group.gate else (p.chip, p.border, p.text)
            )
            body.append(
                f'<rect x="{x:.1f}" y="{chip_y}" width="{chip_w}" height="{chip_h}" rx="14" '
                f'fill="{fill}" stroke="{stroke}" stroke-width="2"/>'
            )
            cx = x + chip_w / 2
            body.append(ts.path(stage.name, cx, chip_y + 39, 24, "SemiBold", colour, "middle"))
            for li, line in enumerate(stage.note):
                body.append(
                    ts.path(
                        line, cx, chip_y + chip_h + 36 + li * 24, 17, "Regular", p.muted, "middle"
                    )
                )
            centres.append(cx)
            x += chip_w + (gap if si < len(group.stages) - 1 else 0)
        g_end = x
        bracket_y = chip_y - 30
        body.append(
            f'<path d="M{g_start + 4:.1f} {bracket_y + 10} V{bracket_y} '
            f'H{g_end - 4:.1f} V{bracket_y + 10}" '
            f'fill="none" stroke="{p.accent if group.gate else p.border}" stroke-width="2"/>'
        )
        body.append(
            ts.path(
                group.title,
                (g_start + g_end) / 2,
                bracket_y - 16,
                20,
                "SemiBold",
                p.accent if group.gate else p.muted,
                "middle",
            )
        )
        if gi < len(PIPELINE) - 1:
            x += group_gap
    arrow_y = chip_y + chip_h / 2
    for a, b in pairwise(centres):
        x0, x1 = a + chip_w / 2 + 3, b - chip_w / 2 - 3
        if x1 - x0 < 6:
            continue
        body.append(
            f'<path d="M{x0:.1f} {arrow_y} H{x1 - 5:.1f} '
            f"M{x1 - 11:.1f} {arrow_y - 6} L{x1 - 3:.1f} {arrow_y} "
            f'L{x1 - 11:.1f} {arrow_y + 6}" fill="none" stroke="{p.muted}" stroke-width="2" '
            f'stroke-linecap="round" stroke-linejoin="round"/>'
        )
    return _svg(
        w,
        h,
        body,
        "The pipeline: probe, define, generate, label, cluster, calibrate, bias, judge, run, "
        "improve",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=(__doc__ or "").splitlines()[0])
    parser.add_argument(
        "--font-dir", type=Path, required=True, help="directory holding Inter-*.ttf"
    )
    parser.add_argument("--out-dir", type=Path, default=OUT_DIR)
    args = parser.parse_args()
    ts = Typesetter(args.font_dir)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    for palette in (LIGHT, DARK):
        (args.out_dir / f"banner-{palette.name}.svg").write_text(
            banner(ts, palette), encoding="utf-8"
        )
        (args.out_dir / f"pipeline-{palette.name}.svg").write_text(
            pipeline(ts, palette), encoding="utf-8"
        )


if __name__ == "__main__":
    main()
