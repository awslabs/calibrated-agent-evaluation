# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Default theme + self-contained, offline assets."""

from __future__ import annotations

import re
from collections.abc import Callable
from typing import Any

from fasthtml.common import to_xml

from calibrated_agent_evaluation.review.renderers.trace_view import labeling_form
from calibrated_agent_evaluation.schema import Trace


def test_page_embeds_bundled_theme_with_distinct_verdict_styles(
    make_client: Callable[..., Any], backend: Any
) -> None:
    client, _app = make_client(backend)
    html = client.get("/").text
    # The theme is inlined as a <style> block (no external stylesheet to fetch).
    assert "<style" in html.lower()
    # Pass and Fail verdict controls are styled distinctly (green vs red).
    assert ".hd-pass" in html
    assert ".hd-fail" in html
    assert "--hd-pass" in html
    assert "--hd-fail" in html


def test_no_external_or_cdn_asset_is_referenced(
    make_client: Callable[..., Any], backend: Any
) -> None:
    client, _app = make_client(backend)
    html = client.get("/").text
    # Every script/stylesheet is local/inlined — no CDN, so the surface works offline.
    # (A `<link rel="canonical">` is fine; only external CSS/JS assets are the concern.)
    assert "cdn.jsdelivr.net" not in html
    assert not re.search(r'<script[^>]+src\s*=\s*["\']https?://', html, re.IGNORECASE)
    external_stylesheet = re.search(
        r'<link[^>]+rel\s*=\s*["\']stylesheet["\'][^>]+href\s*=\s*["\']https?://',
        html,
        re.IGNORECASE,
    )
    assert external_stylesheet is None


def test_htmx_is_served_from_the_local_vendored_copy(
    make_client: Callable[..., Any], backend: Any
) -> None:
    client, _app = make_client(backend)
    html = client.get("/").text
    # htmx is present and inlined (its source is embedded), not linked from a CDN.
    assert "var htmx" in html
    assert "cdn.jsdelivr.net/npm/htmx" not in html


def _rule(css: str, selector: str) -> dict[str, str]:
    # The declarations of the one rule whose selector list is exactly `selector`.
    css = re.sub(r"/\*.*?\*/", "", css, flags=re.DOTALL)
    rules = [
        body for head, body in re.findall(r"([^{}]+)\{([^{}]*)\}", css) if head.strip() == selector
    ]
    assert len(rules) == 1, f"expected one {selector} rule, found {len(rules)}"
    decls = (d.split(":", 1) for d in rules[0].split(";") if ":" in d)
    return {name.strip(): value.strip() for name, value in decls}


def test_pinned_form_is_capped_with_its_own_scroll(
    make_client: Callable[..., Any], backend: Any
) -> None:
    # a crew trace's form is about 700px tall, and pinned with no cap it
    # covered the whole trace on a 768-860px window. The served theme caps it at a fraction of the
    # window and scrolls the overflow inside the form, not the trace.
    client, _app = make_client(backend)
    css = re.search(r"<style>(.*?)</style>", client.get("/").text, re.DOTALL)
    assert css is not None
    form = _rule(css.group(1), ".hd-label-form")
    assert form["position"] == "sticky"
    cap = re.fullmatch(r"(\d+(?:\.\d+)?)vh", form.get("max-height", ""))
    assert cap is not None, (
        f"max-height must be a share of the window, got {form.get('max-height')}"
    )
    assert float(cap.group(1)) <= 50  # the trace keeps at least half the window
    assert form.get("overflow-y") == "auto"


def test_capped_crew_form_leads_with_critique_and_verdict(
    corpus_traces: list[Trace],
) -> None:
    # The cap shows the top of the form first, so what must stay in view without scrolling the
    # form (the critique box and the verdict buttons) comes before the crew attribution fields.
    clm = next(t for t in corpus_traces if t.trace_id == "CLM-4471")
    html = to_xml(labeling_form(clm))
    assert html.index('name="critique"') < html.index('value="FAIL"') < html.index("culprit_span")
    assert html.index('value="FAIL"') < html.index('name="expected"')
