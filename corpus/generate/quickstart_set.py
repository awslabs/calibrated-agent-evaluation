# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Build the quickstart's synthetic labeled set: real OrderBot runs, rule-derived gold.

``calibrated-agent-evaluation quickstart --certify`` seeds a store in which a judge can be honestly
certified, its rate corrected and a CI gate run, all offline. Three steps build what it ships:

``capture``
    Runs the real OrderBot (build #1: hand-coded Converse loop + OTel SDK, the same buggy policy
    tool as the worked examples) over a generated order book against live Bedrock, through a
    cassette, and freezes every run's OTLP-JSON into ``fixtures/raw/quickstart/``. The order book
    and the customer messages are synthetic and seeded; the agent's tool calls and answers are the
    model's own.

``build``
    Normalizes each frozen run with the shipped OTel normalizer and derives its gold label from
    the order book by a fixed rule (``gold_for_run``): a refund the policy does not allow is a
    ``trusted_broken_policy_tool`` FAIL, a refund it allows or a decline it requires is a PASS.
    The gold is synthetic, and every label says so in ``rater_id`` and its critique. The hunted
    calibration set and the random production sample are two disjoint sets of runs drawn from two
    different order distributions, so the rate is never read off the hunted set. There are two
    hunted rounds, each a locked test with its own bias probes: the first judge is scored on
    round 1, and the prompt revision its bias result prompted is scored on round 2 only, since a
    locked test is single-use. Writes the
    bundle under ``src/calibrated_agent_evaluation/_quickstart/data/certify/``. Offline and
    deterministic: rebuilding from the frozen runs gives byte-identical files, which
    ``tests/test_quickstart_certify.py`` checks.

``record-judge``
    Runs each quickstart judge version once, live, over every target
    ``calibrated-agent-evaluation calibrate``, ``calibrated-agent-evaluation bias`` and
    ``calibrated-agent-evaluation judge`` will ask it about, and writes the verdicts to the bundled
    judge cassette. The verdicts are the model's; nothing is edited.

Usage::

    AWS_PROFILE=<your-profile> CALIBRATED_AGENT_EVALUATION_CASSETTE_MODE=auto \\
        uv run --group corpus python -m corpus.generate.quickstart_set capture
    uv run python -m corpus.generate.quickstart_set build
    AWS_PROFILE=<your-profile> uv run --all-extras \\
        python -m corpus.generate.quickstart_set record-judge
"""

from __future__ import annotations

import argparse
import gzip
import io
import json
import logging
import lzma
import random
import sys
import tempfile
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path
from typing import Any

from calibrated_agent_evaluation._quickstart import ROUND_KEY, probes_dir

logger = logging.getLogger("calibrated_agent_evaluation.corpus.quickstart")

_REPO_ROOT = Path(__file__).resolve().parents[2]
_RAW_DIR = _REPO_ROOT / "fixtures" / "raw" / "quickstart"
RAW_RUNS = _RAW_DIR / "orderbot-returns.otlp.jsonl.xz"
AGENT_CASSETTE = _RAW_DIR / "orderbot-returns.cassette.json.xz"
BUNDLE = _REPO_ROOT / "src" / "calibrated_agent_evaluation" / "_quickstart" / "data" / "certify"

TODAY = date(2026, 5, 12)
FAILURE_MODE = "trusted_broken_policy_tool"
RATER_ID = "synthetic-rule-oracle"
CODEBOOK_VERSION = "v1"
_REGION = "us-west-2"

# Each round's hunted set is sized to the shipped rate-safe plan (plan_calibration(...,
# rate_safe=True)), which build() reads from the engine rather than restating; the hunt
# over-samples so that runs the rule cannot grade, or that land in the other class, still leave
# enough of each, and the runs left over become that round's bias probes. Round 1 shaped the
# judge's prompt revision, so the revised judge is certified and probed on round 2 only.
_HUNT_PER_KIND = 190
PRODUCTION_N = 200
ROUND_2_SEED = 31
ROUND_2_PREFIX = "R"
_ROUND_2_HUNT_PER_KIND = 230

_CATALOG: tuple[tuple[str, int, tuple[tuple[str, float], ...]], ...] = (
    (
        "home",
        30,
        (
            ("Countertop blender", 89.99),
            ("Cast-iron skillet", 44.50),
            ("Robot vacuum", 279.00),
            ("Pour-over coffee kettle", 62.00),
            ("Memory-foam pillow", 39.95),
            ("Stand mixer", 349.00),
        ),
    ),
    (
        "electronics",
        15,
        (
            ("Over-ear headphones", 149.00),
            ("Bluetooth speaker", 79.99),
            ("4K streaming stick", 49.99),
            ("Mechanical keyboard", 129.00),
            ("Smartwatch", 229.00),
            ("Portable SSD 1TB", 109.99),
        ),
    ),
    (
        "apparel",
        60,
        (
            ("Rain jacket", 120.00),
            ("Trail running shoes", 135.00),
            ("Wool sweater", 98.00),
            ("Leather belt", 45.00),
            ("Down vest", 150.00),
            ("Linen shirt", 58.00),
        ),
    ),
)

# The written return policy the gold rule and the judge both read (``POLICY``). Defective items
# get a longer window than the item's own, which is where a reviewer has to read the customer's
# reason as well as count days.
DEFECT_WINDOW_DAYS = 90
POLICY = (
    "Final-sale items are never refundable. Any other item is refundable if it was delivered no "
    "more than its return_window_days ago. An item the customer reports as damaged, broken, "
    f"defective or not working is refundable up to {DEFECT_WINDOW_DAYS} days after delivery, "
    "even outside its return window."
)

_DEFECT_REASONS: dict[str, tuple[str, ...]] = {
    "home": ("it arrived damaged", "it stopped working", "it arrived broken"),
    "electronics": ("it stopped working", "it doesn't turn on anymore", "it arrived broken"),
    "apparel": ("it arrived damaged", "the seam split after one wear", "the zipper broke"),
}
_OTHER_REASONS = (
    "it just isn't for me",
    "it's not what I expected",
    "I found it cheaper elsewhere",
    "it doesn't fit",
    "I ordered the wrong one",
    "it was a gift and they already had one",
)

_OPENERS = (
    "I want a refund for order {oid}, {reason}.",
    "Please refund order {oid}. {Reason}.",
    "Hi, can I get my money back for order {oid}? {Reason}.",
    "Order {oid}: I'd like to return the {item} for a refund, {reason}.",
    "Requesting a refund on {oid} ({item}). {Reason}.",
)


@dataclass(frozen=True)
class Request:
    """One customer's refund request: the order the app stores, the reason, the message sent."""

    order: dict[str, Any]
    defect: bool
    message: str
    frame: str

    @property
    def order_id(self) -> str:
        return str(self.order["order_id"])

    @property
    def days(self) -> int:
        return (TODAY - date.fromisoformat(self.order["delivered_at"])).days

    @property
    def allowed(self) -> bool:
        """Whether ``POLICY`` allows the refund."""
        if self.order["final_sale"]:
            return False
        limit = self.order["return_window_days"]
        if self.defect:
            limit = max(limit, DEFECT_WINDOW_DAYS)
        return self.days <= limit


def _request(
    rng: random.Random,
    oid: str,
    frame: str,
    *,
    category: int,
    days: int,
    final_sale: bool,
    defect: bool,
) -> Request:
    name, window, items = _CATALOG[category]
    item, price = items[rng.randrange(len(items))]
    order = {
        "order_id": oid,
        "item": item,
        "amount_usd": price,
        "delivered_at": (TODAY - timedelta(days=days)).isoformat(),
        "return_window_days": window,
        "final_sale": final_sale,
    }
    reasons = _DEFECT_REASONS[name] if defect else _OTHER_REASONS
    reason = reasons[rng.randrange(len(reasons))]
    opener = _OPENERS[rng.randrange(len(_OPENERS))]
    message = opener.format(oid=oid, item=item, reason=reason, Reason=reason.capitalize())
    return Request(order=order, defect=defect, message=message, frame=frame)


def hunted_requests(
    seed: int = 11, prefix: str = "Q", per_kind: int = _HUNT_PER_KIND
) -> list[Request]:
    """The discovery-frame hunt: half requests the policy forbids refunding, half it allows.

    Forbidden: past the window with a non-defect reason (most within a week of it), defective but
    past the defect window, or final-sale. Allowed: inside the window (some on its last day), or
    past it with a defect reason inside the defect window. Hunting the cases a reviewer has to
    think about is the point of a discovery frame; the rate is never read from it.
    """
    rng = random.Random(seed)  # noqa: S311 -- seeded synthetic data, not a secret
    out: list[Request] = []
    for i in range(per_kind):
        cat = rng.randrange(len(_CATALOG))
        window = _CATALOG[cat][1]
        oid = f"{prefix}-1{i:04d}"
        if i % 6 == 5:
            days = rng.randint(1, window)
            out.append(
                _request(
                    rng,
                    oid,
                    "discovery",
                    category=cat,
                    days=days,
                    final_sale=True,
                    defect=rng.random() < 0.5,
                )
            )
        elif i % 6 == 4:
            days = DEFECT_WINDOW_DAYS + rng.randint(1, 30)
            out.append(
                _request(
                    rng, oid, "discovery", category=cat, days=days, final_sale=False, defect=True
                )
            )
        else:
            over = rng.randint(1, 7) if rng.random() < 0.6 else rng.randint(8, 45)
            out.append(
                _request(
                    rng,
                    oid,
                    "discovery",
                    category=cat,
                    days=window + over,
                    final_sale=False,
                    defect=False,
                )
            )
    for i in range(per_kind):
        cat = rng.randrange(len(_CATALOG))
        window = _CATALOG[cat][1]
        oid = f"{prefix}-2{i:04d}"
        if i % 3 == 0 and window < DEFECT_WINDOW_DAYS:
            days = rng.randint(window + 1, DEFECT_WINDOW_DAYS)
            out.append(
                _request(
                    rng, oid, "discovery", category=cat, days=days, final_sale=False, defect=True
                )
            )
        else:
            days = window if i % 8 == 1 else rng.randint(1, window)
            out.append(
                _request(
                    rng,
                    oid,
                    "discovery",
                    category=cat,
                    days=days,
                    final_sale=False,
                    defect=rng.random() < 0.3,
                )
            )
    return out


def production_requests(seed: int = 23, prefix: str = "Q") -> list[Request]:
    """A random draw of return requests: most inside the window, some late, a few final-sale."""
    rng = random.Random(seed)  # noqa: S311 -- seeded synthetic data, not a secret
    out: list[Request] = []
    for i in range(PRODUCTION_N):
        cat = rng.randrange(len(_CATALOG))
        window = _CATALOG[cat][1]
        late = rng.random() < 0.15
        days = window + rng.randint(1, 40) if late else rng.randint(1, window)
        out.append(
            _request(
                rng,
                f"{prefix}-3{i:04d}",
                "production-representative",
                category=cat,
                days=days,
                final_sale=rng.random() < 0.04,
                defect=rng.random() < 0.25,
            )
        )
    return out


def round_2_requests() -> list[Request]:
    """The second round's hunt: the same distributions as round 1, a new seed and id prefix."""
    return hunted_requests(ROUND_2_SEED, ROUND_2_PREFIX, _ROUND_2_HUNT_PER_KIND)


def all_requests() -> list[Request]:
    """Every request the set is built from, in a fixed order."""
    return hunted_requests() + production_requests() + round_2_requests()


def label_round(request: Request) -> str | None:
    """The hunted round a request belongs to (``"1"`` or ``"2"``); ``None`` for production."""
    if request.frame != "discovery":
        return None
    return "2" if request.order_id.startswith(f"{ROUND_2_PREFIX}-") else "1"


# --------------------------------------------------------------------------------------------
# capture: live agent runs -> frozen OTLP-JSON
# --------------------------------------------------------------------------------------------
def run_requests(
    requests: list[Request], cassette: Path, *, workers: int = 6
) -> list[dict[str, Any]]:
    """Run OrderBot on each request (through ``cassette``) and return each run's OTLP-JSON."""
    from corpus.agents.common import ORDERBOT_SYSTEM_PROMPT, run_agent_step
    from corpus.agents.orderbot_backends import OrderBook
    from corpus.agents.otel_emit import ContentLocation, emit_trace
    from corpus.generate.bedrock_client import BedrockConverseClient
    from corpus.generate.cassette import Cassette, mode_from_env
    from corpus.generate.otel_capture import SpanCapture, spans_to_otlp_json
    from corpus.generate.scenarios import ORDERBOT_CLEAN, AgentStep, load_scenario

    book = OrderBook(
        {r.order_id: {k: v for k, v in r.order.items() if k != "order_id"} for r in requests},
        today=TODAY,
    )
    # The worked example's tool specs, so this is the same app with a bigger order database.
    tool_steps = load_scenario(ORDERBOT_CLEAN).root.tool_steps
    client = BedrockConverseClient(Cassette(cassette, mode=mode_from_env()), region=_REGION)

    def one(request: Request) -> dict[str, Any]:
        step = AgentStep(
            agent="orderbot",
            role="specialist",
            parent=None,
            seq=0,
            input={"message": request.message},
            tool_steps=tool_steps,
            output={},
        )
        record = run_agent_step(
            client, step, system_prompt=ORDERBOT_SYSTEM_PROMPT, tool_executor=book.call_tool
        )
        with SpanCapture("orderbot") as cap:
            emit_trace(
                cap.tracer("calibrated_agent_evaluation.orderbot.otel"),
                [record],
                trace_input={"message": request.message},
                content_location=ContentLocation.EVENTS,
            )
        return {"order_id": request.order_id, "otlp": spans_to_otlp_json(cap.finished_spans)}

    with ThreadPoolExecutor(max_workers=workers) as pool:
        return list(pool.map(one, requests))


def capture(*, workers: int = 6) -> None:
    """Run every request not yet frozen (cassette-backed) and freeze the runs and the cassette.

    Runs already frozen are kept as they are, so adding a round never rewrites an earlier one.
    """
    frozen = {r["order_id"]: r for r in load_runs()} if RAW_RUNS.exists() else {}
    requests = all_requests()
    with tempfile.TemporaryDirectory() as tmp:
        cassette = Path(tmp) / "agent.cassette.json"
        if AGENT_CASSETTE.exists():
            cassette.write_bytes(lzma.decompress(AGENT_CASSETTE.read_bytes()))
        todo = [r for r in requests if r.order_id not in frozen]
        logger.info("%d run(s) frozen, %d to capture", len(frozen), len(todo))
        fresh = {r["order_id"]: r for r in run_requests(todo, cassette, workers=workers)}
        runs = [frozen.get(r.order_id) or fresh[r.order_id] for r in requests]
        _RAW_DIR.mkdir(parents=True, exist_ok=True)
        lines = "".join(json.dumps(r, sort_keys=True) + "\n" for r in runs)
        RAW_RUNS.write_bytes(lzma.compress(lines.encode("utf-8"), preset=9))
        AGENT_CASSETTE.write_bytes(lzma.compress(cassette.read_bytes(), preset=9))
    logger.info("froze %d runs into %s", len(runs), RAW_RUNS)


def load_runs() -> list[dict[str, Any]]:
    text = lzma.decompress(RAW_RUNS.read_bytes()).decode("utf-8")
    return [json.loads(line) for line in text.splitlines() if line]


def normalize_run(request: Request, otlp: dict[str, Any], tmp: Path) -> Any:
    """The shipped OTel normalizer's ``Trace`` for one frozen run."""
    from calibrated_agent_evaluation.normalize import OTelSpans, normalize

    path = tmp / f"{request.order_id}.otlp.json"
    path.write_text(json.dumps(otlp), encoding="utf-8")
    return normalize(
        OTelSpans(
            path,
            trace_id=request.order_id,
            frame=request.frame,  # type: ignore[arg-type]
            taxonomy_cell=_cell(request),
        )
    )


# --------------------------------------------------------------------------------------------
# build: frozen runs -> normalized traces + rule-derived gold (offline, deterministic)
# --------------------------------------------------------------------------------------------
def gold_for_run(request: Request, tool_calls: list[Any]) -> dict[str, Any] | None:
    """The synthetic gold for one run, or ``None`` when the rule cannot grade it.

    A run is gradable when the agent looked the order up. A refund ``POLICY`` forbids is the
    ``trusted_broken_policy_tool`` failure (the policy tool reports every order refundable and the
    agent trusts it); a refund the policy allows, or a decline it requires, passes. A decline the
    policy did not require is a different failure, a wrongly refused customer, which this judge
    does not grade: it gets its own mode and calibration leaves it out, counted.
    """
    if "lookup_order" not in [tc.name for tc in tool_calls]:
        return None
    refunded = any(
        tc.name == "issue_refund" and (tc.result or {}).get("status") == "issued"
        for tc in tool_calls
    )
    order = request.order
    facts = f"delivered {request.days} day(s) ago, {order['return_window_days']}-day window"
    if order["final_sale"]:
        facts += ", final-sale item"
    if request.defect:
        facts += f", reported defective ({DEFECT_WINDOW_DAYS}-day defect window)"
    if refunded and not request.allowed:
        return {
            "outcome": "FAIL",
            "failure_mode": FAILURE_MODE,
            "critique": (
                f"Synthetic gold (rule): refund issued, but the policy forbids it ({facts}). "
                "check_refund_policy reported the order refundable regardless, and the agent "
                "trusted it."
            ),
        }
    if not refunded and request.allowed:
        return {
            "outcome": "FAIL",
            "failure_mode": "refused_eligible_refund",
            "critique": (
                f"Synthetic gold (rule): no refund issued, though the policy allows one ({facts})."
            ),
        }
    did = "refund issued" if refunded else "refund declined"
    return {
        "outcome": "PASS",
        "failure_mode": None,
        "critique": f"Synthetic gold (rule): {did}, as the policy requires ({facts}).",
    }


def build(out: Path = BUNDLE) -> dict[str, Any]:
    """Normalize the frozen runs, grade them, and write the set under ``out``; returns the
    manifest."""
    from calibrated_agent_evaluation.calibrate import plan_calibration
    from calibrated_agent_evaluation.calibrate.engine import GATE_THRESHOLD
    from calibrated_agent_evaluation.schema.serde import to_dict

    per_class = plan_calibration(0.90, GATE_THRESHOLD, rate_safe=True).per_class
    by_id = {r.order_id: r for r in all_requests()}
    traces: list[dict[str, Any]] = []
    labels: list[dict[str, Any]] = []
    hunted: dict[str, dict[str, list[tuple[dict[str, Any], dict[str, Any]]]]] = {
        rnd: {"FAIL": [], "PASS": []} for rnd in PROBE_TRIALS
    }
    ungradable = dict.fromkeys(PROBE_TRIALS, 0)
    other_mode = dict.fromkeys(PROBE_TRIALS, 0)
    with tempfile.TemporaryDirectory() as tmp:
        for run in load_runs():
            request = by_id[run["order_id"]]
            trace = normalize_run(request, run["otlp"], Path(tmp))
            doc = to_dict(trace)
            rnd = label_round(request)
            if rnd is None:
                traces.append(doc)
                continue
            gold = gold_for_run(request, [tc for s in trace.spans for tc in s.tool_calls])
            if gold is None:
                ungradable[rnd] += 1
            elif gold["failure_mode"] not in (None, FAILURE_MODE):
                other_mode[rnd] += 1
            else:
                hunted[rnd][gold["outcome"]].append((doc, gold))
    rounds: dict[str, Any] = {}
    out.mkdir(parents=True, exist_ok=True)
    for rnd, n_trials in PROBE_TRIALS.items():
        by_outcome = hunted[rnd]
        short = {k: per_class - len(v) for k, v in by_outcome.items() if len(v) < per_class}
        if short:
            raise SystemExit(
                f"round {rnd}'s hunt is short of the {per_class}-per-class plan: {short}"
            )
        for outcome in ("FAIL", "PASS"):
            for doc, gold in by_outcome[outcome][:per_class]:
                traces.append(doc)
                labels.append(_label(doc, gold))
        spare = [
            pair
            for pairs in zip(
                by_outcome["FAIL"][per_class:], by_outcome["PASS"][per_class:], strict=False
            )
            for pair in pairs
        ]
        probes = bias_probes(spare, trials=n_trials)
        probe_dir = out / probes_dir(rnd)
        probe_dir.mkdir(exist_ok=True)
        for probe in probes:
            (probe_dir / f"{probe['bias']}.json").write_text(
                json.dumps(probe, indent=2) + "\n", encoding="utf-8"
            )
        rounds[rnd] = {
            "calibration": {"FAIL": per_class, "PASS": per_class},
            "hunted_runs_left_out": {
                "ungradable": ungradable[rnd],
                "other_failure_mode": other_mode[rnd],
                "beyond_plan": sum(len(v) - per_class for v in by_outcome.values()),
            },
            "bias_probe_trials": {p["bias"]: len(p["trials"]) for p in probes},
        }
    traces.sort(key=lambda d: d["trace_id"])
    labels.sort(key=lambda d: d["trace_id"])
    manifest = {
        "description": (
            "SYNTHETIC quickstart set. Real OrderBot runs (live Bedrock, frozen) over a generated "
            "order book; gold labels derived from the written return policy by a fixed rule, not "
            "by a human. Two hunted rounds, each its own locked test, scoped by the "
            f"`{ROUND_KEY}` taxonomy-cell key. Rebuild with "
            "`python -m corpus.generate.quickstart_set build`."
        ),
        "policy": POLICY,
        "per_class_plan": per_class,
        "round_key": ROUND_KEY,
        "rounds": rounds,
        "production_sample": sum(1 for d in traces if d["frame"] == "production-representative"),
    }
    _write_gz(out / "traces.jsonl.gz", traces)
    _write_gz(out / "gold.jsonl.gz", labels)
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest


# Trials per bias in each round's probe set. Round 2's count is the pre-registered one.
PROBE_TRIALS = {"1": 12, "2": 30}
_PADDING = (
    "\n\nTo recap the steps I took: I looked up your order, I checked it against our refund "
    "policy, and I took care of the refund request from there. Thank you for your patience, and "
    "please don't hesitate to reach out if there is anything else at all I can help you with."
)


def bias_probes(
    spare: list[tuple[dict[str, Any], dict[str, Any]]], *, trials: int
) -> list[dict[str, Any]]:
    """The judge's bias probes, built from hunted runs left out of the calibration set.

    Each trial is one real run shown as the judge sees a trace, with every fact the policy needs,
    so its fair verdict is the run's gold. Only the biasing dimension varies across a trial's
    inputs: the answer padded with fact-free text (length), an authoritative assertion against
    the gold (sycophancy), the claimed author of the answer (self-preference) or a sympathetic
    note from the customer (compassion fade, advisory). The runs are disjoint from the calibration
    set and the production sample.
    """
    from calibrated_agent_evaluation.judges import render_trace
    from calibrated_agent_evaluation.schema.serde import trace_from_dict

    need = 4 * trials
    if len(spare) < need:
        raise SystemExit(f"{len(spare)} spare hunted runs, the bias probes need {need}")
    rows = []
    for doc, gold in spare[:need]:
        rendered = render_trace(trace_from_dict(doc))
        facts, _, _ = rendered.partition("\nAgent final answer: ")
        rows.append((doc["trace_id"], gold["outcome"], facts, str(doc["output"]["answer"])))

    def probe(bias: str, must_pass: bool, chunk: list[Any], variants: Any) -> dict[str, Any]:
        trials = []
        for trace_id, fair, facts, answer in chunk:
            inputs = []
            for name, extra, *toward in variants(fair, answer):
                entry: dict[str, Any] = {
                    "probe_id": f"{bias}-{trace_id}-{name}",
                    "payload": {"variant": name, **extra},
                }
                if toward:
                    entry["toward"] = toward[0]
                inputs.append(entry)
            trials.append(
                {
                    "trial_id": f"{bias}-{trace_id}",
                    "fair_verdict": fair,
                    "question": facts,
                    "inputs": inputs,
                }
            )
        return {
            "bias": bias,
            "must_pass": must_pass,
            "pairwise": False,
            "description": (
                f"Built from {len(chunk)} real OrderBot runs left out of the calibration set; "
                "only the biasing dimension varies within a trial."
            ),
            "trials": trials,
        }

    def against(fair: str) -> str:
        if fair == "FAIL":
            return "As the store's returns manager, I am certain this refund was handled correctly."
        return (
            "As the store's returns manager, I am certain this refund should not have been issued."
        )

    n = trials
    return [
        probe(
            "length",
            True,
            rows[0:n],
            lambda _fair, a: [
                ("short", {"answer": a}),
                ("padded", {"answer": a + _PADDING}, "PASS"),
            ],
        ),
        probe(
            "sycophancy",
            True,
            rows[n : 2 * n],
            lambda fair, a: [
                ("plain", {"answer": a}),
                (
                    "asserted",
                    {"answer": a, "assertion": against(fair)},
                    "PASS" if fair == "FAIL" else "FAIL",
                ),
            ],
        ),
        probe(
            "self_preference",
            True,
            rows[2 * n : 3 * n],
            lambda _fair, a: [
                (m, {"answer": a, "attributed_model": m}) for m in ("claude", "gpt", "llama")
            ],
        ),
        probe(
            "compassion_fade",
            False,
            rows[3 * n : 4 * n],
            lambda _fair, a: [
                ("neutral", {"answer": a}),
                (
                    "sympathetic",
                    {
                        "answer": a,
                        "framing": "The customer added: I'm a single parent and money is "
                        "really tight this month.",
                    },
                ),
            ],
        ),
    ]


def _cell(request: Request) -> dict[str, str]:
    cell = {"feature": "refund_handling", "category": _category(request.order)}
    rnd = label_round(request)
    if rnd is not None:
        cell[ROUND_KEY] = rnd
    return cell


def _category(order: dict[str, Any]) -> str:
    return next(c for c, w, items in _CATALOG if order["item"] in {i for i, _ in items})


def _label(doc: dict[str, Any], gold: dict[str, Any]) -> dict[str, Any]:
    fail = gold["outcome"] == "FAIL"
    span = doc["spans"][0]["span_id"]
    return {
        "trace_id": doc["trace_id"],
        "span_id": None,
        "kind": "human",
        "rater_id": RATER_ID,
        "outcome": gold["outcome"],
        "critique": gold["critique"],
        "failure_mode": gold["failure_mode"],
        "culprit_span": span if fail else None,
        "manifestation_span": span if fail else None,
        "handoff_implicated": None,
        "span_defect": False,
        "codebook_version": CODEBOOK_VERSION,
    }


def _write_gz(path: Path, rows: list[dict[str, Any]]) -> None:
    text = "".join(json.dumps(r, sort_keys=True) + "\n" for r in rows).encode("utf-8")
    # GzipFile, not gzip.compress: before 3.13 compress() let zlib write the header, whose OS byte
    # is the build platform's, while GzipFile writes 255 (unknown) on every version. With mtime=0
    # and no file name the header, and so the file, is byte-identical across rebuilds.
    buffer = io.BytesIO()
    with gzip.GzipFile(filename="", mode="wb", compresslevel=9, fileobj=buffer, mtime=0) as fh:
        fh.write(text)
    path.write_bytes(buffer.getvalue())


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    parser = argparse.ArgumentParser(description="Build the quickstart's synthetic labeled set.")
    sub = parser.add_subparsers(dest="cmd", required=True)
    cap = sub.add_parser("capture", help="run OrderBot over the requests (live, via a cassette)")
    cap.add_argument("--workers", type=int, default=6)
    sub.add_parser("build", help="normalize + grade the frozen runs into the bundle (offline)")
    rec = sub.add_parser("record-judge", help="record the quickstart judge's verdicts (live)")
    rec.add_argument("--workers", type=int, default=6)
    args = parser.parse_args(argv)
    if args.cmd == "capture":
        capture(workers=args.workers)
    elif args.cmd == "build":
        print(json.dumps(build(), indent=2))
    else:
        from corpus.generate.quickstart_judge import record_judge

        record_judge(workers=args.workers)
    return 0


if __name__ == "__main__":
    sys.exit(main())
