// Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

// Tests for src/calibrated_agent_evaluation/review/static/hotkeys.js, run with Node's built-in runner (`node --test`).
//
// No npm dependency and no browser: the script is loaded into a bare VM context (no `document`),
// which makes it export `CalibratedAgentEvaluationHotkeys` without wiring a page, and each test drives
// `createController` through a fake `env` that records what the page would have done. Driven from
// pytest by tests/review/test_hotkeys_js.py so `just check` runs it.
"use strict";

const test = require("node:test");
const assert = require("node:assert/strict");
const fs = require("node:fs");
const path = require("node:path");
const vm = require("node:vm");

const SRC = path.join(__dirname, "..", "..", "src", "calibrated_agent_evaluation", "review", "static", "hotkeys.js");

function load() {
  const context = { encodeURIComponent };
  vm.createContext(context);
  vm.runInContext(fs.readFileSync(SRC, "utf8"), context);
  return context.CalibratedAgentEvaluationHotkeys;
}

const { createController, MIN_DWELL_MS, modeKey, similarModes, similarHint } = load();

// A deferred promise, so a test controls when a request "comes back".
function deferred() {
  let resolve;
  let reject;
  const promise = new Promise((res, rej) => { resolve = res; reject = rej; });
  return { promise, resolve, reject };
}

function response(html, status) {
  return { headers: { get: (h) => (h === "HD-Label-Status" ? status || null : null) }, text: () => Promise.resolve(html) };
}

// Fake page. `traceId` is whatever the last swap put on screen; a fragment is just "TRACE:<id>".
function makeEnv(opts = {}) {
  const env = {
    clock: 10_000,
    shownTrace: opts.trace || "A",
    critiqueText: opts.critique || "",
    modes: !!opts.modes,
    mode: opts.mode || "",
    modeFocus: [],
    swaps: [],
    focused: 0,
    blurred: 0,
    notices: [],
    requests: [], // { url, opts, d }
    now() { return env.clock; },
    fetch(url, o) {
      const d = deferred();
      env.requests.push({ url, opts: o || {}, d });
      return d.promise;
    },
    traceId() { return env.shownTrace; },
    critique() { return env.critiqueText; },
    focusCritique() { env.focused += 1; },
    modesRequired() { return env.modes; },
    failureMode() { return env.mode; },
    focusModes(notice) { env.modeFocus.push(notice); },
    propose: !!opts.propose,
    newMode: opts.newMode || "",
    newModeOpened: 0,
    canPropose() { return env.propose; },
    newModeName() { return env.newMode; },
    openNewMode() { env.newModeOpened += 1; env.mode = ""; },
    formData(outcome, extra, omit) {
      const data = { trace_id: env.shownTrace, outcome, failure_mode: env.mode, ...(extra || {}) };
      if (env.newMode) {
        data.new_mode_name = env.newMode;
        data.new_mode_note = "a note";
      }
      for (const k of omit || []) delete data[k];
      return data;
    },
    swap(html) {
      env.swaps.push(html);
      const m = /^TRACE:(.+)$/.exec(html);
      if (m) env.shownTrace = m[1];
    },
    blur() { env.blurred += 1; },
    suggested: opts.suggested || null,
    suggestedOutcome() { return env.suggested; },
    legendToggles: 0,
    toggleLegend() { env.legendToggles += 1; },
    notice(t) { env.notices.push(t); },
  };
  return env;
}

const labelPosts = (env) => env.requests.filter((r) => r.url === "/label");
const flush = () => new Promise((r) => setImmediate(r));

// Build a controller whose /next prefetch for the current trace has answered with `nextHtml`.
async function ready(env, nextHtml) {
  const c = createController(env);
  c.prefetchNext();
  const pre = env.requests.find((r) => r.url.startsWith("/next"));
  pre.d.resolve(response(nextHtml || "TRACE:B"));
  await flush();
  env.clock += MIN_DWELL_MS; // the labeler has had the trace on screen long enough
  return c;
}

const key = (k, extra = {}) => ({ key: k, repeat: false, target: { tagName: "BODY" }, preventDefault() {}, ...extra });

// --- key repeat and double-tap -------------------------------------------

test("a key-repeat event never submits a verdict", async () => {
  const env = makeEnv();
  const c = await ready(env);
  c.onKeydown(key("1", { repeat: true }));
  assert.equal(labelPosts(env).length, 0);
});

test("holding 1 down submits exactly one PASS, for the trace that was on screen", async () => {
  const env = makeEnv();
  const c = await ready(env);
  c.onKeydown(key("1"));
  for (let i = 0; i < 20; i += 1) c.onKeydown(key("1", { repeat: true }));
  const posts = labelPosts(env);
  assert.equal(posts.length, 1);
  assert.equal(posts[0].opts.body.trace_id, "A");
  assert.equal(env.shownTrace, "B"); // advanced once
});

test("a double-tap lands one verdict: the second tap is on a trace shown for no time", async () => {
  const env = makeEnv();
  const c = await ready(env);
  c.onKeydown(key("1"));
  labelPosts(env)[0].d.resolve(response("", "recorded")); // a fast backend: write already back
  await flush();
  env.clock += 40; // a double-tap is tens of milliseconds
  c.onKeydown(key("1"));
  assert.equal(labelPosts(env).length, 1);
});

test("no second verdict while the first write is still in flight", async () => {
  const env = makeEnv();
  const c = await ready(env);
  c.onKeydown(key("1"));
  env.clock += 5_000; // plenty of dwell on B, but A's write has not come back
  c.onKeydown(key("1"));
  assert.equal(labelPosts(env).length, 1);
  labelPosts(env)[0].d.resolve(response("", "recorded"));
  await flush();
  c.onKeydown(key("1"));
  assert.equal(labelPosts(env).length, 2);
  assert.equal(labelPosts(env)[1].opts.body.trace_id, "B");
});

test("a browser shortcut (Cmd/Ctrl+1) is not a verdict", async () => {
  const env = makeEnv();
  const c = await ready(env);
  c.onKeydown(key("1", { metaKey: true }));
  c.onKeydown(key("1", { ctrlKey: true }));
  assert.equal(labelPosts(env).length, 0);
});

test("undo waits for an in-flight label write instead of overtaking it", async () => {
  const env = makeEnv();
  const c = await ready(env);
  c.onKeydown(key("1"));
  c.onKeydown(key("u"));
  assert.equal(env.requests.filter((r) => r.url === "/undo").length, 0);
});

// --- only `recorded` keeps the optimistic advance ----------------------

for (const status of ["noop", "refused", "error"]) {
  test(`a ${status} response replaces the optimistic view with the server's`, async () => {
    const env = makeEnv();
    const c = await ready(env);
    c.onKeydown(key("1"));
    assert.equal(env.shownTrace, "B");
    labelPosts(env)[0].d.resolve(response("TRACE:A-with-notice", status));
    await flush();
    assert.equal(env.swaps.at(-1), "TRACE:A-with-notice");
  });
}

test("a recorded response keeps the optimistic advance", async () => {
  const env = makeEnv();
  const c = await ready(env);
  c.onKeydown(key("1"));
  labelPosts(env)[0].d.resolve(response("", "recorded"));
  await flush();
  assert.deepEqual(env.swaps, ["TRACE:B"]);
});

test("an optimistic submit tells the server so; a blocking one does not", async () => {
  const env = makeEnv();
  const c = await ready(env);
  c.onKeydown(key("1"));
  assert.equal(labelPosts(env)[0].opts.headers["HD-Optimistic"], "1");

  const env2 = makeEnv();
  const c2 = createController(env2); // no prefetch answered yet
  env2.clock += MIN_DWELL_MS;
  c2.onKeydown(key("1"));
  assert.equal(labelPosts(env2)[0].opts.headers["HD-Optimistic"], undefined);
  labelPosts(env2)[0].d.resolve(response("TRACE:B", "recorded"));
  await flush();
  assert.equal(env2.shownTrace, "B"); // blocking path shows the server's next trace
});

test("a network failure on the write re-reads the view without moving it", async () => {
  const env = makeEnv();
  const c = await ready(env);
  c.onKeydown(key("1"));
  labelPosts(env)[0].d.reject(new Error("offline"));
  await flush();
  const resync = env.requests.find((r) => r.url === "/view");
  assert.ok(resync, "expected a GET /view resync");
  assert.equal(env.requests.filter((r) => r.url.startsWith("/nav")).length, 0);
});

// --- submitting from the critique box -----------------------------------

const inTextarea = { tagName: "TEXTAREA" };

test("bare 1/2 typed into the critique box are text, not verdicts", async () => {
  const env = makeEnv();
  const c = await ready(env);
  c.onKeydown(key("1", { target: inTextarea }));
  c.onKeydown(key("2", { target: inTextarea }));
  assert.equal(labelPosts(env).length, 0);
});

test("FAIL with no critique focuses the box; Ctrl+Enter then submits the FAIL", async () => {
  const env = makeEnv();
  const c = await ready(env);
  c.onKeydown(key("2"));
  assert.equal(env.focused, 1);
  assert.equal(labelPosts(env).length, 0);
  env.critiqueText = "refunded outside the window";
  c.onKeydown(key("Enter", { ctrlKey: true, target: inTextarea }));
  const posts = labelPosts(env);
  assert.equal(posts.length, 1);
  assert.equal(posts[0].opts.body.outcome, "FAIL");
});

test("Cmd+Enter works the same as Ctrl+Enter", async () => {
  const env = makeEnv();
  const c = await ready(env);
  c.onKeydown(key("2"));
  env.critiqueText = "wrong tool";
  c.onKeydown(key("Enter", { metaKey: true, target: inTextarea }));
  assert.equal(labelPosts(env).length, 1);
});

test("Ctrl+Enter with no verdict asked for submits nothing", async () => {
  const env = makeEnv({ critique: "some notes" });
  const c = await ready(env);
  c.onKeydown(key("Enter", { ctrlKey: true, target: inTextarea }));
  assert.equal(labelPosts(env).length, 0);
});

test("Escape leaves the field so the bare hotkeys work again", async () => {
  const env = makeEnv();
  const c = await ready(env);
  c.onKeydown(key("Escape", { target: inTextarea }));
  assert.equal(env.blurred, 1);
});

test("keys typed in a span <select> are not hotkeys", async () => {
  const env = makeEnv();
  const c = await ready(env);
  c.onKeydown(key("1", { target: { tagName: "SELECT" } }));
  c.onKeydown(key("ArrowRight", { target: { tagName: "SELECT" } }));
  assert.equal(env.requests.filter((r) => r.url === "/label" || r.url.startsWith("/nav")).length, 0);
});

// --- the failure-mode picker -------------------------------------------------

test("with a codebook, FAIL with a critique but no mode focuses the picker and sends nothing", async () => {
  const env = makeEnv({ modes: true, critique: "refunded late" });
  const c = await ready(env);
  c.onKeydown(key("2"));
  assert.equal(labelPosts(env).length, 0);
  assert.equal(env.modeFocus.length, 1);
  assert.match(env.modeFocus[0], /failure mode/);
});

test("once a mode is picked, Ctrl+Enter from the picker submits the FAIL with it", async () => {
  const env = makeEnv({ modes: true, critique: "refunded late" });
  const c = await ready(env);
  c.onKeydown(key("2"));
  env.mode = "refund_outside_window";
  c.onKeydown(key("Enter", { ctrlKey: true, target: { tagName: "SELECT" } }));
  const posts = labelPosts(env);
  assert.equal(posts.length, 1);
  assert.equal(posts[0].opts.body.outcome, "FAIL");
  assert.equal(posts[0].opts.body.failure_mode, "refund_outside_window");
});

test("an empty critique is asked for before the failure mode", async () => {
  const env = makeEnv({ modes: true });
  const c = await ready(env);
  c.onKeydown(key("2"));
  assert.equal(env.focused, 1);
  assert.equal(env.modeFocus.length, 0);
});

test("a PASS never sends a failure mode, even one picked earlier", async () => {
  const env = makeEnv({ modes: true, mode: "refund_outside_window" });
  const c = await ready(env);
  c.onKeydown(key("1"));
  const [post] = labelPosts(env);
  assert.equal(post.opts.body.outcome, "PASS");
  assert.equal("failure_mode" in post.opts.body, false);
});

test("M focuses the picker when the codebook lists modes, and is ignored otherwise", async () => {
  const env = makeEnv({ modes: true });
  const c = await ready(env);
  c.onKeydown(key("m"));
  assert.deepEqual(env.modeFocus, [null]);

  const open = makeEnv();
  const c2 = await ready(open);
  c2.onKeydown(key("m"));
  assert.equal(open.modeFocus.length, 0);
});

test("without a codebook a FAIL with a critique submits with no mode (open coding)", async () => {
  const env = makeEnv({ critique: "refunded late" });
  const c = await ready(env);
  c.onKeydown(key("2"));
  assert.equal(labelPosts(env).length, 1);
});

// --- defer/undo/nav do not leak form nodes ------------------------------

test("defer, undo and nav are plain POSTs whose fragment is swapped in", async () => {
  const env = makeEnv();
  const c = await ready(env);
  c.onKeydown(key("d"));
  c.onKeydown(key("u"));
  c.onKeydown(key("ArrowRight"));
  c.onKeydown(key("ArrowLeft"));
  const urls = env.requests.map((r) => r.url).filter((u) => !u.startsWith("/next"));
  assert.deepEqual(urls, ["/defer", "/undo", "/nav?to=next", "/nav?to=prev"]);
  for (const r of env.requests.filter((x) => !x.url.startsWith("/next"))) {
    assert.equal(r.opts.method, "POST");
    r.d.resolve(response("TRACE:C"));
  }
  await flush();
  assert.equal(env.swaps.length, 4);
});

// --- the legend ---------------------------------------------------------

test("? toggles the hotkey legend, and is plain text inside a field", async () => {
  const env = makeEnv();
  const c = await ready(env);
  c.onKeydown(key("?"));
  c.onKeydown(key("?"));
  c.onKeydown(key("?", { target: inTextarea }));
  assert.equal(env.legendToggles, 2);
});

// --- ratification is its own action -------------------------------------

test("A accepts the suggestion: its verdict, with the explicit ratified flag", async () => {
  const env = makeEnv({ suggested: "FAIL", critique: "the LLM's critique" });
  const c = await ready(env);
  c.onKeydown(key("a"));
  const [post] = labelPosts(env);
  assert.equal(post.opts.body.outcome, "FAIL");
  assert.equal(post.opts.body.ratified, "1");
});

test("a verdict key never sends the ratified flag", async () => {
  const env = makeEnv({ suggested: "PASS" });
  const c = await ready(env);
  c.onKeydown(key("1"));
  assert.equal(labelPosts(env)[0].opts.body.ratified, undefined);
});

test("A with no suggestion on screen does nothing", async () => {
  const env = makeEnv();
  const c = await ready(env);
  c.onKeydown(key("a"));
  assert.equal(labelPosts(env).length, 0);
});

// --- history -------------------------------------------------------------

test("H opens the history view with a GET", async () => {
  const env = makeEnv();
  const c = await ready(env);
  c.onKeydown(key("h"));
  const req = env.requests.find((r) => r.url === "/history");
  assert.ok(req);
  assert.equal(req.opts.method, undefined);
});

test("a verdict key on a view with no trace to label does nothing", async () => {
  const env = makeEnv();
  const c = await ready(env);
  env.shownTrace = null; // the history or completion view has no label form
  c.onKeydown(key("1"));
  assert.equal(labelPosts(env).length, 0);
});

// --- the page bootstrap ------------------------------------------------------------------------

test("on a page, the script registers keydown, click, input, submit and load listeners", () => {
  const listeners = {};
  const document = {
    addEventListener(type, fn) { listeners[type] = fn; },
    getElementById() { return null; },
  };
  const context = { encodeURIComponent, document, window: {} };
  vm.createContext(context);
  vm.runInContext(fs.readFileSync(SRC, "utf8"), context);
  assert.deepEqual(Object.keys(listeners).sort(), ["DOMContentLoaded", "click", "input", "keydown", "submit"]);
  // Enter in a field submits the label form natively; the controller always takes that over.
  let prevented = false;
  listeners.submit({ target: { id: "label-form" }, preventDefault() { prevented = true; } });
  assert.ok(prevented);
});


// --- proposing a new failure mode ------------------------------------------

test("N opens the new-mode fields and clears a picked mode, only when the form offers them", async () => {
  const env = makeEnv({ modes: true, mode: "wrong_tool", propose: true });
  const c = await ready(env);
  c.onKeydown(key("n"));
  assert.equal(env.newModeOpened, 1);
  assert.equal(env.mode, "");
  const closed = makeEnv({ modes: true });
  const c2 = await ready(closed);
  c2.onKeydown(key("N"));
  assert.equal(closed.newModeOpened, 0);
});

test("a FAIL with a new mode and no picked one is submitted, and waits for the server's view", async () => {
  const env = makeEnv({ modes: true, propose: true, critique: "promised a callback", newMode: "promised callback" });
  const c = await ready(env);
  c.onKeydown(key("2"));
  const [post] = labelPosts(env);
  assert.equal(post.opts.body.new_mode_name, "promised callback");
  assert.equal(post.opts.headers["HD-Optimistic"], undefined); // the prefetched view is stale
  assert.equal(env.shownTrace, "A");
  post.d.resolve(response("TRACE:B", "recorded"));
  await flush();
  assert.equal(env.shownTrace, "B");
  assert.deepEqual(env.modeFocus, []);
});

test("a PASS never sends a half-typed new mode", async () => {
  const env = makeEnv({ modes: true, propose: true, newMode: "promised callback" });
  const c = await ready(env);
  c.onKeydown(key("1"));
  const [post] = labelPosts(env);
  assert.equal(post.opts.body.new_mode_name, undefined);
  assert.equal(post.opts.body.new_mode_note, undefined);
});

test("a FAIL with neither a picked mode nor a new one still asks for a mode", async () => {
  const env = makeEnv({ modes: true, propose: true, critique: "late" });
  const c = await ready(env);
  c.onKeydown(key("2"));
  assert.equal(labelPosts(env).length, 0);
  assert.equal(env.modeFocus.length, 1);
});

test("N typed inside a text field is text, not a hotkey", async () => {
  const env = makeEnv({ modes: true, propose: true });
  const c = await ready(env);
  c.onKeydown(key("n", { target: { tagName: "INPUT" } }));
  assert.equal(env.newModeOpened, 0);
});

test("mode names compare without case or spacing", () => {
  assert.equal(modeKey("  Promised   Callback "), "promised callback");
});

test("similar existing modes are found by name, containment and shared words", () => {
  const names = ["refund_outside_window", "wrong_tool", "Promised Callback", "late refund"];
  assert.deepEqual(Array.from(similarModes("promised  callback", names)), ["Promised Callback"]);
  assert.deepEqual(Array.from(similarModes("wrong_tool_call", names)), ["wrong_tool"]);
  assert.deepEqual(Array.from(similarModes("refund issued late", names)), ["refund_outside_window", "late refund"]);
  assert.deepEqual(Array.from(similarModes("ab", names)), []);
  assert.deepEqual(Array.from(similarModes("hallucinated policy", names)), []);
});

test("the hint names an exact match as the mode the label will use", () => {
  const names = ["Promised Callback", "wrong_tool"];
  assert.match(similarHint("promised callback", names), /Already in the picker as Promised Callback/);
  assert.match(similarHint("wrong_tool_call", names), /Similar modes already in the picker: wrong_tool/);
  assert.equal(similarHint("brand new thing", names), "");
});
