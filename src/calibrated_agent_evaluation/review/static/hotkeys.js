// Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

// Calibrated Agent Evaluation review-surface hotkeys + optimistic auto-advance.
//
// The throughput engine for fast labeling: keystrokes drive labeling, and a
// verdict key advances the view IMMEDIATELY from a prefetched next trace, so the labeler never
// waits on the backend write. The label write rides a separate POST. The server tags each
// write response with an `HD-Label-Status` header, and only `recorded` means the write landed as
// the view shows; on anything else (`noop`, `refused`, `error`) the client swaps in the server's
// fragment, so a write that did not land never reads as recorded.
//
// The logic lives in `createController(env)`, which touches the page only through `env`, so it
// can be tested without a browser (tests/js/hotkeys.test.js). The bottom of this file wires `env`
// to the real DOM. One dependency-free vanilla-JS asset: no build step, no framework.
(function (global) {
  "use strict";

  // A verdict on a trace that has been on screen for less than this is ignored. OS key-repeat is
  // caught by `e.repeat`, but a double-tap is two real keydowns, and on a fast backend the first
  // write is done before the second tap, so the in-flight lock alone does not stop it. A labeler
  // reading a trace spends seconds on it; 300 ms is far below that and above a double-tap.
  var MIN_DWELL_MS = 300;

  // How a mode name is compared: case and spacing ignored (the server's `mode_key`).
  function modeKey(name) {
    return String(name || "").trim().split(/\s+/).join(" ").toLowerCase();
  }

  function tokens(name) {
    return modeKey(name).split(/[^a-z0-9]+/).filter(function (t) { return t.length >= 4; });
  }

  // The existing modes a new name may duplicate: the same name, one inside the other, or a
  // shared word of four letters or more ("refund_late" and "late refund"). Shown while the
  // labeler types, so they pick the existing mode instead of proposing it again.
  function similarModes(name, names) {
    var key = modeKey(name);
    if (key.length < 3) return [];
    var words = tokens(name);
    return (names || []).filter(function (other) {
      var otherKey = modeKey(other);
      if (otherKey === key || otherKey.indexOf(key) !== -1 || key.indexOf(otherKey) !== -1) {
        return true;
      }
      var otherWords = tokens(other);
      return words.some(function (w) { return otherWords.indexOf(w) !== -1; });
    });
  }

  function similarHint(name, names) {
    var key = modeKey(name);
    var exact = (names || []).filter(function (n) { return modeKey(n) === key; });
    if (exact.length) return "Already in the picker as " + exact[0] + "; the label will use it.";
    var similar = similarModes(name, names);
    if (!similar.length) return "";
    return "Similar modes already in the picker: " + similar.join(", ") + ". Pick one with M if it fits.";
  }

  function isTypingTarget(target) {
    if (!target) return false;
    var tag = target.tagName;
    return tag === "TEXTAREA" || tag === "INPUT" || tag === "SELECT" || !!target.isContentEditable;
  }

  function createController(env) {
    var prefetched = null; // { after: <trace_id>, html: <string> }
    var prefetching = false;
    var submitting = false; // in-flight lock: one label write at a time
    var shownAt = env.now();
    var lastAttempted = null; // the verdict last asked for on this trace, for Ctrl/Cmd+Enter

    function prefetchNext() {
      if (prefetching) return;
      prefetching = true;
      var after = env.traceId();
      var url = "/next" + (after ? "?after=" + encodeURIComponent(after) : "");
      env.fetch(url, { headers: { "HX-Request": "true" } })
        .then(function (r) { return r.text(); })
        .then(function (html) { prefetched = { after: after, html: html }; prefetching = false; })
        .catch(function () { prefetched = null; prefetching = false; });
    }

    // Every swap is a new trace on screen: restart the dwell clock and the prefetch.
    function shown() {
      shownAt = env.now();
      lastAttempted = null;
      prefetched = null;
      prefetchNext();
    }

    function swap(html) {
      env.swap(html);
      shown();
    }

    // Re-read the server's view without moving it: the recovery path after a failed request.
    function resync() {
      return env.fetch("/view", { headers: { "HX-Request": "true" } })
        .then(function (r) { return r.text(); })
        .then(swap)
        .catch(function () { env.notice("The review server is unreachable; nothing was recorded."); });
    }

    function submitLabel(outcome, optimistic, extra) {
      submitting = true;
      // A PASS carries no failure mode; the server refuses one, so a mode picked before the
      // labeler changed their mind is not sent.
      var omit = outcome === "PASS" ? ["failure_mode", "new_mode_name", "new_mode_note"] : [];
      var body = env.formData(outcome, extra, omit);
      var headers = { "HX-Request": "true" };
      if (optimistic) headers["HD-Optimistic"] = "1";
      return env.fetch("/label", { method: "POST", body: body, headers: headers })
        .then(function (resp) {
          var status = resp.headers.get("HD-Label-Status");
          return resp.text().then(function (html) {
            submitting = false;
            // `recorded` after an optimistic advance: the view already shows the truth. Every
            // other status, or a non-optimistic submit, takes the server's fragment (04).
            if (status !== "recorded" || !optimistic) swap(html);
          });
        })
        .catch(function () {
          submitting = false;
          return resync();
        });
    }

    function onVerdict(outcome, extra) {
      if (submitting) return false; // the previous write has not come back yet
      if (env.now() - shownAt < MIN_DWELL_MS) return false; // not on screen long enough to read
      if (!env.traceId()) return false; // a read-only view (history, completion): nothing to label
      lastAttempted = outcome;
      // Mirrors the server rule: a FAIL needs a critique. Block it before advancing, and leave the cursor in
      // the critique box so Ctrl/Cmd+Enter can submit once it is written.
      if (outcome === "FAIL" && !env.critique().trim()) {
        env.focusCritique("A FAIL needs a critique: what went wrong? (required) Ctrl+Enter submits.");
        return false;
      }
      // Mirrors the server rule: when the codebook lists failure modes, a FAIL names one, or
      // proposes a new one.
      var proposing = outcome === "FAIL" && !!env.newModeName().trim();
      if (outcome === "FAIL" && env.modesRequired() && !env.failureMode() && !proposing) {
        env.focusModes("A FAIL needs a failure mode: pick one, then Ctrl+Enter submits.");
        return false;
      }
      var after = env.traceId();
      // A proposal changes the picker, so the prefetched next trace is stale: wait for the
      // server's view, which shows the new mode and says it was recorded.
      var next = !proposing && prefetched && prefetched.after === after ? prefetched.html : null;
      submitLabel(outcome, next !== null, extra); // reads the form before the swap replaces it
      if (next !== null) swap(next);
      return true;
    }

    // Accept the LLM suggestion as shown, which a human must do explicitly. The only path that sends `ratified=1`; a verdict
    // key on an unedited suggestion is refused by the server.
    function accept() {
      var suggested = env.suggestedOutcome();
      if (!suggested) return false;
      return onVerdict(suggested, { ratified: "1" });
    }

    // Defer / undo / navigate: POST, then show what the server says. Blocked while a label write
    // is in flight, because the server applies them to the session in arrival order and an undo
    // that overtook a write would retract the label before it.
    function action(path) {
      if (submitting) return false;
      env.fetch(path, { method: "POST", headers: { "HX-Request": "true" } })
        .then(function (r) { return r.text(); })
        .then(swap)
        .catch(resync);
      return true;
    }

    // A read-only view (history): GET it and show it.
    function show(path) {
      if (submitting) return false;
      env.fetch(path, { headers: { "HX-Request": "true" } })
        .then(function (r) { return r.text(); })
        .then(swap)
        .catch(resync);
      return true;
    }

    function onKeydown(e) {
      if (e.repeat) return; // a held key is one keypress, never a walk down the work-list (03)
      if ((e.metaKey || e.ctrlKey) && e.key === "Enter") {
        // Submit from inside the critique box (07). Only the verdict the labeler already asked
        // for: Ctrl+Enter never picks one.
        if (lastAttempted) {
          e.preventDefault();
          env.blur(e.target);
          onVerdict(lastAttempted);
        }
        return;
      }
      if (isTypingTarget(e.target)) {
        if (e.key === "Escape") env.blur(e.target); // then the bare hotkeys work again
        return; // otherwise keys belong to the field being typed in
      }
      if (e.metaKey || e.ctrlKey || e.altKey) return; // browser shortcuts (Cmd+1 is a tab switch)
      switch (e.key) {
        case "1": e.preventDefault(); onVerdict("PASS"); break;
        case "2": e.preventDefault(); onVerdict("FAIL"); break;
        case "d": case "D":
          e.preventDefault();
          if (env.now() - shownAt >= MIN_DWELL_MS) action("/defer");
          break;
        case "u": case "U": e.preventDefault(); action("/undo"); break;
        case "ArrowRight": e.preventDefault(); action("/nav?to=next"); break;
        case "ArrowLeft": e.preventDefault(); action("/nav?to=prev"); break;
        case "?": e.preventDefault(); env.toggleLegend(); break;
        case "m": case "M":
          if (env.modesRequired()) { e.preventDefault(); env.focusModes(null); }
          break;
        case "n": case "N":
          if (env.canPropose()) { e.preventDefault(); env.openNewMode(); }
          break;
        case "a": case "A": e.preventDefault(); accept(); break; // explicit accept
        case "h": case "H": e.preventDefault(); show("/history"); break;
        default: break;
      }
    }

    return {
      onKeydown: onKeydown,
      onVerdict: onVerdict,
      accept: accept,
      action: action,
      shown: shown,
      prefetchNext: prefetchNext,
      isSubmitting: function () { return submitting; },
    };
  }

  global.CalibratedAgentEvaluationHotkeys = {
    createController: createController,
    MIN_DWELL_MS: MIN_DWELL_MS,
    modeKey: modeKey,
    similarModes: similarModes,
    similarHint: similarHint,
  };

  if (typeof document === "undefined") return; // loaded by the test harness, not a page

  // --- the real page -----------------------------------------------------------------------
  function root() { return document.getElementById("review-root"); }
  function form() { return document.getElementById("label-form"); }

  var env = {
    now: function () { return Date.now(); },
    fetch: function (url, opts) { return window.fetch(url, opts); },
    traceId: function () {
      var hidden = form() && form().querySelector('input[name="trace_id"]');
      return hidden ? hidden.value : null;
    },
    critique: function () {
      var c = form() && form().querySelector('[name="critique"]');
      return c ? c.value : "";
    },
    focusCritique: function (placeholder) {
      var c = form() && form().querySelector('[name="critique"]');
      if (!c) return;
      c.setAttribute("placeholder", placeholder);
      c.focus();
    },
    modesRequired: function () {
      return !!form() && form().getAttribute("data-modes-required") === "1";
    },
    failureMode: function () {
      var m = form() && form().querySelector('[name="failure_mode"]');
      return m ? m.value : "";
    },
    focusModes: function (notice) {
      var m = form() && form().querySelector('[name="failure_mode"]');
      if (!m) return;
      var hint = form().querySelector(".hd-failure-mode-hint");
      if (notice && hint) hint.textContent = notice;
      m.focus();
    },
    canPropose: function () { return !!form() && !!form().querySelector(".hd-new-mode"); },
    newModeName: function () {
      var n = form() && form().querySelector('[name="new_mode_name"]');
      return n ? n.value : "";
    },
    openNewMode: function () {
      var box = form() && form().querySelector(".hd-new-mode");
      if (!box) return;
      box.hidden = false;
      // One mode per label: opening a proposal clears a pick from the list.
      var m = form().querySelector('[name="failure_mode"]');
      if (m) m.value = "";
      var n = box.querySelector('[name="new_mode_name"]');
      if (n) n.focus();
    },
    formData: function (outcome, extra, omit) {
      var data = new FormData(form() || undefined);
      data.set("outcome", outcome);
      (omit || []).forEach(function (k) { data.delete(k); });
      if (extra) Object.keys(extra).forEach(function (k) { data.set(k, extra[k]); });
      return data;
    },
    swap: function (html) {
      var r = root();
      if (!r) return;
      // nosemgrep: insecure-innerhtml,insecure-document-method  -- `html` is a server-rendered
      // FastHTML fragment from this app's own endpoints (auto-escaped), not attacker input; the
      // review UI is a local, offline, single-user tool.
      r.innerHTML = html;
      if (window.htmx) window.htmx.process(r);
    },
    blur: function (el) { if (el && el.blur) el.blur(); },
    suggestedOutcome: function () {
      var f = form();
      return f ? f.getAttribute("data-suggested-outcome") : null;
    },
    toggleLegend: function () {
      var legend = document.getElementById("hd-legend");
      if (legend) legend.hidden = !legend.hidden;
    },
    notice: function (text) {
      var r = root();
      if (!r) return;
      var n = document.createElement("div");
      n.className = "hd-notice";
      n.setAttribute("role", "alert");
      n.textContent = text;
      r.insertBefore(n, r.firstChild);
    },
  };

  var controller = createController(env);
  document.addEventListener("keydown", controller.onKeydown);
  // Verdict buttons go through the same guards as the keys (a mouse double-click is a double-tap).
  document.addEventListener("click", function (e) {
    if (!e.target || !e.target.closest) return;
    if (e.target.closest("#label-form .hd-accept")) {
      e.preventDefault();
      controller.accept();
      return;
    }
    var btn = e.target.closest('#label-form button[name="outcome"]');
    if (!btn) return;
    e.preventDefault();
    controller.onVerdict(btn.value);
  });
  // While a new mode is typed, show the existing modes it may duplicate.
  document.addEventListener("input", function (e) {
    if (!e.target || e.target.name !== "new_mode_name") return;
    var f = form();
    var hint = f && f.querySelector(".hd-new-mode-similar");
    if (!hint) return;
    var names = [];
    try { names = JSON.parse(f.getAttribute("data-mode-names") || "[]"); } catch (err) { names = []; }
    hint.textContent = similarHint(e.target.value, names);
  });
  // Enter in a text field would submit the form with no verdict; the controller owns submission.
  document.addEventListener("submit", function (e) {
    if (e.target && e.target.id === "label-form") e.preventDefault();
  });
  document.addEventListener("DOMContentLoaded", function () {
    controller.prefetchNext();
    startClock();
    // The history links are htmx swaps; each one is a new view, like a controller swap.
    if (document.body) document.body.addEventListener("htmx:afterSwap", controller.shown);
  });

  // Keep the "on traces" clock running between renders. It is anchored on the value the first
  // page render carried, then advances with wall time. A prefetched fragment's own value dates
  // from when it was prefetched, so it is overwritten rather than trusted.
  function startClock() {
    var first = document.querySelector(".hd-elapsed[data-elapsed-seconds]");
    if (!first || !window.setInterval) return;
    var base = parseFloat(first.getAttribute("data-elapsed-seconds")) || 0;
    var loadedAt = Date.now();
    window.setInterval(function () {
      var el = document.querySelector(".hd-elapsed");
      if (!el) return;
      var total = Math.floor(base + (Date.now() - loadedAt) / 1000);
      var secs = total % 60;
      el.textContent = Math.floor(total / 60) + ":" + (secs < 10 ? "0" : "") + secs + " on traces";
    }, 1000);
  }
})(typeof globalThis !== "undefined" ? globalThis : this);
