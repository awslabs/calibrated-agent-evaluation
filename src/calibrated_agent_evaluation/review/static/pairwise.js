// Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

// Calibrated Agent Evaluation A/B page keys: a or 1 picks the left trace, b or 2 the right one, s skips.
// A key clicks the matching button, so the write goes through htmx like a click (same headers,
// same server checks). One write at a time, and a key within 300 ms of a new pair appearing is
// ignored, so a double-tap cannot record the next pair unseen.
(function () {
  "use strict";
  var MIN_DWELL_MS = 300;
  var KEYS = { a: "a", "1": "a", b: "b", "2": "b", s: "skip" };
  var busy = false;
  var shownAt = Date.now();

  document.addEventListener("htmx:beforeRequest", function () { busy = true; });
  document.addEventListener("htmx:afterRequest", function () { busy = false; });
  document.addEventListener("htmx:afterSwap", function () { shownAt = Date.now(); });

  document.addEventListener("keydown", function (e) {
    if (e.repeat || e.ctrlKey || e.metaKey || e.altKey) return;
    var choice = KEYS[e.key.toLowerCase()];
    if (!choice || busy || Date.now() - shownAt < MIN_DWELL_MS) return;
    var button = document.querySelector('#review-root button[data-choice="' + choice + '"]');
    if (!button) return;
    e.preventDefault();
    button.click();
  });
})();
