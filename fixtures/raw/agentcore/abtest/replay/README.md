# H1: the A/B audit's replay set

Synthetic JSON descriptions of local stores, written by `../../make_replay_roots.py` (which
imports no Calibrated Agent Evaluation code). Each file lists a store's traces, their trace-level labels and, for a
replay root, the run provenance its executor recorded. Build them into stores with
`uv run python docs/site/examples/integrate/agentcore_audit.py replay --set h1 --to <dir>`; the tests build them with
`tests/_replay_fixture.py`.

- `control.json.gz`, `treatment.json.gz` (gzipped JSON, as every description over 8 KB is): 40 confirmation traces, inputs `{"prompt": ...}`, human
  verdicts planted so the target's cells are `(2, 14, 1, 23)` and the guard's `(3, 1, 2, 34)`;
  provenance names the fixture bundle's versions `...0001` and `...0002`.
- `development.json`: 10 other questions as `{"answer": ...}`, one with a human FAIL for a third
  mode no replay root labels. `development-overlap.json` adds confirmation question 7 under another
  id; `development-messages.json` adds a trace whose input has no single prompt text.
