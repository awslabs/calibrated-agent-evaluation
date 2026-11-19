# H2: the recommendation audit's replay set

`input.json.gz` and `result.json.gz` (gzipped JSON): 40 confirmation questions disjoint from H1's and from both
development sets, human verdicts with target cells `(4, 10, 2, 24)` and guard cells `(3, 1, 2, 34)`,
provenance naming the recommendation fixture's input and result versions. `development.json`: the
10 traces whose ids are the recommendation fixtures' session-span trace ids. Synthetic, written by
`../../make_replay_roots.py`; build with `docs/site/examples/integrate/agentcore_audit.py replay --set h2 --to <dir>`.
