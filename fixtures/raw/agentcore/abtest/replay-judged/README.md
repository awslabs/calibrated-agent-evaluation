# H1-judged: the H1 replay set with judge verdicts

H1's two replay roots with trace-level judge verdicts equal to H1's human ones, from the
synthetic raters `target-judge@3f1c0a9e5b7d2468` and `guard-judge@8e2d4b6a1c3f5079`, and a human
label agreeing with the judge on the after-run trace of each of the target's 15 discordant pairs.
`records-certified.json` holds a `CERTIFIED` calibration record and a `CLEAR` bias record for each
judge; `records-uncertified-guard.json` holds the target judge's only. The ids and fingerprints are
made up. The two replay roots are stored gzipped (`control.json.gz`, `treatment.json.gz`). Build with `docs/site/examples/integrate/agentcore_audit.py replay --set h1-judged --to <dir>`.
