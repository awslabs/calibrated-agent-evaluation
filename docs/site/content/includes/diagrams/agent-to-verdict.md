<div class="diagram" markdown>

```mermaid
flowchart TD
    raw["your agent's traces<br/><small>OTLP, Inspect, Strands, ...</small>"] --> check(["converts with nothing lost?"])
    check -- no --> refused["refused,<br/>nothing written"]
    check -- yes --> store[("trace store")]
    store --> gold["your labels"]
    store --> scores["judge verdicts"]
    gold --> calib["how often the judge<br/>agrees with you"]
    calib --> runrec["gate result and<br/>corrected failure rate"]
    scores --> runrec
```

</div>
