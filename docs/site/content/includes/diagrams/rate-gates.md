<div class="diagram" markdown>

```mermaid
flowchart TD
    run["calibrated-agent-evaluation run with a trusted judge"] --> labels(["enough labels for a rate<br/>at this judge's accuracy?"])
    labels -- no --> r1["no rate yet: label N<br/>more of each kind"]
    labels -- yes --> sample(["at least 150 randomly<br/>chosen traces scored?"])
    sample -- no --> r2["no rate yet: score<br/>N more traces"]
    sample -- yes --> sep(["does the judge tell failures<br/>from passes well enough?"])
    sep -- no --> r3["range only,<br/>no single number"]
    sep -- yes --> ok["failure rate with its likely range"]
```

</div>
