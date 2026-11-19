<div class="diagram" markdown>

```mermaid
flowchart LR
    subgraph understand["1. Understand the failures"]
        direction TB
        define --> label --> cluster
    end
    subgraph score["2. Trust a judge"]
        direction TB
        calibrate --> bias --> judge
    end
    subgraph gate["3. Gate and fix"]
        direction TB
        run --> improve
    end
    understand ==> score ==> gate
    gate -. "new kinds of failure" .-> understand
    gate -. "judge prompt or model changed" .-> score
```

</div>
