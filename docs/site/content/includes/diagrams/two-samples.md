<div class="diagram" markdown>

```mermaid
flowchart LR
    subgraph hunted["Hunted calibration set"]
        direction TB
        h1["fail-heavy on purpose:<br/>gold labels per class from the plan"]
    end
    subgraph random["Production-representative sample"]
        direction TB
        r1["drawn at random from real traffic,<br/>judged but not hand-labeled"]
    end
    hunted -- "measures" --> acc["TPR and TNR<br/><small>do not depend on the failure rate,<br/>so they transfer</small>"]
    random -- "measures" --> flag["the judge's flag rate"]
    acc --> rg["Rogan–Gladen corrected rate<br/>with a 95% interval"]
    flag --> rg
    hunted -. "used as the rate sample" .-> refuse["REFUSED<br/>SampleMisuseError"]
```

</div>
