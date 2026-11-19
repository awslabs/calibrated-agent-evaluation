<div class="diagram" markdown>

```mermaid
flowchart TD
    file["judge file"] --> lint(["well-formed?<br/>one PASS or FAIL verdict,<br/>reasoning before it"])
    lint -- no --> r0["refused: the message<br/>names the problem"]
    lint -- yes --> cal(["agrees with your labels<br/>often enough to trust?<br/>calibrated-agent-evaluation calibrate"])
    cal -- no --> r1["refused: label N more<br/>of each kind"]
    cal -- yes --> bias(["verdict holds steady<br/>under the bias checks?<br/>calibrated-agent-evaluation bias"])
    bias -- no --> r2["refused: says<br/>which bias"]
    bias -- yes --> judge["calibrated-agent-evaluation judge writes verdicts"]
```

</div>
