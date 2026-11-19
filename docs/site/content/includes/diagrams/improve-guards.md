<div class="diagram" markdown>

```mermaid
flowchart TD
    reg["register the target metric and minimum reduction<br/><code>calibrated-agent-evaluation improve --register</code><br/><small>frozen before the fix is measured</small>"]
    reg --> held{"confirmation slice disjoint<br/>from the traces the fix was tuned on?"}
    held -- no --> x1["REFUSED<br/>ConfirmationContaminationError"]
    held -- yes --> adj{"judge-produced labels:<br/>discordant pairs adjudicated by a human?"}
    adj -- no --> x2["REFUSED<br/>AdjudicationRequiredError"]
    adj -- yes --> mcnemar["paired McNemar test<br/>on before/after for the same inputs"]
    mcnemar --> panel["guard panel: other slices, Holm-corrected<br/><small>for a crew: the other agents and the workflow outcome rate</small>"]
    panel --> verdict{"significant, clears the registered reduction,<br/>no guard regressed, no relocation?"}
    verdict -- yes --> acc["accepted"]
    verdict -- no --> rej["rejected, with every reason"]
```

</div>
