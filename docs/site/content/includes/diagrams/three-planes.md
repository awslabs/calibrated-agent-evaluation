<div class="diagram" markdown>

```mermaid
flowchart TD
    sources["your agent<br/><small>trace files, Langfuse, a live run,<br/>or a black-box endpoint</small>"]
    exec["run the agent<br/><small>only when you ask it to</small>"]
    norm["convert<br/><small>one trace shape;<br/>refuses to lose content</small>"]
    data[("trace store<br/><small>a local file or Langfuse</small>")]
    method["evaluate<br/><small>label, check the judge,<br/>score, gate, improve</small>"]
    sources --> norm
    sources -. "live run or endpoint" .-> exec --> norm
    norm --> data
    data <--> method
```

</div>
