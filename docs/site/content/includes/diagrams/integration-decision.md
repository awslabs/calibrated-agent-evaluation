<div class="diagram" markdown>

```mermaid
flowchart TD
    traces(["Does the agent already<br/>export traces?"])
    traces -- "as files" --> files["Ingest the files<br/>calibrated-agent-evaluation ingest"]
    traces -- "to Langfuse" --> lf["Read the project<br/>calibrated-agent-evaluation ingest --from-langfuse"]
    traces -- "no" --> code(["Can you run its code<br/>in your process?"])
    code -- no --> probe["Probe the endpoint<br/>calibrated-agent-evaluation probe"]
    code -- yes --> drive(["Should Calibrated Agent Evaluation choose<br/>the inputs? Strands or<br/>Inspect agents only"])
    drive -- yes --> route["Route inputs through it<br/>calibrated-agent-evaluation generate --run"]
    drive -- no --> capture["Capture a run<br/>calibrated-agent-evaluation capture"]
```

</div>
