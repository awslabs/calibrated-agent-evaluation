<div class="diagram" markdown>

```mermaid
flowchart TD
    traces(["Does the agent already<br/>export traces?"])
    traces -- "as files" --> files["Ingest the files<br/><code>ingest</code>"]
    traces -- "to Langfuse" --> lf["Read the project<br/><code>ingest --from-langfuse</code>"]
    traces -- "no" --> code(["Can you run its code<br/>in your process?"])
    code -- no --> probe["Probe the endpoint<br/><code>probe</code>"]
    code -- yes --> drive(["Should the toolkit pick<br/>the inputs? Strands or<br/>Inspect agents only"])
    drive -- yes --> route["Route inputs through it<br/><code>generate --run</code>"]
    drive -- no --> capture["Capture a run<br/><code>capture</code>"]
```

</div>
