# Self-hosted Langfuse stack (the Calibrated Agent Evaluation Langfuse backend)

A pinned, **unmodified** self-hosted Langfuse v3 OSS deployment. Calibrated Agent Evaluation does not fork or patch Langfuse; this compose file stands up the
stock `langfuse:3` images with telemetry disabled.

Every image is pinned by digest, with the tag it was resolved from in a comment above it, so a
`docker compose pull` cannot pick up a different build. To move to a newer build, resolve the
tag's current digest from the registry (for example
`docker buildx imagetools inspect langfuse/langfuse:3`) and replace the digest.

## What it backs

- The opt-in **live integration tests** (`tests/integration/test_langfuse_live.py` and the other
  `@pytest.mark.integration` Langfuse tests), which check that `LangfuseBackend` round-trips against
  a real unmodified instance, not just the unit-test stub.
- `just seed-fixture langfuse`, and the manual acceptance runs that check the local store and
  Langfuse return the same results.
- `just flagship` when it runs without `--local-only`.

## Usage

```sh
just langfuse-up          # start the stack in the background (web on http://localhost:3000)
# ... wait until http://localhost:3000 answers, then run the tests or seed the fixture ...
just test-integration     # runs the @pytest.mark.integration suite against the running stack
just langfuse-down        # stop and remove containers and volumes
```

`just langfuse-up` is `docker compose up -d` and returns before the web container is ready; the
first start pulls images and runs migrations, which takes a minute or more.

Connection settings for the auto-provisioned dev project are in `.env.example`
(`pk-lf-calibrated-agent-evaluation-dev` / `sk-lf-calibrated-agent-evaluation-dev`, host `http://localhost:3000`). `just test-integration`
sets them for you.

## Topology

`langfuse-web`, `langfuse-worker`, Postgres, Clickhouse, Redis and MinIO: the real OSS topology, so
the Clickhouse-backed read paths the adapter uses (`client.api.trace.*`, `client.api.score_v_2.*`)
and the synchronous score write (`client.api.score.create`) run against the same services a
production deployment has.

## Security note

Every published port binds `127.0.0.1`, so the stack is reachable from this machine only.
`docker-compose.yml` holds no credential value; it requires each one from the environment, and
`just langfuse-up` reads the throwaway dev values in `.env.example` (or your own
`docker/langfuse/.env`). Running compose by hand, pass
`--env-file docker/langfuse/.env.example`. Those values are for this loopback dev stack only.
