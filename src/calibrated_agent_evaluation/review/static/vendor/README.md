# Vendored front-end assets

These third-party assets are vendored (served from the local package, not a CDN) so the review
surface works offline and air-gapped: Calibrated Agent Evaluation runs on a laptop with no required cloud.

| File | Version | License | Source |
|---|---|---|---|
| `htmx.min.js` | htmx 2.0.7 | **0BSD** (permissive) | https://cdn.jsdelivr.net/npm/htmx.org@2.0.7/dist/htmx.min.js |

htmx is intrinsic to the FastHTML stack, and FastHTML loads it from a CDN by
default. We disable that (`FastHTML(default_hdrs=False)`) and serve this local copy instead, so
nothing is fetched from the internet at runtime. To update: bump the version here, re-download the
matching `htmx.min.js`, and re-run the review tests.
