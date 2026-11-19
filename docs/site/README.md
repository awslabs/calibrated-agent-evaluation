# Calibrated Agent Evaluation documentation site

A static documentation site for `calibrated-agent-evaluation`, built with
[MkDocs](https://www.mkdocs.org/) and [Material for MkDocs](https://squidfunk.github.io/mkdocs-material/).
The site build is pure Python and installs from the `docs` dependency group.

## Layout

```
docs/site/
  mkdocs.yml            # site config: theme, plugins, nav, strict link validation
  STYLE.md              # the writing standard for every page under content/
  content/              # hand-written pages (docs_dir)
    index.md            # home page
    why/                # why Calibrated Agent Evaluation, the primer, the leaders' page, alternatives, cost
    getting-started/    # installation, quickstart, how it works
    integrate/          # getting traces in: ingest, Langfuse, capture, route, probe, gallery, skills
    guides/             # task how-tos, starting with the lifecycle guide under guides/lifecycle/
    recipes/            # short task recipes
    concepts/           # explanation: stage map, traces, how the numbers work, methods, glossary
    includes/           # snippets: abbreviations (tooltips and glossary), mermaid diagrams
    assets/             # images, javascripts, stylesheets
    bias-probe-format.md, versioning.md, changelog.md, llms.txt
  examples/             # example programs whose output the pages include
  overrides/            # Material template overrides
  inventories/          # vendored Sphinx inventory for standard-library links (python3.inv)
  docs_examples.py      # the captured-output registry: every golden and the command behind it
  gen_examples.py       # build-time: fail the build when a golden no longer matches the code
  docs_charts.py        # the data charts, drawn from committed study data
  data/                 # series the charts read
  gen_charts.py         # build-time: render the charts into assets/charts/
  gen_glossary.py       # build-time: the glossary page, from includes/abbreviations.md
  gen_ref_pages.py      # build-time: one API reference page per public calibrated_agent_evaluation package
  gen_cli_pages.py      # build-time: the calibrated-agent-evaluation CLI reference, from the real argparse parser
  griffe_roles.py       # build-time: rewrite Sphinx :func:/:mod: roles into mkdocstrings autorefs
  hooks.py              # build-time: keep reference/SUMMARY.md (nav input) out of the site
  _site/                # build output (gitignored)
```

Captured output lives under `fixtures/expected/docs/`, listed in its `MANIFEST`. Pages include
those files; nobody hand-writes command output into a page.

The nav follows [Diataxis](https://diataxis.fr/): Guides and Recipes (tasks), Concepts
(explanation), Reference (the generated CLI and API pages), plus Why Calibrated Agent Evaluation, Get started and
Integrate for orientation.

The API reference and CLI reference are generated at build time and are not committed.
`gen_ref_pages.py` walks `src/calibrated_agent_evaluation/` and renders each public package's docstrings through
[mkdocstrings](https://mkdocstrings.github.io/); `gen_cli_pages.py` imports the shipped
`build_parser()` and renders `calibrated-agent-evaluation --help` for every subcommand at a fixed width of 100 columns.
A new package or subcommand appears on the next build with no hand-editing.

## Build and preview

```bash
# from the repo root
just docs          # live-reload dev server at http://127.0.0.1:8000 (skips the golden check)
just docs-build    # strict build into docs/site/_site/; fails on broken links, bad refs or a stale golden
just docs-goldens  # re-run every captured example and rewrite its golden
```

Or directly, from the repo root:

```bash
uv run --group docs --all-extras mkdocs serve -f docs/site/mkdocs.yml
uv run --group docs --all-extras mkdocs build --strict -f docs/site/mkdocs.yml
```

`--all-extras` is needed because mkdocstrings imports modules behind optional extras and the golden
check runs examples that use the `stats` code. Set
`CALIBRATED_AGENT_EVALUATION_DOCS_SKIP_EXAMPLES=1` to skip the golden
check for a fast serve. Run from the repo root, not from `docs/site/`: the vendored inventory is
read from a path relative to the working directory.

### API reference details

- Standard-library links: types such as `str`, `Exception` or `typing.Protocol` link to
  docs.python.org through the Sphinx inventory vendored at `inventories/python3.inv`, so the build
  needs no network. Refresh it with
  `curl -sSfL -o docs/site/inventories/python3.inv https://docs.python.org/3/objects.inv`.
- Vendor types stay plain text: `strands.models.model.Model` and `inspect_ai.model.ModelAPI`
  render unlinked, since no inventory is configured for either project.
- Inherited members of external bases are not listed: Griffe analyses the source statically and
  does not load the standard library or vendor SDKs, so a class deriving from `Exception`,
  `typing.Protocol`, `enum.StrEnum`, `UserWarning`, `RuntimeError` or one of the two vendor types
  shows its own members only. `mkdocs build -v` logs this as `Base class ... is not loaded` at
  DEBUG level (877 lines at the time of writing). The omitted members belong to
  Python or the vendor.
- Private modules are skipped: `gen_ref_pages.py` leaves out any module with a `_`-prefixed
  component in its path, including whole private packages such as `calibrated_agent_evaluation._quickstart`.
- `reference/SUMMARY.md` is nav input: `gen_ref_pages.py` writes it for literate-nav, and
  `hooks.py` excludes it from the built site.
- Two environment variables silence two banners about MkDocs 2 that are not build warnings:
  Material reads `NO_MKDOCS_2_WARNING` and properdocs reads `DISABLE_MKDOCS_2_WARNING`. The `just`
  recipes set both.

## Hosting

The build output (`docs/site/_site/`) is a self-contained static site. By default it is built for
the GitHub Pages site, `https://awslabs.github.io/calibrated-agent-evaluation/`, with a header link
to the GitHub repository. To host it elsewhere, set `CALIBRATED_AGENT_EVALUATION_DOCS_SITE_URL` to
the deployed base URL so Material emits correct absolute asset URLs, including the search worker's.
`CALIBRATED_AGENT_EVALUATION_DOCS_REPO_URL` and `CALIBRATED_AGENT_EVALUATION_DOCS_REPO_NAME` set
the repository link in the header. Page links are relative and need no change.

The header's repository link is Material's `repo_url` widget. On each visitor's first page view
in a browser session it calls the GitHub REST API (`api.github.com/repos/<owner>/<repo>` and its
latest release) to show the star count, fork count and latest version, then caches the answer in
`sessionStorage`. This is the only request the built site makes to a third party. Material 9.7
has no setting that turns the fetch off on its own; its cookie consent feature (`extra.consent`
with a `github` cookie) skips the fetch until the visitor consents, at the cost of a consent
banner.
