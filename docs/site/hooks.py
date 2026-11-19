# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""MkDocs hooks for the Calibrated Agent Evaluation docs site (loaded by path from mkdocs.yml
``hooks:``).

``gen_ref_pages.py`` and ``gen_cli_pages.py`` write ``reference/SUMMARY.md`` and ``cli/SUMMARY.md``
for mkdocs-literate-nav to read as the reference nav. literate-nav marks them "not in nav" but
leaves them pages, so the build used to publish ``reference/SUMMARY/`` (a 136 KB page of bare
links). They are nav input, not content: once literate-nav has read them, exclude them.
"""

from __future__ import annotations

from mkdocs.config.defaults import MkDocsConfig
from mkdocs.plugins import event_priority
from mkdocs.structure.files import Files, InclusionLevel

_NAV_ONLY = ("reference/SUMMARY.md", "cli/SUMMARY.md")


@event_priority(-110)  # after literate-nav's on_files (priority -100) has read the nav file
def on_files(files: Files, config: MkDocsConfig) -> Files:  # noqa: ARG001 - MkDocs passes it by name
    """Exclude the literate-nav source files from the built site."""
    for path in _NAV_ONLY:
        nav_file = files.get_file_from_path(path)
        if nav_file is not None:
            nav_file.inclusion = InclusionLevel.EXCLUDED
    return files
