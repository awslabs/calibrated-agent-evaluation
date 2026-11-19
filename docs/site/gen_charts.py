# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Render the docs site's data charts into ``assets/charts/`` at build time.

Run by the mkdocs ``gen-files`` plugin; the drawing lives in ``docs_charts.py``. Pages show a chart
in the reader's theme with a pair of images:

    ![Labels needed to certify a judge](../assets/charts/calibration-size-light.svg#only-light)
    ![Labels needed to certify a judge](../assets/charts/calibration-size-dark.svg#only-dark)
"""

from __future__ import annotations

import sys
from pathlib import Path

import mkdocs_gen_files

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from docs_charts import render_all

for filename, svg in render_all().items():
    with mkdocs_gen_files.open(f"assets/charts/{filename}", "wb") as fd:
        fd.write(svg)
