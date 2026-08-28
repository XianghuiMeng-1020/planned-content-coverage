"""Shared paths and frozen constants.

OULAD is not redistributed (CC-BY 4.0). Download it separately (see README.md).
Point OULAD_RAW and OULAD_INTERIM at local copies, or use the defaults under data/.
"""
from __future__ import annotations

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

OULAD_RAW = Path(os.environ.get("OULAD_RAW", ROOT / "data" / "raw" / "oulad"))
OULAD_INTERIM = Path(os.environ.get("OULAD_INTERIM", ROOT / "data" / "interim" / "oulad"))

# Presentations used to define the oucontent development set (BBB/EEE/FFF).
# AAA and CCC enter the frozen analyses through v3/v4 constants.
PRIMARY_PRES = {
    ("BBB", "2013J"),
    ("BBB", "2014B"),
    ("BBB", "2014J"),
    ("EEE", "2013J"),
    ("EEE", "2014B"),
    ("EEE", "2014J"),
    ("FFF", "2013J"),
    ("FFF", "2014B"),
    ("FFF", "2014J"),
}

SEED = 20260826
