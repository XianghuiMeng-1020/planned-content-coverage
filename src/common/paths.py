"""Shared paths and frozen constants.

OULAD is not redistributed in this repository (CC-BY 4.0; download separately,
see README.md). Point the environment variables below at your local copies,
or drop the files into `data/raw/oulad` and `data/interim/oulad` and leave the
defaults untouched.
"""
from __future__ import annotations

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

OULAD_RAW = Path(os.environ.get("OULAD_RAW", ROOT / "data" / "raw" / "oulad"))
OULAD_INTERIM = Path(os.environ.get("OULAD_INTERIM", ROOT / "data" / "interim" / "oulad"))

# Development presentations (pre-registered before any outcome was inspected).
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
