#!/usr/bin/env python3
"""Entry point: KU Leuven joint-specificity diagnostic. Does not overwrite frozen headlines."""
from pathlib import Path
import runpy
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "src/strong_accept"))
runpy.run_path(str(ROOT / "src/strong_accept/run_kuleuven_specificity_diagnostic.py"), run_name="__main__")
