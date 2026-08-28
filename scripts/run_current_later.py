#!/usr/bin/env python3
"""Entry point: clustered Wald contrast of current vs later planned-content non-reach."""
from pathlib import Path
import runpy
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
runpy.run_path(str(ROOT / "src/strong_accept/run_current_later_contrast.py"), run_name="__main__")
