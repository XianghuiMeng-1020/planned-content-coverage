#!/usr/bin/env python3
"""Entry point: KU Leuven external probe and S1 singleton sensitivity.

Requires the official Zenodo extract. Set KULEUVEN_DATA to the folder that
contains official/course_info.json and extracted/dataset/.
"""
from pathlib import Path
import runpy
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "src/strong_accept"))
runpy.run_path(str(ROOT / "src/strong_accept/run_kuleuven_external.py"), run_name="__main__")
runpy.run_path(str(ROOT / "src/strong_accept/run_kuleuven_s1.py"), run_name="__main__")
