#!/usr/bin/env python3
"""Reproduce the primary association, FE, recurrence, and predictive-transport analyses."""
from __future__ import annotations

import sys
from pathlib import Path

_SRC = Path(__file__).resolve().parents[1] / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from v7_inference.run_inference import main  # noqa: E402

if __name__ == "__main__":
    main()
