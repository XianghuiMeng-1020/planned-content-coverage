#!/usr/bin/env python3
"""Reproduce submission / low-score association and operational-increment tests."""
from __future__ import annotations

import sys
from pathlib import Path

_SRC = Path(__file__).resolve().parents[1] / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from v9_decision.run_decision_utility import main  # noqa: E402

if __name__ == "__main__":
    main()
