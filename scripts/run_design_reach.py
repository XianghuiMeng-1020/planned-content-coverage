#!/usr/bin/env python3
"""Reproduce breadth-versus-volume identification and design-reach analyses."""
from __future__ import annotations

import sys
from pathlib import Path

_SRC = Path(__file__).resolve().parents[1] / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from v10_value_boundary.run_unit_a_breadth_vs_volume import main as unit_a  # noqa: E402
from v10_value_boundary.run_unit_b_design_reach import main as unit_b  # noqa: E402
from v11_design_replication.run_phase5c_design_replication import main as heldout  # noqa: E402

if __name__ == "__main__":
    unit_a()
    unit_b()
    heldout()
