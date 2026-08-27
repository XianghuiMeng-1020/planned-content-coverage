"""V3 constants. V2 placebo family and oucontent family stay frozen."""
from __future__ import annotations

import sys
from pathlib import Path

_SRC = Path(__file__).resolve().parents[1]
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))
from common.paths import PRIMARY_PRES, SEED  # noqa: E402
from v2.constants import (  # noqa: E402
    FAMILY_OUCONTENT,
    PLACEBO_CIRCULAR_SHIFTS,
    PLACEBO_RANDOM_NAME,
    PRIMARY_ASSESSMENT_TYPE,
    SEED_PLACEBO,
    WATCH_FRAC,
)

# All previously outcome-inspected presentations are V3 development.
V2_CONFIRMATION_INSPECTED = frozenset(
    {("CCC", "2014B"), ("CCC", "2014J"), ("AAA", "2013J"), ("AAA", "2014J")}
)
V1_PEEKED = frozenset({("FFF", "2013B")})
V3_DEVELOPMENT = frozenset(PRIMARY_PRES) | V2_CONFIRMATION_INSPECTED

# Assessment-distance bins from official dates only (days from window end to TMA).
DIST_CLOSE = 14
DIST_MID = 42

MIN_OPP = 1
SEED = SEED
