"""V4 constants. Exact-week timing is not the construct."""
from __future__ import annotations

import sys
from pathlib import Path

_SRC = Path(__file__).resolve().parents[1]
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))
from common.paths import PRIMARY_PRES, SEED  # noqa: E402
from v3.constants import V1_PEEKED, V2_CONFIRMATION_INSPECTED, V3_DEVELOPMENT  # noqa: E402

V4_DEVELOPMENT = frozenset(V3_DEVELOPMENT)
# Frozen only after metadata screen; outcomes must not be inspected before the v4 tag.
V4_CONFIRMATION = frozenset({("GGG", "2013J"), ("GGG", "2014B"), ("GGG", "2014J")})
FAMILY_RESOURCE = "resource"
# Confirmation candidates are decided by metadata screen only.
SEED = SEED
MIN_OPP = 1
WATCH_FRAC = 0.10
B_BOOT = 100
# Predeclared dose bins for never_share.
DOSE_EDGES = (0.0, 0.0, 0.25, 0.50, 0.75, 1.00)
# Sensitivity horizon: one instructional block ≈ 4 weeks (calendar, not outcome-tuned).
PROXIMAL_DAYS = 28
