"""Shared constants for OCNE v2. Do not rewrite the v1 registry."""
from __future__ import annotations

import sys
from pathlib import Path

_SRC = Path(__file__).resolve().parents[1]
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))
from common.paths import PRIMARY_PRES, SEED  # noqa: E402

DEVELOPMENT_PRES = frozenset(PRIMARY_PRES)

# Previously peeked under v1 exploratory coverage (phase 4). Not confirmation.
PEEKED_NONPRIMARY = frozenset({("FFF", "2013B")})

# Activity families (construct-first; not outcome-selected).
FAMILY_ALL_DOCUMENTED = "all_documented"
FAMILY_OUCONTENT = "oucontent"
FAMILY_OUCONTENT_QUIZ = "oucontent_quiz"
FAMILY_CORE_CONTENT = "core_content"  # oucontent, page, ouwiki, resource, dataplus

CORE_CONTENT_TYPES = frozenset(
    {"oucontent", "page", "ouwiki", "resource", "dataplus", "dualpane"}
)
NAV_TYPES = frozenset({"homepage", "forumng", "subpage", "url", "glossary"})

# Placebo family is small and fixed. Do not search for a weak control.
PLACEBO_CIRCULAR_SHIFTS = (3, 7, 14)
PLACEBO_RANDOM_NAME = "random_valid_start"
SEED_PLACEBO = SEED

# Assessment: TMA is the tutor-marked, pedagogically primary in-course assessment.
PRIMARY_ASSESSMENT_TYPE = "TMA"

# Minimum documented opportunities in an assessment window to form a rate.
MIN_OPP_PER_OCCASION = 1

WATCH_FRAC = 0.10
B_BOOT = 200
