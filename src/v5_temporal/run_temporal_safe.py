"""Occasion helpers used by the learner-level analyses.

This module is a minimal extract of the assessment-row collapse used by the
frozen learner-level pipeline. It does not run the older score-association
development analyses.
"""
from __future__ import annotations

import pandas as pd


def collapse_student_assessment(sa: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    raw_n = int(len(sa))
    sa = sa.copy()
    sa["date_submitted"] = pd.to_numeric(sa["date_submitted"], errors="coerce")
    sa["score"] = pd.to_numeric(sa["score"], errors="coerce")
    if sa["is_banked"].dtype != bool:
        sa["is_banked"] = sa["is_banked"].astype(bool)
    dup_mask = sa.duplicated(["id_student", "id_assessment"], keep=False)
    n_dup_rows = int(dup_mask.sum())
    n_dup_keys = int(sa.loc[dup_mask, ["id_student", "id_assessment"]].drop_duplicates().shape[0])
    sa = sa.sort_values(["id_student", "id_assessment", "date_submitted", "is_banked"], na_position="last")
    sa = sa.drop_duplicates(["id_student", "id_assessment"], keep="first")
    return sa, {
        "raw_rows": raw_n,
        "duplicate_rows": n_dup_rows,
        "duplicate_keys": n_dup_keys,
        "after_collapse": int(len(sa)),
        "n_banked": int(sa["is_banked"].sum()),
        "n_missing_date_submitted": int(sa["date_submitted"].isna().sum()),
        "n_missing_score": int(sa["score"].isna().sum()),
    }
