"""Loaders. Confirmatory outcome tables are not loaded by metadata audit."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pyreadr

_SRC = Path(__file__).resolve().parents[1]
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))
from common.paths import OULAD_INTERIM, OULAD_RAW  # noqa: E402


def load_vle() -> pd.DataFrame:
    vle = pd.read_parquet(OULAD_INTERIM / "vle.parquet")
    vle["week_from"] = pd.to_numeric(vle["week_from"], errors="coerce")
    vle["week_to"] = pd.to_numeric(vle["week_to"], errors="coerce")
    vle["planned_valid"] = (
        vle["week_from"].notna()
        & vle["week_to"].notna()
        & (vle["week_to"] >= vle["week_from"])
    )
    vle["window_len"] = np.where(
        vle["planned_valid"],
        vle["week_to"] - vle["week_from"] + 1,
        np.nan,
    )
    return vle


def load_courses() -> pd.DataFrame:
    c = pd.read_parquet(OULAD_INTERIM / "courses.parquet")
    c["n_weeks"] = (c["module_presentation_length"] // 7) + 1
    return c


def load_assessments() -> pd.DataFrame:
    a = pd.read_parquet(OULAD_INTERIM / "assessments.parquet")
    a["date"] = pd.to_numeric(a["date"], errors="coerce")
    return a


def load_registration() -> pd.DataFrame:
    reg = pd.read_parquet(OULAD_INTERIM / "student_registration.parquet")
    reg["date_registration"] = pd.to_numeric(reg["date_registration"], errors="coerce")
    reg["date_unregistration"] = pd.to_numeric(reg["date_unregistration"], errors="coerce")
    return reg


def load_student_assessment() -> pd.DataFrame:
    sa = pd.read_parquet(OULAD_INTERIM / "student_assessment.parquet")
    sa["score"] = pd.to_numeric(sa["score"], errors="coerce")
    return sa


def load_student_vle() -> pd.DataFrame:
    sv = list(pyreadr.read_r(OULAD_RAW / "student_vle.rda").values())[0]
    sv["date"] = pd.to_numeric(sv["date"], errors="coerce")
    sv["sum_click"] = pd.to_numeric(sv["sum_click"], errors="coerce").fillna(0)
    sv["week"] = np.floor(sv["date"].clip(lower=0) / 7.0) + 1
    sv.loc[sv["date"] < 0, "week"] = 0
    sv["week"] = sv["week"].astype("int16")
    return sv


def week_end_day(week: float | int) -> int:
    """Last calendar day of a 1-indexed course week (week 1 = days 0–6)."""
    w = int(week)
    return w * 7 - 1


def week_start_day(week: float | int) -> int:
    w = int(week)
    return (w - 1) * 7
