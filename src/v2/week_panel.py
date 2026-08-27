"""Learner-week raw inactivity for presentations not in the v1 development panel."""
from __future__ import annotations

import pandas as pd

from v2.load import load_courses, load_registration


def build_week_panel(sv: pd.DataFrame, presentations: set[tuple[str, str]]) -> pd.DataFrame:
    courses = load_courses()
    reg = load_registration()
    skel = []
    for _, c in courses.iterrows():
        key = (c.code_module, c.code_presentation)
        if key not in presentations:
            continue
        for w in range(1, int(c.n_weeks) + 1):
            skel.append(
                {
                    "code_module": c.code_module,
                    "code_presentation": c.code_presentation,
                    "week": w,
                    "week_start": (w - 1) * 7,
                    "week_end": w * 7 - 1,
                }
            )
    weeks = pd.DataFrame(skel)
    want = {f"{a}|{b}" for a, b in presentations}
    preg = reg[reg["code_module"].str.cat(reg["code_presentation"], sep="|").isin(want)].copy()
    panel = preg.merge(weeks, on=["code_module", "code_presentation"], how="left")
    panel = panel[panel["date_registration"].notna()]
    panel = panel[panel["date_registration"] <= panel["week_end"]]
    panel = panel[panel["date_unregistration"].isna() | (panel["date_unregistration"] > panel["week_start"])]
    clicks = (
        sv[sv["code_module"].str.cat(sv["code_presentation"], sep="|").isin(want)]
        .groupby(["code_module", "code_presentation", "id_student", "week"], as_index=False)
        .agg(n_clicks=("sum_click", "sum"))
    )
    panel = panel.merge(clicks, on=["code_module", "code_presentation", "id_student", "week"], how="left")
    panel["n_clicks"] = panel["n_clicks"].fillna(0)
    panel["A_raw"] = (panel["n_clicks"] > 0).astype(int)
    panel["raw_inact"] = 1 - panel["A_raw"]
    return panel
