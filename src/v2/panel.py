"""Resource-matched opportunity panel. Missing planned-use metadata never enters as O=0."""
from __future__ import annotations

from typing import Iterable

import numpy as np
import pandas as pd

from v2.constants import (
    CORE_CONTENT_TYPES,
    FAMILY_ALL_DOCUMENTED,
    FAMILY_CORE_CONTENT,
    FAMILY_OUCONTENT,
    FAMILY_OUCONTENT_QUIZ,
    PLACEBO_CIRCULAR_SHIFTS,
    PLACEBO_RANDOM_NAME,
    PRIMARY_ASSESSMENT_TYPE,
    SEED_PLACEBO,
)
from v2.load import load_assessments, load_courses, load_registration, week_end_day, week_start_day


def family_mask(activity: pd.Series, family: str) -> pd.Series:
    if family == FAMILY_ALL_DOCUMENTED:
        return pd.Series(True, index=activity.index)
    if family == FAMILY_OUCONTENT:
        return activity == "oucontent"
    if family == FAMILY_OUCONTENT_QUIZ:
        return activity.isin(["oucontent", "quiz"])
    if family == FAMILY_CORE_CONTENT:
        return activity.isin(CORE_CONTENT_TYPES)
    raise ValueError(family)


def documented_sites(vle: pd.DataFrame, presentations: set[tuple[str, str]], family: str) -> pd.DataFrame:
    keys = vle["code_module"].str.cat(vle["code_presentation"], sep="|")
    want = {f"{a}|{b}" for a, b in presentations}
    sites = vle.loc[vle["planned_valid"] & keys.isin(want)].copy()
    sites = sites.loc[family_mask(sites["activity_type"], family)].copy()
    sites["window_start_day"] = sites["week_from"].map(week_start_day)
    sites["window_end_day"] = sites["week_to"].map(week_end_day)
    sites["window_len"] = sites["week_to"] - sites["week_from"] + 1
    return sites.reset_index(drop=True)


def shift_windows(sites: pd.DataFrame, courses: pd.DataFrame, kind: str, shift: int | None = None) -> pd.DataFrame:
    """Preserve resource identity and window length; break official alignment."""
    out = sites.copy()
    nweeks = courses.set_index(["code_module", "code_presentation"])["n_weeks"].to_dict()
    rng = np.random.default_rng(SEED_PLACEBO)
    new_from = []
    new_to = []
    for _, r in out.iterrows():
        n_w = int(nweeks[(r.code_module, r.code_presentation)])
        length = int(r.window_len)
        max_start = max(1, n_w - length + 1)
        if kind == "circular":
            # stay inside the course; wrap within valid starts
            start = ((int(r.week_from) - 1 + int(shift)) % max_start) + 1
        elif kind == "random":
            start = int(rng.integers(1, max_start + 1))
        else:
            raise ValueError(kind)
        new_from.append(start)
        new_to.append(start + length - 1)
    out["week_from"] = new_from
    out["week_to"] = new_to
    out["window_start_day"] = out["week_from"].map(week_start_day)
    out["window_end_day"] = out["week_to"].map(week_end_day)
    return out


def clicks_by_learner_site_week(sv: pd.DataFrame, site_ids: Iterable[int]) -> pd.DataFrame:
    site_ids = set(int(x) for x in site_ids)
    sub = sv.loc[sv["id_site"].isin(site_ids), ["id_site", "id_student", "week", "sum_click"]].copy()
    if sub.empty:
        return sub
    return sub.groupby(["id_site", "id_student", "week"], as_index=False).agg(sum_click=("sum_click", "sum"))


def action_on_window(clicks: pd.DataFrame, sites: pd.DataFrame) -> pd.DataFrame:
    """A=1 iff the learner clicked this site during this site's current window."""
    if clicks.empty:
        return pd.DataFrame(columns=["id_site", "id_student", "A", "n_clicks_window", "n_clicks_any"])
    any_c = clicks.groupby(["id_site", "id_student"], as_index=False).agg(n_clicks_any=("sum_click", "sum"))
    win = sites[["id_site", "week_from", "week_to"]].copy()
    m = clicks.merge(win, on="id_site", how="inner")
    m = m[(m["week"] >= m["week_from"]) & (m["week"] <= m["week_to"])]
    in_w = m.groupby(["id_site", "id_student"], as_index=False).agg(n_clicks_window=("sum_click", "sum"))
    out = any_c.merge(in_w, on=["id_site", "id_student"], how="left")
    out["n_clicks_window"] = out["n_clicks_window"].fillna(0)
    out["A"] = (out["n_clicks_window"] > 0).astype(int)
    return out


def registered_learner_sites(reg: pd.DataFrame, sites: pd.DataFrame) -> pd.DataFrame:
    """Cartesian registered learners × documented sites, restricted to registration overlap."""
    parts = []
    for (mod, pres), gsite in sites.groupby(["code_module", "code_presentation"]):
        learners = reg[(reg.code_module == mod) & (reg.code_presentation == pres) & reg.date_registration.notna()][
            ["id_student", "date_registration", "date_unregistration"]
        ]
        if learners.empty or gsite.empty:
            continue
        a = learners.assign(_k=1)
        b = gsite[
            [
                "id_site",
                "code_module",
                "code_presentation",
                "activity_type",
                "week_from",
                "week_to",
                "window_start_day",
                "window_end_day",
            ]
        ].assign(_k=1)
        panel = a.merge(b, on="_k").drop(columns="_k")
        panel = panel[panel["date_registration"] <= panel["window_end_day"]]
        panel = panel[
            panel["date_unregistration"].isna() | (panel["date_unregistration"] > panel["window_start_day"])
        ]
        parts.append(panel)
    if not parts:
        return pd.DataFrame()
    return pd.concat(parts, ignore_index=True)


def attach_actions(panel: pd.DataFrame, actions: pd.DataFrame) -> pd.DataFrame:
    out = panel.merge(actions, on=["id_site", "id_student"], how="left")
    out["n_clicks_window"] = out["n_clicks_window"].fillna(0)
    out["n_clicks_any"] = out["n_clicks_any"].fillna(0)
    out["A"] = out["A"].fillna(0).astype(int)
    out["miss"] = 1 - out["A"]
    out["later_same_resource"] = ((out["n_clicks_any"] - out["n_clicks_window"]) > 0).astype(int)
    return out


def map_next_tma(panel: pd.DataFrame, assessments: pd.DataFrame) -> pd.DataFrame:
    tma = assessments[
        (assessments["assessment_type"] == PRIMARY_ASSESSMENT_TYPE) & assessments["date"].notna()
    ][["code_module", "code_presentation", "id_assessment", "date"]].sort_values(
        ["code_module", "code_presentation", "date"]
    )
    rows = []
    for (mod, pres), g in panel.groupby(["code_module", "code_presentation"], sort=False):
        dates = tma[(tma.code_module == mod) & (tma.code_presentation == pres)]
        if dates.empty:
            gg = g.copy()
            gg["next_id_assessment"] = np.nan
            gg["next_tma_date"] = np.nan
            rows.append(gg)
            continue
        arr_d = dates["date"].to_numpy()
        arr_id = dates["id_assessment"].to_numpy()
        end = g["window_end_day"].to_numpy()
        idx = np.searchsorted(arr_d, end, side="right")
        nxt_id = np.where(idx < len(arr_d), arr_id[np.minimum(idx, len(arr_d) - 1)], np.nan)
        nxt_d = np.where(idx < len(arr_d), arr_d[np.minimum(idx, len(arr_d) - 1)], np.nan)
        nxt_id = np.where(idx < len(arr_d), nxt_id, np.nan)
        gg = g.copy()
        gg["next_id_assessment"] = nxt_id
        gg["next_tma_date"] = nxt_d
        rows.append(gg)
    return pd.concat(rows, ignore_index=True)


def map_prev_tma_date(assessments: pd.DataFrame) -> pd.DataFrame:
    tma = assessments[
        (assessments["assessment_type"] == PRIMARY_ASSESSMENT_TYPE) & assessments["date"].notna()
    ][["code_module", "code_presentation", "id_assessment", "date"]].sort_values(
        ["code_module", "code_presentation", "date"]
    )
    recs = []
    for (mod, pres), g in tma.groupby(["code_module", "code_presentation"]):
        prev = None
        for _, r in g.iterrows():
            recs.append(
                {
                    "code_module": mod,
                    "code_presentation": pres,
                    "id_assessment": r.id_assessment,
                    "tma_date": r.date,
                    "prev_tma_date": prev,
                }
            )
            prev = r.date
    return pd.DataFrame(recs)


def occasion_table(site_panel: pd.DataFrame, miss_col: str = "miss") -> pd.DataFrame:
    work = site_panel.dropna(subset=["next_id_assessment"]).copy()
    work["next_id_assessment"] = work["next_id_assessment"].astype(int)
    occ = work.groupby(
        ["code_module", "code_presentation", "id_student", "next_id_assessment"],
        as_index=False,
    ).agg(
        n_opp=(miss_col, "size"),
        n_miss=(miss_col, "sum"),
        miss_rate=(miss_col, "mean"),
        later_same_rate=("later_same_resource", "mean") if "later_same_resource" in work.columns else (miss_col, "size"),
        next_tma_date=("next_tma_date", "first"),
    )
    return occ


def attach_tma_outcomes(occ: pd.DataFrame, sa: pd.DataFrame, tma_meta: pd.DataFrame) -> pd.DataFrame:
    scores = sa.rename(columns={"id_assessment": "next_id_assessment"})[
        ["id_student", "next_id_assessment", "score"]
    ]
    out = occ.merge(tma_meta, left_on=["code_module", "code_presentation", "next_id_assessment"], right_on=["code_module", "code_presentation", "id_assessment"], how="left")
    out = out.merge(scores, on=["id_student", "next_id_assessment"], how="left")
    out["submitted"] = out["score"].notna().astype(int)
    return out


def attach_prior_score(occ: pd.DataFrame, sa: pd.DataFrame, assessments: pd.DataFrame) -> pd.DataFrame:
    tma = assessments[
        (assessments["assessment_type"] == PRIMARY_ASSESSMENT_TYPE) & assessments["date"].notna()
    ][["code_module", "code_presentation", "id_assessment", "date"]]
    sa_t = sa.merge(tma, on="id_assessment", how="inner")
    sa_t = sa_t.dropna(subset=["score", "date"])
    sa_t["pres_student"] = (
        sa_t["code_module"].astype(str)
        + "|"
        + sa_t["code_presentation"].astype(str)
        + "|"
        + sa_t["id_student"].astype(str)
    )
    left = occ.copy()
    left["pres_student"] = (
        left["code_module"].astype(str)
        + "|"
        + left["code_presentation"].astype(str)
        + "|"
        + left["id_student"].astype(str)
    )
    left = left.sort_values("next_tma_date")
    right = sa_t[["pres_student", "date", "score"]].rename(columns={"score": "prior_score", "date": "prior_date"})
    right = right.sort_values("prior_date")
    out = pd.merge_asof(
        left,
        right,
        left_on="next_tma_date",
        right_on="prior_date",
        by="pres_student",
        direction="backward",
        allow_exact_matches=False,
    )
    out["prior_missing"] = out["prior_score"].isna().astype(int)
    out["prior_score_filled"] = out["prior_score"].fillna(0.0)
    return out.drop(columns=["pres_student"], errors="ignore")


def attach_raw_controls(occ: pd.DataFrame, week_panel: pd.DataFrame) -> pd.DataFrame:
    """Raw weekly inactivity/activity in (prev_tma_date, this_tma_date], leakage-safe."""
    out = occ.copy()
    if week_panel.empty:
        out["raw_inact"] = np.nan
        out["active_rate"] = np.nan
        out["log_clicks"] = np.nan
        return out
    wp = week_panel.copy()
    if "week_end" not in wp.columns:
        wp["week_end"] = wp["week"] * 7 - 1
    keys = ["code_module", "code_presentation", "id_student"]
    merged = out.reset_index().merge(
        wp[keys + ["week_end", "raw_inact", "A_raw", "n_clicks"]],
        on=keys,
        how="left",
    )
    lo = merged["prev_tma_date"].fillna(-1e9)
    hi = merged["next_tma_date"]
    merged = merged[(merged["week_end"] > lo) & (merged["week_end"] <= hi)]
    ctrl = merged.groupby("index", as_index=True).agg(
        raw_inact=("raw_inact", "mean"),
        active_rate=("A_raw", "mean"),
        log_clicks=("n_clicks", lambda s: float(np.log1p(s.sum()))),
    )
    out = out.join(ctrl)
    return out


def placebo_specs() -> list[tuple[str, str, int | None]]:
    specs = [("authentic", "authentic", None)]
    for s in PLACEBO_CIRCULAR_SHIFTS:
        specs.append((f"circular_{s}", "circular", s))
    specs.append((PLACEBO_RANDOM_NAME, "random", None))
    return specs
