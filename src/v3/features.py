"""Temporal-state and continuous alignment features. Cutoff is next TMA date (exclusive)."""
from __future__ import annotations

import numpy as np
import pandas as pd

from v2.panel import map_next_tma, registered_learner_sites


def clicks_by_learner_site_date(sv: pd.DataFrame, site_ids) -> pd.DataFrame:
    ids = set(int(x) for x in site_ids)
    sub = sv.loc[sv["id_site"].isin(ids), ["id_site", "id_student", "date", "sum_click"]].copy()
    sub = sub[sub["sum_click"] > 0]
    if sub.empty:
        return sub
    return sub.groupby(["id_site", "id_student", "date"], as_index=False).agg(sum_click=("sum_click", "sum"))


def annotate_site_states(panel: pd.DataFrame, clicks: pd.DataFrame) -> pd.DataFrame:
    """Add first-access state and multi-state flags. Clicks with date >= next_TMA are unused."""
    work = panel.dropna(subset=["next_tma_date"]).copy()
    if work.empty:
        return work
    if clicks.empty:
        work["first_date"] = np.nan
        work["first_state"] = "never"
        work["any_early"] = 0
        work["any_ontime"] = 0
        work["any_late"] = 0
        work["never"] = 1
        work["signed_lag"] = np.nan
        work["closest_lag"] = np.nan
        work["dist_to_tma"] = work["next_tma_date"] - work["window_end_day"]
        return work

    m = clicks.merge(
        work[["id_site", "id_student", "window_start_day", "window_end_day", "next_tma_date"]],
        on=["id_site", "id_student"],
        how="inner",
    )
    m = m[m["date"] < m["next_tma_date"]]
    if m.empty:
        work["first_date"] = np.nan
        work["first_state"] = "never"
        work["any_early"] = 0
        work["any_ontime"] = 0
        work["any_late"] = 0
        work["never"] = 1
        work["signed_lag"] = np.nan
        work["closest_lag"] = np.nan
        work["dist_to_tma"] = work["next_tma_date"] - work["window_end_day"]
        return work

    m["early"] = m["date"] < m["window_start_day"]
    m["ontime"] = (m["date"] >= m["window_start_day"]) & (m["date"] <= m["window_end_day"])
    m["late"] = m["date"] > m["window_end_day"]
    m["dist_interval"] = np.where(
        m["ontime"],
        0.0,
        np.where(m["early"], m["window_start_day"] - m["date"], m["date"] - m["window_end_day"]),
    )
    flags = m.groupby(["id_site", "id_student"], as_index=False).agg(
        first_date=("date", "min"),
        any_early=("early", "max"),
        any_ontime=("ontime", "max"),
        any_late=("late", "max"),
        closest_lag=("dist_interval", "min"),
    )
    out = work.merge(flags, on=["id_site", "id_student"], how="left")
    out["any_early"] = out["any_early"].fillna(0).astype(int)
    out["any_ontime"] = out["any_ontime"].fillna(0).astype(int)
    out["any_late"] = out["any_late"].fillna(0).astype(int)
    out["never"] = (out["first_date"].isna()).astype(int)
    fd = out["first_date"]
    st = np.where(
        fd.isna(),
        "never",
        np.where(
            fd < out["window_start_day"],
            "early",
            np.where(fd <= out["window_end_day"], "ontime", "late"),
        ),
    )
    out["first_state"] = st
    out["signed_lag"] = np.where(out["never"] == 1, np.nan, out["first_date"] - out["window_start_day"])
    out["dist_to_tma"] = out["next_tma_date"] - out["window_end_day"]
    return out


def occasion_from_sites(site: pd.DataFrame) -> pd.DataFrame:
    """Learner × TMA shares and continuous alignment features."""
    work = site.dropna(subset=["next_id_assessment"]).copy()
    work["next_id_assessment"] = work["next_id_assessment"].astype(int)
    work["is_early"] = (work["first_state"] == "early").astype(int)
    work["is_ontime"] = (work["first_state"] == "ontime").astype(int)
    work["is_late"] = (work["first_state"] == "late").astype(int)
    work["is_never"] = (work["first_state"] == "never").astype(int)
    work["late_catchup"] = ((work["any_late"] == 1) & (work["any_ontime"] == 0)).astype(int)

    def _med(s):
        s = s.dropna()
        return float(s.median()) if len(s) else np.nan

    def _iqr(s):
        s = s.dropna()
        if len(s) < 2:
            return np.nan
        return float(s.quantile(0.75) - s.quantile(0.25))

    occ = work.groupby(
        ["code_module", "code_presentation", "id_student", "next_id_assessment"],
        as_index=False,
    ).agg(
        n_opp=("id_site", "size"),
        share_early=("is_early", "mean"),
        share_ontime=("is_ontime", "mean"),
        share_late=("is_late", "mean"),
        share_never=("is_never", "mean"),
        ontime_coverage=("any_ontime", "mean"),
        pre_tma_coverage=("never", lambda s: float(1 - s.mean())),
        late_catchup_share=("late_catchup", "mean"),
        multi_early=("any_early", "mean"),
        multi_ontime=("any_ontime", "mean"),
        multi_late=("any_late", "mean"),
        median_signed_lag=("signed_lag", _med),
        timing_dispersion=("signed_lag", _iqr),
        median_closest_lag=("closest_lag", _med),
        mean_dist_to_tma=("dist_to_tma", "mean"),
        next_tma_date=("next_tma_date", "first"),
    )
    return occ


def build_site_panel(sites, reg, assessments, clicks):
    panel = registered_learner_sites(reg, sites)
    if panel.empty:
        return panel
    panel = map_next_tma(panel, assessments)
    return annotate_site_states(panel, clicks)
