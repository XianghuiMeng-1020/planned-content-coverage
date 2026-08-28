#!/usr/bin/env python3
"""Unit B: planned-content design-reach stability. Writes only results/v10_value_boundary/."""
from __future__ import annotations

import itertools
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

_SRC = Path(__file__).resolve().parents[1]
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from common.paths import ROOT  # noqa: E402
from v2.constants import PRIMARY_ASSESSMENT_TYPE  # noqa: E402
from v2.load import load_assessments, load_courses, load_registration, load_student_vle, load_vle  # noqa: E402
from v2.panel import documented_sites, map_next_tma  # noqa: E402
from v3.constants import FAMILY_OUCONTENT  # noqa: E402
from v3.features import clicks_by_learner_site_date  # noqa: E402
from v4.constants import V4_DEVELOPMENT  # noqa: E402
from v10_value_boundary.shared import LOG, OUT, dump_csv, dump_json, log  # noqa: E402


def eligible_learner_sites(reg: pd.DataFrame, sites: pd.DataFrame) -> pd.DataFrame:
    parts = []
    for (mod, pres), gsite in sites.groupby(["code_module", "code_presentation"], sort=False):
        learners = reg[
            (reg.code_module == mod) & (reg.code_presentation == pres) & reg.date_registration.notna()
        ][["id_student", "date_registration", "date_unregistration"]]
        if learners.empty or gsite.empty:
            continue
        a = learners.assign(_k=1)
        b = gsite[
            [
                "id_site",
                "code_module",
                "code_presentation",
                "week_from",
                "week_to",
                "window_start_day",
                "window_end_day",
                "next_id_assessment",
                "next_tma_date",
            ]
        ].assign(_k=1)
        panel = a.merge(b, on="_k").drop(columns="_k")
        panel = panel[panel["date_registration"] <= panel["window_start_day"]]
        panel = panel[panel["date_unregistration"].isna() | (panel["date_unregistration"] >= panel["next_tma_date"])]
        parts.append(panel)
    if not parts:
        return pd.DataFrame()
    return pd.concat(parts, ignore_index=True)


def classify_b(median_rho: float | None, n_pairs_ge4: int) -> str:
    if n_pairs_ge4 < 3:
        return "B4"
    if median_rho is None:
        return "B4"
    if median_rho >= 0.50:
        return "B1"
    if median_rho >= 0.20:
        return "B2"
    return "B3"


def main() -> None:
    log("UNIT B start")
    vle = load_vle()
    assessments = load_assessments()
    reg = load_registration()
    courses = load_courses()
    sv = load_student_vle()
    sites0 = documented_sites(vle, set(V4_DEVELOPMENT), FAMILY_OUCONTENT)
    sites0 = map_next_tma(sites0, assessments).dropna(subset=["next_id_assessment", "next_tma_date"])
    sites0 = sites0[sites0.window_end_day < sites0.next_tma_date].copy()
    log(f"eligible planned sites (window_end < official next TMA)={len(sites0)}")
    clicks = clicks_by_learner_site_date(sv, sites0.id_site.tolist())
    panel = eligible_learner_sites(reg, sites0)
    log(f"eligible learner-site rows={len(panel)}")
    if panel.empty:
        dump_json("unit_b_summary.json", {"scenario": "B4", "reason": "empty eligible panel"})
        return
    acc = clicks.merge(sites0[["id_site", "next_tma_date"]], on="id_site", how="inner")
    acc = acc[acc.date < acc.next_tma_date]
    hit = acc.groupby(["id_site", "id_student"], as_index=False).size()
    hit["accessed"] = 1
    panel = panel.merge(hit[["id_site", "id_student", "accessed"]], on=["id_site", "id_student"], how="left")
    panel["accessed"] = panel["accessed"].fillna(0).astype(int)
    item = panel.groupby(
        ["code_module", "code_presentation", "id_site", "week_from", "week_to", "window_end_day", "next_tma_date"],
        as_index=False,
    ).agg(n_eligible=("id_student", "nunique"), n_reached=("accessed", "sum"))
    item = item[item.n_eligible > 0].copy()
    item["reach"] = item.n_reached / item.n_eligible
    item["gap"] = 1.0 - item.reach
    item["days_to_tma"] = item.next_tma_date - item.window_end_day
    nw = courses.set_index(["code_module", "code_presentation"])["n_weeks"].to_dict()
    item["n_weeks"] = item.apply(lambda r: nw.get((r.code_module, r.code_presentation), np.nan), axis=1)
    dump_csv("unit_b_item_reach.csv", item)

    week = (
        item.groupby(["code_module", "code_presentation", "week_from"], as_index=False)
        .agg(mean_gap=("gap", "mean"), n_items=("id_site", "nunique"), mean_reach=("reach", "mean"))
    )
    pairs = []
    for mod, g in week.groupby("code_module"):
        pres = sorted(g.code_presentation.unique())
        for a, b in itertools.combinations(pres, 2):
            wa = g[g.code_presentation == a][["week_from", "mean_gap"]].rename(columns={"mean_gap": "gap_a"})
            wb = g[g.code_presentation == b][["week_from", "mean_gap"]].rename(columns={"mean_gap": "gap_b"})
            m = wa.merge(wb, on="week_from")
            n_overlap = int(len(m))
            rho = p = None
            if n_overlap >= 2 and m.gap_a.nunique() > 1 and m.gap_b.nunique() > 1:
                rho, p = spearmanr(m.gap_a, m.gap_b)
                rho = float(rho)
                p = float(p)
            elif n_overlap >= 2:
                rho, p = None, None
            pairs.append(
                {
                    "code_module": mod,
                    "presentation_a": a,
                    "presentation_b": b,
                    "n_overlap_weeks": n_overlap,
                    "spearman_rho": rho,
                    "spearman_p": p,
                    "defined_ge4": n_overlap >= 4,
                }
            )
    pair_df = pd.DataFrame(pairs)
    dump_csv("unit_b_cross_presentation_stability.csv", pair_df)
    defined = pair_df[pair_df.defined_ge4 & pair_df.spearman_rho.notna()]
    median_rho = float(defined.spearman_rho.median()) if len(defined) else None
    n_pairs_ge4 = int((pair_df.defined_ge4).sum())
    scenario = classify_b(median_rho, n_pairs_ge4)

    # Secondary 1: week position
    item["third"] = np.ceil(item["n_weeks"] / 3.0)
    item["week_pos"] = np.where(
        item.week_from <= item.third,
        "early",
        np.where(item.week_from > (item.n_weeks - item.third), "late", "middle"),
    )
    pos = (
        item.groupby("week_pos", as_index=False)
        .agg(n_items=("id_site", "nunique"), mean_gap=("gap", "mean"), mean_reach=("reach", "mean"))
    )
    early = item[item.week_pos == "early"]["gap"]
    late = item[item.week_pos == "late"]["gap"]
    pos_diff = float(late.mean() - early.mean()) if len(early) and len(late) else None

    # Secondary 2: assessment proximity
    item["prox"] = np.where(item.days_to_tma <= 14, "le14", "gt14")
    prox = (
        item.groupby("prox", as_index=False)
        .agg(n_items=("id_site", "nunique"), mean_gap=("gap", "mean"), mean_reach=("reach", "mean"))
    )
    close = item[item.prox == "le14"]["gap"]
    far = item[item.prox == "gt14"]["gap"]
    prox_diff = float(close.mean() - far.mean()) if len(close) and len(far) else None

    struct = pd.concat(
        [
            pos.assign(property="week_position", contrast="late_minus_early", diff=pos_diff),
            prox.assign(property="assessment_proximity", contrast="le14_minus_gt14", diff=prox_diff),
        ],
        ignore_index=True,
    )
    dump_csv("unit_b_content_structure_results.csv", struct)

    dump_json(
        "unit_b_summary.json",
        {
            "unit": "planned oucontent site x presentation; official next TMA cutoff",
            "n_eligible_items": int(len(item)),
            "n_contexts_presentations": int(item.groupby(["code_module", "code_presentation"]).ngroups),
            "n_modules": int(item.code_module.nunique()),
            "comparable_unit": "mean Gap by week_from within module-presentation",
            "n_same_module_pairs": int(len(pair_df)),
            "n_pairs_overlap_ge4": n_pairs_ge4,
            "n_pairs_with_rho": int(len(defined)),
            "primary_metric": "median Spearman rho of week-level Gap across same-module pairs with >=4 overlapping weeks",
            "median_rho": median_rho,
            "rho_min": float(defined.spearman_rho.min()) if len(defined) else None,
            "rho_max": float(defined.spearman_rho.max()) if len(defined) else None,
            "week_position_late_minus_early_gap": pos_diff,
            "proximity_le14_minus_gt14_gap": prox_diff,
            "scenario": scenario,
            "required_optional_status": "not documented in OULAD VLE; not tested",
            "high_gap_means_poor_quality": False,
        },
    )
    log(f"UNIT B done scenario={scenario} median_rho={median_rho}")
    (OUT / "unit_b_run.log").write_text("\n".join(LOG) + "\n")


if __name__ == "__main__":
    main()
