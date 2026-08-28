#!/usr/bin/env python3
"""Held-out design-reach versus generic course position. No student outcomes."""
from __future__ import annotations

import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

_SRC = Path(__file__).resolve().parents[1]
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from common.paths import ROOT  # noqa: E402
from v2.load import load_assessments, load_courses, load_registration, load_student_vle, load_vle  # noqa: E402
from v2.panel import documented_sites, map_next_tma  # noqa: E402
from v3.constants import FAMILY_OUCONTENT  # noqa: E402
from v3.features import clicks_by_learner_site_date  # noqa: E402
from v4.constants import V4_DEVELOPMENT  # noqa: E402
from v10_value_boundary.run_unit_b_design_reach import eligible_learner_sites  # noqa: E402

OUT = ROOT / "results" / "v11_design_replication"
FORBIDDEN = [
    ROOT / "results" / "v9_decision_utility",
    ROOT / "results" / "v10_value_boundary",
]
SEED = 20260831
B_PLACEBO = 2000
MIN_SLOTS = 4
MIN_PLACEBO_SLOTS = 8
CHRONO = {"2013B": 0, "2013J": 1, "2014B": 2, "2014J": 3}
FOLDS = [
    {"fold": "CCC_2014J", "module": "CCC", "train": ["2014B"], "holdout": "2014J"},
    {"fold": "EEE_2014B", "module": "EEE", "train": ["2013J"], "holdout": "2014B"},
    {"fold": "EEE_2014J", "module": "EEE", "train": ["2013J", "2014B"], "holdout": "2014J"},
    {"fold": "FFF_2014B", "module": "FFF", "train": ["2013J"], "holdout": "2014B"},
    {"fold": "FFF_2014J", "module": "FFF", "train": ["2013J", "2014B"], "holdout": "2014J"},
]
LOG: list[str] = []


def log(msg: str) -> None:
    LOG.append(f"{datetime.now(timezone.utc).isoformat()} {msg}")
    print(msg, flush=True)


def _jd(x):
    if isinstance(x, (np.floating,)):
        return float(x) if np.isfinite(x) else None
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, (np.bool_,)):
        return bool(x)
    raise TypeError(type(x))


def _guard(p: Path) -> Path:
    p = p.resolve()
    assert str(p).startswith(str(OUT.resolve()))
    for bad in FORBIDDEN:
        assert not str(p).startswith(str(bad.resolve()))
    return p


def dump_json(name: str, obj) -> Path:
    OUT.mkdir(parents=True, exist_ok=True)
    p = _guard(OUT / name)
    p.write_text(json.dumps(obj, indent=2, default=_jd))
    log(f"wrote {p.relative_to(ROOT)}")
    return p


def dump_csv(name: str, df: pd.DataFrame) -> Path:
    OUT.mkdir(parents=True, exist_ok=True)
    p = _guard(OUT / name)
    df.to_csv(p, index=False)
    log(f"wrote {p.relative_to(ROOT)} n={len(df)}")
    return p


def pos_bin(week_from: float, n_weeks: float) -> str:
    pos = float(week_from) / float(n_weeks)
    if pos <= 1.0 / 3.0:
        return "EARLY"
    if pos <= 2.0 / 3.0:
        return "MIDDLE"
    return "LATE"


def rho_safe(a, b) -> tuple[float | None, float | None]:
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    if len(a) < 2 or np.unique(a).size < 2 or np.unique(b).size < 2:
        return None, None
    r, p = spearmanr(a, b)
    if not np.isfinite(r):
        return None, None
    return float(r), float(p)


def top_set(values: pd.Series, frac=0.25) -> set:
    k = max(1, int(np.ceil(frac * len(values))))
    ordered = values.sort_values(ascending=False)
    cutoff = ordered.iloc[k - 1]
    return set(ordered.index[ordered >= cutoff])


def classify(med_h, med_p, med_d, med_res, n_pos, n_ok) -> str:
    if n_ok < 2:
        return "E"
    if med_h is None:
        return "E"
    half = n_ok / 2.0
    n_h_pos = None  # filled by caller if needed
    if med_h < 0.20:
        return "D"
    # A
    if (
        med_h >= 0.50
        and med_d is not None
        and med_d >= 0.10
        and n_pos >= 4
        and med_res is not None
        and med_res >= 0.30
    ):
        return "A"
    # B
    if med_h >= 0.30 and med_d is not None and med_d < 0.10 and (med_res is None or med_res < 0.30):
        return "B"
    return "C"


def build_item_gap() -> pd.DataFrame:
    log("rebuild item Gap (Phase-5B Unit B definition; v11 only)...")
    vle = load_vle()
    assessments = load_assessments()
    reg = load_registration()
    courses = load_courses()
    sv = load_student_vle()
    sites0 = documented_sites(vle, set(V4_DEVELOPMENT), FAMILY_OUCONTENT)
    sites0 = map_next_tma(sites0, assessments).dropna(subset=["next_id_assessment", "next_tma_date"])
    sites0 = sites0[sites0.window_end_day < sites0.next_tma_date].copy()
    clicks = clicks_by_learner_site_date(sv, sites0.id_site.tolist())
    panel = eligible_learner_sites(reg, sites0)
    acc = clicks.merge(sites0[["id_site", "next_tma_date"]], on="id_site", how="inner")
    acc = acc[acc.date < acc.next_tma_date]
    hit = acc.groupby(["id_site", "id_student"], as_index=False).size()
    hit["accessed"] = 1
    panel = panel.merge(hit[["id_site", "id_student", "accessed"]], on=["id_site", "id_student"], how="left")
    panel["accessed"] = panel["accessed"].fillna(0).astype(int)
    item = panel.groupby(
        ["code_module", "code_presentation", "id_site", "week_from", "window_end_day", "next_tma_date"],
        as_index=False,
    ).agg(n_eligible=("id_student", "nunique"), n_reached=("accessed", "sum"))
    item = item[item.n_eligible > 0].copy()
    item["reach"] = item.n_reached / item.n_eligible
    item["gap"] = 1.0 - item.reach
    nw = courses.set_index(["code_module", "code_presentation"])["n_weeks"].to_dict()
    item["n_weeks"] = item.apply(lambda r: nw.get((r.code_module, r.code_presentation), np.nan), axis=1)
    return item


def week_table(item: pd.DataFrame) -> pd.DataFrame:
    w = item.groupby(["code_module", "code_presentation", "week_from", "n_weeks"], as_index=False).agg(
        gap=("gap", "mean"), n_items=("id_site", "nunique")
    )
    w["chrono"] = w.code_presentation.map(CHRONO)
    w["pos_bin"] = [pos_bin(r.week_from, r.n_weeks) for r in w.itertuples()]
    return w


def main() -> None:
    log("PHASE 5C start")
    item = build_item_gap()
    week = week_table(item)
    dump_csv("_week_level_gap.csv", week)

    pred_rows = []
    pos_rows = []
    res_rows = []
    topk_rows = []
    placebo_rows = []
    fold_summ = []

    rng_master = np.random.default_rng(SEED)

    for spec in FOLDS:
        fold, mod, train_pres, ho = spec["fold"], spec["module"], spec["train"], spec["holdout"]
        t_ho = CHRONO[ho]
        ho_w = week[(week.code_module == mod) & (week.code_presentation == ho)].copy()
        train_w = week[(week.code_module == mod) & (week.code_presentation.isin(train_pres))].copy()
        if ho_w.empty or train_w.empty:
            fold_summ.append({"fold": fold, "status": "SKIP_EMPTY"})
            continue
        counts = train_w.groupby("week_from").code_presentation.nunique()
        common = set(ho_w.week_from) & set(counts[counts == len(train_pres)].index)
        if len(common) < MIN_SLOTS:
            fold_summ.append({"fold": fold, "status": "SKIP", "n_slots": int(len(common))})
            continue
        ho_s = ho_w[ho_w.week_from.isin(common)].set_index("week_from").sort_index()
        hist = (
            train_w[train_w.week_from.isin(common)]
            .groupby("week_from", as_index=True)
            .agg(gap_h=("gap", "mean"), n_train=("code_presentation", "nunique"))
            .sort_index()
        )
        # P: all V4 presentations with chrono < t_ho
        p_train = week[week.chrono < t_ho].copy()
        bin_means = p_train.groupby("pos_bin")["gap"].mean().to_dict()
        grand = float(p_train["gap"].mean()) if len(p_train) else np.nan
        gap_p = ho_s["pos_bin"].map(lambda b: bin_means.get(b, grand)).astype(float)
        gap_h = hist.reindex(ho_s.index)["gap_h"].astype(float)
        y = ho_s["gap"].astype(float)
        rho_h, p_h = rho_safe(gap_h, y)
        rho_p, p_p = rho_safe(gap_p, y)
        d_rho = None if (rho_h is None or rho_p is None) else rho_h - rho_p
        # residuals: subtract this fold's P bin means
        train_mod = train_w[train_w.week_from.isin(common)].copy()
        train_mod["gap_p"] = train_mod["pos_bin"].map(lambda b: bin_means.get(b, grand))
        train_mod["resid"] = train_mod["gap"] - train_mod["gap_p"]
        hist_res = train_mod.groupby("week_from")["resid"].mean().reindex(ho_s.index)
        ho_res = y - gap_p
        rho_r, p_r = rho_safe(hist_res, ho_res)

        for w in ho_s.index:
            pred_rows.append(
                {
                    "fold": fold,
                    "code_module": mod,
                    "holdout": ho,
                    "week_from": int(w),
                    "pos_bin": ho_s.loc[w, "pos_bin"],
                    "n_weeks_holdout": float(ho_s.loc[w, "n_weeks"]),
                    "gap_holdout": float(y.loc[w]),
                    "gap_hat_H": float(gap_h.loc[w]),
                    "gap_hat_P": float(gap_p.loc[w]),
                    "residual_hist": float(hist_res.loc[w]),
                    "residual_holdout": float(ho_res.loc[w]),
                }
            )
        nom = top_set(gap_h)
        obs = top_set(y)
        prec = len(nom & obs) / len(nom) if nom else None
        jac = len(nom & obs) / len(nom | obs) if (nom or obs) else None
        topk_rows.append(
            {
                "fold": fold,
                "n_slots": int(len(ho_s)),
                "k_nominated": int(len(nom)),
                "k_observed": int(len(obs)),
                "precision": prec,
                "jaccard": jac,
                "nominated_weeks": ",".join(str(int(x)) for x in sorted(nom)),
                "observed_top_weeks": ",".join(str(int(x)) for x in sorted(obs)),
            }
        )
        placebo_pct = None
        if len(ho_s) >= MIN_PLACEBO_SLOTS:
            bins = ho_s["pos_bin"].to_numpy()
            weeks = ho_s.index.to_numpy()
            h_vals = gap_h.to_numpy()
            y_vals = y.to_numpy()
            authentic = rho_h
            ge = 0
            for b in range(B_PLACEBO):
                perm = h_vals.copy()
                for lab in np.unique(bins):
                    idx = np.where(bins == lab)[0]
                    if len(idx) > 1:
                        perm[idx] = rng_master.permutation(perm[idx])
                r, _ = rho_safe(perm, y_vals)
                if r is not None and authentic is not None and r >= authentic:
                    ge += 1
                placebo_rows.append({"fold": fold, "b": b, "rho_h_placebo": r})
            placebo_pct = ge / B_PLACEBO if authentic is not None else None
        fold_summ.append(
            {
                "fold": fold,
                "status": "OK",
                "code_module": mod,
                "train": "+".join(train_pres),
                "holdout": ho,
                "n_slots": int(len(ho_s)),
                "rho_H": rho_h,
                "p_H": p_h,
                "rho_P": rho_p,
                "p_P": p_p,
                "delta_rho": d_rho,
                "rho_residual": rho_r,
                "p_residual": p_r,
                "topk_precision": prec,
                "topk_jaccard": jac,
                "placebo_feasible": len(ho_s) >= MIN_PLACEBO_SLOTS,
                "placebo_frac_ge_authentic": placebo_pct,
            }
        )
        pos_rows.append(
            {
                "fold": fold,
                "n_p_train_slots": int(len(p_train)),
                "bin_means": json.dumps({k: float(v) for k, v in bin_means.items()}),
                "grand_mean": grand,
                "rho_P": rho_p,
            }
        )
        res_rows.append({"fold": fold, "n_slots": int(len(ho_s)), "rho_residual": rho_r, "p_residual": p_r})
        log(f"{fold} n={len(ho_s)} rhoH={rho_h} rhoP={rho_p} d={d_rho} res={rho_r}")

    folds = pd.DataFrame(fold_summ)
    ok = folds[folds.status == "OK"].copy()
    med = lambda s: float(s.median()) if s.notna().any() else None
    med_h = med(ok["rho_H"]) if len(ok) else None
    med_p = med(ok["rho_P"]) if len(ok) else None
    med_d = med(ok["delta_rho"]) if len(ok) else None
    med_r = med(ok["rho_residual"]) if len(ok) else None
    n_pos = int((ok["delta_rho"] > 0).sum()) if len(ok) and ok["delta_rho"].notna().any() else 0
    n_h_pos = int((ok["rho_H"] > 0).sum()) if len(ok) else 0
    n_ok = int(len(ok))
    scenario = classify(med_h, med_p, med_d, med_r, n_pos, n_ok)
    if scenario == "D" and n_ok >= 2 and n_h_pos >= (n_ok / 2.0) and med_h is not None and med_h >= 0.20:
        scenario = "C"
    # module contextual check already handled by classify falling to C

    dump_csv("heldout_predictions.csv", pd.DataFrame(pred_rows))
    dump_csv("position_baseline.csv", pd.DataFrame(pos_rows))
    dump_csv("residual_stability.csv", pd.DataFrame(res_rows))
    dump_csv("topk_inspection.csv", pd.DataFrame(topk_rows))
    if placebo_rows:
        dump_csv("placebo_results.csv", pd.DataFrame(placebo_rows))
    else:
        dump_csv("placebo_results.csv", pd.DataFrame(columns=["fold", "b", "rho_h_placebo"]))

    dump_json(
        "summary.json",
        {
            "seed": SEED,
            "B_placebo": B_PLACEBO,
            "folds": fold_summ,
            "n_analyzable": n_ok,
            "median_rho_H": med_h,
            "median_rho_P": med_p,
            "median_delta_rho": med_d,
            "n_delta_positive": n_pos,
            "n_rhoH_positive": n_h_pos,
            "median_rho_residual": med_r,
            "median_topk_precision": med(ok["topk_precision"]) if len(ok) else None,
            "median_topk_jaccard": med(ok["topk_jaccard"]) if len(ok) else None,
            "scenario": scenario,
            "aaa_bbb": "NOT_IDENTIFIABLE_FOR_HELDOUT_REPLICATION",
            "student_outcome_used": False,
        },
    )
    log(f"PHASE 5C done scenario={scenario}")
    _guard(OUT / "run.log").write_text("\n".join(LOG) + "\n")


if __name__ == "__main__":
    main()
