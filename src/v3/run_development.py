#!/usr/bin/env python3
"""V3 development search on all previously inspected presentations.

Does not load outcomes for presentations outside V3_DEVELOPMENT.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from scipy import stats

_SRC = Path(__file__).resolve().parents[1]
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from common.paths import ROOT, SEED  # noqa: E402
from v2.load import (  # noqa: E402
    load_assessments,
    load_courses,
    load_registration,
    load_student_assessment,
    load_student_vle,
    load_vle,
)
from v2.panel import (  # noqa: E402
    attach_prior_score,
    attach_raw_controls,
    attach_tma_outcomes,
    documented_sites,
    map_prev_tma_date,
    placebo_specs,
    shift_windows,
)
from v2.week_panel import build_week_panel  # noqa: E402
from v3.constants import DIST_CLOSE, DIST_MID, FAMILY_OUCONTENT, MIN_OPP, V3_DEVELOPMENT, WATCH_FRAC  # noqa: E402
from v3.features import build_site_panel, clicks_by_learner_site_date, occasion_from_sites  # noqa: E402

OUT = ROOT / "results" / "v3"
OUT.mkdir(parents=True, exist_ok=True)

# Candidate feature lists (risk-positive features enter as-is; on-time is protective).
CANDIDATES = {
    "A_four_state": ["share_early", "share_ontime", "share_late", "share_never"],
    "B_coverage_decomp": ["ontime_coverage", "late_catchup_share", "share_never"],
    "C_continuous": ["median_signed_lag", "pre_tma_coverage"],
    "D_hybrid": ["ontime_coverage", "share_never", "median_signed_lag"],
}
# Extra compact contrast (construct reason: never vs aligned is the critical contrast).
CANDIDATES["E_never_ontime"] = ["ontime_coverage", "share_never"]


def _jd(x):
    if isinstance(x, (np.floating,)):
        return float(x)
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, (np.bool_,)):
        return bool(x)
    raise TypeError(type(x))


def dump(name, obj):
    p = OUT / name
    p.write_text(json.dumps(obj, indent=2, default=_jd))
    print("wrote", p, flush=True)


def scored(occ):
    return occ.dropna(subset=["score"]).query("n_opp >= @MIN_OPP").copy()


def gradient(site: pd.DataFrame) -> dict:
    sub = site.dropna(subset=["score_site"] if "score_site" in site.columns else [])
    # attach score at site rows via next assessment — caller merges
    if "score" not in site.columns:
        return {}
    g = site.dropna(subset=["score", "first_state"])
    rows = []
    for st, gg in g.groupby("first_state"):
        rows.append(
            {
                "state": st,
                "n": int(len(gg)),
                "mean_score": float(gg["score"].mean()),
                "sd": float(gg["score"].std()),
            }
        )
    by = {r["state"]: r for r in rows}
    out = {"by_state": rows}
    if "ontime" in by and "never" in by:
        out["ontime_minus_never"] = by["ontime"]["mean_score"] - by["never"]["mean_score"]
    if "ontime" in by and "late" in by:
        out["ontime_minus_late"] = by["ontime"]["mean_score"] - by["late"]["mean_score"]
    if "late" in by and "never" in by:
        out["late_minus_never"] = by["late"]["mean_score"] - by["never"]["mean_score"]
    # presentation heterogeneity of ontime-never
    het = []
    for (mod, pres), gg in g.groupby(["code_module", "code_presentation"]):
        means = gg.groupby("first_state")["score"].mean()
        if "ontime" in means and "never" in means:
            het.append(
                {
                    "code_module": mod,
                    "code_presentation": pres,
                    "ontime_minus_never": float(means["ontime"] - means["never"]),
                    "n": int(len(gg)),
                }
            )
    out["by_presentation_ontime_minus_never"] = het
    out["n_pres_ontime_better_than_never"] = int(sum(x["ontime_minus_never"] > 0 for x in het))
    return out


def fit_m3(df: pd.DataFrame, feats: list[str]) -> dict:
    work = df.dropna(subset=["score", "raw_inact", "log_clicks"] + [f for f in feats if f in df.columns]).copy()
    work = work[work["n_opp"] >= MIN_OPP]
    # lag missing => no pre-TMA access on any resource; fill 0 and flag
    if "median_signed_lag" in feats:
        work["lag_missing"] = work["median_signed_lag"].isna().astype(int)
        work["median_signed_lag"] = work["median_signed_lag"].fillna(0.0)
    work["pres"] = work["code_module"] + "_" + work["code_presentation"]
    if len(work) < 80:
        return {"insufficient": True, "n": int(len(work))}
    rhs_ctrl = "prior_score_filled + prior_missing + C(pres) + log_clicks + active_rate + n_opp + raw_inact"
    rhs_m3 = rhs_ctrl + " + " + " + ".join(feats)
    if "median_signed_lag" in feats:
        rhs_m3 += " + lag_missing"
    formulas = {
        "M0": "score ~ prior_score_filled + prior_missing + C(pres)",
        "M1": "score ~ prior_score_filled + prior_missing + C(pres) + log_clicks + active_rate + n_opp",
        "M2": "score ~ " + rhs_ctrl,
        "M3": "score ~ " + rhs_m3,
    }
    in_sample = {}
    for name, fml in formulas.items():
        fit = smf.ols(fml, data=work).fit(cov_type="cluster", cov_kwds={"groups": work["id_student"]})
        rec = {"n": int(fit.nobs), "r2": float(fit.rsquared), "rmse": float(np.sqrt(np.mean(fit.resid**2)))}
        if name == "M3":
            rec["coefs"] = {}
            ysd = float(work["score"].std())
            for f in feats:
                if f in fit.params:
                    rec["coefs"][f] = {
                        "coef": float(fit.params[f]),
                        "se": float(fit.bse[f]),
                        "p": float(fit.pvalues[f]),
                        "ci": [float(x) for x in fit.conf_int().loc[f].tolist()],
                        "std_xy": float(fit.params[f] * work[f].std() / ysd) if ysd and work[f].std() else np.nan,
                    }
        in_sample[name] = rec
    # LOO presentation with module FE
    loo_f = {
        "M2": "score ~ prior_score_filled + prior_missing + C(code_module) + log_clicks + active_rate + n_opp + raw_inact",
        "M3": "score ~ prior_score_filled + prior_missing + C(code_module) + log_clicks + active_rate + n_opp + raw_inact + "
        + " + ".join(feats)
        + (" + lag_missing" if "median_signed_lag" in feats else ""),
    }
    loo = {k: [] for k in loo_f}
    for pres, te in work.groupby("pres"):
        tr = work[work.pres != pres]
        if len(tr) < 80 or len(te) < 20:
            continue
        for name, fml in loo_f.items():
            fit = smf.ols(fml, data=tr).fit()
            pred = fit.predict(te)
            y = te.score.to_numpy()
            p = pred.to_numpy()
            loo[name].append(
                {
                    "pres": pres,
                    "n": int(len(te)),
                    "rmse": float(np.sqrt(np.mean((y - p) ** 2))),
                    "r2": float(1 - np.sum((y - p) ** 2) / np.sum((y - y.mean()) ** 2)) if y.std() else np.nan,
                }
            )
    loo_mean = {
        name: {
            "rmse": float(np.mean([r["rmse"] for r in rows])),
            "r2": float(np.nanmean([r["r2"] for r in rows])),
            "by_presentation": rows,
        }
        for name, rows in loo.items()
        if rows
    }
    delta = None
    if "M2" in loo_mean and "M3" in loo_mean:
        delta = {
            "r2_M3_minus_M2": loo_mean["M3"]["r2"] - loo_mean["M2"]["r2"],
            "rmse_M2_minus_M3": loo_mean["M2"]["rmse"] - loo_mean["M3"]["rmse"],
        }
    # residualized timing: M3 increment is the nested test
    return {
        "n": int(len(work)),
        "in_sample": in_sample,
        "loo": loo_mean,
        "incremental_loo": delta,
        "in_sample_delta_r2": in_sample["M3"]["r2"] - in_sample["M2"]["r2"],
    }


def within_learner(df: pd.DataFrame, feat: str) -> dict:
    work = df.dropna(subset=["score", feat]).copy()
    if feat == "median_signed_lag":
        work = work.dropna(subset=["median_signed_lag"])
    work["lid"] = work["code_module"] + "|" + work["code_presentation"] + "|" + work["id_student"].astype(str)
    cnt = work.groupby("lid").size()
    keep = set(cnt[cnt >= 2].index)
    work = work[work.lid.isin(keep)]
    if len(work) < 100:
        return {"insufficient": True, "n": int(len(work))}
    wvar = float(work.groupby("lid")[feat].var().mean())
    # demean
    for c in ["score", feat, "log_clicks", "raw_inact", "n_opp"]:
        if c in work.columns:
            work[c + "_w"] = work[c] - work.groupby("lid")[c].transform("mean")
    work["tma_c"] = work.groupby("lid")["next_tma_date"].transform(lambda s: (s - s.mean()) / (s.std() + 1e-6))
    fml = f"score_w ~ {feat}_w + log_clicks_w + raw_inact_w + n_opp_w + tma_c"
    fit = smf.ols(fml, data=work).fit(cov_type="cluster", cov_kwds={"groups": work["lid"]})
    key = f"{feat}_w"
    rec = {
        "n_occasions": int(len(work)),
        "n_learners": int(work.lid.nunique()),
        "within_var": wvar,
        "coef": float(fit.params.get(key, np.nan)),
        "se": float(fit.bse.get(key, np.nan)) if key in fit.bse else np.nan,
        "p": float(fit.pvalues.get(key, np.nan)) if key in fit.pvalues else np.nan,
        "ci": [float(x) for x in fit.conf_int().loc[key].tolist()] if key in fit.params else [np.nan, np.nan],
    }
    # presentation-level within signs
    signs = []
    for (mod, pres), g in work.groupby(["code_module", "code_presentation"]):
        if g.lid.nunique() < 20 or g[feat].nunique() < 2:
            continue
        for c in ["score", feat, "log_clicks", "raw_inact", "n_opp"]:
            g[c + "_w"] = g[c] - g.groupby("lid")[c].transform("mean")
        g["tma_c"] = g.groupby("lid")["next_tma_date"].transform(lambda s: (s - s.mean()) / (s.std() + 1e-6))
        try:
            f2 = smf.ols(fml, data=g).fit()
            signs.append(
                {
                    "code_module": mod,
                    "code_presentation": pres,
                    "coef": float(f2.params.get(key, np.nan)),
                    "n": int(len(g)),
                }
            )
        except Exception:
            continue
    rec["by_presentation"] = signs
    # expected direction: ontime +, never -, lag -
    rec["n_pres"] = len(signs)
    return rec


def dose(df: pd.DataFrame, feat: str) -> dict:
    work = scored(df).dropna(subset=[feat])
    rows = []
    for (mod, pres), g in work.groupby(["code_module", "code_presentation"]):
        try:
            g = g.copy()
            g["q"] = pd.qcut(g[feat], 4, labels=["Q1", "Q2", "Q3", "Q4"], duplicates="drop")
        except ValueError:
            continue
        means = g.groupby("q", observed=False)["score"].mean()
        rows.append(
            {
                "code_module": mod,
                "code_presentation": pres,
                "means": {str(k): float(v) for k, v in means.items()},
                "q4_minus_q1": float(means.iloc[-1] - means.iloc[0]) if len(means) >= 2 else np.nan,
            }
        )
    return {"by_presentation": rows, "n": len(rows)}


def matching(df: pd.DataFrame, align_feat: str, high_good: bool) -> dict:
    """Within-presentation tercile cells of activity × prior; compare high vs low alignment."""
    work = scored(df).dropna(subset=[align_feat, "log_clicks"])
    diffs = []
    rng = np.random.default_rng(SEED)
    for (mod, pres), g in work.groupby(["code_module", "code_presentation"]):
        g = g.copy()
        try:
            g["act_t"] = pd.qcut(g["log_clicks"], 3, labels=False, duplicates="drop")
            g["pri_t"] = pd.qcut(g["prior_score_filled"], 3, labels=False, duplicates="drop")
            g["aln_t"] = pd.qcut(g[align_feat], 3, labels=False, duplicates="drop")
        except ValueError:
            continue
        cell_d = []
        for (a, p), gg in g.groupby(["act_t", "pri_t"]):
            lo = gg[gg.aln_t == 0]
            hi = gg[gg.aln_t == gg.aln_t.max()]
            if len(lo) < 5 or len(hi) < 5:
                continue
            d = float(hi.score.mean() - lo.score.mean())
            if not high_good:
                d = -d
            cell_d.append(d)
        if cell_d:
            diffs.append({"code_module": mod, "code_presentation": pres, "mean_cell_diff": float(np.mean(cell_d)), "n_cells": len(cell_d)})
    if not diffs:
        return {"insufficient": True}
    vals = [d["mean_cell_diff"] for d in diffs]
    # bootstrap presentations
    boots = [float(np.mean(rng.choice(vals, size=len(vals), replace=True))) for _ in range(200)]
    return {
        "n_presentations": len(diffs),
        "mean_high_minus_low": float(np.mean(vals)),
        "ci": [float(np.quantile(boots, 0.025)), float(np.quantile(boots, 0.975))],
        "n_pres_positive": int(sum(v > 0 for v in vals)),
        "by_presentation": diffs,
    }


def practical(occ: pd.DataFrame, week_panel: pd.DataFrame, timing_risk: str) -> dict:
    """Rank on early occasions; evaluate last TMA. timing_risk higher = worse."""
    work = occ.dropna(subset=["next_tma_date"]).copy()
    recs = []
    for (mod, pres), g in work.groupby(["code_module", "code_presentation"]):
        last = g["next_tma_date"].max()
        early = g[g.next_tma_date < last]
        late = g[g.next_tma_date == last]
        if early.empty or late.empty:
            continue
        feat = early.groupby("id_student", as_index=False).agg(
            timing_risk=(timing_risk, "mean"),
            ontime=("ontime_coverage", "mean"),
            never=("share_never", "mean"),
        )
        y = late.groupby("id_student", as_index=False).agg(score=("score", "mean"), submitted=("submitted", "max"))
        wp = week_panel[(week_panel.code_module == mod) & (week_panel.code_presentation == pres)]
        raw = wp[wp.week_end < last].groupby("id_student", as_index=False).agg(raw_inact=("raw_inact", "mean"))
        m = feat.merge(y, on="id_student").merge(raw, on="id_student", how="left")
        m["adverse"] = ((m.submitted == 0) | (m.score < 40)).astype(int)
        if len(m) < 40:
            continue
        k = max(1, int(round(WATCH_FRAC * len(m))))
        # z-scores within presentation
        for c in ["raw_inact", "timing_risk"]:
            sd = m[c].std()
            m["z_" + c] = 0.0 if not sd or sd == 0 else (m[c] - m[c].mean()) / sd
        m["combo"] = m["z_raw_inact"] + m["z_timing_risk"]

        def top(col):
            ids = set(m.sort_values([col, "id_student"], ascending=[False, True]).id_student.iloc[:k])
            sub = m[m.id_student.isin(ids)]
            return float(sub.adverse.mean()), float(sub.score.mean()) if sub.score.notna().any() else np.nan

        base = float(m.adverse.mean())
        r_a, r_s = top("raw_inact")
        t_a, t_s = top("timing_risk")
        c_a, c_s = top("combo")
        recs.append(
            {
                "code_module": mod,
                "code_presentation": pres,
                "n": int(len(m)),
                "k": int(k),
                "base": base,
                "raw_adverse": r_a,
                "timing_adverse": t_a,
                "combo_adverse": c_a,
                "raw_enrich": r_a - base,
                "timing_enrich": t_a - base,
                "combo_enrich": c_a - base,
            }
        )
    if not recs:
        return {"n_presentations": 0}
    df = pd.DataFrame(recs)
    return {
        "n_presentations": int(len(df)),
        "mean_raw_enrich": float(df.raw_enrich.mean()),
        "mean_timing_enrich": float(df.timing_enrich.mean()),
        "mean_combo_enrich": float(df.combo_enrich.mean()),
        "mean_raw_adverse": float(df.raw_adverse.mean()),
        "mean_timing_adverse": float(df.timing_adverse.mean()),
        "mean_combo_adverse": float(df.combo_adverse.mean()),
        "n_combo_beats_raw": int((df.combo_adverse > df.raw_adverse).sum()),
        "by_presentation": recs,
    }


def interaction(df: pd.DataFrame) -> dict:
    work = scored(df).dropna(subset=["share_never", "ontime_coverage", "raw_inact", "log_clicks"])
    work["pres"] = work.code_module + "_" + work.code_presentation
    work["prior_c"] = work.prior_score_filled - work.prior_score_filled.mean()
    fml = (
        "score ~ prior_score_filled + prior_missing + C(pres) + log_clicks + active_rate + n_opp + raw_inact "
        "+ share_never + ontime_coverage + share_never:prior_c + ontime_coverage:prior_c"
    )
    fit = smf.ols(fml, data=work).fit(cov_type="cluster", cov_kwds={"groups": work.id_student})
    keys = ["share_never:prior_c", "ontime_coverage:prior_c"]
    out = {}
    for k in keys:
        if k in fit.params:
            out[k] = {
                "coef": float(fit.params[k]),
                "se": float(fit.bse[k]),
                "p": float(fit.pvalues[k]),
                "ci": [float(x) for x in fit.conf_int().loc[k].tolist()],
            }
    return out


def distance_mech(site: pd.DataFrame) -> dict:
    g = site.dropna(subset=["score", "first_state", "dist_to_tma"]).copy()
    g["dist_bin"] = np.where(g.dist_to_tma <= DIST_CLOSE, "close", np.where(g.dist_to_tma <= DIST_MID, "mid", "far"))
    rows = []
    for (b, st), gg in g.groupby(["dist_bin", "first_state"]):
        rows.append({"dist_bin": b, "state": st, "n": int(len(gg)), "mean_score": float(gg.score.mean())})
    contrasts = {}
    for b in ["close", "mid", "far"]:
        sub = {r["state"]: r["mean_score"] for r in rows if r["dist_bin"] == b}
        if "ontime" in sub and "never" in sub:
            contrasts[b] = sub["ontime"] - sub["never"]
    return {"cells": rows, "ontime_minus_never_by_bin": contrasts}


def main() -> None:
    print("Loading V3 development inputs...", flush=True)
    vle = load_vle()
    courses = load_courses()
    reg = load_registration()
    sa = load_student_assessment()
    assessments = load_assessments()
    tma_meta = map_prev_tma_date(assessments)
    sv = load_student_vle()
    pres = set(V3_DEVELOPMENT)
    sites0 = documented_sites(vle, pres, FAMILY_OUCONTENT)
    print("oucontent documented sites", len(sites0), flush=True)
    clicks = clicks_by_learner_site_date(sv, sites0.id_site.tolist())
    print("click-date rows", len(clicks), flush=True)
    week_panel = build_week_panel(sv, pres)
    del sv

    timings = {}
    authentic_occ = None
    authentic_site = None
    for spec in placebo_specs():
        name, kind, shift = spec
        print("timing", name, flush=True)
        sites = sites0
        if kind == "circular":
            sites = shift_windows(sites0, courses, "circular", shift)
        elif kind == "random":
            sites = shift_windows(sites0, courses, "random")
        site = build_site_panel(sites, reg, assessments, clicks)
        occ = occasion_from_sites(site)
        occ = attach_tma_outcomes(occ, sa, tma_meta)
        if name == "authentic":
            occ = attach_prior_score(occ, sa, assessments)
            occ = attach_raw_controls(occ, week_panel)
            authentic_occ = occ
            # site-level scores for gradient
            site = site.merge(
                occ[["code_module", "code_presentation", "id_student", "next_id_assessment", "score"]],
                on=["code_module", "code_presentation", "id_student", "next_id_assessment"],
                how="left",
            )
            authentic_site = site
        timings[name] = occ

    occ = authentic_occ
    site = authentic_site
    print("occasions", len(occ), "scored", occ.score.notna().sum(), flush=True)

    grad = gradient(site)
    dump("timing_gradient.json", grad)

    cand_eval = {}
    for cname, feats in CANDIDATES.items():
        print("candidate", cname, flush=True)
        models = fit_m3(occ, feats)
        # placebo incremental / spearman of a scalar risk
        # scalar: never - ontime, or lag
        if "share_never" in occ.columns:
            occ["_risk"] = occ["share_never"] - occ["ontime_coverage"]
        placebos = {}
        for pname, pocc in timings.items():
            if pname == "authentic":
                continue
            tmp = pocc.copy()
            # merge scores/controls from authentic occ keys
            keys = ["code_module", "code_presentation", "id_student", "next_id_assessment"]
            tmp = tmp.merge(
                occ[keys + ["score", "prior_score_filled", "prior_missing", "log_clicks", "active_rate", "raw_inact"]],
                on=keys,
                how="left",
                suffixes=("", "_y"),
            )
            if "score" not in tmp.columns and "score_y" in tmp.columns:
                tmp["score"] = tmp["score_y"]
            use_feats = [f for f in feats if f in tmp.columns]
            try:
                placebos[pname] = {
                    "spearman_never": float(stats.spearmanr(tmp.dropna(subset=["score", "share_never"]).share_never, tmp.dropna(subset=["score", "share_never"]).score).statistic)
                    if tmp.dropna(subset=["score", "share_never"]).share_never.nunique() > 1
                    else np.nan,
                    "models": fit_m3(tmp, use_feats) if len(tmp.dropna(subset=["score"])) > 80 else {"insufficient": True},
                }
            except Exception as e:
                placebos[pname] = {"error": str(e)}
        auth_s = float(
            stats.spearmanr(scored(occ).share_never, scored(occ).score).statistic
        )
        cand_eval[cname] = {
            "features": feats,
            "models": models,
            "authentic_spearman_never": auth_s,
            "placebos": {
                k: {
                    "spearman_never": v.get("spearman_never"),
                    "delta_r2": (v.get("models") or {}).get("incremental_loo"),
                    "in_sample_delta_r2": (v.get("models") or {}).get("in_sample_delta_r2"),
                }
                for k, v in placebos.items()
            },
        }

    dump("candidate_models.json", cand_eval)

    within = {
        "ontime_coverage": within_learner(occ, "ontime_coverage"),
        "share_never": within_learner(occ, "share_never"),
        "median_signed_lag": within_learner(occ, "median_signed_lag"),
    }
    dump("within_learner.json", within)

    dump(
        "dose_response.json",
        {
            "ontime_coverage": dose(occ, "ontime_coverage"),
            "share_never": dose(occ, "share_never"),
            "median_signed_lag": dose(occ, "median_signed_lag"),
        },
    )

    match = {
        "ontime_coverage": matching(occ, "ontime_coverage", high_good=True),
        "share_never": matching(occ, "share_never", high_good=False),
    }
    dump("same_activity_matching.json", match)

    # practical: authentic never-share as timing risk; placebo never-share
    prac_auth = practical(occ, week_panel, "share_never")
    prac_placebo = {}
    for pname, pocc in timings.items():
        if pname == "authentic":
            continue
        keys = ["code_module", "code_presentation", "id_student", "next_id_assessment"]
        merged = pocc.merge(
            occ[keys + ["submitted", "score"]],
            on=keys,
            how="left",
            suffixes=("", "_y"),
        )
        if "submitted" not in merged.columns and "submitted_y" in merged.columns:
            merged["submitted"] = merged["submitted_y"]
        prac_placebo[pname] = practical(merged, week_panel, "share_never")
    dump("practical_increment.json", {"authentic": prac_auth, "placebo": prac_placebo})

    dump("interaction.json", interaction(occ))
    dump("assessment_distance.json", distance_mech(site))

    # by-presentation spearman for H1
    byp = []
    for (mod, pres), g in scored(occ).groupby(["code_module", "code_presentation"]):
        if g.share_never.nunique() < 2:
            continue
        r, _ = stats.spearmanr(g.share_never, g.score)
        r2, _ = stats.spearmanr(g.ontime_coverage, g.score)
        byp.append(
            {
                "code_module": mod,
                "code_presentation": pres,
                "n": int(len(g)),
                "rho_never": float(r),
                "rho_ontime": float(r2),
                "mean_ontime": float(g.ontime_coverage.mean()),
                "mean_never": float(g.share_never.mean()),
                "mean_early": float(g.share_early.mean()),
                "mean_late": float(g.share_late.mean()),
            }
        )
    dump("by_presentation.json", byp)

    # headline for ranking
    headline = []
    for cname, ev in cand_eval.items():
        m = ev["models"]
        inc = m.get("incremental_loo") or {}
        headline.append(
            {
                "candidate": cname,
                "in_sample_delta_r2": m.get("in_sample_delta_r2"),
                "loo_delta_r2": inc.get("r2_M3_minus_M2"),
                "loo_delta_rmse": inc.get("rmse_M2_minus_M3"),
                "coefs": (m.get("in_sample") or {}).get("M3", {}).get("coefs"),
            }
        )
    dump("development_headline.json", headline)
    print(json.dumps(headline, indent=2, default=_jd), flush=True)
    print("gradient", {k: grad.get(k) for k in ["ontime_minus_never", "ontime_minus_late", "late_minus_never", "n_pres_ontime_better_than_never"]}, flush=True)
    print("within never", within["share_never"], flush=True)
    print("practical auth", {k: prac_auth.get(k) for k in prac_auth if k != "by_presentation"}, flush=True)
    print("matching ontime", match["ontime_coverage"], flush=True)


if __name__ == "__main__":
    main()
