#!/usr/bin/env python3
"""V4 development: planned-resource pre-TMA coverage on inspected presentations only."""
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
    map_next_tma,
    map_prev_tma_date,
    registered_learner_sites,
)
from v2.week_panel import build_week_panel  # noqa: E402
from v3.constants import FAMILY_OUCONTENT  # noqa: E402
from v3.features import annotate_site_states, clicks_by_learner_site_date, occasion_from_sites  # noqa: E402
from v4.constants import B_BOOT, MIN_OPP, V4_DEVELOPMENT  # noqa: E402

OUT = ROOT / "results" / "v4"
OUT.mkdir(parents=True, exist_ok=True)


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


def dose_bins(occ: pd.DataFrame) -> dict:
    sub = scored(occ)
    sub["bin"] = pd.cut(
        sub["share_never"],
        bins=[-0.01, 0.0, 0.25, 0.50, 0.75, 1.01],
        labels=["0", ">0-25", ">25-50", ">50-75", ">75-100"],
        include_lowest=True,
    )
    rows = []
    for b, g in sub.groupby("bin", observed=False):
        rows.append({"bin": str(b), "n": int(len(g)), "mean_score": float(g.score.mean()) if len(g) else np.nan})
    # trend: Spearman never vs score overall and by presentation
    rho, _ = stats.spearmanr(sub.share_never, sub.score)
    het = []
    for (mod, pres), g in sub.groupby(["code_module", "code_presentation"]):
        if g.share_never.nunique() < 2:
            continue
        r, _ = stats.spearmanr(g.share_never, g.score)
        het.append({"code_module": mod, "code_presentation": pres, "n": int(len(g)), "rho": float(r)})
    return {
        "bins": rows,
        "spearman": float(rho),
        "n_pres_negative": int(sum(x["rho"] < 0 for x in het)),
        "n_pres": len(het),
        "by_presentation": het,
        "n": int(len(sub)),
    }


def fit_hierarchy(occ: pd.DataFrame, feat: str = "share_never") -> dict:
    work = scored(occ).dropna(subset=["raw_inact", "log_clicks", feat])
    work["pres"] = work.code_module + "_" + work.code_presentation
    work["coverage"] = 1 - work[feat]
    f = {
        "M0": "score ~ prior_score_filled + prior_missing + C(pres)",
        "M1": "score ~ prior_score_filled + prior_missing + C(pres) + log_clicks + active_rate + n_opp",
        "M2": "score ~ prior_score_filled + prior_missing + C(pres) + log_clicks + active_rate + n_opp + raw_inact",
        "M3": "score ~ prior_score_filled + prior_missing + C(pres) + log_clicks + active_rate + n_opp + raw_inact + "
        + feat,
    }
    ins = {}
    fits = {}
    for name, fml in f.items():
        fit = smf.ols(fml, data=work).fit(cov_type="cluster", cov_kwds={"groups": work.id_student})
        fits[name] = fit
        rec = {
            "n": int(fit.nobs),
            "r2": float(fit.rsquared),
            "rmse": float(np.sqrt(np.mean(fit.resid**2))),
            "mae": float(np.mean(np.abs(fit.resid))),
        }
        if name == "M3" and feat in fit.params:
            ysd = float(work.score.std())
            rec["coef"] = float(fit.params[feat])
            rec["se"] = float(fit.bse[feat])
            rec["p"] = float(fit.pvalues[feat])
            rec["ci"] = [float(x) for x in fit.conf_int().loc[feat].tolist()]
            rec["std_xy"] = float(fit.params[feat] * work[feat].std() / ysd) if ysd else np.nan
            rec["points_per_25pp"] = float(fit.params[feat] * 0.25)
        ins[name] = rec
    # partial R2 and LR
    partial = (ins["M3"]["r2"] - ins["M2"]["r2"]) / (1 - ins["M2"]["r2"])
    # semi-partial: corr of resid_M2 with never
    work["e2"] = fits["M2"].resid
    spr, _ = stats.pearsonr(work["e2"], work[feat])
    lr = 2 * (fits["M3"].llf - fits["M2"].llf)
    # LOO presentation
    loo = {"M2": [], "M3": []}
    loo_f = {
        "M2": "score ~ prior_score_filled + prior_missing + C(code_module) + log_clicks + active_rate + n_opp + raw_inact",
        "M3": "score ~ prior_score_filled + prior_missing + C(code_module) + log_clicks + active_rate + n_opp + raw_inact + "
        + feat,
    }
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
                    "r2": float(1 - np.sum((y - p) ** 2) / np.sum((y - y.mean()) ** 2)) if y.std() else np.nan,
                    "rmse": float(np.sqrt(np.mean((y - p) ** 2))),
                    "coef": float(fit.params.get(feat, np.nan)) if name == "M3" else np.nan,
                }
            )
    loo_mean = {
        k: {
            "r2": float(np.nanmean([r["r2"] for r in v])),
            "rmse": float(np.mean([r["rmse"] for r in v])),
            "by_presentation": v,
        }
        for k, v in loo.items()
        if v
    }
    # bootstrap ΔR² (learner cluster)
    rng = np.random.default_rng(SEED)
    lids = work.id_student.unique()
    deltas = []
    fml2 = f["M2"]
    fml3 = f["M3"]
    for _ in range(B_BOOT):
        draw = rng.choice(lids, size=len(lids), replace=True)
        # approximate: sample rows of drawn learners
        idx = np.concatenate([work.index[work.id_student == i].to_numpy() for i in draw[: max(1, len(draw) // 8)]])
        # faster: sample occasions
        samp = work.sample(n=len(work), replace=True, random_state=int(rng.integers(1e9)))
        try:
            r2 = smf.ols(fml2, data=samp).fit().rsquared
            r3 = smf.ols(fml3, data=samp).fit().rsquared
            deltas.append(float(r3 - r2))
        except Exception:
            continue
    return {
        "n": int(len(work)),
        "in_sample": ins,
        "partial_r2": float(partial),
        "semipartial_r": float(spr),
        "lr_stat": float(lr),
        "loo": loo_mean,
        "loo_delta_r2": loo_mean["M3"]["r2"] - loo_mean["M2"]["r2"] if "M3" in loo_mean else np.nan,
        "bootstrap_delta_r2": {
            "mean": float(np.mean(deltas)) if deltas else np.nan,
            "ci": [float(np.quantile(deltas, 0.025)), float(np.quantile(deltas, 0.975))] if len(deltas) >= 20 else None,
            "B": len(deltas),
        },
    }


def nonlinearity(occ: pd.DataFrame) -> dict:
    work = scored(occ).dropna(subset=["share_never", "raw_inact", "log_clicks"])
    work["pres"] = work.code_module + "_" + work.code_presentation
    base = "score ~ prior_score_filled + prior_missing + C(pres) + log_clicks + active_rate + n_opp + raw_inact"
    lin = smf.ols(base + " + share_never", data=work).fit()
    # restricted cubic-ish: never + never^2 (df=2 polynomial, not a search)
    work["never2"] = work.share_never**2
    quad = smf.ols(base + " + share_never + never2", data=work).fit()
    work["nq"] = pd.qcut(work.share_never, 4, duplicates="drop")
    qmeans = work.groupby("nq", observed=False).score.mean()
    return {
        "linear_r2": float(lin.rsquared),
        "quad_r2": float(quad.rsquared),
        "quad_minus_linear": float(quad.rsquared - lin.rsquared),
        "never2_p": float(quad.pvalues.get("never2", np.nan)),
        "quartile_means": {str(k): float(v) for k, v in qmeans.items()},
    }


def within_learner(occ: pd.DataFrame) -> dict:
    work = scored(occ).dropna(subset=["share_never", "log_clicks", "raw_inact"])
    work["lid"] = work.code_module + "|" + work.code_presentation + "|" + work.id_student.astype(str)
    keep = set(work.groupby("lid").size().pipe(lambda s: s[s >= 2].index))
    work = work[work.lid.isin(keep)]
    if len(work) < 50 or work.lid.nunique() < 20 or work.share_never.nunique() < 2:
        return {
            "n_occasions": int(len(work)),
            "n_learners": int(work.lid.nunique()) if len(work) else 0,
            "insufficient": True,
        }
    wsd = float(work.groupby("lid").share_never.std().mean())
    for c in ["score", "share_never", "log_clicks", "raw_inact", "n_opp"]:
        work[c + "_w"] = work[c] - work.groupby("lid")[c].transform("mean")
    work["tma_c"] = work.groupby("lid").next_tma_date.transform(lambda s: (s - s.mean()) / (s.std() + 1e-6))
    fit = smf.ols(
        "score_w ~ share_never_w + log_clicks_w + raw_inact_w + n_opp_w + tma_c",
        data=work,
    ).fit(cov_type="cluster", cov_kwds={"groups": work.lid})
    # Mundlak: within + between
    work["never_bar"] = work.groupby("lid").share_never.transform("mean")
    mund = smf.ols(
        "score ~ share_never + never_bar + log_clicks + raw_inact + n_opp + C(code_module)",
        data=work,
    ).fit(cov_type="cluster", cov_kwds={"groups": work.lid})
    return {
        "n_occasions": int(len(work)),
        "n_learners": int(work.lid.nunique()),
        "within_sd": wsd,
        "within_coef": float(fit.params["share_never_w"]),
        "within_se": float(fit.bse["share_never_w"]),
        "within_ci": [float(x) for x in fit.conf_int().loc["share_never_w"]],
        "within_p": float(fit.pvalues["share_never_w"]),
        "mundlak_within": float(mund.params["share_never"]),
        "mundlak_between": float(mund.params["never_bar"]),
        "mundlak_within_ci": [float(x) for x in mund.conf_int().loc["share_never"]],
        "mundlak_between_ci": [float(x) for x in mund.conf_int().loc["never_bar"]],
    }


def matching(occ: pd.DataFrame) -> dict:
    work = scored(occ).dropna(subset=["share_never", "log_clicks", "active_rate"])
    diffs = []
    rng = np.random.default_rng(SEED)
    for (mod, pres), g in work.groupby(["code_module", "code_presentation"]):
        g = g.copy()
        try:
            g["pri"] = pd.qcut(g.prior_score_filled, 3, labels=False, duplicates="drop")
            g["act"] = pd.qcut(g.log_clicks, 3, labels=False, duplicates="drop")
            g["ar"] = pd.qcut(g.active_rate, 3, labels=False, duplicates="drop")
            g["nop"] = pd.qcut(g.n_opp, 3, labels=False, duplicates="drop")
            g["cov"] = pd.qcut(1 - g.share_never, 3, labels=False, duplicates="drop")
        except ValueError:
            continue
        cell = []
        for _, gg in g.groupby(["pri", "act", "ar", "nop"]):
            lo = gg[gg["cov"] == 0]
            hi = gg[gg["cov"] == gg["cov"].max()]
            if len(lo) < 4 or len(hi) < 4:
                continue
            cell.append(float(hi.score.mean() - lo.score.mean()))
        if cell:
            diffs.append(
                {
                    "code_module": mod,
                    "code_presentation": pres,
                    "diff": float(np.mean(cell)),
                    "n_cells": len(cell),
                    "n": int(len(g)),
                }
            )
    if not diffs:
        return {"insufficient": True}
    vals = np.array([d["diff"] for d in diffs])
    w = np.array([d["n"] for d in diffs], dtype=float)
    pooled = float(np.average(vals, weights=w))
    boots = [float(np.average(rng.choice(vals, size=len(vals), replace=True))) for _ in range(200)]
    return {
        "n_presentations": len(diffs),
        "unweighted_mean": float(vals.mean()),
        "weighted_mean": pooled,
        "ci": [float(np.quantile(boots, 0.025)), float(np.quantile(boots, 0.975))],
        "n_pres_positive": int((vals > 0).sum()),
        "by_presentation": diffs,
    }


def resource_identity_placebo(site: pd.DataFrame, occ: pd.DataFrame, clicks: pd.DataFrame) -> dict:
    """Wrong-plan identity: same family/presentation/count, different documented sites.

    Matching is frozen before outcome comparison:
    - donor pool = other planned sites in the same presentation (same activity family);
    - calendar match = nearest |window_end − median(actual window_end)|;
    - tie-break = id_site;
    - undocumented sites are not donors (missing plan ≠ unplanned).
    Access uses clicks with date < this occasion's TMA (no other-TMA never reuse).
    If a TMA's planned set is the entire presentation pool, that TMA is dropped.
    """
    keys = ["code_module", "code_presentation", "id_student", "next_id_assessment"]
    work = site.dropna(subset=["next_id_assessment", "next_tma_date"]).copy()
    work["next_id_assessment"] = work.next_id_assessment.astype(int)
    site_cal = work.groupby(["code_module", "code_presentation", "id_site"], as_index=False).agg(
        window_end=("window_end_day", "first")
    )
    tma_sets = work.groupby(["code_module", "code_presentation", "next_id_assessment"]).id_site.apply(
        lambda s: set(int(x) for x in s.unique())
    )
    match_rows = []
    skipped = []
    for (mod, pres, tma), actual in tma_sets.items():
        pool = site_cal[(site_cal.code_module == mod) & (site_cal.code_presentation == pres)]
        alt = pool[~pool.id_site.isin(actual)]
        k = len(actual)
        if k < 1 or len(alt) < k:
            skipped.append(
                {
                    "code_module": mod,
                    "code_presentation": pres,
                    "next_id_assessment": int(tma),
                    "n_actual": k,
                    "n_alt": int(len(alt)),
                    "reason": "fewer alternative planned sites than n_opp",
                }
            )
            continue
        med = float(pool[pool.id_site.isin(actual)].window_end.median())
        pick = alt.assign(dist=(alt.window_end - med).abs()).sort_values(["dist", "id_site"]).id_site.iloc[:k]
        for sid in pick.tolist():
            match_rows.append(
                {
                    "code_module": mod,
                    "code_presentation": pres,
                    "next_id_assessment": int(tma),
                    "id_site": int(sid),
                    "k": k,
                }
            )
    if not match_rows:
        return {
            "feasible": False,
            "reason": (
                "No TMA occasion has a same-presentation leftover planned-site set of equal size. "
                "When every documented planned site maps to the same TMA, identity replacement "
                "is identical to the actual coverage measure. Undocumented sites are not used as donors."
            ),
            "skipped_tmas": skipped,
        }
    matched = pd.DataFrame(match_rows)
    o = scored(occ).merge(
        matched[["code_module", "code_presentation", "next_id_assessment"]].drop_duplicates(),
        on=["code_module", "code_presentation", "next_id_assessment"],
        how="inner",
    )
    if len(o) < 100:
        return {"feasible": False, "n": int(len(o)), "reason": "matched occasions < 100", "skipped_tmas": skipped}
    pairs = o[keys + ["next_tma_date"]].drop_duplicates().merge(
        matched, on=["code_module", "code_presentation", "next_id_assessment"], how="inner"
    )
    acc = pairs.merge(clicks, on=["id_site", "id_student"], how="left")
    acc = acc[acc.date.notna() & (acc.date < acc.next_tma_date)]
    hit = acc.groupby(keys, as_index=False).id_site.nunique().rename(columns={"id_site": "n_hit"})
    kk = matched.groupby(["code_module", "code_presentation", "next_id_assessment"], as_index=False).k.first()
    plc = o[keys].drop_duplicates().merge(kk, on=["code_module", "code_presentation", "next_id_assessment"])
    plc = plc.merge(hit, on=keys, how="left")
    plc["n_hit"] = plc.n_hit.fillna(0)
    plc["never_placebo"] = 1.0 - plc.n_hit / plc.k
    m = scored(occ).merge(plc[keys + ["never_placebo"]], on=keys, how="inner")
    ra, _ = stats.spearmanr(m.share_never, m.score)
    rp, _ = stats.spearmanr(m.never_placebo, m.score)
    m["pres"] = m.code_module + "_" + m.code_presentation
    sub = m.dropna(subset=["raw_inact", "log_clicks"])
    fml = "score ~ prior_score_filled + prior_missing + C(pres) + log_clicks + active_rate + n_opp + raw_inact + {f}"
    fa = smf.ols(fml.format(f="share_never"), data=sub).fit(cov_type="cluster", cov_kwds={"groups": sub.id_student})
    fp = smf.ols(fml.format(f="never_placebo"), data=sub).fit(cov_type="cluster", cov_kwds={"groups": sub.id_student})
    ca, cp = float(fa.params["share_never"]), float(fp.params["never_placebo"])
    return {
        "feasible": True,
        "construction": (
            "Within presentation and documented planned oucontent, replace the TMA's planned "
            "site set with a same-size set of other planned sites whose window_end is closest "
            "to the actual set's median window_end (tie-break id_site). Pre-TMA access is "
            "recomputed on the matched sites. Undocumented oucontent is excluded."
        ),
        "n": int(len(m)),
        "n_tmas_matched": int(matched.next_id_assessment.nunique()),
        "n_tmas_skipped": len(skipped),
        "rho_actual": float(ra),
        "rho_placebo": float(rp),
        "coef_actual": ca,
        "coef_placebo": cp,
        "ci_actual": [float(x) for x in fa.conf_int().loc["share_never"]],
        "ci_placebo": [float(x) for x in fp.conf_int().loc["never_placebo"]],
        "actual_more_negative_rho": bool(ra < rp),
        "actual_more_negative_coef": bool(ca < cp),
        "skipped_tmas": skipped,
    }


def proximal_sensitivity(site: pd.DataFrame, occ: pd.DataFrame, horizon_days: int = 28) -> dict:
    """Sensitivity: planned sites whose window ends within `horizon_days` of the next TMA.

    Horizon is one instructional block (28 days) from calendar structure, not outcome-tuned.
    """
    work = site.dropna(subset=["next_id_assessment", "next_tma_date"]).copy()
    work["next_id_assessment"] = work.next_id_assessment.astype(int)
    work["prox"] = (work.next_tma_date - work.window_end_day) <= horizon_days
    prox = (
        work[work.prox]
        .groupby(["code_module", "code_presentation", "id_student", "next_id_assessment"], as_index=False)
        .agg(n_prox=("id_site", "size"), share_never_prox=("never", "mean"))
    )
    m = scored(occ).merge(
        prox,
        on=["code_module", "code_presentation", "id_student", "next_id_assessment"],
        how="inner",
    )
    if len(m) < 100:
        return {"n": int(len(m)), "horizon_days": horizon_days, "insufficient": True}
    m = m.dropna(subset=["raw_inact", "log_clicks", "share_never_prox"])
    m["pres"] = m.code_module + "_" + m.code_presentation
    fml = "score ~ prior_score_filled + prior_missing + C(pres) + log_clicks + active_rate + n_opp + raw_inact + {f}"
    ff = smf.ols(fml.format(f="share_never"), data=m).fit(cov_type="cluster", cov_kwds={"groups": m.id_student})
    fp = smf.ols(fml.format(f="share_never_prox"), data=m).fit(cov_type="cluster", cov_kwds={"groups": m.id_student})
    return {
        "horizon_days": horizon_days,
        "n": int(len(m)),
        "frac_occasions_with_proximal": float(len(m) / max(len(scored(occ)), 1)),
        "coef_full_on_overlap": float(ff.params["share_never"]),
        "ci_full": [float(x) for x in ff.conf_int().loc["share_never"]],
        "coef_proximal": float(fp.params["share_never_prox"]),
        "ci_proximal": [float(x) for x in fp.conf_int().loc["share_never_prox"]],
        "note": "Sensitivity only. Primary remains the full eligible planned set.",
    }


def module_holdout(occ: pd.DataFrame) -> dict:
    work = scored(occ).dropna(subset=["raw_inact", "log_clicks", "share_never"])
    rows = []
    fml = "score ~ prior_score_filled + prior_missing + log_clicks + active_rate + n_opp + raw_inact + share_never"
    for mod, te in work.groupby("code_module"):
        tr = work[work.code_module != mod]
        if len(tr) < 80 or len(te) < 20:
            continue
        fit = smf.ols(fml, data=tr).fit()
        pred = fit.predict(te)
        y = te.score.to_numpy()
        p = pred.to_numpy()
        r2 = float(1 - np.sum((y - p) ** 2) / np.sum((y - y.mean()) ** 2)) if y.std() else np.nan
        rows.append(
            {
                "code_module": mod,
                "n": int(len(te)),
                "coef": float(fit.params["share_never"]),
                "r2": r2,
                "rmse": float(np.sqrt(np.mean((y - p) ** 2))),
            }
        )
    return {
        "n_modules": len(rows),
        "n_coef_negative": int(sum(r["coef"] < 0 for r in rows)),
        "by_module": rows,
    }


def unplanned_feasibility() -> dict:
    return {
        "feasible": False,
        "reason": (
            "Empty week_from/week_to is missing documentation, not a documented unplanned resource "
            "(Kuzilek 2017; V1 Sensitivity C found zero confident non-plan weeks). "
            "Manufacturing an unplanned set from NA fields is forbidden."
        ),
    }


def temporal_specificity(occ: pd.DataFrame) -> dict:
    """Compare never_share association with next vs prior vs later TMA score."""
    work = occ.dropna(subset=["share_never"]).copy()
    # prior score already on occ; later TMA: next next
    work = work.sort_values(["code_module", "code_presentation", "id_student", "next_tma_date"])
    work["later_score"] = work.groupby(["code_module", "code_presentation", "id_student"]).score.shift(-1)
    sub = work.dropna(subset=["score", "share_never"])
    r_next, _ = stats.spearmanr(sub.share_never, sub.score)
    subp = work.dropna(subset=["prior_score", "share_never"])
    r_prior, _ = stats.spearmanr(subp.share_never, subp.prior_score) if len(subp) > 50 else (np.nan, np.nan)
    subl = work.dropna(subset=["later_score", "share_never"])
    r_later, _ = stats.spearmanr(subl.share_never, subl.later_score) if len(subl) > 50 else (np.nan, np.nan)
    return {
        "rho_next": float(r_next),
        "rho_prior": float(r_prior) if np.isfinite(r_prior) else None,
        "rho_later_tma": float(r_later) if np.isfinite(r_later) else None,
        "n_next": int(len(sub)),
        "n_prior": int(len(subp)),
        "n_later": int(len(subl)),
        "next_stronger_than_prior": bool(np.isfinite(r_next) and np.isfinite(r_prior) and r_next < r_prior),
    }


def residual_risk(occ: pd.DataFrame) -> dict:
    work = scored(occ).dropna(subset=["raw_inact", "log_clicks", "share_never"])
    work["pres"] = work.code_module + "_" + work.code_presentation
    work["fold"] = work.pres
    # cross-fit residuals by presentation
    work["resid"] = np.nan
    fml = "score ~ prior_score_filled + prior_missing + log_clicks + active_rate + n_opp + raw_inact + C(code_module)"
    for pres, te in work.groupby("fold"):
        tr = work[work.fold != pres]
        if len(tr) < 80:
            continue
        fit = smf.ols(fml, data=tr).fit()
        work.loc[te.index, "resid"] = te.score.to_numpy() - fit.predict(te).to_numpy()
    sub = work.dropna(subset=["resid"])
    r, p = stats.pearsonr(sub.resid, sub.share_never)
    fit = smf.ols("resid ~ share_never", data=sub).fit(cov_type="cluster", cov_kwds={"groups": sub.id_student})
    het = []
    for (mod, pres), g in sub.groupby(["code_module", "code_presentation"]):
        if g.share_never.nunique() < 2:
            continue
        rr, _ = stats.pearsonr(g.resid, g.share_never)
        het.append({"code_module": mod, "code_presentation": pres, "r": float(rr), "n": int(len(g))})
    return {
        "n": int(len(sub)),
        "resid_corr": float(r),
        "resid_p": float(p),
        "resid_coef": float(fit.params["share_never"]),
        "resid_ci": [float(x) for x in fit.conf_int().loc["share_never"]],
        "partial_r2_approx": float(r**2),
        "n_pres_neg": int(sum(x["r"] < 0 for x in het)),
        "by_presentation": het,
    }


def complementary(occ: pd.DataFrame, week_panel: pd.DataFrame) -> dict:
    """Unique coverage-selected vs unique raw-selected last-TMA groups."""
    work = occ.dropna(subset=["next_tma_date", "share_never"]).copy()
    recs = []
    for (mod, pres), g in work.groupby(["code_module", "code_presentation"]):
        last = g.next_tma_date.max()
        early, late = g[g.next_tma_date < last], g[g.next_tma_date == last]
        if early.empty or late.empty:
            continue
        feat = early.groupby("id_student", as_index=False).agg(never=("share_never", "mean"))
        y = late.groupby("id_student", as_index=False).agg(score=("score", "mean"), submitted=("submitted", "max"))
        wp = week_panel[(week_panel.code_module == mod) & (week_panel.code_presentation == pres)]
        raw = wp[wp.week_end < last].groupby("id_student", as_index=False).agg(raw_inact=("raw_inact", "mean"))
        m = feat.merge(y, on="id_student").merge(raw, on="id_student", how="left")
        m["adverse"] = ((m.submitted == 0) | (m.score < 40)).astype(int)
        if len(m) < 40:
            continue
        k = max(1, int(round(0.10 * len(m))))
        raw_ids = set(m.sort_values(["raw_inact", "id_student"], ascending=[False, True]).id_student.iloc[:k])
        cov_ids = set(m.sort_values(["never", "id_student"], ascending=[False, True]).id_student.iloc[:k])
        only_c = cov_ids - raw_ids
        only_r = raw_ids - cov_ids
        both = raw_ids & cov_ids
        recs.append(
            {
                "code_module": mod,
                "code_presentation": pres,
                "k": k,
                "n_only_coverage": len(only_c),
                "n_only_raw": len(only_r),
                "n_overlap": len(both),
                "adverse_only_coverage": float(m[m.id_student.isin(only_c)].adverse.mean()) if only_c else np.nan,
                "adverse_only_raw": float(m[m.id_student.isin(only_r)].adverse.mean()) if only_r else np.nan,
                "mean_score_only_coverage": float(m[m.id_student.isin(only_c)].score.mean()) if only_c else np.nan,
                "mean_score_only_raw": float(m[m.id_student.isin(only_r)].score.mean()) if only_r else np.nan,
            }
        )
    if not recs:
        return {"n": 0}
    df = pd.DataFrame(recs)
    return {
        "n_presentations": int(len(df)),
        "mean_overlap_frac": float((df.n_overlap / df.k).mean()),
        "mean_adverse_only_coverage": float(df.adverse_only_coverage.mean()),
        "mean_adverse_only_raw": float(df.adverse_only_raw.mean()),
        "mean_score_only_coverage": float(df.mean_score_only_coverage.mean()),
        "mean_score_only_raw": float(df.mean_score_only_raw.mean()),
        "by_presentation": recs,
    }


def ledger(occ: pd.DataFrame, sites: pd.DataFrame) -> list:
    rows = []
    for (mod, pres), g in occ.groupby(["code_module", "code_presentation"]):
        st = sites[(sites.code_module == mod) & (sites.code_presentation == pres)]
        rows.append(
            {
                "code_module": mod,
                "code_presentation": pres,
                "resource_family": "oucontent",
                "n_eligible_learners": int(g.id_student.nunique()),
                "n_eligible_resources": int(st.id_site.nunique()) if len(st) else 0,
                "n_learner_tma_occasions": int(len(g)),
                "n_scored_occasions": int(g.score.notna().sum()),
                "prior_status": "v4_development",
            }
        )
    return rows


def spec_compare(occ: pd.DataFrame) -> dict:
    work = scored(occ).dropna(subset=["raw_inact", "log_clicks"])
    work["pres"] = work.code_module + "_" + work.code_presentation
    work["coverage"] = 1 - work.share_never
    work["any_never"] = (work.share_never > 0).astype(int)
    work["n_uncovered"] = work.share_never * work.n_opp
    base = "score ~ prior_score_filled + prior_missing + C(pres) + log_clicks + active_rate + n_opp + raw_inact"
    out = {}
    for name, extra in {
        "A_never_share": " + share_never",
        "B_coverage": " + coverage",
        "D_any_never": " + any_never",
        "E_count_uncovered": " + n_uncovered",
    }.items():
        fit = smf.ols(base + extra, data=work).fit(cov_type="cluster", cov_kwds={"groups": work.id_student})
        key = extra.strip(" +")
        out[name] = {
            "r2": float(fit.rsquared),
            "coef": float(fit.params[key]),
            "ci": [float(x) for x in fit.conf_int().loc[key]],
        }
    return out


def main() -> None:
    print("V4 development load...", flush=True)
    vle = load_vle()
    courses = load_courses()
    reg = load_registration()
    sa = load_student_assessment()
    assessments = load_assessments()
    tma_meta = map_prev_tma_date(assessments)
    sv = load_student_vle()
    pres = set(V4_DEVELOPMENT)
    sites0 = documented_sites(vle, pres, FAMILY_OUCONTENT)
    clicks = clicks_by_learner_site_date(sv, sites0.id_site.tolist())
    week_panel = build_week_panel(sv, pres)
    del sv
    panel = registered_learner_sites(reg, sites0)
    panel = map_next_tma(panel, assessments)
    site = annotate_site_states(panel, clicks)
    occ = occasion_from_sites(site)
    occ = attach_tma_outcomes(occ, sa, tma_meta)
    occ = attach_prior_score(occ, sa, assessments)
    occ = attach_raw_controls(occ, week_panel)
    print("occasions", len(occ), "scored", occ.score.notna().sum(), "presentations", occ.groupby(["code_module", "code_presentation"]).ngroups, flush=True)

    dump("presentation_ledger.json", ledger(occ, sites0))
    dump("dose_response.json", dose_bins(occ))
    dump("nonlinearity.json", nonlinearity(occ))
    print("fitting hierarchy...", flush=True)
    hier = fit_hierarchy(occ)
    dump("controlled_association.json", hier)
    dump("spec_compare.json", spec_compare(occ))
    dump("within_learner.json", within_learner(occ))
    dump("same_activity.json", matching(occ))
    print("specificity...", flush=True)
    dump("unplanned_feasibility.json", unplanned_feasibility())
    dump("resource_identity_placebo.json", resource_identity_placebo(site, occ, clicks))
    dump("proximal_sensitivity.json", proximal_sensitivity(site, occ))
    dump("temporal_specificity.json", temporal_specificity(occ))
    dump("residual_risk.json", residual_risk(occ))
    dump("complementary_lists.json", complementary(occ, week_panel))
    dump(
        "module_holdout.json",
        {
            "presentation_loo": hier.get("loo"),
            "n_pres_M3_coef_negative": int(
                sum(r.get("coef", 0) < 0 for r in (hier.get("loo") or {}).get("M3", {}).get("by_presentation", []))
            ),
            "module_loo": module_holdout(occ),
        },
    )
    print("M3", hier["in_sample"]["M3"], "partial", hier["partial_r2"], "loo_d", hier["loo_delta_r2"], flush=True)


if __name__ == "__main__":
    main()
