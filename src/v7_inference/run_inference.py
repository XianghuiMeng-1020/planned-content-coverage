#!/usr/bin/env python3
"""Phase-3 inference. Writes only results/v7_inference/. Uses OPP_END_SAFE from v6."""
from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
import statsmodels.api as sm
from scipy import stats

_SRC = Path(__file__).resolve().parents[1]
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from common.paths import ROOT  # noqa: E402
from v2.load import load_registration  # noqa: E402

OUT = ROOT / "results" / "v7_inference"
FIG = OUT / "figures"
SRC_TAB = ROOT / "results" / "v6_construct_validity" / "occasion_table.parquet"
FORBIDDEN = [ROOT / "results" / "v4", ROOT / "results" / "v5_temporal_safe", ROOT / "results" / "v6_construct_validity"]
SEED = 20260828
B_BOOT = 1000
PLANNED = {
    ("AAA", "2013J"): 3, ("AAA", "2014J"): 3,
    ("BBB", "2013J"): 1, ("BBB", "2014B"): 1, ("BBB", "2014J"): 24,
    ("CCC", "2014B"): 5, ("CCC", "2014J"): 5,
    ("EEE", "2013J"): 26, ("EEE", "2014B"): 26, ("EEE", "2014J"): 27,
    ("FFF", "2013J"): 88, ("FFF", "2014B"): 82, ("FFF", "2014J"): 82,
}
LOG: list[str] = []


def log(msg: str) -> None:
    LOG.append(f"{datetime.now(timezone.utc).isoformat()} {msg}")
    print(msg, flush=True)


def _jd(x):
    if isinstance(x, (np.floating,)):
        return float(x)
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, (np.bool_,)):
        return bool(x)
    raise TypeError(type(x))


def dump(name: str, obj) -> Path:
    OUT.mkdir(parents=True, exist_ok=True)
    p = (OUT / name).resolve()
    assert str(p).startswith(str(OUT.resolve()))
    for bad in FORBIDDEN:
        assert not str(p).startswith(str(bad.resolve()))
    p.write_text(json.dumps(obj, indent=2, default=_jd))
    log(f"wrote {p.relative_to(ROOT)}")
    return p


def prep(df: pd.DataFrame) -> pd.DataFrame:
    w = df[df.n_opp_end >= 1].copy()
    w["share_never"] = w["never_end"]
    w["n_opp"] = w["n_opp_end"]
    w["pres"] = w.code_module + "_" + w.code_presentation
    w["lid"] = w.code_module + "|" + w.code_presentation + "|" + w.id_student.astype(str)
    w["log1p_nopp"] = np.log1p(w.n_opp)
    w["early"] = (w.date_submitted < w.next_tma_date).astype(int)
    return w


LEGACY = "score ~ prior_score_filled + prior_missing + C(pres) + log_clicks + active_rate + n_opp + raw_inact + share_never"
NONRED = "score ~ prior_score_filled + prior_missing + C(pres) + log_clicks + active_rate + n_opp + share_never"
BASE_NR = "score ~ prior_score_filled + prior_missing + C(pres) + log_clicks + active_rate + n_opp"


def pack_fit(fit, feat="share_never") -> dict:
    out = {"n": int(fit.nobs), "r2": float(fit.rsquared), "rank": int(getattr(fit.model, "rank", np.nan))}
    if feat in fit.params:
        out["coef"] = float(fit.params[feat])
        out["se"] = float(fit.bse[feat]) if feat in fit.bse else None
        out["p"] = float(fit.pvalues[feat]) if feat in fit.pvalues else None
        if feat in fit.conf_int().index:
            out["ci"] = [float(x) for x in fit.conf_int().loc[feat]]
    return out


def fit_cluster(df, fml, groups, feat="share_never"):
    fit = smf.ols(fml, data=df).fit(cov_type="cluster", cov_kwds={"groups": groups})
    rec = pack_fit(fit, feat)
    rec["n_learners"] = int(df.id_student.nunique())
    return rec, fit


def two_way_ci(df, fml, feat="share_never"):
    """CGM two-way: learner + assessment. V = V1+V2-V12."""
    fit1 = smf.ols(fml, data=df).fit(cov_type="cluster", cov_kwds={"groups": df.id_student})
    fit2 = smf.ols(fml, data=df).fit(cov_type="cluster", cov_kwds={"groups": df.next_id_assessment})
    sa = df.id_student.astype(str) + "|" + df.next_id_assessment.astype(str)
    fit12 = smf.ols(fml, data=df).fit(cov_type="cluster", cov_kwds={"groups": sa})
    if feat not in fit1.params:
        return {"used": False, "reason": "feat missing"}
    i = list(fit1.params.index).index(feat)
    v = fit1.cov_params().to_numpy() + fit2.cov_params().to_numpy() - fit12.cov_params().to_numpy()
    se = float(np.sqrt(max(v[i, i], 0.0)))
    b = float(fit1.params[feat])
    z = stats.norm.ppf(0.975)
    return {
        "used": True,
        "n_assessments": int(df.next_id_assessment.nunique()),
        "coef": b,
        "se": se,
        "ci": [b - z * se, b + z * se],
        "note": "CGM V1+V2-V12; learner x assessment",
    }


def learner_bootstrap(df, fml, feat="share_never", B=B_BOOT, seed=SEED):
    """Cluster-resample learners; OLS via lstsq on a frozen design matrix."""
    from patsy import dmatrices

    y, X = dmatrices(fml, df, return_type="dataframe")
    yv = y.to_numpy().ravel()
    Xv = X.to_numpy()
    feat_i = list(X.columns).index(feat)
    lids = df.loc[X.index, "id_student"].to_numpy()
    uniq, inv = np.unique(lids, return_inverse=True)
    groups = [np.flatnonzero(inv == i) for i in range(len(uniq))]
    rng = np.random.default_rng(seed)
    coefs = np.empty(B, dtype=float)
    n_ok = 0
    n_u = len(uniq)
    for b in range(B):
        draw = rng.integers(0, n_u, size=n_u)
        idx = np.concatenate([groups[i] for i in draw])
        beta, *_ = np.linalg.lstsq(Xv[idx], yv[idx], rcond=None)
        coefs[n_ok] = float(beta[feat_i])
        n_ok += 1
        if (b + 1) % 250 == 0:
            log(f"  bootstrap {b+1}/{B}")
    arr = coefs[:n_ok]
    return {
        "B": int(len(arr)),
        "mean": float(arr.mean()) if len(arr) else None,
        "ci_percentile": [float(np.quantile(arr, 0.025)), float(np.quantile(arr, 0.975))] if len(arr) else None,
        "seed": seed,
        "method": "learner-cluster percentile; numpy lstsq on frozen nonredundant design",
    }


def vif_continuous(df, cols):
    X = df[cols].dropna()
    out = {}
    for c in cols:
        y = X[c].to_numpy()
        xx = sm.add_constant(X.drop(columns=[c]), has_constant="add")
        r = sm.OLS(y, xx).fit()
        out[c] = float(1 / (1 - r.rsquared)) if r.rsquared < 1 else float("inf")
    return out


def meta_iv(betas, ses):
    b = np.asarray(betas, float)
    se = np.asarray(ses, float)
    w = 1.0 / se**2
    fe = float(np.sum(w * b) / np.sum(w))
    se_fe = float(np.sqrt(1 / np.sum(w)))
    q = float(np.sum(w * (b - fe) ** 2))
    k = len(b)
    df = k - 1
    c = float(np.sum(w) - np.sum(w**2) / np.sum(w))
    tau2 = max(0.0, (q - df) / c) if c > 0 else 0.0
    wr = 1.0 / (se**2 + tau2)
    re = float(np.sum(wr * b) / np.sum(wr))
    se_re = float(np.sqrt(1 / np.sum(wr)))
    i2 = max(0.0, (q - df) / q) if q > 0 else 0.0
    z = 1.96
    pi = [re - z * np.sqrt(tau2 + se_re**2), re + z * np.sqrt(tau2 + se_re**2)]
    return {
        "k": k,
        "fixed_mean": fe,
        "fixed_ci": [fe - z * se_fe, fe + z * se_fe],
        "random_mean": re,
        "random_ci": [re - z * se_re, re + z * se_re],
        "tau2": tau2,
        "I2": i2,
        "Q": q,
        "prediction_interval_95": [float(pi[0]), float(pi[1])],
    }


def context_fit(df, fml_base, extra_fe=""):
    """Nonredundant within a single context; cluster learner."""
    work = df.dropna(subset=["score", "share_never", "log_clicks", "active_rate", "n_opp", "prior_score_filled"])
    if len(work) < 40:
        return {"insufficient": True, "n": int(len(work))}
    fml = fml_base + extra_fe
    fit = smf.ols(fml, data=work).fit(cov_type="cluster", cov_kwds={"groups": work.id_student})
    base = smf.ols(fml.replace(" + share_never", ""), data=work).fit()
    rec = pack_fit(fit)
    rec["n"] = int(len(work))
    rec["n_learners"] = int(work.id_student.nunique())
    rec["partial_r2"] = float((fit.rsquared - base.rsquared) / (1 - base.rsquared)) if base.rsquared < 1 else None
    rec["mean_n_opp"] = float(work.n_opp.mean())
    rec["sd_score"] = float(work.score.std())
    rec["sd_never"] = float(work.share_never.std())
    rec["sign"] = int(np.sign(rec.get("coef", 0)))
    return rec


def fe_a(work):
    keep = set(work.groupby("lid").size().pipe(lambda s: s[s >= 2].index))
    d = work[work.lid.isin(keep)].copy()
    if len(d) < 50:
        return {"insufficient": True, "n_occasions": int(len(d))}
    wsd = d.groupby("lid").share_never.std()
    for c in ["score", "share_never", "log_clicks", "raw_inact", "n_opp"]:
        d[c + "_w"] = d[c] - d.groupby("lid")[c].transform("mean")
    d["tma_c"] = d.groupby("lid").next_tma_date.transform(lambda s: (s - s.mean()) / (s.std() + 1e-6))
    fit = smf.ols(
        "score_w ~ share_never_w + log_clicks_w + raw_inact_w + n_opp_w + tma_c",
        data=d,
    ).fit(cov_type="cluster", cov_kwds={"groups": d.lid})
    return {
        "n_occasions": int(len(d)),
        "n_learners": int(d.lid.nunique()),
        "within_sd_mean": float(wsd.mean()),
        "prop_learners_with_variation": float((wsd > 1e-12).mean()),
        "coef": float(fit.params["share_never_w"]),
        "ci": [float(x) for x in fit.conf_int().loc["share_never_w"]],
        "p": float(fit.pvalues["share_never_w"]),
    }


def fe_b(work):
    d = work[work.prior_missing == 0].copy()
    keep = set(d.groupby("lid").size().pipe(lambda s: s[s >= 2].index))
    d = d[d.lid.isin(keep)].copy()
    if len(d) < 5000:
        return {"insufficient": True, "n_occasions": int(len(d)), "reason": "N<5000 after prior complete-case + repeater"}
    for c in ["score", "share_never", "log_clicks", "raw_inact", "n_opp", "prior_score_filled"]:
        d[c + "_w"] = d[c] - d.groupby("lid")[c].transform("mean")
    d["tma_c"] = d.groupby("lid").next_tma_date.transform(lambda s: (s - s.mean()) / (s.std() + 1e-6))
    fit = smf.ols(
        "score_w ~ share_never_w + log_clicks_w + raw_inact_w + n_opp_w + prior_score_filled_w + tma_c",
        data=d,
    ).fit(cov_type="cluster", cov_kwds={"groups": d.lid})
    return {
        "n_occasions": int(len(d)),
        "n_learners": int(d.lid.nunique()),
        "coef": float(fit.params["share_never_w"]),
        "ci": [float(x) for x in fit.conf_int().loc["share_never_w"]],
        "p": float(fit.pvalues["share_never_w"]),
    }


def mundlak(work):
    keep = set(work.groupby("lid").size().pipe(lambda s: s[s >= 2].index))
    d = work[work.lid.isin(keep)].copy()
    d["never_bar"] = d.groupby("lid").share_never.transform("mean")
    fit = smf.ols(
        "score ~ share_never + never_bar + log_clicks + active_rate + n_opp + C(code_module)",
        data=d,
    ).fit(cov_type="cluster", cov_kwds={"groups": d.lid})
    return {
        "n_occasions": int(len(d)),
        "n_learners": int(d.lid.nunique()),
        "within": float(fit.params["share_never"]),
        "within_ci": [float(x) for x in fit.conf_int().loc["share_never"]],
        "between": float(fit.params["never_bar"]),
        "between_ci": [float(x) for x in fit.conf_int().loc["never_bar"]],
    }


def metrics(y, p):
    y, p = np.asarray(y, float), np.asarray(p, float)
    sst = np.sum((y - y.mean()) ** 2)
    r2 = float(1 - np.sum((y - p) ** 2) / sst) if sst > 0 else np.nan
    return {"r2": r2, "rmse": float(np.sqrt(np.mean((y - p) ** 2))), "mae": float(np.mean(np.abs(y - p)))}


def smd(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    a, b = a[np.isfinite(a)], b[np.isfinite(b)]
    sd = np.sqrt(((len(a) - 1) * a.var() + (len(b) - 1) * b.var()) / max(len(a) + len(b) - 2, 1))
    return float((a.mean() - b.mean()) / sd) if sd > 0 else 0.0


def main() -> None:
    log("PHASE3 inference start")
    raw = pd.read_parquet(SRC_TAB)
    df = prep(raw)
    log(f"primary n={len(df)} learners={df.id_student.nunique()}")
    assert (ROOT / "results" / "v4" / "controlled_association.json").exists()
    assert (ROOT / "results" / "v6_construct_validity" / "controlled_association.json").exists() or (ROOT / "results" / "v6_construct_validity" / "construct_comparison.json").exists()

    # --- primary models ---
    log("legacy + nonredundant...")
    rec_l, fit_l = fit_cluster(df, LEGACY, df.id_student)
    rec_n, fit_n = fit_cluster(df, NONRED, df.id_student)
    fit_base = smf.ols(BASE_NR, data=df).fit(cov_type="cluster", cov_kwds={"groups": df.id_student})
    rec_n["partial_r2"] = float((fit_n.rsquared - fit_base.rsquared) / (1 - fit_base.rsquared))
    rec_l["partial_r2"] = float((fit_l.rsquared - smf.ols(BASE_NR + " + raw_inact", data=df).fit().rsquared) / max(1e-12, 1 - smf.ols(BASE_NR + " + raw_inact", data=df).fit().rsquared))
    tw = two_way_ci(df, NONRED) if df.next_id_assessment.nunique() >= 20 else {"used": False, "reason": "n_assessments<20"}
    log("learner bootstrap B=1000...")
    boot = learner_bootstrap(df, NONRED)
    vif = vif_continuous(df, ["prior_score_filled", "log_clicks", "active_rate", "n_opp", "share_never", "raw_inact"])
    dump(
        "primary_inference.json",
        {
            "legacy": rec_l,
            "nonredundant": rec_n,
            "two_way": tw,
            "bootstrap": boot,
            "vif": vif,
            "delta_legacy_minus_nonred": rec_l.get("coef", 0) - rec_n.get("coef", 0),
            "formula_legacy": LEGACY,
            "formula_nonredundant": NONRED,
        },
    )

    # --- FE ---
    log("FE hierarchy...")
    fa = fe_a(df)
    fb = fe_b(df)
    fc = mundlak(df)
    dump("within_learner.json", {"FE_A": fa, "FE_B": fb, "FE_C": fc})

    # --- presentation effects ---
    log("presentation-specific...")
    pres_rows = []
    fml_p = "score ~ prior_score_filled + prior_missing + log_clicks + active_rate + n_opp + share_never"
    for (mod, pres), g in df.groupby(["code_module", "code_presentation"]):
        rec = context_fit(g, fml_p)
        rec.update({"code_module": mod, "code_presentation": pres, "pres": f"{mod}_{pres}"})
        pres_rows.append(rec)
    pres_df = pd.DataFrame(pres_rows)
    pres_path = OUT / "presentation_effects.csv"
    OUT.mkdir(parents=True, exist_ok=True)
    pres_df.to_csv(pres_path, index=False)
    okp = pres_df.dropna(subset=["coef", "se"])
    meta_p = meta_iv(okp.coef, okp.se)
    dump(
        "presentation_heterogeneity.json",
        {
            "n_negative": int((okp.coef < 0).sum()),
            "n_positive": int((okp.coef > 0).sum()),
            "n_ci_excl0": int(
                sum(1 for _, r in okp.iterrows() if isinstance(r.get("ci"), list) and r.coef < 0 and r.ci[1] < 0)
            ),
            "median": float(okp.coef.median()),
            "min": float(okp.coef.min()),
            "max": float(okp.coef.max()),
            "iqr": float(okp.coef.quantile(0.75) - okp.coef.quantile(0.25)),
            "meta": meta_p,
            "rows": pres_rows,
        },
    )

    # --- module ---
    log("module-specific...")
    mod_rows = []
    for mod, g in df.groupby("code_module"):
        extra = " + C(code_presentation)" if g.code_presentation.nunique() > 1 else ""
        rec = context_fit(g, fml_p, extra)
        rec["code_module"] = mod
        rec["score_mean"] = float(g.score.mean())
        rec["n_opp_median"] = float(g.n_opp.median())
        mod_rows.append(rec)
    mod_df = pd.DataFrame(mod_rows)
    meta_m = meta_iv(mod_df.coef.dropna(), mod_df.se.dropna()) if mod_df.coef.notna().all() else {}
    dump("module_heterogeneity.json", {"rows": mod_rows, "meta": meta_m})

    # --- moderators exploratory ---
    log("moderators...")
    ctx = []
    for (mod, pres), g in df.groupby(["code_module", "code_presentation"]):
        row = okp[(okp.code_module == mod) & (okp.code_presentation == pres)]
        if row.empty:
            continue
        tma_ord = g.next_tma_date.rank(method="dense")
        ctx.append(
            {
                "pres": f"{mod}_{pres}",
                "beta": float(row.coef.iloc[0]),
                "median_n_opp": float(g.n_opp.median()),
                "mean_n_opp": float(g.n_opp.mean()),
                "mean_score": float(g.score.mean()),
                "sd_score": float(g.score.std()),
                "mean_log_clicks": float(g.log_clicks.mean()),
                "prop_early": float(g.early.mean()),
                "n_planned_oucontent": PLANNED.get((mod, pres), np.nan),
                "mean_tma_order": float(tma_ord.mean()),
            }
        )
    ctx_df = pd.DataFrame(ctx)
    mods = [c for c in ctx_df.columns if c not in ("pres", "beta")]
    corrs = []
    for m in mods:
        r, p = stats.spearmanr(ctx_df.beta, ctx_df[m])
        corrs.append({"moderator": m, "spearman": float(r), "p": float(p)})
    pvals = np.array([c["p"] for c in corrs])
    order = np.argsort(pvals)
    adj = np.empty_like(pvals)
    n = len(pvals)
    prev = 1.0
    for rank, i in enumerate(order[::-1], start=0):
        adj_i = min(prev, pvals[i] * n / (n - rank))
        adj[i] = adj_i
        prev = adj_i
    # BH
    from statsmodels.stats.multitest import multipletests

    _, fdr, _, _ = multipletests(pvals, method="fdr_bh")
    for c, q in zip(corrs, fdr):
        c["fdr_bh"] = float(q)
        c["label"] = "exploratory"
    dump("moderators_exploratory.json", {"n_contexts": len(ctx_df), "correlations": corrs})

    # --- interaction ---
    log("denominator interaction...")
    inter_p, _ = fit_cluster(df, NONRED + " + share_never:log1p_nopp", df.id_student)
    inter_fit = smf.ols(NONRED + " + share_never:log1p_nopp", data=df).fit(cov_type="cluster", cov_kwds={"groups": df.id_student})
    ik = [k for k in inter_fit.params.index if "share_never" in k and "log1p" in k]
    inter_rec = pack_fit(inter_fit)
    if ik:
        inter_rec["interaction_coef"] = float(inter_fit.params[ik[0]])
        inter_rec["interaction_ci"] = [float(x) for x in inter_fit.conf_int().loc[ik[0]]]
    # FE interaction
    keep = set(df.groupby("lid").size().pipe(lambda s: s[s >= 2].index))
    dfe = df[df.lid.isin(keep)].copy()
    dfe["sxn"] = dfe.share_never * dfe.log1p_nopp
    for c in ["score", "share_never", "log_clicks", "raw_inact", "n_opp", "sxn"]:
        dfe[c + "_w"] = dfe[c] - dfe.groupby("lid")[c].transform("mean")
    dfe["tma_c"] = dfe.groupby("lid").next_tma_date.transform(lambda s: (s - s.mean()) / (s.std() + 1e-6))
    fei = smf.ols(
        "score_w ~ share_never_w + sxn_w + log_clicks_w + raw_inact_w + n_opp_w + tma_c",
        data=dfe,
    ).fit(cov_type="cluster", cov_kwds={"groups": dfe.lid})
    strat = []
    for thr in [1, 3, 5, 10]:
        sub = df[df.n_opp >= thr]
        r, _ = fit_cluster(sub, NONRED, sub.id_student)
        r["threshold"] = thr
        strat.append(r)
    dump(
        "denominator_interaction.json",
        {
            "pooled": inter_rec,
            "fe_interaction": {
                "coef": float(fei.params["sxn_w"]),
                "ci": [float(x) for x in fei.conf_int().loc["sxn_w"]],
                "n": int(len(dfe)),
            },
            "stratified": strat,
        },
    )

    # --- LOPO ---
    log("LOPO transport...")
    lopo_fml_b = "score ~ prior_score_filled + prior_missing + C(code_module) + log_clicks + active_rate + n_opp"
    lopo_fml_c = lopo_fml_b + " + share_never"
    lopo = []
    for pres, te in df.groupby("pres"):
        tr = df[df.pres != pres]
        fit_b = smf.ols(lopo_fml_b, data=tr).fit()
        fit_c = smf.ols(lopo_fml_c, data=tr).fit()
        mb = metrics(te.score, fit_b.predict(te))
        mc = metrics(te.score, fit_c.predict(te))
        # test direction: residualize with train control model
        te_e_y = te.score - fit_b.predict(te)
        # residualize never: regress never on controls in train, apply
        fn = smf.ols("share_never ~ prior_score_filled + prior_missing + C(code_module) + log_clicks + active_rate + n_opp", data=tr).fit()
        te_e_x = te.share_never - fn.predict(te)
        rho = float(np.corrcoef(te_e_x, te_e_y)[0, 1]) if te_e_x.std() and te_e_y.std() else np.nan
        lopo.append(
            {
                "pres": pres,
                "module": pres.split("_")[0],
                "n": int(len(te)),
                "r2_base": mb["r2"],
                "r2_cov": mc["r2"],
                "delta_r2": mc["r2"] - mb["r2"],
                "rmse_base": mb["rmse"],
                "rmse_cov": mc["rmse"],
                "delta_rmse": mc["rmse"] - mb["rmse"],
                "mae_base": mb["mae"],
                "mae_cov": mc["mae"],
                "delta_mae": mc["mae"] - mb["mae"],
                "train_coef": float(fit_c.params["share_never"]),
                "test_partial_sign": int(np.sign(rho)) if np.isfinite(rho) else None,
            }
        )
    lopo_df = pd.DataFrame(lopo)
    rng = np.random.default_rng(SEED)
    dlt = lopo_df.delta_r2.to_numpy()
    fold_ci = [float(np.quantile(rng.choice(dlt, size=len(dlt), replace=True), q)) for q in (0.025, 0.975)]
    dump(
        "lopo.json",
        {
            "folds": lopo,
            "mean_delta_r2": float(lopo_df.delta_r2.mean()),
            "median_delta_r2": float(lopo_df.delta_r2.median()),
            "weighted_mean_delta_r2": float(np.average(lopo_df.delta_r2, weights=lopo_df.n)),
            "n_pos": int((lopo_df.delta_r2 > 0).sum()),
            "n_neg": int((lopo_df.delta_r2 < 0).sum()),
            "median_delta_rmse": float(lopo_df.delta_rmse.median()),
            "median_delta_mae": float(lopo_df.delta_mae.median()),
            "fold_bootstrap_ci_mean_note": "k=13 fold resample only; severe small-N",
            "fold_bootstrap_ci_on_mean_proxy": fold_ci,
        },
    )

    # --- LOMO ---
    log("LOMO transport...")
    lomo_b = "score ~ prior_score_filled + prior_missing + log_clicks + active_rate + n_opp"
    lomo_c = lomo_b + " + share_never"
    lomo = []
    for mod, te in df.groupby("code_module"):
        tr = df[df.code_module != mod]
        fit_b = smf.ols(lomo_b, data=tr).fit()
        fit_c = smf.ols(lomo_c, data=tr).fit()
        mb, mc = metrics(te.score, fit_b.predict(te)), metrics(te.score, fit_c.predict(te))
        in_mod = next(r for r in mod_rows if r["code_module"] == mod)
        cls = "NO_GAIN"
        if mb["r2"] < 0 and mc["r2"] < 0:
            cls = "PREDICTIVE_FAILURE_BOTH_MODELS"
        elif (mc["r2"] - mb["r2"]) < 0:
            cls = "COVERAGE_WORSENS"
        elif (mc["r2"] - mb["r2"]) > 0 and (mc["rmse"] - mb["rmse"]) < 0:
            cls = "PREDICTIVE_GAIN"
        lomo.append(
            {
                "code_module": mod,
                "n": int(len(te)),
                "r2_base": mb["r2"],
                "r2_cov": mc["r2"],
                "delta_r2": mc["r2"] - mb["r2"],
                "rmse_base": mb["rmse"],
                "rmse_cov": mc["rmse"],
                "delta_rmse": mc["rmse"] - mb["rmse"],
                "mae_base": mb["mae"],
                "mae_cov": mc["mae"],
                "delta_mae": mc["mae"] - mb["mae"],
                "train_coef": float(fit_c.params["share_never"]),
                "heldout_assoc_sign": in_mod.get("sign"),
                "class": cls,
            }
        )
    dump("lomo.json", {"modules": lomo})

    # --- AAA/BBB shift ---
    log("shift diagnostics...")
    shifts = []
    cols = ["score", "prior_score_filled", "log_clicks", "active_rate", "share_never", "n_opp"]
    for label, mask_te in [("AAA", df.code_module == "AAA"), ("BBB", df.code_module == "BBB")]:
        te, tr = df[mask_te], df[~mask_te]
        rec = {"context": label, "n_test": int(len(te)), "n_train": int(len(tr))}
        for c in cols:
            rec[f"smd_{c}"] = smd(te[c], tr[c])
            rec[f"mean_te_{c}"] = float(te[c].mean())
            rec[f"mean_tr_{c}"] = float(tr[c].mean())
            rec[f"sd_te_{c}"] = float(te[c].std())
            rec[f"sd_tr_{c}"] = float(tr[c].std())
            rec[f"ks_{c}"] = float(stats.ks_2samp(te[c].dropna(), tr[c].dropna()).statistic)
        fit_b = smf.ols(lomo_b, data=tr).fit()
        rec["mean_test_residual_baseline"] = float((te.score - fit_b.predict(te)).mean())
        rec["baseline_r2"] = metrics(te.score, fit_b.predict(te))["r2"]
        shifts.append(rec)
    dump("shift_aaa_bbb.json", shifts)

    # --- robustness ---
    log("robustness family...")
    rob = {}
    if ((df.score < 0) | (df.score > 100)).any():
        dw = df.copy()
        dw["score"] = dw.score.clip(0, 100)
        rob["winsor"], _ = fit_cluster(dw, NONRED, dw.id_student)
    else:
        rob["winsor"] = {"skipped": True, "reason": "all scores in [0,100]"}
    try:
        rlm = sm.RLM.from_formula(NONRED, data=df, M=sm.robust.norms.HuberT()).fit()
        rob["huber"] = {"coef": float(rlm.params["share_never"]), "se": float(rlm.bse["share_never"])}
        z = 1.96
        rob["huber"]["ci"] = [rob["huber"]["coef"] - z * rob["huber"]["se"], rob["huber"]["coef"] + z * rob["huber"]["se"]]
    except Exception as e:
        rob["huber"] = {"failed": str(e)}
    # presentation-weighted: weight = 1/n_pres
    cnt = df.groupby("pres").size()
    dw = df.copy()
    dw["w"] = 1.0 / dw.pres.map(cnt)
    wls = smf.wls(NONRED, data=dw, weights=dw.w).fit(cov_type="cluster", cov_kwds={"groups": dw.id_student})
    rob["presentation_weighted"] = pack_fit(wls)
    rob["equal_presentation_mean"] = float(okp.coef.mean())
    # z-score
    dz = df.copy()
    dz["score"] = dz.groupby(["code_module", "code_presentation", "next_id_assessment"]).score.transform(lambda s: (s - s.mean()) / (s.std() + 1e-9))
    fmlz = NONRED.replace("score ~", "score ~")
    rob["z_within_assessment"], _ = fit_cluster(dz, fmlz, dz.id_student)
    dump("robustness.json", rob)

    # --- functional form ---
    log("functional form...")
    requested = np.quantile(df.share_never, [0.05, 0.35, 0.65, 0.95])
    knots = np.unique(np.round(requested, 8))
    knot_note = "empirical 5/35/65/95 of never_share"
    if len(knots) < 3:
        knots = np.array([0.25, 0.50, 0.75])
        knot_note += "; <3 distinct requested knots so frozen fallback 0.25/0.50/0.75"
    dump("rcs_knots_pre_outcome.json", {"requested": [float(x) for x in requested], "knots_used": [float(x) for x in knots], "note": knot_note})
    q = pd.qcut(df.share_never, 4, duplicates="drop")
    dq = df.copy()
    dq["q"] = q.astype(str)
    fq = smf.ols(BASE_NR + " + C(q)", data=dq).fit(cov_type="cluster", cov_kwds={"groups": dq.id_student})

    def rcs_basis(x, kn):
        x = np.asarray(x, float)
        kn = np.sort(np.asarray(kn, float))
        t1, tk, tkm = kn[0], kn[-1], kn[-2]
        cols = [x]
        for tj in kn[:-2]:
            def p(z, t):
                return np.maximum(z - t, 0.0) ** 3
            cols.append(p(x, tj) - p(x, tkm) * (tk - tj) / (tk - tkm) + p(x, tk) * (tkm - tj) / (tk - tkm))
        return np.column_stack(cols)

    ds = df.copy()
    basis = rcs_basis(ds.share_never.to_numpy(), knots)
    for j in range(basis.shape[1]):
        ds[f"sp{j}"] = basis[:, j]
    sp_terms = " + ".join(f"sp{j}" for j in range(basis.shape[1]))
    fs = smf.ols(BASE_NR + " + " + sp_terms, data=ds).fit(cov_type="cluster", cov_kwds={"groups": ds.id_student})
    extra = [f"sp{j}" for j in range(1, basis.shape[1])]
    wald = None
    if extra:
        hyp = " + ".join(f"{e}=0" for e in extra)
        try:
            w = fs.wald_test(hyp, scalar=True)
            wald = {"stat": float(np.asarray(w.statistic).ravel()[0]), "p": float(np.asarray(w.pvalue).ravel()[0])}
        except Exception as e:
            wald = {"failed": str(e)}
    modal = df.pres.mode().iloc[0]
    grid_x = [0.0, 0.25, 0.50, 0.75, 1.00]
    pred_lin, pred_sp = [], []
    means = {c: float(df[c].mean()) for c in ["prior_score_filled", "log_clicks", "active_rate", "n_opp"]}
    base_row = {
        "prior_score_filled": means["prior_score_filled"],
        "prior_missing": 0,
        "pres": modal,
        "log_clicks": means["log_clicks"],
        "active_rate": means["active_rate"],
        "n_opp": means["n_opp"],
        "raw_inact": 1 - means["active_rate"],
    }
    for x in grid_x:
        row = pd.DataFrame([{**base_row, "share_never": x}])
        pred_lin.append(float(fit_n.predict(row).iloc[0]))
        b = rcs_basis([x], knots)[0]
        rs = {**base_row, "share_never": x}
        for j, val in enumerate(b):
            rs[f"sp{j}"] = float(val)
        pred_sp.append(float(fs.predict(pd.DataFrame([rs])).iloc[0]))
    mono = all(pred_sp[i] >= pred_sp[i + 1] - 1e-6 for i in range(len(pred_sp) - 1))
    q_means = dq.groupby("q", observed=False).score.mean().to_dict()
    dump(
        "functional_form.json",
        {
            "requested_knots": [float(x) for x in requested],
            "knots_used": [float(x) for x in knots],
            "knot_note": knot_note,
            "linear": rec_n,
            "quartile_params": {k: float(v) for k, v in fq.params.items() if k.startswith("C(q)")},
            "quartile_mean_score": {str(k): float(v) for k, v in q_means.items()},
            "spline_r2": float(fs.rsquared),
            "linear_r2": float(fit_n.rsquared),
            "wald_nonlinear": wald,
            "pred_linear": [[float(a), float(b)] for a, b in zip(grid_x, pred_lin)],
            "pred_spline": [[float(a), float(b)] for a, b in zip(grid_x, pred_sp)],
            "spline_monotonic_decreasing": bool(mono),
        },
    )

    # --- missing / withdrawal ---
    log("sample sensitivities...")
    miss_b = df[df.prior_missing == 0]
    rec_mb, _ = fit_cluster(miss_b, NONRED, miss_b.id_student) if len(miss_b) > 50 else ({"insufficient": True}, None)
    dfc = df.copy()
    dfc["first_tma"] = dfc.prior_missing
    rec_mc, fit_mc = fit_cluster(dfc, NONRED + " + share_never:first_tma", dfc.id_student)
    ik2 = [k for k in fit_mc.params.index if "first_tma" in k and "share_never" in k]
    miss_c = pack_fit(fit_mc)
    if ik2:
        miss_c["interaction"] = float(fit_mc.params[ik2[0]])
        miss_c["interaction_ci"] = [float(x) for x in fit_mc.conf_int().loc[ik2[0]]]
    # withdrawal
    reg = load_registration()
    mrg = df.merge(
        reg[["code_module", "code_presentation", "id_student", "date_unregistration"]],
        on=["code_module", "code_presentation", "id_student"],
        how="left",
    )
    wb = mrg[mrg.date_unregistration.isna() | (mrg.date_unregistration > mrg.date_submitted)]
    wc = mrg[mrg.date_unregistration.isna()]
    rec_wb, _ = fit_cluster(wb, NONRED, wb.id_student)
    rec_wc, _ = fit_cluster(wc, NONRED, wc.id_student)
    dump(
        "sample_sensitivity.json",
        {
            "missing_A_current": rec_n,
            "missing_B_complete_prior": rec_mb,
            "missing_C_interact_first_tma": miss_c,
            "W_A_current_n": int(len(df)),
            "W_B_registered_at_submission": rec_wb,
            "W_C_never_withdrawn": rec_wc,
            "n_WB": int(len(wb)),
            "n_WC": int(len(wc)),
        },
    )

    # --- figures ---
    log("figure data...")
    FIG.mkdir(parents=True, exist_ok=True)
    import matplotlib.pyplot as plt

    # A forest
    pplot = okp.sort_values("coef")
    fig, ax = plt.subplots(figsize=(7, 5))
    y = np.arange(len(pplot))
    lo = [c[0] for c in pplot.ci]
    hi = [c[1] for c in pplot.ci]
    ax.hlines(y, lo, hi, color="#333", lw=1.4)
    ax.plot(pplot.coef, y, "o", color="#111", ms=5)
    ax.axvline(0, color="#888", lw=1)
    ax.set_yticks(y)
    ax.set_yticklabels(pplot.pres)
    ax.set_xlabel("Adjusted β (never_share)")
    ax.set_title("Presentation-specific controlled associations")
    fig.tight_layout()
    fig.savefig(FIG / "figA_presentation_forest.png", dpi=150)
    plt.close()
    # B module assoc vs lomo dR2
    fig, ax = plt.subplots(figsize=(6, 4))
    lomo_df = pd.DataFrame(lomo)
    mm = mod_df.merge(lomo_df, on="code_module")
    ax.axhline(0, color="#888", lw=1)
    ax.axvline(0, color="#888", lw=1)
    ax.scatter(mm.coef, mm.delta_r2, c="#111")
    for _, r in mm.iterrows():
        ax.text(r.coef, r.delta_r2, r.code_module, fontsize=8, ha="left")
    ax.set_xlabel("Module-specific β")
    ax.set_ylabel("LOMO ΔR²")
    ax.set_title("Associational vs predictive transport")
    fig.tight_layout()
    fig.savefig(FIG / "figB_module_transport.png", dpi=150)
    plt.close()
    # C functional form
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(grid_x, pred_lin, "o-", color="#111", label="linear")
    ax.plot(grid_x, pred_sp, "s--", color="#555", label="RCS (4 knots)")
    ax.set_xlabel("never_share")
    ax.set_ylabel("Predicted score (controls at means)")
    ax.legend(frameon=False)
    ax.set_title("Functional form of coverage")
    fig.tight_layout()
    fig.savefig(FIG / "figC_functional_form.png", dpi=150)
    plt.close()

    # --- gates (computed after all) ---
    def neg_ok(rec):
        return rec.get("coef", 1) < 0 and rec.get("ci") and rec["ci"][1] < 0

    i1 = "FAIL"
    if neg_ok(rec_n) and neg_ok(rec_l) and boot.get("ci_percentile") and boot["ci_percentile"][1] < 0:
        mag = abs(rec_n["coef"] - rec_l["coef"]) / max(abs(rec_l["coef"]), 1e-9)
        i1 = "STRONG PASS" if mag <= 0.25 else "PASS"
    elif neg_ok(rec_n):
        i1 = "PASS"
    elif rec_n.get("coef", 1) < 0:
        i1 = "MIXED"

    i2 = "FAIL"
    fe_c_ok = fc.get("within", 1) < 0 and isinstance(fc.get("within_ci"), list) and fc["within_ci"][1] < 0
    fe_b_ok = (not fb.get("insufficient")) and neg_ok(fb)
    if neg_ok(fa) and (fe_b_ok or fe_c_ok):
        i2 = "STRONG PASS"
    elif neg_ok(fa):
        i2 = "PASS"

    pi = meta_p["prediction_interval_95"]
    n_cross = int(((okp.coef < 0) & okp.ci.map(lambda x: isinstance(x, list) and x[1] >= 0)).sum()) if len(okp) else 0
    n_pos_ci = int((okp.ci.map(lambda x: isinstance(x, list) and x[0] > 0)).sum()) if len(okp) else 0
    if n_pos_ci >= 2:
        i3 = "CONTRADICTORY"
    elif pi[1] < 0:
        i3 = "LOW"
    elif (okp.coef < 0).mean() >= 0.7 and pi[1] >= 0:
        i3 = "MODERATE"
    else:
        i3 = "HIGH"

    i4 = "FAIL"
    if (okp.coef < 0).all() and (mod_df.coef < 0).all() and meta_p["random_ci"][1] < 0:
        i4 = "STRONG PASS"
    elif (okp.coef < 0).mean() >= 0.8:
        i4 = "PASS"
    elif (okp.coef < 0).mean() >= 0.5:
        i4 = "MIXED"

    pos_f = int((lopo_df.delta_r2 > 0).sum())
    if pos_f >= 11:
        i5 = "CONSISTENT"
    elif pos_f <= 3:
        i5 = "POOR"
    else:
        i5 = "HETEROGENEOUS"

    i6 = "PASS" if mono and (pred_lin[0] > pred_lin[-1]) else ("MIXED" if pred_lin[0] > pred_lin[-1] else "FAIL")
    i7 = "STRONG PASS" if neg_ok(rec_mb) and neg_ok(rec_wb) and neg_ok(rec_wc) else ("PASS" if neg_ok(rec_n) and rec_mb.get("coef", 1) < 0 else "MIXED")

    if i1 == "FAIL":
        verdict = "CURRENT_CORE_ASSOCIATION_NOT_ROBUST"
    elif i2 in ("MIXED", "FAIL"):
        verdict = "PROCEED_WITH_BETWEEN_LEARNER_EMPHASIS"
    elif i4 == "MIXED" and i1 in ("PASS", "STRONG PASS"):
        verdict = "PI_REVIEW_REQUIRED_CONTEXT_DEPENDENCE"
    elif i1 in ("PASS", "STRONG PASS") and i2 in ("PASS", "STRONG PASS") and i4 in ("PASS", "STRONG PASS") and i7 in ("PASS", "STRONG PASS"):
        verdict = "PROCEED_TO_PHASE4_TIMING_THEORY"
    else:
        verdict = "PROCEED_TO_PHASE4_TIMING_THEORY" if i1 in ("PASS", "STRONG PASS") and i2 in ("PASS", "STRONG PASS") else "PI_REVIEW_REQUIRED_CONTEXT_DEPENDENCE"

    dump(
        "gates.json",
        {
            "I1": i1,
            "I2": i2,
            "I3": i3,
            "I4": i4,
            "I5": i5,
            "I6": i6,
            "I7": i7,
            "verdict": verdict,
            "n_cross_zero": n_cross,
            "n_pos_ci": n_pos_ci,
        },
    )
    df[["row_id", "pres", "share_never", "n_opp", "score"]].to_csv(OUT / "primary_ids.csv", index=False)
    (OUT / "run.log").write_text("\n".join(LOG) + "\n")
    log("done")


if __name__ == "__main__":
    main()
