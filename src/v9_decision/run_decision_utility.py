#!/usr/bin/env python3
"""Phase-5 decision utility. Writes only results/v9_decision_utility/."""
from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf

_SRC = Path(__file__).resolve().parents[1]
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from common.paths import OULAD_INTERIM, ROOT  # noqa: E402
from v2.constants import PRIMARY_ASSESSMENT_TYPE  # noqa: E402
from v2.load import load_assessments, load_registration, load_student_vle, load_vle  # noqa: E402
from v2.panel import (  # noqa: E402
    attach_prior_score,
    documented_sites,
    map_next_tma,
    map_prev_tma_date,
    registered_learner_sites,
)
from v3.constants import FAMILY_OUCONTENT  # noqa: E402
from v3.features import clicks_by_learner_site_date  # noqa: E402
from v4.constants import V4_DEVELOPMENT  # noqa: E402
from v5_temporal.run_temporal_safe import collapse_student_assessment  # noqa: E402

OUT = ROOT / "results" / "v9_decision_utility"
FORBIDDEN = [
    ROOT / "results" / "v4",
    ROOT / "results" / "v5_temporal_safe",
    ROOT / "results" / "v6_construct_validity",
    ROOT / "results" / "v7_inference",
    ROOT / "results" / "v8_timing",
]
SEED = 20260830
KEYS = ["code_module", "code_presentation", "id_student", "next_id_assessment"]
BUDGETS = (0.05, 0.10, 0.20)
DCA_PTS = tuple(np.round(np.arange(0.05, 0.41, 0.05), 2))
RECLASS = (10, 20)
LOW_CUT = 40.0
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


def load_sa() -> pd.DataFrame:
    return pd.read_parquet(OULAD_INTERIM / "student_assessment.parquet")


def auroc(y, p) -> float:
    y = np.asarray(y)
    p = np.asarray(p)
    pos, neg = y == 1, y == 0
    if pos.sum() == 0 or neg.sum() == 0:
        return float("nan")
    order = np.argsort(p)
    ys = y[order]
    tps = np.cumsum(ys[::-1])[::-1]
    # Mann-Whitney
    ranks = pd.Series(p).rank(method="average").to_numpy()
    return float((ranks[pos].sum() - pos.sum() * (pos.sum() + 1) / 2) / (pos.sum() * neg.sum()))


def auprc(y, p) -> float:
    y = np.asarray(y)
    p = np.asarray(p)
    if y.sum() == 0:
        return float("nan")
    order = np.argsort(-p)
    ys = y[order]
    tp = np.cumsum(ys)
    prec = tp / np.arange(1, len(ys) + 1)
    rec = tp / ys.sum()
    rec = np.r_[0, rec]
    prec = np.r_[1, prec]
    return float(np.trapz(prec, rec))


def brier(y, p) -> float:
    return float(np.mean((np.asarray(p) - np.asarray(y)) ** 2))


def logloss(y, p) -> float:
    p = np.clip(np.asarray(p), 1e-6, 1 - 1e-6)
    y = np.asarray(y)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def calib_ab(y, p) -> dict:
    p = np.clip(np.asarray(p, dtype=float), 1e-6, 1 - 1e-6)
    y = np.asarray(y, dtype=float)
    lp = np.log(p / (1 - p))
    if np.unique(y).size < 2 or np.std(lp) < 1e-12:
        return {"intercept": None, "slope": None}
    X = sm.add_constant(lp)
    try:
        fit = sm.Logit(y, X).fit(disp=False, maxiter=100)
        return {"intercept": float(fit.params[0]), "slope": float(fit.params[1])}
    except Exception as e:
        return {"intercept": None, "slope": None, "error": str(e)[:120]}


def reliability(y, p, n_bins=10) -> list[dict]:
    p = np.asarray(p, dtype=float)
    y = np.asarray(y, dtype=float)
    bins = np.linspace(0, 1, n_bins + 1)
    rows = []
    for i in range(n_bins):
        m = (p >= bins[i]) & (p < bins[i + 1] if i < n_bins - 1 else p <= bins[i + 1])
        if m.sum() == 0:
            continue
        rows.append({"bin": i, "n": int(m.sum()), "pred": float(p[m].mean()), "obs": float(y[m].mean())})
    return rows


def budget_stats(y, p, frac: float) -> dict:
    y = np.asarray(y)
    p = np.asarray(p)
    n = len(y)
    k = max(1, int(round(frac * n)))
    top = np.argsort(-p)[:k]
    captured = int(y[top].sum())
    tot = int(y.sum())
    prec = captured / k if k else None
    rec = captured / tot if tot else None
    prev = tot / n if n else None
    return {
        "n": n,
        "k": k,
        "captured": captured,
        "total_pos": tot,
        "recall": rec,
        "precision": prec,
        "lift": (prec / prev) if prec is not None and prev else None,
        "nnr": (1 / prec) if prec else None,
        "prevalence": prev,
    }


def dca_curve(y, p) -> list[dict]:
    y = np.asarray(y)
    p = np.asarray(p)
    n = len(y)
    prev = float(y.mean()) if n else None
    rows = []
    for pt in DCA_PTS:
        pred = p >= pt
        tp = int(((y == 1) & pred).sum())
        fp = int(((y == 0) & pred).sum())
        w = pt / (1 - pt)
        rows.append(
            {
                "pt": float(pt),
                "nb_model": float(tp / n - fp / n * w),
                "nb_all": float(prev - (1 - prev) * w) if prev is not None else None,
                "nb_none": 0.0,
            }
        )
    return rows


def fit_logit(df, fml, groups=None):
    try:
        if groups is None:
            fit = smf.logit(fml, data=df).fit(disp=False, maxiter=200)
        else:
            fit = smf.logit(fml, data=df).fit(disp=False, maxiter=200, cov_type="cluster", cov_kwds={"groups": groups})
        return fit
    except Exception as e:
        log(f"logit fail {e}")
        return None


def pack_logit(fit, feat="never_share") -> dict:
    if fit is None or feat not in fit.params:
        return {"ok": False}
    b = float(fit.params[feat])
    ci = [float(x) for x in fit.conf_int().loc[feat]] if feat in fit.conf_int().index else [None, None]
    or_ = float(np.exp(b))
    or_ci = [float(np.exp(ci[0])), float(np.exp(ci[1]))] if ci[0] is not None else None
    # AME for logit: mean(p*(1-p))*beta
    p = np.asarray(fit.predict())
    ame = float(np.mean(p * (1 - p)) * b)
    return {
        "ok": True,
        "n": int(fit.nobs),
        "coef": b,
        "or": or_,
        "ci": ci,
        "or_ci": or_ci,
        "p": float(fit.pvalues[feat]) if feat in fit.pvalues else None,
        "ame": ame,
        "pseudo_r2": float(fit.prsquared),
    }


def predict_ok(fit, df):
    if fit is None:
        return None
    try:
        return np.asarray(fit.predict(df), dtype=float)
    except Exception:
        return None


def attach_controls_cutoff(occ: pd.DataFrame, sv: pd.DataFrame, cutoff_col: str) -> pd.DataFrame:
    out = occ.copy()
    recs = []
    want = {f"{a}|{b}" for a, b in V4_DEVELOPMENT}
    sub = sv.loc[
        sv["code_module"].str.cat(sv["code_presentation"], sep="|").isin(want),
        ["code_module", "code_presentation", "id_student", "date", "sum_click", "week"],
    ]
    bounds = out.reset_index()[["index", "code_module", "code_presentation", "id_student", "prev_tma_date", cutoff_col]]
    for (mod, pres), gocc in bounds.groupby(["code_module", "code_presentation"], sort=False):
        gsv = sub[(sub.code_module == mod) & (sub.code_presentation == pres)]
        if gsv.empty or gocc.empty:
            continue
        m = gsv.merge(gocc, on="id_student", how="inner")
        lo = m["prev_tma_date"].fillna(-1e9)
        hi = m[cutoff_col]
        keep = m["date"].notna() & (m["date"] < hi) & (m["date"] > lo)
        used = m.loc[keep].copy()
        if used.empty:
            continue
        used["week_start"] = (used["week"] - 1) * 7
        used["week_end"] = used["week"] * 7 - 1
        used = used[(used["week_end"] > used["prev_tma_date"].fillna(-1e9)) & (used["week_start"] < used[cutoff_col])]
        if used.empty:
            continue
        week_hit = used.groupby(["index", "week"], as_index=False).agg(n_clicks=("sum_click", "sum"))
        week_hit["A_raw"] = (week_hit.n_clicks > 0).astype(int)
        click_sum = used.groupby("index")["sum_click"].sum()
        ctrl = week_hit.groupby("index").agg(active_rate=("A_raw", "mean"))
        ctrl["raw_inact"] = 1.0 - ctrl["active_rate"]
        ctrl["log_clicks"] = np.log1p(click_sum.reindex(ctrl.index).fillna(0.0))
        recs.append(ctrl)
    if recs:
        ctrl = pd.concat(recs)
        ctrl = ctrl[~ctrl.index.duplicated(keep="first")]
        out = out.join(ctrl)
    out["log_clicks"] = out.get("log_clicks", np.nan).fillna(0.0)
    out["active_rate"] = out.get("active_rate", np.nan).fillna(0.0)
    out["raw_inact"] = out.get("raw_inact", np.nan).fillna(1.0)
    return out


def coverage_for_cutoff(site: pd.DataFrame, clicks: pd.DataFrame, occ: pd.DataFrame, cutoff_col: str) -> pd.DataFrame:
    m = site.merge(occ[KEYS + [cutoff_col]], on=KEYS, how="inner")
    m = m[m.window_end_day < m[cutoff_col]]
    if m.empty:
        out = occ[KEYS].copy()
        out["n_opp"] = 0
        out["never_share"] = np.nan
        return out
    acc = clicks.merge(m[KEYS + ["id_site", cutoff_col]], on=["id_site", "id_student"], how="inner")
    acc = acc[(acc.date < acc[cutoff_col]) & (acc.next_id_assessment == acc.next_id_assessment)]
    # next_id on clicks merge: need next_id on acc from m
    hit = acc.groupby(KEYS + ["id_site"], as_index=False).size()
    hit["accessed"] = 1
    sites = m[KEYS + ["id_site"]].drop_duplicates()
    sites = sites.merge(hit[KEYS + ["id_site", "accessed"]], on=KEYS + ["id_site"], how="left")
    sites["accessed"] = sites.accessed.fillna(0)
    sites["never"] = 1 - sites.accessed
    agg = sites.groupby(KEYS, as_index=False).agg(n_opp=("id_site", "nunique"), never_share=("never", "mean"))
    return agg


def classify_states(reg, tmas, sa) -> pd.DataFrame:
    parts = []
    for (mod, pres), gt in tmas.groupby(["code_module", "code_presentation"]):
        learners = reg[(reg.code_module == mod) & (reg.code_presentation == pres) & reg.date_registration.notna()][
            ["id_student", "date_registration", "date_unregistration"]
        ]
        if learners.empty:
            continue
        a = learners.assign(_k=1)
        b = gt[["id_assessment", "date"]].rename(columns={"id_assessment": "next_id_assessment", "date": "next_tma_date"}).assign(_k=1)
        sk = a.merge(b, on="_k").drop(columns="_k")
        sk["code_module"] = mod
        sk["code_presentation"] = pres
        parts.append(sk)
    skel = pd.concat(parts, ignore_index=True)
    sa_j = sa.rename(columns={"id_assessment": "next_id_assessment"})[
        ["id_student", "next_id_assessment", "date_submitted", "is_banked", "score"]
    ]
    out = skel.merge(sa_j, on=["id_student", "next_id_assessment"], how="left")
    banked = out.is_banked == True  # noqa: E712
    submitted = (~banked) & out.date_submitted.notna() & out.score.notna()
    no_sub = ~submitted & ~banked
    withdrew = no_sub & out.date_unregistration.notna() & (out.date_unregistration < out.next_tma_date)
    late_reg = no_sub & ~withdrew & (out.date_registration > out.next_tma_date)
    eligible = no_sub & ~withdrew & ~late_reg & (out.date_registration <= out.next_tma_date) & (
        out.date_unregistration.isna() | (out.date_unregistration >= out.next_tma_date)
    )
    unscore = (~banked) & out.date_submitted.notna() & out.score.isna()
    state = np.full(len(out), "ambiguous", dtype=object)
    state[banked.to_numpy()] = "banked"
    state[submitted.to_numpy()] = "submitted"
    state[eligible.to_numpy()] = "eligible_nonsubmit"
    state[withdrew.to_numpy()] = "withdrew_before_risk"
    state[late_reg.to_numpy()] = "ambiguous"
    state[unscore.to_numpy()] = "ambiguous"
    out["state"] = state
    out["submitted"] = submitted.astype(int)
    return out


def metrics_pair(y, p0, p1) -> dict:
    return {
        "n": int(len(y)),
        "prevalence": float(np.mean(y)),
        "B0": {"auroc": auroc(y, p0), "auprc": auprc(y, p0), "brier": brier(y, p0), "logloss": logloss(y, p0), **calib_ab(y, p0)},
        "B1": {"auroc": auroc(y, p1), "auprc": auprc(y, p1), "brier": brier(y, p1), "logloss": logloss(y, p1), **calib_ab(y, p1)},
        "d_auroc": auroc(y, p1) - auroc(y, p0),
        "d_auprc": auprc(y, p1) - auprc(y, p0),
        "d_brier": brier(y, p1) - brier(y, p0),
    }


def oos_eval(df, ycol, context: str):
    """LOPO or LOMO. Returns fold rows + stacked predictions."""
    folds = []
    stack = []
    if context == "LOPO":
        keys = sorted(df.pres.unique())
        fml0 = f"{ycol} ~ prior_score_filled + prior_missing + log_clicks + active_rate + n_opp + C(code_module)"
        split_col = "pres"
    else:
        keys = sorted(df.code_module.unique())
        fml0 = f"{ycol} ~ prior_score_filled + prior_missing + log_clicks + active_rate + n_opp"
        split_col = "code_module"
    fml1 = fml0 + " + never_share"
    for key in keys:
        te = df[df[split_col] == key].copy()
        tr = df[df[split_col] != key].copy()
        rec = {"fold": str(key), "n_test": int(len(te)), "prev": float(te[ycol].mean()) if len(te) else None}
        if te[ycol].nunique() < 2 or tr[ycol].nunique() < 2 or len(tr) < 50 or len(te) < 20:
            rec["status"] = "SKIP"
            folds.append(rec)
            continue
        f0 = fit_logit(tr, fml0)
        f1 = fit_logit(tr, fml1)
        p0 = predict_ok(f0, te)
        p1 = predict_ok(f1, te)
        if p0 is None or p1 is None:
            rec["status"] = "FIT_FAIL"
            folds.append(rec)
            continue
        m = metrics_pair(te[ycol].to_numpy(), p0, p1)
        rec.update(m)
        rec["status"] = "OK"
        folds.append(rec)
        chunk = te[KEYS + [ycol, "pres", "code_module"]].copy()
        chunk["p0"] = p0
        chunk["p1"] = p1
        chunk["fold"] = str(key)
        stack.append(chunk)
    stacked = pd.concat(stack, ignore_index=True) if stack else pd.DataFrame()
    return folds, stacked


def reclass_table(st: pd.DataFrame, ycol: str) -> dict:
    rows = []
    for fold, g in st.groupby("fold"):
        r0 = pd.Series(g.p0).rank(pct=True) * 100
        r1 = pd.Series(g.p1).rank(pct=True) * 100
        delta = r1.to_numpy() - r0.to_numpy()
        y = g[ycol].to_numpy()
        for thr in RECLASS:
            up = delta >= thr
            down = delta <= -thr
            rows.append(
                {
                    "fold": fold,
                    "thr": thr,
                    "n_up": int(up.sum()),
                    "risk_up": float(y[up].mean()) if up.any() else None,
                    "n_down": int(down.sum()),
                    "risk_down": float(y[down].mean()) if down.any() else None,
                }
            )
    # pooled ranks within fold already; also overall
    r0 = pd.Series(st.p0).rank(pct=True) * 100
    r1 = pd.Series(st.p1).rank(pct=True) * 100
    delta = r1.to_numpy() - r0.to_numpy()
    y = st[ycol].to_numpy()
    pooled = []
    for thr in RECLASS:
        up = delta >= thr
        down = delta <= -thr
        pooled.append(
            {
                "fold": "POOLED_LOPO_RANKS",
                "thr": thr,
                "n_up": int(up.sum()),
                "risk_up": float(y[up].mean()) if up.any() else None,
                "n_down": int(down.sum()),
                "risk_down": float(y[down].mean()) if down.any() else None,
            }
        )
    return {"folds": rows, "pooled": pooled}


def nri_idi(y, p0, p1) -> dict:
    y = np.asarray(y)
    p0, p1 = np.asarray(p0), np.asarray(p1)
    ev, ne = y == 1, y == 0
    if ev.sum() == 0 or ne.sum() == 0:
        return {"nri": None, "idi": None}
    nri_e = float(np.mean(p1[ev] > p0[ev]) - np.mean(p1[ev] < p0[ev]))
    nri_n = float(np.mean(p1[ne] < p0[ne]) - np.mean(p1[ne] > p0[ne]))
    idi = float((p1[ev].mean() - p0[ev].mean()) - (p1[ne].mean() - p0[ne].mean()))
    return {"nri": nri_e + nri_n, "nri_event": nri_e, "nri_nonevent": nri_n, "idi": idi}


def main() -> None:
    log("PHASE5 decision utility start")
    vle = load_vle()
    assessments = load_assessments()
    reg = load_registration()
    sa, _ = collapse_student_assessment(load_sa())
    sv = load_student_vle()
    tmas = assessments[
        (assessments.assessment_type == PRIMARY_ASSESSMENT_TYPE)
        & assessments.date.notna()
        & assessments.code_module.str.cat(assessments.code_presentation, sep="|").isin({f"{a}|{b}" for a, b in V4_DEVELOPMENT})
    ][["code_module", "code_presentation", "id_assessment", "date"]]
    log("classify cohort...")
    skel = classify_states(reg, tmas, sa)
    flow = skel.state.value_counts().to_dict()
    flow["total"] = int(len(skel))
    dump("sample_flow.json", {str(k): int(v) if not isinstance(v, int) else v for k, v in flow.items()})

    tma_meta = map_prev_tma_date(assessments)
    skel = skel.merge(
        tma_meta.rename(columns={"id_assessment": "next_id_assessment"})[
            ["code_module", "code_presentation", "next_id_assessment", "prev_tma_date"]
        ],
        on=["code_module", "code_presentation", "next_id_assessment"],
        how="left",
    )
    skel["pres"] = skel.code_module + "_" + skel.code_presentation
    primary_states = skel[skel.state.isin(["submitted", "eligible_nonsubmit"])].copy()

    log("site panel...")
    sites0 = documented_sites(vle, set(V4_DEVELOPMENT), FAMILY_OUCONTENT)
    clicks = clicks_by_learner_site_date(sv, sites0.id_site.tolist())
    panel = registered_learner_sites(reg, sites0)
    panel = map_next_tma(panel, assessments).dropna(subset=["next_id_assessment"])
    panel["next_id_assessment"] = panel.next_id_assessment.astype(int)
    site = panel[KEYS + ["id_site", "window_start_day", "window_end_day", "next_tma_date"]].copy()

    def build_task(occ_in: pd.DataFrame, cutoff_col: str, name: str) -> pd.DataFrame:
        log(f"coverage+controls {name}...")
        cov = coverage_for_cutoff(site, clicks, occ_in, cutoff_col)
        occ = occ_in.merge(cov, on=KEYS, how="left")
        occ["n_opp"] = occ.n_opp.fillna(0).astype(int)
        occ = attach_prior_score(occ, sa, assessments)
        occ = attach_controls_cutoff(occ, sv, cutoff_col)
        occ["low_score"] = np.where(occ.score.notna(), (occ.score < LOW_CUT).astype(int), np.nan)
        return occ

    a = primary_states.copy()
    a["cutoff"] = np.where(a.state == "submitted", a.date_submitted, a.next_tma_date)
    b = primary_states.copy()
    b["cutoff"] = b.next_tma_date
    occ_a = build_task(a, "cutoff", "SUB-A")
    occ_b = build_task(b, "cutoff", "SUB-B")

    def denom_report(occ, label):
        d = occ[occ.n_opp >= 1]
        return {
            "label": label,
            "n_defined": int(len(d)),
            "n_undefined": int((occ.n_opp == 0).sum()),
            "submitters_n": int(((d.state == "submitted")).sum()),
            "nonsubmit_n": int((d.state == "eligible_nonsubmit").sum()),
            "never_submitters": float(d.loc[d.state == "submitted", "never_share"].mean()) if (d.state == "submitted").any() else None,
            "never_nonsubmit": float(d.loc[d.state == "eligible_nonsubmit", "never_share"].mean()) if (d.state == "eligible_nonsubmit").any() else None,
            "nopp_submit_median": float(d.loc[d.state == "submitted", "n_opp"].median()) if (d.state == "submitted").any() else None,
            "nopp_non_median": float(d.loc[d.state == "eligible_nonsubmit", "n_opp"].median()) if (d.state == "eligible_nonsubmit").any() else None,
        }

    dump("denominator.json", {"SUB_A": denom_report(occ_a, "SUB-A"), "SUB_B": denom_report(occ_b, "SUB-B")})

    # Q5-A association
    sub_assoc = {}
    for name, occ in [("SUB_A", occ_a), ("SUB_B", occ_b)]:
        d = occ[(occ.n_opp >= 1) & occ.state.isin(["submitted", "eligible_nonsubmit"])].copy()
        d["y"] = (d.state == "submitted").astype(int)
        fml = "y ~ never_share + prior_score_filled + prior_missing + log_clicks + active_rate + n_opp + C(pres)"
        fit = fit_logit(d, fml, groups=d.id_student)
        rec = pack_logit(fit)
        rec["n_learners"] = int(d.id_student.nunique())
        rec["submit_rate"] = float(d.y.mean())
        # presentation-specific
        pres_rows = []
        for pres, g in d.groupby("pres"):
            if g.y.nunique() < 2 or len(g) < 80:
                continue
            pf = fit_logit(g, "y ~ never_share + prior_score_filled + prior_missing + log_clicks + active_rate + n_opp", groups=g.id_student)
            pr = pack_logit(pf)
            pr["pres"] = pres
            pres_rows.append(pr)
        rec["by_presentation"] = pres_rows
        rec["n"] = int(len(d))
        sub_assoc[name] = rec
        log(f"{name} submit assoc ok={rec.get('ok')} coef={rec.get('coef')}")
    dump("submission_association.json", sub_assoc)

    # Q5-B scored sample (SUB-A cutoff == date_submitted)
    scored = occ_a[(occ_a.state == "submitted") & (occ_a.n_opp >= 1) & occ_a.score.notna()].copy()
    scored["y"] = (scored.score < LOW_CUT).astype(int)
    log(f"scored n={len(scored)} low_score_prev={scored.y.mean():.4f}")
    dump(
        "low_score_definition.json",
        {
            "rule": "score < 40",
            "rationale": "OULAD documented pass mark",
            "n": int(len(scored)),
            "n_learners": int(scored.id_student.nunique()),
            "prevalence": float(scored.y.mean()),
        },
    )
    fml_s = "y ~ never_share + prior_score_filled + prior_missing + log_clicks + active_rate + n_opp + C(pres)"
    fit_s = fit_logit(scored, fml_s, groups=scored.id_student)
    dump("lowscore_association.json", pack_logit(fit_s))

    # secondary continuous
    lin0 = smf.ols("score ~ prior_score_filled + prior_missing + log_clicks + active_rate + n_opp + C(pres)", data=scored).fit()
    lin1 = smf.ols("score ~ prior_score_filled + prior_missing + log_clicks + active_rate + n_opp + C(pres) + never_share", data=scored).fit()
    dump("continuous_secondary.json", {"B0_r2": float(lin0.rsquared), "B1_r2": float(lin1.rsquared), "d_r2": float(lin1.rsquared - lin0.rsquared)})

    log("LOPO/LOMO Q5-B low-score...")
    lopo_s, st_s = oos_eval(scored, "y", "LOPO")
    lomo_s, st_s_m = oos_eval(scored, "y", "LOMO")
    dump("lopo_lowscore.json", lopo_s)
    dump("lomo_lowscore.json", lomo_s)

    log("LOPO/LOMO Q5-A submission...")
    subA = occ_a[(occ_a.n_opp >= 1)].copy()
    subA["y"] = (subA.state == "submitted").astype(int)
    # rank non-submission risk: use 1-P(submit) later
    lopo_u, st_u = oos_eval(subA, "y", "LOPO")
    lomo_u, st_u_m = oos_eval(subA, "y", "LOMO")
    dump("lopo_submit.json", lopo_u)
    dump("lomo_submit.json", lomo_u)

    def fold_summary(folds):
        ok = [f for f in folds if f.get("status") == "OK"]
        if not ok:
            return {}
        return {
            "n_ok": len(ok),
            "median_B0_auroc": float(np.nanmedian([f["B0"]["auroc"] for f in ok])),
            "median_B1_auroc": float(np.nanmedian([f["B1"]["auroc"] for f in ok])),
            "median_d_auroc": float(np.nanmedian([f["d_auroc"] for f in ok])),
            "median_B0_auprc": float(np.nanmedian([f["B0"]["auprc"] for f in ok])),
            "median_B1_auprc": float(np.nanmedian([f["B1"]["auprc"] for f in ok])),
            "median_d_auprc": float(np.nanmedian([f["d_auprc"] for f in ok])),
            "n_improve_auroc": int(sum(f["d_auroc"] > 0 for f in ok)),
            "n_worsen_auroc": int(sum(f["d_auroc"] < 0 for f in ok)),
        }

    dump("lopo_lomo_summary.json", {"lowscore_LOPO": fold_summary(lopo_s), "lowscore_LOMO": fold_summary(lomo_s), "submit_LOPO": fold_summary(lopo_u), "submit_LOMO": fold_summary(lomo_u)})

    # budgets on stacked LOPO
    def budgets_from_stack(st, ycol, invert=False, label=""):
        if st.empty:
            return {}
        y = st[ycol].to_numpy().astype(int)
        if invert:
            y = 1 - y
        p0 = 1 - st.p0.to_numpy() if invert else st.p0.to_numpy()
        p1 = 1 - st.p1.to_numpy() if invert else st.p1.to_numpy()
        out = {"label": label, "invert_to_risk": invert, "pooled": {}, "folds": []}
        for fr in BUDGETS:
            b0 = budget_stats(y, p0, fr)
            b1 = budget_stats(y, p1, fr)
            out["pooled"][str(fr)] = {
                "B0": b0,
                "B1": b1,
                "d_recall": (b1["recall"] - b0["recall"]) if b0["recall"] is not None and b1["recall"] is not None else None,
                "extra_true": b1["captured"] - b0["captured"],
            }
        for fold, g in st.groupby("fold"):
            yy = g[ycol].to_numpy().astype(int)
            if invert:
                yy = 1 - yy
            q0 = 1 - g.p0.to_numpy() if invert else g.p0.to_numpy()
            q1 = 1 - g.p1.to_numpy() if invert else g.p1.to_numpy()
            rec = {"fold": fold}
            for fr in BUDGETS:
                bb0, bb1 = budget_stats(yy, q0, fr), budget_stats(yy, q1, fr)
                rec[str(fr)] = {"B0_recall": bb0["recall"], "B1_recall": bb1["recall"], "extra_true": bb1["captured"] - bb0["captured"]}
            out["folds"].append(rec)
        return out

    bud_s = budgets_from_stack(st_s, "y", invert=False, label="low_score_LOPO")
    bud_u = budgets_from_stack(st_u, "y", invert=True, label="nonsusubmit_LOPO")
    dump("fixed_budget.json", {"low_score": bud_s, "nonsusubmit": bud_u})

    rec_s = reclass_table(st_s, "y") if not st_s.empty else {}
    dump("reclassification.json", rec_s)
    dump("nri_idi.json", nri_idi(st_s.y, st_s.p0, st_s.p1) if not st_s.empty else {})

    dca = {}
    if not st_s.empty:
        dca["low_score"] = {
            "B0": dca_curve(st_s.y, st_s.p0),
            "B1": dca_curve(st_s.y, st_s.p1),
        }
    dump("decision_curve.json", dca)

    if not st_s.empty:
        dump(
            "calibration_pooled.json",
            {
                "B0": {**calib_ab(st_s.y, st_s.p0), "brier": brier(st_s.y, st_s.p0), "bins": reliability(st_s.y, st_s.p0)},
                "B1": {**calib_ab(st_s.y, st_s.p1), "brier": brier(st_s.y, st_s.p1), "bins": reliability(st_s.y, st_s.p1)},
            },
        )

    # context heterogeneity: LOPO ΔAUROC by module
    het = []
    for f in lopo_s:
        if f.get("status") != "OK":
            continue
        het.append({"fold": f["fold"], "module": f["fold"].split("_")[0], "d_auroc": f["d_auroc"], "d_auprc": f["d_auprc"], "extra10": None})
    extra_map = {r["fold"]: r.get("0.1", {}).get("extra_true") for r in bud_s.get("folds", [])}
    for h in het:
        h["extra10"] = extra_map.get(h["fold"])
    dump("context_heterogeneity.json", het)

    # gates
    def p1_gate(a, b):
        if not a.get("ok"):
            return "FAIL"
        a_ok = a["ci"][1] < 0 if a["coef"] < 0 else a["ci"][0] > 0
        if not a_ok:
            return "FAIL"
        if b.get("ok"):
            same = np.sign(a["coef"]) == np.sign(b["coef"])
            b_ok = (b["ci"][1] < 0 if b["coef"] < 0 else b["ci"][0] > 0)
            if same and b_ok:
                return "STRONG PASS"
            if same:
                return "PASS"
            return "MIXED"
        return "PASS"

    def p2_gate(summ):
        if not summ:
            return "FAIL"
        d = summ.get("median_d_auroc")
        da = summ.get("median_d_auprc")
        imp, wor = summ.get("n_improve_auroc", 0), summ.get("n_worsen_auroc", 0)
        if d is None:
            return "FAIL"
        if d > 0 and da is not None and da > 0 and imp > wor:
            return "STRONG PASS" if imp >= wor + 3 else "PASS"
        if d > 0 and imp > wor:
            return "PASS"
        if imp and wor:
            return "MIXED"
        return "FAIL"

    def p3_gate():
        if not st_s.empty:
            c0 = calib_ab(st_s.y, st_s.p0)
            c1 = calib_ab(st_s.y, st_s.p1)
            br0, br1 = brier(st_s.y, st_s.p0), brier(st_s.y, st_s.p1)
            if c1.get("slope") is None:
                return "MIXED"
            worse = br1 > br0 + 0.005
            if worse:
                return "FAIL"
            return "PASS"
        return "MIXED"

    def p4_gate(bud):
        folds = bud.get("folds", [])
        if not folds:
            return "FAIL"
        extras = [f.get("0.1", {}).get("extra_true") for f in folds if f.get("0.1")]
        extras = [e for e in extras if e is not None]
        if not extras:
            return "FAIL"
        pos = sum(e > 0 for e in extras)
        neg = sum(e < 0 for e in extras)
        pooled_extra = bud.get("pooled", {}).get("0.1", {}).get("extra_true")
        if pos > neg and pos >= max(1, len(extras) // 2 + 1) and (pooled_extra or 0) > 0:
            return "STRONG PASS" if pos >= neg + 3 else "PASS"
        if (pooled_extra or 0) > 0 and pos >= neg:
            return "PASS"
        if pos and neg:
            return "MIXED"
        return "FAIL"

    def p5_gate(rec):
        pooled = {r["thr"]: r for r in rec.get("pooled", [])}
        r = pooled.get(10)
        if not r or r["risk_up"] is None or r["risk_down"] is None or r["n_up"] < 20 or r["n_down"] < 20:
            return "MIXED"
        if r["risk_up"] > r["risk_down"] + 0.02:
            return "PASS"
        if r["risk_up"] < r["risk_down"] - 0.02:
            return "FAIL"
        return "MIXED"

    g1 = p1_gate(sub_assoc["SUB_A"], sub_assoc["SUB_B"])
    g2s = p2_gate(fold_summary(lopo_s))
    g2m = p2_gate(fold_summary(lomo_s))
    g2 = g2s if g2s == g2m else ("MIXED" if {g2s, g2m} != {"FAIL"} else "FAIL")
    if g2s in ("PASS", "STRONG PASS") and g2m == "MIXED":
        g2 = "MIXED"
    g3 = p3_gate()
    g4 = p4_gate(bud_s)
    g5 = p5_gate(rec_s)

    def ok(x):
        return x in ("PASS", "STRONG PASS")

    if ok(g2) and ok(g4) and g3 != "FAIL":
        g6 = "OPERATIONALLY_INCREMENTAL"
    elif ok(g1) and (ok(g2) ^ ok(g4)):
        g6 = "MODESTLY_INCREMENTAL"
    elif ok(g1) and not ok(g2) and not ok(g4):
        g6 = "INTERPRETIVE_ONLY"
    elif not ok(g1) and not ok(g2) and not ok(g4):
        g6 = "NO_PRACTICAL_SUPPORT"
    else:
        g6 = "INTERPRETIVE_ONLY" if not (ok(g2) or ok(g4)) else "MODESTLY_INCREMENTAL"

    # empirical pattern
    assoc_score = fit_s is not None and pack_logit(fit_s).get("ok") and pack_logit(fit_s)["ci"][0] > 0 or (
        pack_logit(fit_s).get("ok") and pack_logit(fit_s)["coef"] > 0 and pack_logit(fit_s)["ci"][0] > 0
    )
    # never_share high => more low_score => coef > 0 for low_score; for submit, never high => less submit => coef < 0
    sa_ok = sub_assoc["SUB_A"].get("ok") and sub_assoc["SUB_A"]["coef"] < 0 and sub_assoc["SUB_A"]["ci"][1] < 0
    sc_pack = pack_logit(fit_s)
    sc_ok = sc_pack.get("ok") and sc_pack["coef"] > 0 and sc_pack["ci"][0] > 0
    oos_score = ok(g2s)
    oos_sub = ok(p2_gate(fold_summary(lopo_u)))
    if not oos_score and not oos_sub:
        pattern = "P4"
    elif oos_score and oos_sub:
        pattern = "P1"
    elif oos_score and not oos_sub:
        pattern = "P2"
    else:
        pattern = "P3"

    if g6 in ("OPERATIONALLY_INCREMENTAL", "MODESTLY_INCREMENTAL") and g2 != "MIXED":
        verdict = "PROCEED_TO_PHASE6_NOVELTY_THEORY"
    elif g6 == "INTERPRETIVE_ONLY" or g6 == "NO_PRACTICAL_SUPPORT":
        verdict = "PROCEED_WITH_INTERPRETIVE_ONLY_VALUE"
    else:
        verdict = "PI_REVIEW_REQUIRED_PRACTICAL_HETEROGENEITY"

    # extra: if P4/P2 MIXED due to context split
    if g2 == "MIXED" or g4 == "MIXED":
        extras = [h.get("extra10") for h in het if h.get("extra10") is not None]
        if extras and min(extras) < 0 < max(extras):
            verdict = "PI_REVIEW_REQUIRED_PRACTICAL_HETEROGENEITY"

    dump(
        "gates.json",
        {
            "P1": g1,
            "P2": g2,
            "P2_LOPO": g2s,
            "P2_LOMO": g2m,
            "P3": g3,
            "P4": g4,
            "P5": g5,
            "P6": g6,
            "pattern": pattern,
            "verdict": verdict,
            "submit_assoc_supported": sa_ok,
            "lowscore_assoc_supported": sc_ok,
        },
    )
    (OUT / "run.log").write_text("\n".join(LOG) + "\n")
    log("PHASE5 done " + verdict + " " + g6)


if __name__ == "__main__":
    main()
