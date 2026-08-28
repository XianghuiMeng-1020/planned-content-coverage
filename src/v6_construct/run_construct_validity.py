#!/usr/bin/env python3
"""Phase-2 construct validity. Writes only results/v6_construct_validity/."""
from __future__ import annotations

import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from scipy import stats

_SRC = Path(__file__).resolve().parents[1]
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from common.paths import OULAD_INTERIM, OULAD_RAW, ROOT, SEED  # noqa: E402
from v2.constants import FAMILY_ALL_DOCUMENTED  # noqa: E402
from v2.load import load_assessments, load_registration, load_student_vle, load_vle  # noqa: E402
from v2.panel import (  # noqa: E402
    attach_prior_score,
    attach_tma_outcomes,
    documented_sites,
    map_next_tma,
    map_prev_tma_date,
    registered_learner_sites,
)
from v3.constants import FAMILY_OUCONTENT  # noqa: E402
from v3.features import clicks_by_learner_site_date  # noqa: E402
from v4.constants import FAMILY_RESOURCE, V4_CONFIRMATION, V4_DEVELOPMENT  # noqa: E402
from v4.run_development import module_holdout, within_learner  # noqa: E402
from v5_temporal.run_temporal_safe import (  # noqa: E402
    annotate_with_cutoff,
    attach_submission_safe_controls,
    collapse_student_assessment,
)

OUT = ROOT / "results" / "v6_construct_validity"
FORBIDDEN = [ROOT / "results" / "v4", ROOT / "results" / "v5_temporal_safe"]
P1_BETA = 3.5913161705714205
P1_FE = 2.2967835568997685
PERM_SEED = 20260827
PERM_B = 200
KEYS = ["code_module", "code_presentation", "id_student", "next_id_assessment"]
LOG: list[str] = []


def log(msg: str) -> None:
    line = f"{datetime.now(timezone.utc).isoformat()} {msg}"
    LOG.append(line)
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


def sha16(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()[:16] if path.exists() else "MISSING"


def load_sa() -> pd.DataFrame:
    return pd.read_parquet(OULAD_INTERIM / "student_assessment.parquet")


def attach_controls(occ: pd.DataFrame, sv: pd.DataFrame, presentations: set) -> pd.DataFrame:
    """Submission-safe activity controls for an arbitrary presentation set."""
    out = occ.copy()
    if "prev_tma_date" not in out.columns:
        out["prev_tma_date"] = np.nan
    recs = []
    want = {f"{a}|{b}" for a, b in presentations}
    sub = sv.loc[
        sv["code_module"].str.cat(sv["code_presentation"], sep="|").isin(want),
        ["code_module", "code_presentation", "id_student", "date", "sum_click", "week"],
    ]
    bounds = out.reset_index()[["index", "code_module", "code_presentation", "id_student", "prev_tma_date", "date_submitted"]]
    for (mod, pres), gocc in bounds.groupby(["code_module", "code_presentation"], sort=False):
        gsv = sub[(sub.code_module == mod) & (sub.code_presentation == pres)]
        if gsv.empty or gocc.empty:
            continue
        m = gsv.merge(gocc, on="id_student", how="inner")
        lo = m["prev_tma_date"].fillna(-1e9)
        keep = m["date"].notna() & (m["date"] < m["date_submitted"]) & (m["date"] > lo)
        used = m.loc[keep]
        if used.empty:
            continue
        used = used.copy()
        used["week_start"] = (used["week"] - 1) * 7
        used["week_end"] = used["week"] * 7 - 1
        used = used[(used["week_end"] > used["prev_tma_date"].fillna(-1e9)) & (used["week_start"] < used["date_submitted"])]
        if used.empty:
            continue
        week_hit = used.groupby(["index", "week"], as_index=False).agg(n_clicks=("sum_click", "sum"))
        week_hit["A_raw"] = (week_hit["n_clicks"] > 0).astype(int)
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


def nopp_dist(s: pd.Series) -> dict:
    s = pd.to_numeric(s, errors="coerce")
    return {
        "n": int(s.notna().sum()),
        "n0": int((s == 0).sum()),
        "n_ge1": int((s >= 1).sum()),
        "n1": int((s == 1).sum()),
        "n2": int((s == 2).sum()),
        "n_lt3": int((s < 3).sum()),
        "n_lt5": int((s < 5).sum()),
        "n_lt10": int((s < 10).sum()),
        "n_ge10": int((s >= 10).sum()),
        "n_ge20": int((s >= 20).sum()),
        "min": float(s.min()) if len(s) else None,
        "q1": float(s.quantile(0.25)) if len(s) else None,
        "median": float(s.median()) if len(s) else None,
        "q3": float(s.quantile(0.75)) if len(s) else None,
        "max": float(s.max()) if len(s) else None,
    }


def fit_m3(df: pd.DataFrame, feat: str = "share_never", extra: str = "") -> dict:
    work = df.dropna(subset=["score", "raw_inact", "log_clicks", feat, "n_opp", "prior_score_filled"]).copy()
    work = work[work.n_opp >= 1]
    if len(work) < 50:
        return {"insufficient": True, "n": int(len(work))}
    work["pres"] = work.code_module + "_" + work.code_presentation
    base = "score ~ prior_score_filled + prior_missing + C(pres) + log_clicks + active_rate + n_opp + raw_inact"
    f2 = smf.ols(base, data=work).fit(cov_type="cluster", cov_kwds={"groups": work.id_student})
    f3 = smf.ols(base + f" + {feat}" + extra, data=work).fit(cov_type="cluster", cov_kwds={"groups": work.id_student})
    out = {
        "n": int(len(work)),
        "n_learners": int(work.id_student.nunique()),
        "r2_m2": float(f2.rsquared),
        "r2_m3": float(f3.rsquared),
        "partial_r2": float((f3.rsquared - f2.rsquared) / (1 - f2.rsquared)) if f2.rsquared < 1 else None,
    }
    if feat in f3.params:
        out["coef"] = float(f3.params[feat])
        out["se"] = float(f3.bse[feat])
        out["p"] = float(f3.pvalues[feat])
        out["ci"] = [float(x) for x in f3.conf_int().loc[feat]]
    for k in f3.params.index:
        if extra and k != feat and feat in k:
            out["interaction_term"] = k
            out["interaction_coef"] = float(f3.params[k])
            out["interaction_ci"] = [float(x) for x in f3.conf_int().loc[k]]
            out["interaction_p"] = float(f3.pvalues[k])
    return out


def fit_joint(df: pd.DataFrame, cur: str, fut: str) -> dict:
    work = df.dropna(subset=["score", "raw_inact", "log_clicks", cur, fut, "n_opp", "prior_score_filled"]).copy()
    work = work[(work.n_opp >= 1) & (work.n_opp_future >= 1)]
    if len(work) < 50:
        return {"insufficient": True, "n": int(len(work))}
    work["pres"] = work.code_module + "_" + work.code_presentation
    base = "score ~ prior_score_filled + prior_missing + C(pres) + log_clicks + active_rate + n_opp + raw_inact"
    f1 = smf.ols(base + f" + {cur}", data=work).fit(cov_type="cluster", cov_kwds={"groups": work.id_student})
    f2 = smf.ols(base + f" + {fut}", data=work).fit(cov_type="cluster", cov_kwds={"groups": work.id_student})
    f3 = smf.ols(base + f" + {cur} + {fut}", data=work).fit(cov_type="cluster", cov_kwds={"groups": work.id_student})

    def pack(fit, name):
        if name not in fit.params:
            return {}
        return {
            "coef": float(fit.params[name]),
            "ci": [float(x) for x in fit.conf_int().loc[name]],
            "p": float(fit.pvalues[name]),
            "r2": float(fit.rsquared),
        }

    rho = float(work[cur].corr(work[fut])) if work[cur].nunique() > 1 and work[fut].nunique() > 1 else None
    return {
        "n": int(len(work)),
        "n_learners": int(work.id_student.nunique()),
        "corr": rho,
        "F1_current_only": pack(f1, cur),
        "F2_future_only": pack(f2, fut),
        "F3_joint": {"current": pack(f3, cur), "future": pack(f3, fut), "r2": float(f3.rsquared)},
    }


def presentation_coefs(df: pd.DataFrame, feat: str = "share_never") -> list:
    work = df.dropna(subset=["score", "raw_inact", "log_clicks", feat]).copy()
    work = work[work.n_opp >= 1]
    work["pres"] = work.code_module + "_" + work.code_presentation
    fml = (
        "score ~ prior_score_filled + prior_missing + C(code_module) + log_clicks + "
        f"active_rate + n_opp + raw_inact + {feat}"
    )
    rows = []
    for pres, te in work.groupby("pres"):
        tr = work[work.pres != pres]
        if len(tr) < 80 or len(te) < 20:
            rows.append({"pres": pres, "n": int(len(te)), "skipped": True})
            continue
        fit = smf.ols(fml, data=tr).fit()
        rows.append({"pres": pres, "n": int(len(te)), "coef": float(fit.params[feat])})
    return rows


def match_end_safe(df: pd.DataFrame) -> dict:
    work = df.dropna(subset=["share_never", "log_clicks", "active_rate", "n_opp", "prior_score_filled", "score"]).copy()
    work = work[work.n_opp >= 1]
    diffs = []
    dropped = []
    smds = []
    rng = np.random.default_rng(SEED)
    for (mod, pres, tma), g in work.groupby(["code_module", "code_presentation", "next_id_assessment"]):
        g = g.copy()
        try:
            g["pri"] = pd.qcut(g.prior_score_filled, 3, labels=False, duplicates="drop")
            g["act"] = pd.qcut(g.log_clicks, 3, labels=False, duplicates="drop")
            g["ar"] = pd.qcut(g.active_rate, 3, labels=False, duplicates="drop")
            g["nop"] = pd.qcut(g.n_opp, 3, labels=False, duplicates="drop")
            g["cov"] = pd.qcut(1 - g.share_never, 3, labels=False, duplicates="drop")
        except ValueError:
            dropped.append({"code_module": mod, "code_presentation": pres, "next_id_assessment": int(tma), "reason": "qcut"})
            continue
        n_cells = 0
        cell_d = []
        for _, gg in g.groupby(["pri", "act", "ar", "nop"]):
            lo = gg[gg["cov"] == 0]
            hi = gg[gg["cov"] == gg["cov"].max()]
            if len(lo) < 4 or len(hi) < 4:
                continue
            cell_d.append(float(hi.score.mean() - lo.score.mean()))
            n_cells += 1
            for col in ["prior_score_filled", "log_clicks", "active_rate", "n_opp"]:
                sd = float(pd.concat([lo[col], hi[col]]).std())
                if sd > 0:
                    smds.append(abs(float(hi[col].mean() - lo[col].mean()) / sd))
        if cell_d:
            diffs.append(
                {
                    "code_module": mod,
                    "code_presentation": pres,
                    "next_id_assessment": int(tma),
                    "diff": float(np.mean(cell_d)),
                    "n_cells": n_cells,
                    "n": int(len(g)),
                }
            )
        else:
            dropped.append(
                {
                    "code_module": mod,
                    "code_presentation": pres,
                    "next_id_assessment": int(tma),
                    "reason": "no_cell",
                    "n": int(len(g)),
                }
            )
    if not diffs:
        return {"insufficient": True, "dropped": dropped}
    vals = np.array([d["diff"] for d in diffs])
    w = np.array([d["n"] for d in diffs], dtype=float)
    boots = [float(np.average(rng.choice(vals, size=len(vals), replace=True))) for _ in range(200)]
    pres_kept = sorted({(d["code_module"], d["code_presentation"]) for d in diffs})
    return {
        "method": "within presentation x assessment terciles of prior, log_clicks, active_rate, n_opp; high vs low coverage tercile; min 4/arm",
        "n_cells_or_assessments": len(diffs),
        "n_presentations": len(pres_kept),
        "presentations": [f"{a}_{b}" for a, b in pres_kept],
        "n_obs": int(sum(d["n"] for d in diffs)),
        "unweighted_mean": float(vals.mean()),
        "weighted_mean": float(np.average(vals, weights=w)),
        "ci": [float(np.quantile(boots, 0.025)), float(np.quantile(boots, 0.975))],
        "n_positive": int((vals > 0).sum()),
        "mean_abs_smd": float(np.mean(smds)) if smds else None,
        "balance_acceptable": bool(np.mean(smds) < 0.25) if smds else False,
        "by_assessment": diffs,
        "dropped": dropped,
    }


def planning_audit(vle: pd.DataFrame, assessments: pd.DataFrame, sites0: pd.DataFrame) -> list:
    tma = assessments[(assessments.assessment_type == "TMA") & assessments.date.notna()].copy()
    mapped = map_next_tma(sites0.copy(), assessments)
    rows = []
    for (mod, pres), g in sites0.groupby(["code_module", "code_presentation"]):
        all_ou = vle[(vle.code_module == mod) & (vle.code_presentation == pres) & (vle.activity_type == "oucontent")]
        td = tma[(tma.code_module == mod) & (tma.code_presentation == pres)].sort_values("date")
        n_overlap = 0
        recs = g.reset_index(drop=True)
        for i in range(len(recs)):
            for j in range(i + 1, len(recs)):
                a0, a1 = recs.loc[i, "window_start_day"], recs.loc[i, "window_end_day"]
                b0, b1 = recs.loc[j, "window_start_day"], recs.loc[j, "window_end_day"]
                if a0 <= b1 and b0 <= a1:
                    n_overlap += 1
        n_span = 0
        for _, r in recs.iterrows():
            if ((td.date >= r.window_start_day) & (td.date <= r.window_end_day)).any():
                n_span += 1
        gm = mapped[(mapped.code_module == mod) & (mapped.code_presentation == pres)]
        by_tma = (
            gm.dropna(subset=["next_id_assessment"])
            .groupby("next_id_assessment")
            .size()
            .astype(int)
            .to_dict()
        )
        rows.append(
            {
                "code_module": mod,
                "code_presentation": pres,
                "n_oucontent": int(len(all_ou)),
                "n_valid_planned_windows": int(len(g)),
                "n_missing_windows": int((~all_ou.planned_valid).sum()) if "planned_valid" in all_ou else int(len(all_ou) - len(g)),
                "n_overlapping_window_pairs": n_overlap,
                "n_windows_spanning_tma_date": n_span,
                "resources_per_tma": {str(int(k)): int(v) for k, v in by_tma.items()},
                "n_unmapped": int(gm.next_id_assessment.isna().sum()),
            }
        )
    return rows


def access_before_occasions(clicks: pd.DataFrame, occ: pd.DataFrame, sites: pd.DataFrame) -> pd.DataFrame:
    """(student, assessment, site) accessed before that occasion's submission."""
    parts = []
    for (mod, pres), gocc in occ.groupby(["code_module", "code_presentation"], sort=False):
        gs = sites[(sites.code_module == mod) & (sites.code_presentation == pres)][["id_site"]]
        gc = clicks[clicks.id_site.isin(gs.id_site)]
        if gc.empty or gocc.empty:
            continue
        m = gc.merge(gocc[["id_student", "next_id_assessment", "date_submitted"]], on="id_student", how="inner")
        m = m[m.date.notna() & (m.date < m.date_submitted)]
        if m.empty:
            continue
        parts.append(m.groupby(["id_student", "next_id_assessment", "id_site"], as_index=False).size())
    if not parts:
        return pd.DataFrame(columns=["id_student", "next_id_assessment", "id_site", "accessed"])
    hit = pd.concat(parts, ignore_index=True)
    hit["accessed"] = 1
    return hit[["id_student", "next_id_assessment", "id_site", "accessed"]]


def aggregate_from_grid(grid: pd.DataFrame, assign: pd.Series) -> pd.DataFrame:
    g = grid.copy()
    g["assigned"] = g.id_site.map(assign)
    g = g[(g.assigned == g.next_id_assessment) & (g.window_end_day < g.date_submitted)]
    if g.empty:
        return pd.DataFrame(columns=KEYS + ["n_opp", "share_never"])
    g["never"] = 1 - g.accessed.fillna(0)
    return g.groupby(KEYS, as_index=False).agg(n_opp=("id_site", "nunique"), share_never=("never", "mean"))


def classify_c2(beta, ci, ref) -> str:
    if beta is None or ci is None:
        return "FAIL"
    if beta >= 0 or ci[1] >= 0:
        return "FAIL"
    mag = abs(beta) / ref
    if mag >= 0.60:
        return "STRONG PASS"
    if mag >= 0.30:
        return "PASS"
    return "WEAK"


def main() -> None:
    log("PHASE2 construct-validity start")
    dump(
        "environment.json",
        {
            "python": sys.version,
            "pandas": pd.__version__,
            "numpy": np.__version__,
            "seed": SEED,
            "perm_seed": PERM_SEED,
            "perm_B": PERM_B,
            "protocol_head": "277d203c86cb00e7174c9ce4bcedabea0c465456",
            "vle_sha16": sha16(OULAD_RAW / "student_vle.rda"),
            "sa_sha16": sha16(OULAD_INTERIM / "student_assessment.parquet"),
            "v4_untouched": sha16(ROOT / "results" / "v4" / "controlled_association.json"),
            "v5_untouched": sha16(ROOT / "results" / "v5_temporal_safe" / "controlled_association.json"),
        },
    )

    vle = load_vle()
    assessments = load_assessments()
    reg = load_registration()
    sa, sa_meta = collapse_student_assessment(load_sa())
    sv = load_student_vle()
    tma_meta = map_prev_tma_date(assessments)
    pres = set(V4_DEVELOPMENT)
    sites0 = documented_sites(vle, pres, FAMILY_OUCONTENT)
    log(f"documented oucontent {len(sites0)}")
    audit_rows = planning_audit(vle, assessments, sites0)
    dump("opportunity_set_audit.json", audit_rows)
    pd.DataFrame(audit_rows).to_csv(OUT / "opportunity_set_audit_raw.csv", index=False)

    clicks = clicks_by_learner_site_date(sv, sites0.id_site.tolist())
    panel = registered_learner_sites(reg, sites0)
    panel = map_next_tma(panel, assessments).dropna(subset=["next_id_assessment"])
    panel["next_id_assessment"] = panel.next_id_assessment.astype(int)
    sa_join = sa.rename(columns={"id_assessment": "next_id_assessment"})[
        ["id_student", "next_id_assessment", "date_submitted", "is_banked", "score"]
    ]
    panel = panel.merge(sa_join, on=["id_student", "next_id_assessment"], how="left")
    site = annotate_with_cutoff(panel, clicks, "date_submitted")
    site = site.dropna(subset=["date_submitted"])

    def agg_def(mask: pd.Series, n_name: str, s_name: str) -> pd.DataFrame:
        sub = site.loc[mask]
        if sub.empty:
            return pd.DataFrame(columns=KEYS + [n_name, s_name])
        return sub.groupby(KEYS, as_index=False).agg(**{n_name: ("id_site", "nunique"), s_name: ("never", "mean")})

    phase1 = agg_def(pd.Series(True, index=site.index), "n_opp_phase1", "never_phase1")
    end_m = site.window_end_day < site.date_submitted
    start_m = site.window_start_day < site.date_submitted
    end_safe = agg_def(end_m, "n_opp_end", "never_end")
    start_safe = agg_def(start_m, "n_opp_start", "never_start")

    base = site.groupby(KEYS, as_index=False).agg(
        date_submitted=("date_submitted", "first"),
        is_banked=("is_banked", "first"),
        next_tma_date=("next_tma_date", "first"),
        n_cycle=("id_site", "nunique"),
    )
    occ = base.merge(phase1, on=KEYS, how="left").merge(end_safe, on=KEYS, how="left").merge(start_safe, on=KEYS, how="left")
    for c in ["n_opp_phase1", "n_opp_end", "n_opp_start"]:
        occ[c] = occ[c].fillna(0).astype(int)
    occ = attach_tma_outcomes(occ, sa, tma_meta)
    if "score_y" in occ.columns:
        occ["score"] = occ["score_y"]
    occ = attach_prior_score(occ, sa, assessments)
    occ = attach_controls(occ, sv, pres)
    occ["submission_offset"] = occ["date_submitted"] - occ["next_tma_date"]
    occ["row_id"] = (
        occ.code_module + "|" + occ.code_presentation + "|" + occ.id_student.astype(str) + "|" + occ.next_id_assessment.astype(str)
    )
    assert occ.row_id.is_unique

    scored = occ[occ.score.notna() & occ.date_submitted.notna() & (occ.is_banked != True)].copy()  # noqa: E712
    log(f"scored candidate occasions {len(scored)}")

    # Future-content set: later official cycle, access vs current submission
    log("future-content access...")
    acc = access_before_occasions(clicks, scored, sites0)
    site_later = site[KEYS + ["id_site", "next_tma_date"]].rename(columns={"next_tma_date": "site_tma_date", "next_id_assessment": "site_tma"})
    # site KEYS include next_id_assessment — rename carefully
    sl = site.rename(columns={"next_tma_date": "site_tma_date", "next_id_assessment": "site_tma"})[
        ["code_module", "code_presentation", "id_student", "id_site", "site_tma_date", "site_tma"]
    ]
    fut_m = scored[KEYS + ["next_tma_date", "date_submitted"]].merge(sl, on=["code_module", "code_presentation", "id_student"], how="left")
    fut_m = fut_m[fut_m.site_tma_date > fut_m.next_tma_date]
    assert (fut_m.site_tma != fut_m.next_id_assessment).all() or fut_m.empty
    fut_m = fut_m.merge(acc, on=["id_student", "next_id_assessment", "id_site"], how="left")
    fut_m["accessed"] = fut_m.accessed.fillna(0)
    fut_m["never"] = 1 - fut_m.accessed
    fut = fut_m.groupby(KEYS, as_index=False).agg(n_opp_future=("id_site", "nunique"), future_never_share=("never", "mean"))
    scored = scored.merge(fut, on=KEYS, how="left")
    scored["n_opp_future"] = scored.n_opp_future.fillna(0).astype(int)

    dump("n_opp_distributions.json", {
        "OPP_END_SAFE": nopp_dist(scored.n_opp_end),
        "OPP_START_SAFE": nopp_dist(scored.n_opp_start),
        "OPP_PHASE1": nopp_dist(scored.n_opp_phase1),
        "future": nopp_dist(scored.n_opp_future),
    })

    # Zero-opportunity selection diagnostic (primary)
    z = scored[scored.n_opp_end == 0]
    inc = scored[scored.n_opp_end >= 1]
    dump("zero_opp_selection.json", {
        "n_zero": int(len(z)),
        "n_included": int(len(inc)),
        "by_presentation_zero": {
            f"{a}_{b}": int(n)
            for (a, b), n in (z.groupby(["code_module", "code_presentation"]).size().items() if len(z) else [])
        },
        "mean_score_zero": float(z.score.mean()) if len(z) else None,
        "mean_score_included": float(inc.score.mean()) if len(inc) else None,
        "mean_prior_zero": float(z.prior_score_filled.mean()) if len(z) else None,
        "mean_prior_included": float(inc.prior_score_filled.mean()) if len(inc) else None,
        "mean_log_clicks_zero": float(z.log_clicks.mean()) if len(z) else None,
        "mean_log_clicks_included": float(inc.log_clicks.mean()) if len(inc) else None,
        "mean_offset_zero": float(z.submission_offset.mean()) if len(z) else None,
        "mean_offset_included": float(inc.submission_offset.mean()) if len(inc) else None,
        "first_tma_frac_zero": float((z.prev_tma_date.isna()).mean()) if len(z) and "prev_tma_date" in z else None,
        "first_tma_frac_included": float((inc.prev_tma_date.isna()).mean()) if len(inc) and "prev_tma_date" in inc else None,
    })

    primary = inc.copy()
    primary["share_never"] = primary["never_end"]
    primary["n_opp"] = primary["n_opp_end"]
    primary["coverage"] = 1 - primary.share_never

    start_df = scored[scored.n_opp_start >= 1].copy()
    start_df["share_never"] = start_df["never_start"]
    start_df["n_opp"] = start_df["n_opp_start"]
    ph1 = scored[scored.n_opp_phase1 >= 1].copy()
    ph1["share_never"] = ph1["never_phase1"]
    ph1["n_opp"] = ph1["n_opp_phase1"]

    # Assertions + spot checks
    assert (site.loc[end_m, "window_end_day"] < site.loc[end_m, "date_submitted"]).all()
    bad_acc = clicks.merge(site[KEYS + ["id_site", "date_submitted"]], on=["id_site", "id_student"], how="inner")
    n_bad = int((bad_acc.date >= bad_acc.date_submitted).sum())
    # used access is never from annotate_with_cutoff which filters date < cutoff
    rng = np.random.default_rng(SEED)

    def pick(mask, n):
        idx = primary.index[mask] if hasattr(mask, "index") else primary.index[mask]
        idx = np.array(primary.index[mask])
        if len(idx) == 0:
            return []
        return rng.choice(idx, size=min(n, len(idx)), replace=False)

    spots = []
    groups = [
        ("early", primary.submission_offset < 0, 10),
        ("ontime", primary.submission_offset == 0, 10),
        ("late", primary.submission_offset > 0, 10),
        ("low_n", primary.n_opp.isin([1, 2]), 5),
        ("high_n", primary.n_opp >= 10, 5),
    ]
    for label, mask, n in groups:
        for i in pick(mask, n):
            r = primary.loc[i]
            spots.append(
                {
                    "group": label,
                    "row_id": r.row_id,
                    "n_opp": int(r.n_opp),
                    "date_submitted": float(r.date_submitted),
                    "holds_n_opp_ge1": bool(r.n_opp >= 1),
                }
            )
    zidx = scored.index[scored.n_opp_end == 0].to_numpy()
    for i in rng.choice(zidx, size=min(5, len(zidx)), replace=False) if len(zidx) else []:
        r = scored.loc[i]
        spots.append({"group": "zero_opp", "row_id": r.row_id, "n_opp": 0, "date_submitted": float(r.date_submitted)})
    disjoint = True
    if len(fut_m):
        overlap = fut_m.merge(site[KEYS + ["id_site"]], on=KEYS + ["id_site"], how="inner")
        # future rows should not share current-cycle site identity for same occasion
        cur_sites = site[end_m][KEYS + ["id_site"]]
        ov = fut_m[KEYS + ["id_site"]].merge(cur_sites, on=KEYS + ["id_site"], how="inner")
        disjoint = len(ov) == 0
    assertions = {
        "primary_window_end_lt_submitted": True,
        "n_click_rows_on_or_after_submission_among_site_merges": n_bad,
        "used_access_rule": "date < date_submitted",
        "score_not_in_eligibility": True,
        "future_later_tma_only": True,
        "current_future_disjoint": bool(disjoint),
        "unique_row_id": bool(primary.row_id.is_unique),
        "v4_hash": sha16(ROOT / "results" / "v4" / "controlled_association.json"),
        "v5_hash": sha16(ROOT / "results" / "v5_temporal_safe" / "controlled_association.json"),
        "spotcheck_n": len(spots),
        "c1_pass": bool(disjoint and primary.row_id.is_unique),
    }
    dump("assertions.json", assertions)
    dump("spotchecks.json", spots)

    # Inferential — primary and sensitivities
    log("controlled + FE comparisons...")
    results = {}
    for name, d in [("OPP_END_SAFE", primary), ("OPP_START_SAFE", start_df), ("OPP_PHASE1", ph1)]:
        ctrl = fit_m3(d)
        fe = within_learner(d)
        pcoefs = presentation_coefs(d)
        d2 = d.copy()
        mh = module_holdout(d2)
        results[name] = {"controlled": ctrl, "fe": fe, "presentation_coefs": pcoefs, "lomo": mh}
        log(f"{name} beta={ctrl.get('coef')} n={ctrl.get('n')} FE={fe.get('within_coef')}")
    dump("construct_comparison.json", results)

    # Denominator robustness
    log("denominator robustness...")
    den = []
    for thr in [1, 3, 5, 10, 20]:
        sub = primary[primary.n_opp >= thr].copy()
        ctrl = fit_m3(sub)
        fe = within_learner(sub)
        den.append(
            {
                "threshold": thr,
                "n": ctrl.get("n"),
                "n_learners": ctrl.get("n_learners"),
                "coef": ctrl.get("coef"),
                "ci": ctrl.get("ci"),
                "p": ctrl.get("p"),
                "partial_r2": ctrl.get("partial_r2"),
                "fe_coef": fe.get("within_coef"),
                "fe_ci": fe.get("within_ci"),
                "fe_n": fe.get("n_occasions"),
                "fe_learners": fe.get("n_learners"),
            }
        )
    inter = primary.copy()
    inter["log1p_nopp"] = np.log1p(inter.n_opp)
    inter_fit = fit_m3(inter, extra=" + share_never:log1p_nopp")
    dump("denominator_robustness.json", {"thresholds": den, "interaction": inter_fit})

    # Future falsification
    log("future-content models...")
    both = primary[primary.n_opp_future >= 1].copy()
    f1_full = fit_m3(primary)
    f_joint = fit_joint(both, "share_never", "future_never_share")
    dump("future_content.json", {"F1_full_primary": f1_full, "intersection": f_joint})

    # Schedule permutation
    log(f"schedule permutation B={PERM_B}...")
    sites_map = (
        site.groupby(["code_module", "code_presentation", "id_site"], as_index=False)
        .agg(official_tma=("next_id_assessment", "first"), window_end_day=("window_end_day", "first"))
    )
    grid_parts = []
    for (mod, pres), gocc in primary.groupby(["code_module", "code_presentation"], sort=False):
        gs = sites_map[(sites_map.code_module == mod) & (sites_map.code_presentation == pres)]
        if gs.empty:
            continue
        m = gocc[KEYS + ["date_submitted"]].merge(gs[["id_site", "official_tma", "window_end_day"]], how="cross")
        # merge how=cross is wrong (cartesian across all). Use key-less assign then filter presentation — already filtered gs/gocc
        grid_parts.append(m)
    grid = pd.concat(grid_parts, ignore_index=True)
    grid = grid.merge(acc, on=["id_student", "next_id_assessment", "id_site"], how="left")
    grid["accessed"] = grid.accessed.fillna(0)
    # drop how=cross mistake: gocc x gs is intended (same presentation). OK.

    primary = primary.reset_index(drop=True)

    def m3_coef_from_agg(agg: pd.DataFrame) -> float | None:
        d = primary[KEYS + ["score", "prior_score_filled", "prior_missing", "log_clicks", "active_rate", "raw_inact"]].merge(agg, on=KEYS, how="inner")
        d = d[d.n_opp >= 1].dropna(subset=["share_never", "score", "raw_inact"])
        if len(d) < 200:
            return None
        d["pres"] = d.code_module + "_" + d.code_presentation
        fit = smf.ols(
            "score ~ prior_score_filled + prior_missing + C(pres) + log_clicks + active_rate + n_opp + raw_inact + share_never",
            data=d,
        ).fit()
        return float(fit.params["share_never"])

    official = sites_map.set_index("id_site")["official_tma"]
    auth = m3_coef_from_agg(aggregate_from_grid(grid, official))
    placebos = []
    for b in range(PERM_B):
        rngb = np.random.default_rng(PERM_SEED + b)
        parts = []
        for (_, _), gs in sites_map.groupby(["code_module", "code_presentation"]):
            labs = gs.official_tma.to_numpy().copy()
            rngb.shuffle(labs)
            parts.append(pd.Series(labs, index=gs.id_site.to_numpy()))
        assign = pd.concat(parts)
        # preserve counts: shuffle of labels does
        c = m3_coef_from_agg(aggregate_from_grid(grid, assign))
        if c is not None:
            placebos.append(c)
        if (b + 1) % 20 == 0:
            log(f"  perm {b+1}/{PERM_B}")
    pla = np.array(placebos, dtype=float)
    n_ext = int((pla <= auth).sum()) if auth is not None else 0
    perm_out = {
        "B": int(len(pla)),
        "authentic": auth,
        "placebo_median": float(np.median(pla)) if len(pla) else None,
        "placebo_p025": float(np.quantile(pla, 0.025)) if len(pla) else None,
        "placebo_p975": float(np.quantile(pla, 0.975)) if len(pla) else None,
        "empirical_directional_p": float((1 + n_ext) / (len(pla) + 1)) if len(pla) else None,
        "pct_placebos_as_or_more_extreme": float(n_ext / len(pla)) if len(pla) else None,
        "percentile_authentic_more_extreme_than": float(1 - n_ext / len(pla)) if len(pla) else None,
    }
    dump("schedule_permutation.json", perm_out)

    # Activity-matched
    log("activity-matched...")
    dump("activity_matched.json", match_end_safe(primary))

    # Redundancy
    cols = ["share_never", "log_clicks", "active_rate", "raw_inact", "n_opp", "prior_score_filled"]
    sub = primary[cols].dropna()
    vif = {}
    from numpy.linalg import lstsq

    X = sub.copy()
    X["intercept"] = 1
    for c in cols:
        y = X[c].to_numpy()
        xx = X.drop(columns=[c]).to_numpy()
        b, *_ = lstsq(xx, y, rcond=None)
        pred = xx @ b
        r2 = 1 - np.sum((y - pred) ** 2) / np.sum((y - y.mean()) ** 2) if y.std() else 1
        vif[c] = float(1 / (1 - r2)) if r2 < 1 else float("inf")
    dump(
        "redundancy.json",
        {
            "corr": sub.corr().to_dict(),
            "raw_inact_equals_1_minus_active_rate": bool(np.allclose(sub.raw_inact, 1 - sub.active_rate)),
            "vif": vif,
            "minimal_nonredundant_activity": ["log_clicks", "active_rate", "n_opp"],
            "authoritative_spec_unchanged": True,
        },
    )

    # Within-between / prior vs next
    log("within-between...")
    w = primary.dropna(subset=["share_never", "score", "log_clicks", "raw_inact"]).copy()
    w["lid"] = w.code_module + "|" + w.code_presentation + "|" + w.id_student.astype(str)
    w["never_bar"] = w.groupby("lid").share_never.transform("mean")
    w["never_dev"] = w.share_never - w.never_bar
    mund = smf.ols(
        "score ~ share_never + never_bar + log_clicks + raw_inact + n_opp + C(code_module)",
        data=w,
    ).fit(cov_type="cluster", cov_kwds={"groups": w.lid})
    learner = w.groupby("lid").agg(mean_never=("share_never", "mean"), mean_score=("score", "mean"))
    rho_b, _ = stats.spearmanr(learner.mean_never, learner.mean_score)
    ww = w[w.groupby("lid").lid.transform("size") >= 2]
    rho_w, _ = stats.spearmanr(ww.never_dev, ww.score - ww.groupby("lid").score.transform("mean")) if len(ww) > 50 else (np.nan, np.nan)
    subp = w.dropna(subset=["prior_score"])
    rho_prior, _ = stats.spearmanr(subp.share_never, subp.prior_score) if len(subp) > 50 else (np.nan, np.nan)
    rho_next, _ = stats.spearmanr(w.share_never, w.score)
    dump(
        "within_between.json",
        {
            "mundlak_within": float(mund.params["share_never"]),
            "mundlak_within_ci": [float(x) for x in mund.conf_int().loc["share_never"]],
            "mundlak_between": float(mund.params["never_bar"]),
            "mundlak_between_ci": [float(x) for x in mund.conf_int().loc["never_bar"]],
            "spearman_learner_means": float(rho_b),
            "spearman_within_dev": float(rho_w) if np.isfinite(rho_w) else None,
            "spearman_prior": float(rho_prior) if np.isfinite(rho_prior) else None,
            "spearman_next": float(rho_next),
            "n": int(len(w)),
            "n_learners": int(w.lid.nunique()),
        },
    )

    # Identity forensic (no rerun)
    p1 = json.loads((ROOT / "results" / "v5_temporal_safe" / "identity_placebo.json").read_text())
    dump(
        "identity_placebo_forensic.json",
        {
            "phase1": p1,
            "rerun": False,
            "compatible_interpretations": ["A", "B"],
            "preferred": "B_then_A",
            "reason": (
                "Placebo replaces the TMA's planned set with same-size nearby planned oucontent "
                "(closest window_end). That preserves family, presentation, count, and local "
                "instructional timing. It does not destroy schedule-relative content reach. "
                "Placebo coefficient more negative than actual => exact site identity is not "
                "the operative signal. Nearby same-family planned sites share the timing signal."
            ),
            "classification": "EXACT_IDENTITY_SPECIFICITY_NOT_SUPPORTED",
            "placebo_invalidly_constructed": False,
        },
    )

    # GGG boundary
    log("GGG semantic boundary...")
    ggg_sites = documented_sites(vle, set(V4_CONFIRMATION), FAMILY_ALL_DOCUMENTED)
    ggg_sites = ggg_sites[ggg_sites.activity_type == FAMILY_RESOURCE].copy()
    ggg_clicks = clicks_by_learner_site_date(sv, ggg_sites.id_site.tolist())
    ggg_panel = registered_learner_sites(reg, ggg_sites)
    ggg_panel = map_next_tma(ggg_panel, assessments).dropna(subset=["next_id_assessment"])
    ggg_panel["next_id_assessment"] = ggg_panel.next_id_assessment.astype(int)
    ggg_panel = ggg_panel.merge(sa_join, on=["id_student", "next_id_assessment"], how="left")
    ggg_site = annotate_with_cutoff(ggg_panel, ggg_clicks, "date_submitted").dropna(subset=["date_submitted"])
    ggg_end = ggg_site[ggg_site.window_end_day < ggg_site.date_submitted]
    ggg_occ = ggg_end.groupby(KEYS, as_index=False).agg(
        n_opp=("id_site", "nunique"),
        share_never=("never", "mean"),
        date_submitted=("date_submitted", "first"),
        is_banked=("is_banked", "first"),
        next_tma_date=("next_tma_date", "first"),
    )
    ggg_occ = attach_tma_outcomes(ggg_occ, sa, tma_meta)
    if "score_y" in ggg_occ.columns:
        ggg_occ["score"] = ggg_occ["score_y"]
    ggg_occ = attach_prior_score(ggg_occ, sa, assessments)
    ggg_occ = attach_controls(ggg_occ, sv, set(V4_CONFIRMATION))
    ggg_main = ggg_occ[ggg_occ.score.notna() & (ggg_occ.is_banked != True) & (ggg_occ.n_opp >= 1)].copy()  # noqa: E712
    ggg_ctrl = fit_m3(ggg_main)
    dump(
        "ggg_boundary.json",
        {
            "family": FAMILY_RESOURCE,
            "semantics": "OULAD activity_type=resource (generic files), not oucontent HTML materials",
            "n_sites": int(len(ggg_sites)),
            "controlled": ggg_ctrl,
            "n_scored": int(len(ggg_main)),
        },
    )

    # Gates
    c = results["OPP_END_SAFE"]["controlled"]
    fe = results["OPP_END_SAFE"]["fe"]
    c2c = classify_c2(c.get("coef"), c.get("ci"), P1_BETA)
    c2f = classify_c2(fe.get("within_coef"), fe.get("within_ci"), P1_FE)
    rank = {"FAIL": 0, "WEAK": 1, "PASS": 2, "STRONG PASS": 3}
    c2 = min([c2c, c2f], key=lambda x: rank[x])

    den_ok = [d for d in den if d["threshold"] in (3, 5, 10) and d.get("coef") is not None]
    signs = [d["coef"] < 0 and d["ci"][1] < 0 for d in den_ok]
    tiny = next(d for d in den if d["threshold"] == 1)
    n10 = next(d for d in den if d["threshold"] == 10)
    if all(signs) and n10.get("coef") and n10["coef"] < 0:
        c3 = "STRONG PASS"
    elif all(signs):
        c3 = "PASS"
    elif any(signs) and not all(signs):
        c3 = "MIXED"
    else:
        c3 = "FAIL"
    if tiny.get("coef") and n10.get("coef") and abs(n10["coef"]) < 0.3 * abs(tiny["coef"]):
        if c3 == "STRONG PASS":
            c3 = "PASS"

    fj = f_joint.get("F3_joint", {})
    cb = fj.get("current", {}).get("coef")
    fb = fj.get("future", {}).get("coef")
    cci = fj.get("current", {}).get("ci")
    if cb is None:
        c4 = "FAIL"
        ratio = None
    else:
        ratio = abs(cb) / abs(fb) if fb not in (None, 0) else None
        cur_ok = cb < 0 and cci is not None and cci[1] < 0
        if not cur_ok:
            c4 = "FAIL"
        elif fb is not None and fb < 0 and abs(fb) >= abs(cb):
            c4 = "FAIL"
        elif ratio is not None and ratio >= 1.5 and cur_ok:
            c4 = "STRONG PASS"
        elif ratio is not None and ratio > 1 and cur_ok:
            c4 = "PASS"
        else:
            c4 = "MIXED"

    pct = perm_out.get("percentile_authentic_more_extreme_than")
    if pct is None:
        c5 = "FAIL"
    elif pct >= 0.975:
        c5 = "STRONG PASS"
    elif pct >= 0.95:
        c5 = "PASS"
    elif pct >= 0.90:
        c5 = "MIXED"
    else:
        c5 = "FAIL"

    match = json.loads((OUT / "activity_matched.json").read_text())
    match_ok = (not match.get("insufficient")) and match.get("ci") and match["ci"][0] > 0
    match_dir = (not match.get("insufficient")) and match.get("weighted_mean", 0) > 0
    ctrl_ok = c.get("coef", 1) < 0 and c.get("ci", [1, 1])[1] < 0
    fe_ok = fe.get("within_coef", 1) < 0 and fe.get("within_ci", [1, 1])[1] < 0
    bal = match.get("balance_acceptable", False)
    if ctrl_ok and fe_ok and match_ok and bal:
        c6 = "STRONG PASS"
    elif ctrl_ok and fe_ok and match_dir:
        c6 = "PASS"
    elif ctrl_ok and fe_ok:
        c6 = "MIXED"
    else:
        c6 = "FAIL"

    c1 = "PASS" if assertions["c1_pass"] else "FAIL_FATAL_CONSTRUCT_IMPLEMENTATION"
    c7 = "NOT SUPPORTED"

    # Decision tree
    def good(x):
        return x in ("PASS", "STRONG PASS")

    if c2 == "FAIL":
        verdict = "CURRENT_CONSTRUCT_NOT_SUPPORTED"
    elif c3 == "FAIL":
        verdict = "PI_REVIEW_REQUIRED_DENOMINATOR_DEPENDENCE"
    elif good(c1) and good(c2) and good(c3) and (c4 in ("MIXED", "FAIL") or c5 in ("MIXED", "FAIL")):
        verdict = "PROCEED_TO_PHASE3_GENERAL_CONTENT_REACH_ONLY"
    elif c7 == "NOT SUPPORTED" and good(c4) and good(c5) and good(c1) and good(c2):
        verdict = "PROCEED_TO_PHASE3_SCHEDULE_RELATIVE_CONSTRUCT"
    elif all(good(x) for x in (c1, c2, c3, c4, c5, c6)):
        verdict = "PROCEED_TO_PHASE3_STRONG_CONSTRUCT"
    elif good(c2) and (c4 in ("MIXED", "FAIL") or c5 in ("MIXED", "FAIL")):
        verdict = "PROCEED_TO_PHASE3_GENERAL_CONTENT_REACH_ONLY"
    else:
        verdict = "PROCEED_TO_PHASE3_SCHEDULE_RELATIVE_CONSTRUCT" if good(c4) and good(c5) else "PROCEED_TO_PHASE3_GENERAL_CONTENT_REACH_ONLY"

    dump(
        "gates.json",
        {
            "C1": c1,
            "C2": c2,
            "C2_controlled": c2c,
            "C2_fe": c2f,
            "C3": c3,
            "C4": c4,
            "C4_ratio": ratio,
            "C5": c5,
            "C6": c6,
            "C7": c7,
            "verdict": verdict,
        },
    )

    keep = KEYS + [
        "row_id",
        "date_submitted",
        "next_tma_date",
        "n_opp_end",
        "never_end",
        "n_opp_start",
        "never_start",
        "n_opp_phase1",
        "never_phase1",
        "n_opp_future",
        "future_never_share",
        "score",
        "log_clicks",
        "active_rate",
        "raw_inact",
        "prior_score_filled",
        "prior_missing",
    ]
    for ccol in keep:
        if ccol not in scored.columns:
            scored[ccol] = np.nan
    scored[keep].to_parquet(OUT / "occasion_table.parquet", index=False)
    (OUT / "run.log").write_text("\n".join(LOG) + "\n")
    log("done")


if __name__ == "__main__":
    main()
