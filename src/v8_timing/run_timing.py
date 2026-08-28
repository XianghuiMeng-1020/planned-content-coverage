#!/usr/bin/env python3
"""Phase-4 timing theory. Writes only results/v8_timing/. Uses OPP_END_SAFE from v6."""
from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

_SRC = Path(__file__).resolve().parents[1]
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from common.paths import ROOT  # noqa: E402
from v2.load import load_assessments, load_registration, load_student_vle, load_vle  # noqa: E402
from v2.panel import documented_sites, map_next_tma, registered_learner_sites  # noqa: E402
from v3.constants import FAMILY_OUCONTENT  # noqa: E402
from v3.features import clicks_by_learner_site_date  # noqa: E402
from v4.constants import V4_DEVELOPMENT  # noqa: E402
from v5_temporal.run_temporal_safe import annotate_with_cutoff, collapse_student_assessment  # noqa: E402

OUT = ROOT / "results" / "v8_timing"
FIG = OUT / "figures"
SRC_TAB = ROOT / "results" / "v6_construct_validity" / "occasion_table.parquet"
FORBIDDEN = [
    ROOT / "results" / "v4",
    ROOT / "results" / "v5_temporal_safe",
    ROOT / "results" / "v6_construct_validity",
    ROOT / "results" / "v7_inference",
]
SEED = 20260829
KEYS = ["code_module", "code_presentation", "id_student", "next_id_assessment"]
CTRL = "prior_score_filled + prior_missing + C(pres) + log_clicks + active_rate + n_opp"
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


def pack_fit(fit, feat: str) -> dict:
    out = {"n": int(fit.nobs), "r2": float(fit.rsquared)}
    if feat in fit.params:
        out["coef"] = float(fit.params[feat])
        out["se"] = float(fit.bse[feat]) if feat in fit.bse else None
        out["p"] = float(fit.pvalues[feat]) if feat in fit.pvalues else None
        if feat in fit.conf_int().index:
            out["ci"] = [float(x) for x in fit.conf_int().loc[feat]]
    return out


def state_means(df: pd.DataFrame, col: str, score="score") -> list[dict]:
    rows = []
    for st, g in df.groupby(col, dropna=False):
        rows.append(
            {
                "state": str(st),
                "n": int(len(g)),
                "mean_score": float(g[score].mean()),
                "sd_score": float(g[score].std()),
            }
        )
    return sorted(rows, key=lambda r: r["state"])


def contrast(means: dict, a: str, b: str):
    if a not in means or b not in means:
        return None
    return float(means[a] - means[b])


def load_sa() -> pd.DataFrame:
    from common.paths import OULAD_INTERIM

    return pd.read_parquet(OULAD_INTERIM / "student_assessment.parquet")


def rebuild_end_safe_sites(primary: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    log("rebuild OPP_END_SAFE site panel...")
    vle = load_vle()
    assessments = load_assessments()
    reg = load_registration()
    sa, _ = collapse_student_assessment(load_sa())
    sv = load_student_vle()
    sites0 = documented_sites(vle, set(V4_DEVELOPMENT), FAMILY_OUCONTENT)
    clicks = clicks_by_learner_site_date(sv, sites0.id_site.tolist())
    panel = registered_learner_sites(reg, sites0)
    panel = map_next_tma(panel, assessments).dropna(subset=["next_id_assessment"])
    panel["next_id_assessment"] = panel.next_id_assessment.astype(int)
    sa_join = sa.rename(columns={"id_assessment": "next_id_assessment"})[
        ["id_student", "next_id_assessment", "date_submitted", "is_banked"]
    ]
    panel = panel.merge(sa_join, on=["id_student", "next_id_assessment"], how="left")
    site = annotate_with_cutoff(panel, clicks, "date_submitted")
    site = site.dropna(subset=["date_submitted"])
    end_m = site.window_end_day < site.date_submitted
    site = site.loc[end_m].copy()
    site = site.merge(primary[KEYS + ["score"]], on=KEYS, how="inner")
    audit = {
        "n_documented_sites": int(len(sites0)),
        "n_end_safe_rows_after_primary_join": int(len(site)),
        "n_primary_keys_with_sites": int(site.groupby(KEYS).ngroups),
    }
    return site, audit


def occasion_from_end_safe(site: pd.DataFrame, primary: pd.DataFrame) -> pd.DataFrame:
    work = site.copy()
    work["is_early"] = (work.first_state == "early").astype(int)
    work["is_ontime"] = (work.first_state == "ontime").astype(int)
    work["is_late"] = (work.first_state == "late").astype(int)
    work["is_never"] = (work.first_state == "never").astype(int)
    work["late_catchup"] = ((work.any_late == 1) & (work.any_ontime == 0)).astype(int)
    accessed = work[work.first_state != "never"]

    def _mode(s: pd.Series):
        if s.empty:
            return "never"
        vc = s.value_counts()
        if len(vc) >= 2 and vc.iloc[0] == vc.iloc[1]:
            return "tie"
        return str(vc.index[0])

    shares = work.groupby(KEYS, as_index=False).agg(
        n_opp_chk=("id_site", "size"),
        share_early=("is_early", "mean"),
        share_ontime=("is_ontime", "mean"),
        share_late=("is_late", "mean"),
        share_never_chk=("is_never", "mean"),
        late_catchup_share=("late_catchup", "mean"),
        first_date_min=("first_date", "min"),
    )
    modes = (
        accessed.groupby(KEYS, as_index=False)
        .agg(occ_mode=("first_state", _mode))
        if len(accessed)
        else pd.DataFrame(columns=KEYS + ["occ_mode"])
    )
    occ = primary.merge(shares, on=KEYS, how="left").merge(modes, on=KEYS, how="left")
    occ["occ_state"] = np.where(occ.share_never >= 1.0 - 1e-12, "never", occ.occ_mode.fillna("never"))
    occ["days_first_to_submit"] = occ.date_submitted - occ.first_date_min
    occ["high_never"] = (occ.share_never >= 0.5).astype(int)
    occ["any_late"] = (occ.share_late > 0).astype(int)
    return occ


def t1_gate(site_c: dict, modal_c: dict) -> str:
    def hold(c):
        if c["late_minus_ontime"] is None or c["late_minus_never"] is None:
            return False
        lo = abs(c["late_minus_ontime"])
        ln = abs(c["late_minus_never"])
        signed = c["late_minus_never"]  # late - never; expect positive
        return lo < 0.5 * ln and signed > 3

    s, m = hold(site_c), hold(modal_c)
    if s and m:
        return "STRONG PASS"
    if s:
        return "PASS"
    if s != m:
        return "MIXED"
    return "FAIL"


def t2_gate(late_ci, never_is_low: bool, late_ontime_small: bool, late_remains_large: bool) -> str:
    late_null = late_ci is not None and late_ci[0] <= 0 <= late_ci[1]
    if late_null and never_is_low:
        return "STRONG PASS"
    if late_ontime_small and never_is_low:
        return "PASS"
    if late_remains_large:
        return "FAIL"
    return "MIXED"


def t3_gate(bn, be, bl, late_ci) -> str:
    if bn is None:
        return "FAIL"
    mx = max(abs(be or 0), abs(bl or 0))
    late_null = late_ci is not None and late_ci[0] <= 0 <= late_ci[1]
    late_small = bl is not None and abs(bl) < 1
    never_dom = abs(bn) > 2 * mx if mx > 0 else abs(bn) > 0
    if never_dom and (late_null or late_small) and bn < 0:
        return "STRONG PASS"
    if never_dom and bn < 0:
        return "PASS"
    if bl is not None and bn is not None and abs(bl) >= 0.75 * abs(bn):
        return "FAIL"
    return "MIXED"


def main() -> None:
    log("PHASE4 timing start")
    primary = pd.read_parquet(SRC_TAB)
    primary = primary[primary.n_opp_end >= 1].copy()
    primary["share_never"] = primary["never_end"]
    primary["n_opp"] = primary["n_opp_end"]
    primary["pres"] = primary.code_module + "_" + primary.code_presentation
    primary["lid"] = primary.code_module + "|" + primary.code_presentation + "|" + primary.id_student.astype(str)
    assert len(primary) == 38662, len(primary)
    log(f"primary n={len(primary)}")

    site, audit = rebuild_end_safe_sites(primary)
    site["late_catchup"] = ((site.any_late == 1) & (site.any_ontime == 0)).astype(int)
    occ = occasion_from_end_safe(site, primary)
    # integrity: reconstructed never share vs v6
    delta = (occ.share_never_chk - occ.share_never).abs()
    audit["never_share_max_abs_delta"] = float(delta.max())
    audit["never_share_mean_abs_delta"] = float(delta.mean())
    assert float(delta.max()) < 1e-9, float(delta.max())
    audit["n_ties"] = int((occ.occ_state == "tie").sum())
    dump("site_audit.json", audit)

    site_m = {r["state"]: r["mean_score"] for r in state_means(site, "first_state")}
    site_c = {
        "unit": "U1_site",
        "by_state": state_means(site, "first_state"),
        "ontime_minus_never": contrast(site_m, "ontime", "never"),
        "late_minus_never": contrast(site_m, "late", "never"),
        "late_minus_ontime": contrast(site_m, "late", "ontime"),
        "early_minus_ontime": contrast(site_m, "early", "ontime"),
        "legacy_v3": {"early": 78.43, "ontime": 77.02, "late": 76.54, "never": 70.57},
    }
    modal = occ[occ.occ_state.isin(["early", "ontime", "late", "never"])].copy()
    modal_m = {r["state"]: r["mean_score"] for r in state_means(modal, "occ_state")}
    modal_comp = (
        modal.groupby("occ_state")
        .agg(n=("score", "size"), mean_score=("score", "mean"), mean_never=("share_never", "mean"))
        .reset_index()
    )
    modal_c = {
        "unit": "U3_modal",
        "n_ties_excluded": int((occ.occ_state == "tie").sum()),
        "by_state": state_means(modal, "occ_state"),
        "mean_never_share_by_state": {
            str(r.occ_state): float(r.mean_never) for r in modal_comp.itertuples(index=False)
        },
        "ontime_minus_never": contrast(modal_m, "ontime", "never"),
        "late_minus_never": contrast(modal_m, "late", "never"),
        "late_minus_ontime": contrast(modal_m, "late", "ontime"),
        "early_minus_ontime": contrast(modal_m, "early", "ontime"),
    }
    dump("unadjusted_gradient.json", {"site": site_c, "modal": modal_c, "shares_mean": {
        "early": float(occ.share_early.mean()),
        "ontime": float(occ.share_ontime.mean()),
        "late": float(occ.share_late.mean()),
        "never": float(occ.share_never.mean()),
    }})

    log("T2 composition + modal adjusted...")
    mfit = smf.ols(
        f"score ~ C(occ_state, Treatment('ontime')) + share_never + {CTRL}",
        data=modal,
    ).fit(cov_type="cluster", cov_kwds={"groups": modal.id_student})
    munadj = smf.ols(
        "score ~ C(occ_state, Treatment('ontime'))",
        data=modal,
    ).fit(cov_type="cluster", cov_kwds={"groups": modal.id_student})
    t2 = {
        "unadjusted_modal": {
            k: pack_fit(munadj, k) for k in munadj.params.index if k.startswith("C(occ_state")
        },
        "adjusted_plus_never": {
            k: pack_fit(mfit, k) for k in mfit.params.index if k.startswith("C(occ_state") or k == "share_never"
        },
        "composition": modal_c["mean_never_share_by_state"],
    }
    dump("composition.json", t2)

    log("T3 share model...")
    t3_fit = smf.ols(
        f"score ~ share_early + share_late + share_never + {CTRL}",
        data=occ,
    ).fit(cov_type="cluster", cov_kwds={"groups": occ.id_student})
    t3 = {k: pack_fit(t3_fit, k) for k in ["share_early", "share_late", "share_never"]}
    t3["r2"] = float(t3_fit.rsquared)
    t3["n"] = int(t3_fit.nobs)
    dump("adjusted_shares.json", t3)

    log("T4 within-learner...")
    keep = set(occ.groupby("lid").size().pipe(lambda s: s[s >= 2].index))
    d = occ[occ.lid.isin(keep)].copy()
    for c in ["score", "share_early", "share_late", "share_never", "log_clicks", "active_rate", "n_opp"]:
        d[c + "_w"] = d[c] - d.groupby("lid")[c].transform("mean")
    d["tma_c"] = d.groupby("lid").next_tma_date.transform(lambda s: (s - s.mean()) / (s.std() + 1e-6))
    fe = smf.ols(
        "score_w ~ share_early_w + share_late_w + share_never_w + log_clicks_w + active_rate_w + n_opp_w + tma_c",
        data=d,
    ).fit(cov_type="cluster", cov_kwds={"groups": d.lid})
    t4 = {
        "n_occasions": int(len(d)),
        "n_learners": int(d.lid.nunique()),
        **{k: pack_fit(fe, k + "_w") for k in ["share_early", "share_late", "share_never"]},
    }
    dump("within_learner_timing.json", t4)

    log("T5 first_state vs late_catchup...")
    lc = site.copy()
    lc["late_def"] = np.where(lc.first_state == "never", "never", np.where(lc.late_catchup == 1, "late_catchup", lc.first_state))
    # late_catchup rows are a subset of late-or-multi; keep first_state late separately
    t5 = {
        "first_state": state_means(site, "first_state"),
        "late_catchup_vs_other": state_means(lc, "late_def"),
        "n_first_state_late": int((site.first_state == "late").sum()),
        "n_late_catchup": int(site.late_catchup.sum()),
        "n_late_and_also_ontime": int(((site.first_state == "late") & (site.any_ontime == 1)).sum()),
    }
    dump("late_definitions.json", t5)

    log("T6 recency...")
    rec = occ[occ.share_never < 1].dropna(subset=["days_first_to_submit"]).copy()
    t6_fit = smf.ols(
        f"score ~ days_first_to_submit + share_never + {CTRL}",
        data=rec,
    ).fit(cov_type="cluster", cov_kwds={"groups": rec.id_student})
    t6 = {
        "n": int(len(rec)),
        "median_days_first_to_submit": float(rec.days_first_to_submit.median()),
        "days": pack_fit(t6_fit, "days_first_to_submit"),
        "share_never": pack_fit(t6_fit, "share_never"),
    }
    dump("recency.json", t6)

    log("T7 decision utility...")
    base = smf.ols(f"score ~ high_never + {CTRL}", data=occ).fit()
    full = smf.ols(f"score ~ high_never + any_late + {CTRL}", data=occ).fit()
    cells = (
        occ.assign(cell=np.where(occ.high_never == 1, "high_never", "low_never") + np.where(occ.any_late == 1, "|late", "|no_late"))
        .groupby("cell")
        .agg(n=("score", "size"), mean_score=("score", "mean"), mean_never=("share_never", "mean"))
        .reset_index()
    )
    t7 = {
        "incremental_r2": float(full.rsquared - base.rsquared),
        "any_late_given_high_never": pack_fit(
            smf.ols(f"score ~ high_never + any_late + {CTRL}", data=occ).fit(
                cov_type="cluster", cov_kwds={"groups": occ.id_student}
            ),
            "any_late",
        ),
        "cells": cells.to_dict(orient="records"),
        "threshold": 0.005,
    }
    dump("decision_utility.json", t7)

    log("presentation site gradient...")
    pres_rows = []
    for (mod, pres), g in site.groupby(["code_module", "code_presentation"]):
        mm = g.groupby("first_state").score.mean()
        pres_rows.append(
            {
                "code_module": mod,
                "code_presentation": pres,
                "n": int(len(g)),
                "ontime_minus_never": float(mm.get("ontime", np.nan) - mm.get("never", np.nan)),
                "late_minus_never": float(mm.get("late", np.nan) - mm.get("never", np.nan)),
                "late_minus_ontime": float(mm.get("late", np.nan) - mm.get("ontime", np.nan)),
                "ontime_gt_never": bool(mm.get("ontime", np.nan) > mm.get("never", np.nan)),
            }
        )
    dump("presentation_site_gradient.json", {"rows": pres_rows, "n_ontime_gt_never": int(sum(r["ontime_gt_never"] for r in pres_rows))})

    late_ci = t2["adjusted_plus_never"].get("C(occ_state, Treatment('ontime'))[T.late]", {}).get("ci")
    never_ci = t2["adjusted_plus_never"].get("C(occ_state, Treatment('ontime'))[T.never]", {}).get("ci")
    never_coef = t2["adjusted_plus_never"].get("C(occ_state, Treatment('ontime'))[T.never]", {}).get("coef")
    late_coef = t2["adjusted_plus_never"].get("C(occ_state, Treatment('ontime'))[T.late]", {}).get("coef")
    never_is_low = never_coef is not None and never_coef < 0 and never_ci is not None and never_ci[1] < 0
    late_ontime_small = late_coef is not None and abs(late_coef) < 2
    late_remains_large = late_coef is not None and abs(late_coef) >= 3 and (late_ci is None or late_ci[1] < 0 or late_ci[0] > 0)

    g1 = t1_gate(site_c, modal_c)
    g2 = t2_gate(late_ci, never_is_low, late_ontime_small, late_remains_large)
    g3 = t3_gate(t3.get("share_never", {}).get("coef"), t3.get("share_early", {}).get("coef"), t3.get("share_late", {}).get("coef"), t3.get("share_late", {}).get("ci"))
    fe_never = t4.get("share_never", {}).get("coef")
    fe_never_ci = t4.get("share_never", {}).get("ci")
    fe_late = t4.get("share_late", {}).get("coef")
    fe_late_ci = t4.get("share_late", {}).get("ci")
    fe_never_ok = fe_never is not None and fe_never < 0 and fe_never_ci is not None and fe_never_ci[1] < 0
    fe_late_small = fe_late is None or abs(fe_late) < 1 or (fe_late_ci is not None and fe_late_ci[0] <= 0 <= fe_late_ci[1])
    g4 = "PASS" if fe_never_ok and fe_late_small else ("MIXED" if fe_never_ok else "FAIL")

    fs = {r["state"]: r["mean_score"] for r in t5["first_state"]}
    lcd = {r["state"]: r["mean_score"] for r in t5["late_catchup_vs_other"]}
    d_fs = (fs.get("late") - fs.get("never")) if "late" in fs and "never" in fs else None
    d_lc = (lcd.get("late_catchup") - lcd.get("never")) if "late_catchup" in lcd and "never" in lcd else None
    g5 = "PASS" if d_fs is not None and d_lc is not None and abs(d_fs - d_lc) <= 2 else "MIXED"

    rec_ci = t6["days"].get("ci")
    t6_excl0 = rec_ci is not None and not (rec_ci[0] <= 0 <= rec_ci[1])
    g6 = "RECENCY_REMAINS" if t6_excl0 else "NO_RECENCY_INCREMENT"

    g7 = "SUPPORTED" if t7["incremental_r2"] > 0.005 else "NOT SUPPORTED"

    def ok(x):
        return x in ("PASS", "STRONG PASS")

    if g1 == "FAIL":
        verdict = "WITHDRAW_RQ4_UNADJUSTED_GRADIENT"
    elif g1 == "MIXED":
        verdict = "PI_REVIEW_REQUIRED_UNIT_DEPENDENCE"
    elif (g3 in ("MIXED", "FAIL") or g4 in ("MIXED", "FAIL")) and not (ok(g3) and ok(g4)):
        if g3 == "FAIL" or g4 == "FAIL":
            verdict = "REWRITE_RQ4_TIMING_INFORMATIVE"
        elif ok(g2) and ok(g3):
            verdict = "PROCEED_TO_PHASE5_REACH_NOT_SCHEDULE"
        else:
            verdict = "REWRITE_RQ4_TIMING_INFORMATIVE"
    elif ok(g1) and ok(g2) and ok(g3):
        verdict = "PROCEED_TO_PHASE5_REACH_NOT_SCHEDULE"
    else:
        verdict = "REWRITE_RQ4_TIMING_INFORMATIVE"

    gates = {"T1": g1, "T2": g2, "T3": g3, "T4": g4, "T5": g5, "T6": g6, "T7": g7, "verdict": verdict}
    dump("gates.json", gates)

    FIG.mkdir(parents=True, exist_ok=True)
    order = ["early", "ontime", "late", "never"]
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    x = np.arange(len(order))
    site_vals = [site_m.get(s, np.nan) for s in order]
    modal_vals = [modal_m.get(s, np.nan) for s in order]
    ax.bar(x - 0.18, site_vals, 0.36, label="U1 site-row")
    ax.bar(x + 0.18, modal_vals, 0.36, label="U3 occasion modal")
    ax.set_xticks(x)
    ax.set_xticklabels(order)
    ax.set_ylabel("Mean next-TMA score")
    ax.legend()
    ax.set_ylim(60, 85)
    fig.tight_layout()
    fig.savefig(FIG / "figA_state_means.png", dpi=140)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.4, 3.6))
    labs = ["share_early", "share_late", "share_never"]
    coefs = [t3[k]["coef"] for k in labs]
    los = [t3[k]["ci"][0] for k in labs]
    his = [t3[k]["ci"][1] for k in labs]
    y = np.arange(len(labs))
    ax.errorbar(coefs, y, xerr=[np.array(coefs) - np.array(los), np.array(his) - np.array(coefs)], fmt="o")
    ax.axvline(0, color="0.5", lw=0.8)
    ax.set_yticks(y)
    ax.set_yticklabels(labs)
    ax.set_xlabel("Controlled β (ontime share omitted)")
    fig.tight_layout()
    fig.savefig(FIG / "figB_adjusted_shares.png", dpi=140)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.0, 3.6))
    sts = [s for s in order if s in modal_c["mean_never_share_by_state"]]
    ax.bar(sts, [modal_c["mean_never_share_by_state"][s] for s in sts])
    ax.set_ylabel("Mean never_share")
    ax.set_title("Coverage composition of modal states")
    fig.tight_layout()
    fig.savefig(FIG / "figC_composition.png", dpi=140)
    plt.close(fig)

    (OUT / "run.log").write_text("\n".join(LOG) + "\n")
    log("PHASE4 done " + verdict)


if __name__ == "__main__":
    main()
