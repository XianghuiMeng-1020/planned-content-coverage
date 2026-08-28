#!/usr/bin/env python3
"""Unit A: activity-matched breadth vs volume. Writes only results/v10_value_boundary/."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

_SRC = Path(__file__).resolve().parents[1]
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from v10_value_boundary.shared import (  # noqa: E402
    LOG,
    OUT,
    build_suba_panel,
    dump_csv,
    dump_json,
    log,
    nopp_band,
    sha256_file,
)


MIN_ARM = 15
SMD_ACCEPT = 0.15
ATTENUATION_A2 = 2.0 / 3.0


def _qcut3(s: pd.Series) -> pd.Series:
    out = pd.Series(np.nan, index=s.index, dtype=float)
    if s.nunique(dropna=True) < 3:
        return out
    try:
        cats = pd.qcut(s, 3, labels=False, duplicates="drop")
    except ValueError:
        return out
    if pd.Series(cats).nunique(dropna=True) < 3:
        return out
    return cats.astype(float) + 1.0


def assign_cells(df: pd.DataFrame) -> pd.DataFrame:
    work = df.copy()
    work["vol_tertile"] = np.nan
    work["prior_band"] = "missing"
    for _, idx in work.groupby(["pres", "next_id_assessment"], sort=False).groups.items():
        g = work.loc[idx]
        work.loc[idx, "vol_tertile"] = _qcut3(g["log_clicks"]).to_numpy()
        obs = g["prior_missing"] == 0
        work.loc[idx, "prior_band"] = np.where(g["prior_missing"] == 1, "missing", "P_unsplit")
        if obs.sum() >= 3:
            bands = _qcut3(g.loc[obs, "prior_score_filled"])
            mapped = bands.map({1.0: "P1", 2.0: "P2", 3.0: "P3"})
            work.loc[g.index[obs.to_numpy()], "prior_band"] = mapped.fillna("P_unsplit").to_numpy()
    work["nopp_band"] = work["n_opp"].map(nopp_band)
    work["tma_cell"] = work["pres"].astype(str) + "|" + work["next_id_assessment"].astype(str)
    work["stratum"] = (
        work["tma_cell"]
        + "|V"
        + work["vol_tertile"].astype(str)
        + "|"
        + work["prior_band"].astype(str)
        + "|"
        + work["nopp_band"].astype(str)
    )
    work["cov_tertile"] = np.nan
    for _, idx in work.groupby("stratum", sort=False).groups.items():
        work.loc[idx, "cov_tertile"] = _qcut3(work.loc[idx, "never_share"]).to_numpy()
    work["arm"] = work["cov_tertile"].map({1.0: "T1", 3.0: "T3"})
    return work


def _cell_rd(g: pd.DataFrame, ycol: str) -> dict | None:
    t1 = g[g.arm == "T1"]
    t3 = g[g.arm == "T3"]
    n1, n3 = int(len(t1)), int(len(t3))
    if n1 < MIN_ARM or n3 < MIN_ARM:
        return None
    p1, p3 = float(t1[ycol].mean()), float(t3[ycol].mean())
    v1 = max(p1 * (1 - p1), 1e-6) / n1
    v3 = max(p3 * (1 - p3), 1e-6) / n3
    rd = p3 - p1
    var = v1 + v3
    return {
        "stratum": g["stratum"].iloc[0],
        "pres": g["pres"].iloc[0],
        "code_module": g["code_module"].iloc[0],
        "code_presentation": g["code_presentation"].iloc[0],
        "next_id_assessment": int(g["next_id_assessment"].iloc[0]),
        "n_t1": n1,
        "n_t3": n3,
        "n": n1 + n3,
        "p_t1": p1,
        "p_t3": p3,
        "rd": rd,
        "var": var,
        "weight": 1.0 / var,
    }


def pool_rds(rows: list[dict]) -> dict:
    if not rows:
        return {"n_cells": 0, "n": 0, "rd_weighted": None, "rd_unweighted": None, "se": None, "ci": [None, None]}
    w = np.array([r["weight"] for r in rows], dtype=float)
    rd = np.array([r["rd"] for r in rows], dtype=float)
    rd_w = float(np.sum(w * rd) / np.sum(w))
    se = float(1.0 / np.sqrt(np.sum(w)))
    rd_u = float(np.mean(rd))
    n = int(sum(r["n"] for r in rows))
    return {
        "n_cells": int(len(rows)),
        "n": n,
        "n_t1": int(sum(r["n_t1"] for r in rows)),
        "n_t3": int(sum(r["n_t3"] for r in rows)),
        "rd_weighted": rd_w,
        "rd_unweighted": rd_u,
        "se": se,
        "ci": [rd_w - 1.96 * se, rd_w + 1.96 * se],
        "n_pos_cells": int((rd > 0).sum()),
        "n_neg_cells": int((rd < 0).sum()),
        "n_zero_cells": int((rd == 0).sum()),
    }


def crude_rd(df: pd.DataFrame, ycol: str) -> dict:
    work = df.copy()
    work["gter"] = _qcut3(work["never_share"])
    t1 = work[work.gter == 1]
    t3 = work[work.gter == 3]
    if len(t1) < 2 or len(t3) < 2:
        return {"n_t1": int(len(t1)), "n_t3": int(len(t3)), "rd": None, "ci": [None, None]}
    p1, p3 = float(t1[ycol].mean()), float(t3[ycol].mean())
    n1, n3 = int(len(t1)), int(len(t3))
    se = float(np.sqrt(max(p1 * (1 - p1), 1e-6) / n1 + max(p3 * (1 - p3), 1e-6) / n3))
    rd = p3 - p1
    return {
        "n_t1": n1,
        "n_t3": n3,
        "p_t1": p1,
        "p_t3": p3,
        "rd": rd,
        "se": se,
        "ci": [rd - 1.96 * se, rd + 1.96 * se],
        "note": "global never_share tertiles; descriptor only",
    }


def smd(a: np.ndarray, b: np.ndarray) -> float | None:
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    a = a[np.isfinite(a)]
    b = b[np.isfinite(b)]
    if len(a) < 2 or len(b) < 2:
        return None
    s = np.sqrt(((len(a) - 1) * a.var(ddof=1) + (len(b) - 1) * b.var(ddof=1)) / (len(a) + len(b) - 2))
    if s == 0:
        return 0.0
    return float((a.mean() - b.mean()) / s)


def balance_table(support: pd.DataFrame) -> pd.DataFrame:
    t1 = support[support.arm == "T1"]
    t3 = support[support.arm == "T3"]
    rows = []
    for col, mask in [
        ("log_clicks", None),
        ("n_opp", None),
        ("prior_missing", None),
        ("prior_score_filled_observed", "observed"),
    ]:
        if mask == "observed":
            a = t3.loc[t3.prior_missing == 0, "prior_score_filled"]
            b = t1.loc[t1.prior_missing == 0, "prior_score_filled"]
            name = "prior_score_filled"
        else:
            a, b = t3[col], t1[col]
            name = col
        rows.append(
            {
                "covariate": name,
                "mean_t3": float(a.mean()) if len(a) else None,
                "mean_t1": float(b.mean()) if len(b) else None,
                "smd_t3_minus_t1": smd(a.to_numpy(), b.to_numpy()),
                "n_t3": int(len(a)),
                "n_t1": int(len(b)),
            }
        )
    return pd.DataFrame(rows)


def lpm_cluster(support: pd.DataFrame, ycol: str) -> dict:
    d = support[support.arm.isin(["T1", "T3"])].copy()
    d["t3"] = (d.arm == "T3").astype(int)
    d["y"] = d[ycol].astype(float)
    try:
        fit = smf.ols("y ~ t3 + C(stratum)", data=d).fit(
            cov_type="cluster", cov_kwds={"groups": d.id_student}
        )
        ci = fit.conf_int().loc["t3"]
        return {
            "ok": True,
            "n": int(fit.nobs),
            "coef": float(fit.params["t3"]),
            "se": float(fit.bse["t3"]),
            "ci": [float(ci.iloc[0]), float(ci.iloc[1])],
            "p": float(fit.pvalues["t3"]),
        }
    except Exception as e:
        return {"ok": False, "error": str(e)[:200]}


def classify_a(pool: dict, crude: dict, bal_ok: bool, n_cells: int, hypothesized: str) -> str:
    if n_cells < 8 or not bal_ok:
        return "A4"
    rd = pool.get("rd_weighted")
    ci = pool.get("ci") or [None, None]
    if rd is None or ci[0] is None:
        return "A4"
    excludes0 = (ci[0] > 0) or (ci[1] < 0)
    direction_ok = (rd < 0) if hypothesized == "neg" else (rd > 0)
    if (not excludes0) or (not direction_ok):
        return "A3"
    cr = crude.get("rd")
    if cr is not None and abs(rd) < ATTENUATION_A2 * abs(cr):
        return "A2"
    return "A1"


def analyze(df: pd.DataFrame, ycol: str, hypothesized: str) -> tuple[pd.DataFrame, dict, pd.DataFrame, dict]:
    tagged = assign_cells(df)
    feas_cells = []
    for s, g in tagged.groupby("stratum", sort=False):
        n1 = int((g.arm == "T1").sum())
        n3 = int((g.arm == "T3").sum())
        feas_cells.append(
            {
                "stratum": s,
                "n": int(len(g)),
                "n_t1": n1,
                "n_t3": n3,
                "n_distinct_never": int(g.never_share.nunique()),
                "analyzable": bool(n1 >= MIN_ARM and n3 >= MIN_ARM),
            }
        )
    feas = pd.DataFrame(feas_cells)
    analyzable = set(feas.loc[feas.analyzable, "stratum"])
    tagged["analyzable"] = tagged.stratum.isin(analyzable)
    support = tagged[tagged.analyzable & tagged.arm.isin(["T1", "T3"])].copy()
    rows = []
    for s, g in support.groupby("stratum", sort=False):
        rec = _cell_rd(g, ycol)
        if rec:
            rows.append(rec)
    pooled = pool_rds(rows)
    crude = crude_rd(df, ycol)
    bal = balance_table(support) if len(support) else pd.DataFrame()
    lc = bal.set_index("covariate")["smd_t3_minus_t1"] if len(bal) else pd.Series(dtype=float)
    bal_ok = True
    for cov in ("log_clicks", "n_opp"):
        if cov not in lc.index or lc[cov] is None or abs(float(lc[cov])) > SMD_ACCEPT:
            bal_ok = False
    secondary = lpm_cluster(support, ycol) if len(support) else {"ok": False}
    scenario = classify_a(pooled, crude, bal_ok, pooled["n_cells"], hypothesized)
    ctx_rows = []
    for level, col in [("module", "code_module"), ("presentation", "pres")]:
        for key, sub in (pd.DataFrame(rows).groupby(col) if rows else []):
            pr = pool_rds(sub.to_dict("records"))
            pr.update({"level": level, "context": key, "outcome": ycol})
            ctx_rows.append(pr)
    summary = {
        "outcome": ycol,
        "hypothesized_rd_sign": hypothesized,
        "n_source": int(len(df)),
        "n_support": int(len(support)),
        "retention": float(len(support) / len(df)) if len(df) else None,
        "n_strata_formed": int(tagged.stratum.nunique()),
        "n_analyzable_cells": int(len(analyzable)),
        "n_contexts_pres": int(support.pres.nunique()) if len(support) else 0,
        "n_contexts_module": int(support.code_module.nunique()) if len(support) else 0,
        "balance_accepted": bal_ok,
        "pooled": pooled,
        "crude": crude,
        "secondary_lpm": secondary,
        "scenario": scenario,
        "analyzable_presentations": sorted(support.pres.unique().tolist()) if len(support) else [],
    }
    cell_df = pd.DataFrame(rows)
    return feas, summary, cell_df, support


def main() -> None:
    log("UNIT A start")
    occ = build_suba_panel()
    defined = occ[(occ.n_opp >= 1) & occ.never_share.notna()].copy()
    sub = defined[defined.state.isin(["submitted", "eligible_nonsubmit"])].copy()
    scored = defined[(defined.state == "submitted") & defined.score.notna()].copy()
    log(f"defined={len(defined)} submit_sample={len(sub)} scored={len(scored)}")

    feas_s, sum_s, cells_s, support_s = analyze(sub, "submitted", "neg")
    dump_json(
        "unit_a_feasibility.json",
        {
            "design": "Option 2 within-cell stratification (frozen)",
            "volume": "within-TMA tertile of Phase-5 log_clicks",
            "prior": "missing vs within-TMA tertiles of observed prior_score_filled",
            "n_opp_bands": ["1-4", "5-8", "9+"],
            "coverage_contrast": "within-stratum never_share T3 vs T1",
            "min_arm": MIN_ARM,
            "smd_accept": SMD_ACCEPT,
            "submission": {
                "n_source": sum_s["n_source"],
                "n_strata_formed": sum_s["n_strata_formed"],
                "n_analyzable_cells": sum_s["n_analyzable_cells"],
                "n_support": sum_s["n_support"],
                "retention": sum_s["retention"],
                "presentations": sum_s["analyzable_presentations"],
            },
            "note": "Feasibility uses cell occupancy only; outcomes are not used to select the design.",
        },
    )
    feas_l, sum_l, cells_l, support_l = analyze(scored, "low_score", "pos")

    bal = balance_table(support_s) if len(support_s) else pd.DataFrame()
    if len(support_l):
        bal_l = balance_table(support_l)
        bal_l["sample"] = "low_score"
        bal["sample"] = "submission"
        bal = pd.concat([bal, bal_l], ignore_index=True)
    elif len(bal):
        bal["sample"] = "submission"
    dump_csv("unit_a_balance.csv", bal)

    def result_rows(label: str, summary: dict, cells: pd.DataFrame) -> pd.DataFrame:
        recs = [
            {
                "level": "pooled_weighted",
                "outcome": label,
                "rd": summary["pooled"]["rd_weighted"],
                "se": summary["pooled"]["se"],
                "ci_lo": summary["pooled"]["ci"][0],
                "ci_hi": summary["pooled"]["ci"][1],
                "n_cells": summary["pooled"]["n_cells"],
                "n": summary["pooled"]["n"],
                "scenario": summary["scenario"],
            },
            {
                "level": "pooled_unweighted",
                "outcome": label,
                "rd": summary["pooled"]["rd_unweighted"],
                "se": None,
                "ci_lo": None,
                "ci_hi": None,
                "n_cells": summary["pooled"]["n_cells"],
                "n": summary["pooled"]["n"],
                "scenario": summary["scenario"],
            },
            {
                "level": "crude_unmatched",
                "outcome": label,
                "rd": summary["crude"].get("rd"),
                "se": summary["crude"].get("se"),
                "ci_lo": (summary["crude"].get("ci") or [None, None])[0],
                "ci_hi": (summary["crude"].get("ci") or [None, None])[1],
                "n_cells": None,
                "n": (summary["crude"].get("n_t1") or 0) + (summary["crude"].get("n_t3") or 0),
                "scenario": "descriptor_only",
            },
            {
                "level": "secondary_lpm_cellFE",
                "outcome": label,
                "rd": summary["secondary_lpm"].get("coef"),
                "se": summary["secondary_lpm"].get("se"),
                "ci_lo": (summary["secondary_lpm"].get("ci") or [None, None])[0],
                "ci_hi": (summary["secondary_lpm"].get("ci") or [None, None])[1],
                "n_cells": summary["pooled"]["n_cells"],
                "n": summary["secondary_lpm"].get("n"),
                "scenario": summary["scenario"],
            },
        ]
        out = pd.DataFrame(recs)
        if len(cells):
            extra = cells.rename(columns={"var": "cell_var"})[
                ["stratum", "pres", "code_module", "n_t1", "n_t3", "n", "p_t1", "p_t3", "rd"]
            ].copy()
            extra["level"] = "cell"
            extra["outcome"] = label
            extra["ci_lo"] = extra["rd"] - 1.96 * np.sqrt(cells["var"].to_numpy())
            extra["ci_hi"] = extra["rd"] + 1.96 * np.sqrt(cells["var"].to_numpy())
            extra["se"] = np.sqrt(cells["var"].to_numpy())
            extra["n_cells"] = 1
            extra["scenario"] = summary["scenario"]
            out = pd.concat([out, extra], ignore_index=True, sort=False)
        return out

    dump_csv("unit_a_submission_results.csv", result_rows("submitted", sum_s, cells_s))
    dump_csv("unit_a_low_score_results.csv", result_rows("low_score", sum_l, cells_l))

    ctx = []
    for label, cells in [("submitted", cells_s), ("low_score", cells_l)]:
        if cells is None or cells.empty:
            continue
        for level, col in [("module", "code_module"), ("presentation", "pres")]:
            for key, sub in cells.groupby(col):
                pr = pool_rds(sub.to_dict("records"))
                ctx.append(
                    {
                        "outcome": label,
                        "level": level,
                        "context": key,
                        "n_cells": pr["n_cells"],
                        "n": pr["n"],
                        "rd_weighted": pr["rd_weighted"],
                        "ci_lo": pr["ci"][0],
                        "ci_hi": pr["ci"][1],
                        "n_neg_cells": pr.get("n_neg_cells"),
                        "n_pos_cells": pr.get("n_pos_cells"),
                    }
                )
    dump_csv("unit_a_context_results.csv", pd.DataFrame(ctx))
    dump_json(
        "unit_a_summary.json",
        {
            "submission": sum_s,
            "low_score": sum_l,
            "primary_scenario_uses": "submission",
            "attenuation_rule": (
                "A2 if CI excludes 0 in hypothesized direction and "
                f"|matched RD| < {ATTENUATION_A2} × |crude RD|; else A1 if persists"
            ),
            "log": LOG[-20:],
        },
    )
    log("UNIT A done")
    (OUT / "unit_a_run.log").write_text("\n".join(LOG) + "\n")


if __name__ == "__main__":
    main()
