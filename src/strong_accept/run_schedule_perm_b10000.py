#!/usr/bin/env python3
"""Schedule permutation B=10000. Does not overwrite the v6 B=200 output."""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from common.paths import OULAD_INTERIM, SEED  # noqa: E402
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
from v4.constants import V4_DEVELOPMENT  # noqa: E402
from v5_temporal.run_temporal_safe import annotate_with_cutoff, collapse_student_assessment  # noqa: E402
from v6_construct.run_construct_validity import (  # noqa: E402
    KEYS,
    PERM_SEED,
    access_before_occasions,
    aggregate_from_grid,
    attach_controls,
    load_sa,
)

OUT = ROOT / "results" / "strong_accept"
B = 10_000
AUTH_TARGET = -3.495471324370518


def log(msg: str) -> None:
    print(msg, flush=True)


def build_primary_and_grid():
    vle = load_vle()
    assessments = load_assessments()
    reg = load_registration()
    sa, _ = collapse_student_assessment(load_sa())
    sv = load_student_vle()
    tma_meta = map_prev_tma_date(assessments)
    pres = set(V4_DEVELOPMENT)
    sites0 = documented_sites(vle, pres, FAMILY_OUCONTENT)
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
    end_m = site.window_end_day < site.date_submitted
    end_safe = site.loc[end_m].groupby(KEYS, as_index=False).agg(
        n_opp_end=("id_site", "nunique"), never_end=("never", "mean")
    )
    base = site.groupby(KEYS, as_index=False).agg(
        date_submitted=("date_submitted", "first"),
        is_banked=("is_banked", "first"),
        next_tma_date=("next_tma_date", "first"),
    )
    occ = base.merge(end_safe, on=KEYS, how="left")
    occ["n_opp_end"] = occ.n_opp_end.fillna(0).astype(int)
    occ = attach_tma_outcomes(occ, sa, tma_meta)
    if "score_y" in occ.columns:
        occ["score"] = occ["score_y"]
    occ = attach_prior_score(occ, sa, assessments)
    occ = attach_controls(occ, sv, pres)
    scored = occ[occ.score.notna() & occ.date_submitted.notna() & (occ.is_banked != True)].copy()  # noqa: E712
    primary = scored[scored.n_opp_end >= 1].copy()
    primary["share_never"] = primary["never_end"]
    primary["n_opp"] = primary["n_opp_end"]
    acc = access_before_occasions(clicks, scored, sites0)
    sites_map = (
        site.groupby(["code_module", "code_presentation", "id_site"], as_index=False)
        .agg(official_tma=("next_id_assessment", "first"), window_end_day=("window_end_day", "first"))
    )
    grid_parts = []
    for (mod, pres_), gocc in primary.groupby(["code_module", "code_presentation"], sort=False):
        gs = sites_map[(sites_map.code_module == mod) & (sites_map.code_presentation == pres_)]
        if gs.empty:
            continue
        m = gocc[KEYS + ["date_submitted"]].merge(
            gs[["id_site", "official_tma", "window_end_day"]], how="cross"
        )
        grid_parts.append(m)
    grid = pd.concat(grid_parts, ignore_index=True)
    grid = grid.merge(acc, on=["id_student", "next_id_assessment", "id_site"], how="left")
    grid["accessed"] = grid.accessed.fillna(0)
    primary = primary.reset_index(drop=True)
    return primary, sites_map, grid


def m3_coef_from_agg(primary: pd.DataFrame, agg: pd.DataFrame) -> float | None:
    d = primary[KEYS + ["score", "prior_score_filled", "prior_missing", "log_clicks", "active_rate", "raw_inact"]].merge(
        agg, on=KEYS, how="inner"
    )
    d = d[d.n_opp >= 1].dropna(subset=["share_never", "score", "raw_inact"])
    if len(d) < 200:
        return None
    d["pres"] = d.code_module + "_" + d.code_presentation
    fit = smf.ols(
        "score ~ prior_score_filled + prior_missing + C(pres) + log_clicks + active_rate + n_opp + raw_inact + share_never",
        data=d,
    ).fit()
    return float(fit.params["share_never"])


def main() -> None:
    t0 = time.time()
    OUT.mkdir(parents=True, exist_ok=True)
    log("building primary + grid (same v6 path)...")
    primary, sites_map, grid = build_primary_and_grid()
    official = sites_map.set_index("id_site")["official_tma"]
    auth = m3_coef_from_agg(primary, aggregate_from_grid(grid, official))
    log(f"authentic={auth} target={AUTH_TARGET} abs_diff={None if auth is None else abs(auth - AUTH_TARGET)}")
    placebos = []
    for b in range(B):
        rngb = np.random.default_rng(PERM_SEED + b)
        parts = []
        for (_, _), gs in sites_map.groupby(["code_module", "code_presentation"]):
            labs = gs.official_tma.to_numpy().copy()
            rngb.shuffle(labs)
            parts.append(pd.Series(labs, index=gs.id_site.to_numpy()))
        assign = pd.concat(parts)
        c = m3_coef_from_agg(primary, aggregate_from_grid(grid, assign))
        if c is not None:
            placebos.append(c)
        if (b + 1) % 200 == 0:
            log(f"  perm {b + 1}/{B} elapsed={time.time() - t0:.1f}s")
    pla = np.array(placebos, dtype=float)
    n_ext = int((pla <= auth).sum()) if auth is not None else 0
    p = float((1 + n_ext) / (len(pla) + 1)) if len(pla) else None
    out = {
        "protocol": "within-presentation TMA-label shuffle; B=10000; seed 20260827",
        "B_requested": B,
        "B": int(len(pla)),
        "perm_seed": PERM_SEED,
        "construction_seed": SEED,
        "authentic": auth,
        "authentic_v6_target": AUTH_TARGET,
        "authentic_abs_diff_vs_v6": None if auth is None else abs(auth - AUTH_TARGET),
        "placebo_mean": float(np.mean(pla)) if len(pla) else None,
        "placebo_median": float(np.median(pla)) if len(pla) else None,
        "placebo_p025": float(np.quantile(pla, 0.025)) if len(pla) else None,
        "placebo_p975": float(np.quantile(pla, 0.975)) if len(pla) else None,
        "empirical_directional_p": p,
        "pct_placebos_as_or_more_extreme": float(n_ext / len(pla)) if len(pla) else None,
        "percentile_authentic_more_extreme_than": float(1 - n_ext / len(pla)) if len(pla) else None,
        "monte_carlo_se": float(np.sqrt(p * (1 - p) / len(pla))) if p is not None else None,
        "elapsed_sec": float(time.time() - t0),
        "interpretation_rule": "unique-partition only if p<0.05 or 95% placebo interval excludes authentic; else preserve null",
        "null_preserved_if": "p>=0.05 and authentic inside [p025,p975]",
    }
    dest = OUT / "schedule_permutation_b10000.json"
    dest.write_text(json.dumps(out, indent=2))
    log(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
