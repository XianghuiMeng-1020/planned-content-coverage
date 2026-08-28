#!/usr/bin/env python3
"""Verify frozen reference artifacts against the manuscript headline numbers."""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
V6 = ROOT / "results" / "v6_construct_validity"
V7 = ROOT / "results" / "v7_inference"
V8 = ROOT / "results" / "v8_timing"
V9 = ROOT / "results" / "v9_decision_utility"
REF = ROOT / "results" / "reference" / "headlines.json"


def _load(path: Path):
    return json.loads(path.read_text())


def _close(a: float, b: float, tol: float = 5e-3) -> bool:
    return abs(float(a) - float(b)) <= tol


def main() -> None:
    h = _load(REF)
    prim = _load(V7 / "primary_inference.json")["nonredundant"]
    fe = _load(V7 / "within_learner.json")["FE_A"]
    match = _load(V6 / "activity_matched.json")
    fut = _load(V6 / "future_content.json")["intersection"]["F3_joint"]
    perm = _load(V6 / "schedule_permutation.json")
    ident = _load(V6 / "identity_placebo_forensic.json")
    # key name may be nested; accept either top-level or first analysis block
    if "coef_actual" not in ident:
        ident = next(v for v in ident.values() if isinstance(v, dict) and "coef_actual" in v)
    ggg = _load(V6 / "ggg_boundary.json")["controlled"]
    pres = _load(V7 / "presentation_heterogeneity.json")
    lopo = _load(V9 / "lopo_lomo_summary.json")["lowscore_LOPO"]
    bud = _load(V9 / "fixed_budget.json")["low_score"]["pooled"]["0.1"]
    modal = {row["state"]: row["mean_score"] for row in _load(V8 / "unadjusted_gradient.json")["modal"]["by_state"]}

    checks = [
        ("n occasions", prim["n"], h["n_occasions"]),
        ("n learners", prim["n_learners"], h["n_learners"]),
        ("primary beta", round(prim["coef"], 2), h["primary_beta"]),
        ("primary CI lo", round(prim["ci"][0], 2), h["primary_ci"][0]),
        ("primary CI hi", round(prim["ci"][1], 2), h["primary_ci"][1]),
        ("partial R2", round(prim["partial_r2"], 4), h["partial_r2"]),
        ("FE beta", round(fe["coef"], 2), h["fe_beta"]),
        ("FE n occasions", fe["n_occasions"], h["fe_n_occasions"]),
        ("FE n learners", fe["n_learners"], h["fe_n_learners"]),
        ("matched contrast", round(match["weighted_mean"], 2), h["matched_contrast"]),
        ("current joint", round(fut["current"]["coef"], 2), h["current_joint"]),
        ("later joint", round(fut["future"]["coef"], 2), h["later_joint"]),
        ("schedule perm p", round(perm["empirical_directional_p"], 2), h["schedule_perm_p"]),
        ("identity actual", round(ident["coef_actual"], 2), h["identity_actual"]),
        ("identity placebo", round(ident["coef_placebo"], 2), h["identity_placebo"]),
        ("GGG beta", round(ggg["coef"], 2), h["ggg_beta"]),
        ("pres negative", pres["n_negative"], h["n_pres_negative"]),
        ("pres n", pres["n_negative"] + pres["n_positive"], h["n_pres"]),
        ("dAUROC", round(lopo["median_d_auroc"], 4), h["delta_auroc"]),
        ("budget 10 extra", bud["extra_true"], h["budget_extra_10"]),
        ("modal early", round(modal["early"], 2), h["modal_early"]),
        ("modal ontime", round(modal["ontime"], 2), h["modal_ontime"]),
        ("modal late", round(modal["late"], 2), h["modal_late"]),
        ("modal never", round(modal["never"], 2), h["modal_never"]),
    ]
    failed = []
    for name, got, exp in checks:
        if isinstance(exp, float) and isinstance(got, (int, float)):
            ok = _close(got, exp)
        else:
            ok = got == exp
        if not ok:
            failed.append((name, got, exp))
            print(f"FAIL {name}: got {got} expected {exp}")
        else:
            print(f"PASS {name}: {got}")
    if failed:
        raise SystemExit(f"{len(failed)} headline check(s) failed")
    print("all headline reference checks passed")


if __name__ == "__main__":
    main()
