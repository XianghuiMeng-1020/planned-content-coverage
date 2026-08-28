#!/usr/bin/env python3
"""Verify frozen reference artifacts against the manuscript headline numbers."""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
V9 = ROOT / "results" / "v9_decision_utility"
V10 = ROOT / "results" / "v10_value_boundary"
V11 = ROOT / "results" / "v11_design_replication"
REF = ROOT / "results" / "reference" / "headlines.json"


def _load(path: Path):
    return json.loads(path.read_text())


def _close(a: float, b: float, tol: float = 5e-4) -> bool:
    return abs(float(a) - float(b)) <= tol


def main() -> None:
    h = _load(REF)
    suba = _load(V9 / "submission_association.json")["SUB_A"]
    subb = _load(V9 / "submission_association.json")["SUB_B"]
    low = _load(V9 / "lowscore_association.json")
    prev = _load(V9 / "low_score_definition.json")
    lopo = _load(V9 / "lopo_lomo_summary.json")["lowscore_LOPO"]
    lomo = {row["fold"]: row for row in _load(V9 / "lomo_lowscore.json")}
    cal = _load(V9 / "calibration_pooled.json")
    bud = _load(V9 / "fixed_budget.json")
    ua = _load(V10 / "unit_a_summary.json")
    ub = _load(V10 / "unit_b_summary.json")
    hold = _load(V11 / "summary.json")

    checks = [
        ("SUB-A OR", round(suba["or"], 3), h["sub_a_or"]),
        ("SUB-A CI lo", round(suba["or_ci"][0], 3), h["sub_a_or_ci"][0]),
        ("SUB-A CI hi", round(suba["or_ci"][1], 3), h["sub_a_or_ci"][1]),
        ("SUB-B OR", round(subb["or"], 3), h["sub_b_or"]),
        ("low-score n", prev["n"], h["low_score_n"]),
        ("low-score prevalence", round(prev["prevalence"], 4), h["low_score_prevalence"]),
        ("low-score OR", round(low["or"], 3), h["low_score_or"]),
        ("dAUROC", round(lopo["median_d_auroc"], 5), h["delta_auroc"]),
        ("dAUPRC", round(lopo["median_d_auprc"], 5), h["delta_auprc"]),
        ("BBB LOMO", round(lomo["BBB"]["d_auroc"], 4), h["bbb_lomo"]),
        ("cal B0 slope", round(cal["B0"]["slope"], 3), h["cal_b0_slope"]),
        ("cal B1 slope", round(cal["B1"]["slope"], 3), h["cal_b1_slope"]),
        ("budget 5 extra", bud["low_score"]["pooled"]["0.05"]["extra_true"], h["budget_extra"]["0.05"]),
        ("budget 10 extra", bud["low_score"]["pooled"]["0.1"]["extra_true"], h["budget_extra"]["0.10"]),
        ("budget 20 extra", bud["low_score"]["pooled"]["0.2"]["extra_true"], h["budget_extra"]["0.20"]),
        ("nonsubmit 10 extra", bud["nonsusubmit"]["pooled"]["0.1"]["extra_true"], 0),
        ("unit A balance accepted", ua["submission"]["balance_accepted"], False),
        ("pairwise items", ub["n_eligible_items"], 310),
        ("pairwise rho", round(ub["median_rho"], 3), h["pairwise_rho"]),
        ("held-out rho_H", round(hold["median_rho_H"], 3), h["heldout_rho_h"]),
        ("position rho_P", round(hold["median_rho_P"], 3), h["position_rho_p"]),
        ("delta rho", round(hold["median_delta_rho"], 3), h["delta_rho"]),
        ("residual rho", round(hold["median_rho_residual"], 3), h["residual_rho"]),
        ("top-25 precision", hold["median_topk_precision"], h["topk_precision"]),
        ("AAA/BBB", hold["aaa_bbb"], "NOT_IDENTIFIABLE_FOR_HELDOUT_REPLICATION"),
        ("CCC n", hold["folds"][0]["n_slots"], 4),
    ]
    failed = []
    for name, got, exp in checks:
        ok = got == exp if not isinstance(got, float) else _close(got, exp)
        if isinstance(exp, float) and isinstance(got, (int, float)):
            ok = _close(got, exp)
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
