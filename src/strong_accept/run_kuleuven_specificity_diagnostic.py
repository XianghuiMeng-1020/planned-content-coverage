#!/usr/bin/env python3
"""Certification diagnostic for the frozen KU Leuven joint specificity model.

Does not overwrite kuleuven_external_results.json.
Does not search alternative specifications.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_kuleuven_external import (  # noqa: E402
    OUT,
    build_environment,
    fit_logit,
    load_cutoffs,
    log,
    or_block,
)


def vif_from_corr(x: pd.DataFrame) -> dict[str, float]:
    c = x.corr().to_numpy()
    inv = np.linalg.pinv(c)
    return {col: float(inv[i, i]) for i, col in enumerate(x.columns)}


def main() -> None:
    cutoffs = load_cutoffs()
    pairs = [
        ("1819", "Accountancy"),
        ("1819", "Global economics"),
        ("1920", "Accountancy"),
        ("1920", "Global economics"),
        ("2021", "Accountancy"),
        ("2021", "Global economics 1"),
        ("2021", "Global economics 2"),
    ]
    frames = [build_environment(y, c, cutoffs)["frame"] for y, c in pairs]
    df = pd.concat(frames, ignore_index=True)
    d = df[df.defined].dropna(
        subset=["PASSED_FIRST_ATTEMPT", "material_reach", "mainpage_reach", "n_sessions", "n_active_days"]
    ).copy()
    d["log_sessions"] = np.log1p(d.n_sessions)

    support = {
        "n": int(len(d)),
        "mainpage_mean": float(d.mainpage_reach.mean()),
        "mainpage_sd": float(d.mainpage_reach.std()),
        "mainpage_min": float(d.mainpage_reach.min()),
        "mainpage_p25": float(d.mainpage_reach.quantile(0.25)),
        "mainpage_median": float(d.mainpage_reach.median()),
        "mainpage_p75": float(d.mainpage_reach.quantile(0.75)),
        "mainpage_max": float(d.mainpage_reach.max()),
        "mainpage_zero_share": float((d.mainpage_reach == 0).mean()),
        "mainpage_one_share": float((d.mainpage_reach == 1).mean()),
        "mainpage_nunique": int(d.mainpage_reach.nunique()),
        "material_mean": float(d.material_reach.mean()),
    }
    corr = {
        "material_mainpage": float(d.material_reach.corr(d.mainpage_reach)),
        "mainpage_sessions": float(d.mainpage_reach.corr(d.n_sessions)),
        "mainpage_active_days": float(d.mainpage_reach.corr(d.n_active_days)),
        "material_sessions": float(d.material_reach.corr(d.n_sessions)),
        "material_active_days": float(d.material_reach.corr(d.n_active_days)),
    }
    vif = vif_from_corr(d[["material_reach", "mainpage_reach", "log_sessions", "n_active_days"]])

    f_m2 = "PASSED_FIRST_ATTEMPT ~ mainpage_reach + np.log1p(n_sessions) + n_active_days + C(environment)"
    f_m1 = "PASSED_FIRST_ATTEMPT ~ material_reach + np.log1p(n_sessions) + n_active_days + C(environment)"
    f_m3 = (
        "PASSED_FIRST_ATTEMPT ~ material_reach + mainpage_reach + "
        "np.log1p(n_sessions) + n_active_days + C(environment)"
    )
    m1 = fit_logit(d, f_m1)
    m2 = fit_logit(d, f_m2)
    m3 = fit_logit(d, f_m3)

    loeo = []
    for env in sorted(d.environment.unique()):
        sub = d[d.environment != env]
        r = fit_logit(sub, f_m3)
        loeo.append(
            {
                "dropped": env,
                "n": r["n"],
                "material": or_block(r, "material_reach"),
                "mainpage": or_block(r, "mainpage_reach"),
            }
        )

    mat_m1 = or_block(m1, "material_reach")
    mat_m3 = or_block(m3, "material_reach")
    main_m2 = or_block(m2, "mainpage_reach")
    main_m3 = or_block(m3, "mainpage_reach")
    suppression = bool(mat_m3["or"] / mat_m1["or"] >= 1.5 or (main_m2["or"] >= 1 and main_m3["or"] < 1))

    out = {
        "n": int(len(d)),
        "support": support,
        "correlations": corr,
        "vif": vif,
        "M1_material_only": mat_m1,
        "M2_mainpage_only": main_m2,
        "M3_joint_material": mat_m3,
        "M3_joint_mainpage": main_m3,
        "loeo_joint": loeo,
        "suppression": suppression,
        "verdict": "PASS_B" if suppression else "PASS_A",
        "note": "Diagnostic only. Frozen headlines unchanged.",
    }
    dest = OUT / "kuleuven_specificity_diagnostic.json"
    dest.write_text(json.dumps(out, indent=2))
    log(json.dumps({
        "support": support,
        "correlations": corr,
        "vif": vif,
        "M1": mat_m1,
        "M2": main_m2,
        "M3_material": mat_m3,
        "M3_mainpage": main_m3,
        "suppression": suppression,
        "verdict": out["verdict"],
    }, indent=2))
    log(f"wrote {dest}")


if __name__ == "__main__":
    main()
