#!/usr/bin/env python3
"""H3: clustered Wald contrast β_current = β_later on the frozen joint model.

Decision rule (pre-declared):
  Support a 'more informative' claim only if the clustered 95% CI for
  (β_current − β_later) excludes 0 and the difference is negative
  (current more negatively associated with score, matching the frozen
  non-reach orientation).
  If the CI includes 0, prose may report point-estimate direction only.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "results" / "v6_construct_validity" / "occasion_table.parquet"
OUT = ROOT / "results" / "strong_accept"
SEED = 20260828


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    df = pd.read_parquet(SRC)
    work = df.dropna(
        subset=["score", "raw_inact", "log_clicks", "never_end", "future_never_share", "n_opp_end", "prior_score_filled"]
    ).copy()
    work = work[(work.n_opp_end >= 1) & (work.n_opp_future >= 1)]
    work["share_never"] = work["never_end"]
    work["n_opp"] = work["n_opp_end"]
    work["pres"] = work.code_module + "_" + work.code_presentation
    # Frozen joint formula from v6 fit_joint (legacy-comparable RHS).
    fml = (
        "score ~ prior_score_filled + prior_missing + C(pres) + log_clicks + "
        "active_rate + n_opp + raw_inact + share_never + future_never_share"
    )
    fit = smf.ols(fml, data=work).fit(cov_type="cluster", cov_kwds={"groups": work.id_student})
    hyp = "share_never = future_never_share"
    wald = fit.wald_test(hyp, scalar=True)
    diff = float(fit.params["share_never"] - fit.params["future_never_share"])
    # Cluster-robust variance of the contrast.
    cov = fit.cov_params()
    var = (
        float(cov.loc["share_never", "share_never"])
        + float(cov.loc["future_never_share", "future_never_share"])
        - 2.0 * float(cov.loc["share_never", "future_never_share"])
    )
    se = float(np.sqrt(var))
    z = diff / se if se > 0 else np.nan
    ci = [diff - 1.96 * se, diff + 1.96 * se]
    out = {
        "decision_rule": "support current>later only if 95% CI of (beta_current-beta_later) excludes 0 and difference<0",
        "formula": fml,
        "clustering": "id_student",
        "n": int(fit.nobs),
        "n_learners": int(work.id_student.nunique()),
        "beta_current": float(fit.params["share_never"]),
        "beta_later": float(fit.params["future_never_share"]),
        "ci_current": [float(x) for x in fit.conf_int().loc["share_never"]],
        "ci_later": [float(x) for x in fit.conf_int().loc["future_never_share"]],
        "diff_current_minus_later": diff,
        "se_diff": se,
        "z": float(z),
        "wald_stat": float(wald.statistic),
        "wald_p": float(wald.pvalue),
        "ci95_diff": ci,
        "excludes_zero": bool(ci[1] < 0 or ci[0] > 0),
        "supports_current_more_negative": bool(diff < 0 and ci[1] < 0),
        "seed_note": SEED,
    }
    dest = OUT / "current_later_wald.json"
    dest.write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
