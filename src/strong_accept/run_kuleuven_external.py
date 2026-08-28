#!/usr/bin/env python3
"""KU Leuven external-validity probe. Protocol frozen before this run."""
from __future__ import annotations

import json
import os
import re
from datetime import date, datetime
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

ROOT = Path(__file__).resolve().parents[2]
DATA = Path(os.environ.get("KULEUVEN_DATA", ROOT / "data/external/kuleuven_tiukhova2026"))
EXTRACT = DATA / "extracted/dataset"
OFFICIAL = DATA / "official"
OUT = ROOT / "results/strong_accept"
OUT.mkdir(parents=True, exist_ok=True)

YEAR_FILES = {
    "1819": "1819",
    "1920": "1920",
    "2021": "2021",
}
INFO_COURSE = {
    "Accountancy": "Accountancy",
    "Global economics": "Global economics",
    "Global economics 1": "Global economics",
    "Global economics 2": "Global economics",
}
WEEK_RE = re.compile(r"Week (\d+) of (\d+)")


def log(msg: str) -> None:
    print(msg, flush=True)


def parse_exam_monday(label: str) -> date:
    m = WEEK_RE.fullmatch(label.strip())
    if not m:
        raise ValueError(label)
    week, year = int(m.group(1)), int(m.group(2))
    return date.fromisocalendar(year, week, 1)


def load_cutoffs() -> dict[tuple[str, str], pd.Timestamp]:
    info = json.loads((OFFICIAL / "course_info.json").read_text())
    out = {}
    for ykey, courses in info.items():
        for cname, meta in courses.items():
            t = pd.Timestamp(parse_exam_monday(meta["exam_weeks"][0]))
            out[(ykey, cname)] = t
    return out


def parse_content_dates(s: pd.Series) -> pd.Series:
    return pd.to_datetime(s, dayfirst=True, errors="coerce")


def cascade_keep(logs: pd.DataFrame, content: pd.DataFrame) -> pd.DataFrame:
    """Keep singleton timestamps and cascade roots. Outcomes unused."""
    g = logs.copy()
    g["TIMESTAMP"] = pd.to_datetime(g["TIMESTAMP"], errors="coerce")
    g = g.dropna(subset=["TIMESTAMP", "CONTENT_ID", "USER_ID"])
    cols = ["COURSE_ID", "USER_ID", "CONTENT_ID", "SESSION_ID", "TIMESTAMP"]
    sizes = g.groupby(["COURSE_ID", "USER_ID", "TIMESTAMP"]).CONTENT_ID.transform("nunique")
    singles = g.loc[sizes == 1, cols].copy()
    multi = g.loc[sizes > 1].copy()
    if multi.empty:
        return singles.drop_duplicates()
    cmap = content[["COURSE_ID", "CONTENT_ID", "PARENT_ID"]].copy()
    cmap["CONTENT_ID"] = pd.to_numeric(cmap.CONTENT_ID, errors="coerce")
    cmap["PARENT_ID"] = pd.to_numeric(cmap.PARENT_ID, errors="coerce")
    multi["CONTENT_ID"] = pd.to_numeric(multi.CONTENT_ID, errors="coerce")
    multi = multi.merge(cmap, on=["COURSE_ID", "CONTENT_ID"], how="left")
    present = multi[["COURSE_ID", "USER_ID", "TIMESTAMP", "CONTENT_ID"]].drop_duplicates()
    present = present.rename(columns={"CONTENT_ID": "PARENT_ID"}).assign(_parent_in_bundle=1)
    present["PARENT_ID"] = pd.to_numeric(present.PARENT_ID, errors="coerce")
    flag = multi.merge(
        present,
        on=["COURSE_ID", "USER_ID", "TIMESTAMP", "PARENT_ID"],
        how="left",
    )
    roots = flag.loc[flag._parent_in_bundle.isna(), cols]
    kept = pd.concat([singles, roots], ignore_index=True)
    return kept.drop_duplicates()


def eligible_items(content: pd.DataFrame, cutoff: pd.Timestamp, content_type: str) -> pd.DataFrame:
    d = content[content.CONTENT_TYPE == content_type].copy()
    d = d[d.UNAVAILABLE.fillna(0) != 1]
    d = d[d.UNAVAILABLE_PARENT.fillna(0) != 1]
    d = d[d.GROUP_ASSIGNMENT.fillna(0) != 1]
    start = parse_content_dates(d.START_DATE)
    keep = start.isna() | (start < cutoff)
    return d.loc[keep]


def env_name(year: str, course: str) -> str:
    return f"{course}|{year}"


def build_environment(year: str, course: str, cutoffs: dict) -> dict:
    part = pd.read_excel(EXTRACT / f"{year}_course_participation.xlsx")
    cont = pd.read_excel(EXTRACT / f"{year}_course_content.xlsx")
    logs = pd.read_csv(EXTRACT / f"{year}_log_activity.csv")
    part = part[part.COURSE_ID == course].copy()
    cont = cont[cont.COURSE_ID == course].copy()
    logs = logs[logs.COURSE_ID == course].copy()
    info_key = INFO_COURSE[course]
    cutoff = cutoffs[(year, info_key)]
    emat = eligible_items(cont, cutoff, "Course Material")
    emain = eligible_items(cont, cutoff, "Course Main Page")
    kept = cascade_keep(logs, cont)
    kept_pre = kept[kept.TIMESTAMP < cutoff].copy()
    raw_pre = pd.to_datetime(logs.TIMESTAMP, errors="coerce")
    n_raw_pre = int((raw_pre < cutoff).sum())

    def reach_map(elig: pd.DataFrame) -> pd.Series:
        n = len(elig)
        if n == 0:
            return pd.Series(dtype=float)
        acc = kept_pre[kept_pre.CONTENT_ID.isin(set(elig.CONTENT_ID))]
        hits = acc.groupby("USER_ID").CONTENT_ID.nunique()
        return (hits / n).reindex(part.USER_ID).fillna(0.0)

    material_reach = reach_map(emat)
    mainpage_reach = reach_map(emain)
    sess = kept_pre.groupby("USER_ID").SESSION_ID.nunique()
    days = kept_pre.groupby("USER_ID").TIMESTAMP.apply(lambda s: s.dt.normalize().nunique())
    n_kept = kept_pre.groupby("USER_ID").size()
    out = part.copy()
    out["environment"] = env_name(year, course)
    out["year"] = year
    out["course"] = course
    out["covid_year"] = year == "2021"
    out["n_eligible_material"] = len(emat)
    out["n_eligible_mainpage"] = len(emain)
    out["material_reach"] = material_reach.values
    out["mainpage_reach"] = mainpage_reach.reindex(out.USER_ID).fillna(0.0).values
    out["n_sessions"] = sess.reindex(out.USER_ID).fillna(0).astype(int).values
    out["n_active_days"] = days.reindex(out.USER_ID).fillna(0).astype(int).values
    out["n_kept_events"] = n_kept.reindex(out.USER_ID).fillna(0).astype(int).values
    out["defined"] = out.n_eligible_material >= 1
    return {
        "frame": out,
        "stats": {
            "environment": env_name(year, course),
            "year": year,
            "course": course,
            "cutoff": str(cutoff.date()),
            "enrolled": int(len(part)),
            "with_first_attempt_outcome": int(part.PASSED_FIRST_ATTEMPT.notna().sum()),
            "pass_first_rate": float(part.PASSED_FIRST_ATTEMPT.mean()),
            "pass_any_rate": float(part.PASSED.mean()),
            "content_items": int(len(cont)),
            "material_items": int((cont.CONTENT_TYPE == "Course Material").sum()),
            "eligible_material": int(len(emat)),
            "eligible_mainpage": int(len(emain)),
            "raw_log_rows": int(len(logs)),
            "raw_log_before_cutoff": n_raw_pre,
            "kept_events_before_cutoff": int(len(kept_pre)),
            "cascade_timestamps": None,
            "active_learners": int((out.n_kept_events > 0).sum()),
            "median_eligible_material": int(len(emat)),
            "reach_mean": float(out.loc[out.defined, "material_reach"].mean()) if out.defined.any() else None,
            "reach_median": float(out.loc[out.defined, "material_reach"].median()) if out.defined.any() else None,
            "reach_zero_share": float((out.loc[out.defined, "material_reach"] == 0).mean()) if out.defined.any() else None,
            "reach_one_share": float((out.loc[out.defined, "material_reach"] == 1).mean()) if out.defined.any() else None,
            "undefined_n": int((~out.defined).sum()),
        },
    }


def fit_logit(df: pd.DataFrame, formula: str) -> dict:
    d = df[df.defined].copy()
    d = d.dropna(subset=["PASSED_FIRST_ATTEMPT", "material_reach", "n_sessions", "n_active_days"])
    fit = smf.logit(formula, data=d).fit(disp=False, maxiter=200)
    params = {k: float(v) for k, v in fit.params.items()}
    ci = fit.conf_int()
    cis = {k: [float(ci.loc[k, 0]), float(ci.loc[k, 1])] for k in fit.params.index}
    pvals = {k: float(v) for k, v in fit.pvalues.items()}
    bse = {k: float(v) for k, v in fit.bse.items()}
    return {
        "n": int(fit.nobs),
        "formula": formula,
        "params": params,
        "se": bse,
        "pvalues": pvals,
        "ci95": cis,
        "llf": float(fit.llf),
        "llnull": float(fit.llnull) if fit.llnull is not None else None,
        "prsquared": float(fit.prsquared),
        "converged": bool(fit.mle_retvals.get("converged", True)),
    }


def or_block(res: dict, key: str) -> dict | None:
    if key not in res["params"]:
        return None
    b = res["params"][key]
    lo, hi = res["ci95"][key]
    return {
        "coef": b,
        "or": float(np.exp(b)),
        "or_025": float(np.exp(lo)),
        "or_975": float(np.exp(hi)),
        "se": res["se"][key],
        "p": res["pvalues"][key],
        "or_plus25": float(np.exp(0.25 * b)),
    }


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
    frames = []
    sample = []
    for year, course in pairs:
        log(f"building {year} {course}")
        built = build_environment(year, course, cutoffs)
        frames.append(built["frame"])
        sample.append(built["stats"])
    df = pd.concat(frames, ignore_index=True)
    defined = df[df.defined].copy()

    sample_gate = {
        "n_environments": len(sample),
        "n_learners_with_outcome": int(df.PASSED_FIRST_ATTEMPT.notna().sum()),
        "n_defined": int(defined.shape[0]),
        "environments": sample,
        "viability_preferred": {
            "min_environments": 4,
            "min_learners": 1000,
            "environments_ok": len(sample) >= 4,
            "learners_ok": int(df.PASSED_FIRST_ATTEMPT.notna().sum()) >= 1000,
        },
    }
    gate_path = OUT / "kuleuven_sample_gate.json"
    gate_path.write_text(json.dumps(sample_gate, indent=2))
    log(json.dumps(sample_gate["viability_preferred"]))

    # --- models (protocol already frozen) ---
    f0 = "PASSED_FIRST_ATTEMPT ~ np.log1p(n_sessions) + n_active_days + C(environment)"
    f1 = "PASSED_FIRST_ATTEMPT ~ material_reach + np.log1p(n_sessions) + n_active_days + C(environment)"
    f2 = (
        "PASSED_FIRST_ATTEMPT ~ material_reach + mainpage_reach + "
        "np.log1p(n_sessions) + n_active_days + C(environment)"
    )
    m0 = fit_logit(defined, f0)
    m1 = fit_logit(defined, f1)
    m2 = fit_logit(defined, f2)
    lr_stat = 2 * (m1["llf"] - m0["llf"])
    from math import erf, sqrt

    def chi2_sf(x: float, df_: int = 1) -> float:
        # survival of chi2(1) = 2*(1-Phi(sqrt(x)))
        z = sqrt(max(x, 0.0))
        return float(erfc_approx(z / sqrt(2.0)))

    def erfc_approx(x: float) -> float:
        return float(1 - erf(x))

    env_models = []
    for env, g in defined.groupby("environment"):
        if g.material_reach.nunique() < 2 or g.PASSED_FIRST_ATTEMPT.nunique() < 2 or len(g) < 80:
            env_models.append({"environment": env, "status": "skipped", "n": int(len(g))})
            continue
        fe = "PASSED_FIRST_ATTEMPT ~ material_reach + np.log1p(n_sessions) + n_active_days"
        try:
            r = fit_logit(g, fe)
            env_models.append({"environment": env, "status": "ok", **r, "material": or_block(r, "material_reach")})
        except Exception as e:
            env_models.append({"environment": env, "status": "fail", "error": str(e), "n": int(len(g))})

    loeo = []
    for env in sorted(defined.environment.unique()):
        sub = defined[defined.environment != env]
        r = fit_logit(sub, f1)
        loeo.append({"dropped": env, "n": r["n"], "material": or_block(r, "material_reach")})

    pre = defined[~defined.covid_year]
    covid = defined[defined.covid_year]
    split = {
        "pre_2020": {**fit_logit(pre, f1), "material": or_block(fit_logit(pre, f1), "material_reach")},
        "y2020_21": {**fit_logit(covid, f1), "material": or_block(fit_logit(covid, f1), "material_reach")},
    }

    primary = {
        "protocol": "frozen-before-coefficient; construct is end-of-instruction course-material reach",
        "construct": "end-of-instruction course-material reach",
        "n": m1["n"],
        "baseline": m0,
        "primary": m1,
        "primary_material": or_block(m1, "material_reach"),
        "increment_prsquared": float(m1["prsquared"] - m0["prsquared"]),
        "lr_stat_vs_baseline": float(lr_stat),
        "lr_p_chi2_1": chi2_sf(lr_stat),
        "specificity": m2,
        "specificity_material": or_block(m2, "material_reach"),
        "specificity_mainpage": or_block(m2, "mainpage_reach"),
        "environment_models": env_models,
        "loeo": loeo,
        "covid_split": {
            "pre_2020": {"n": split["pre_2020"]["n"], "material": split["pre_2020"]["material"], "prsquared": split["pre_2020"]["prsquared"]},
            "y2020_21": {"n": split["y2020_21"]["n"], "material": split["y2020_21"]["material"], "prsquared": split["y2020_21"]["prsquared"]},
        },
    }
    dest = OUT / "kuleuven_external_results.json"
    dest.write_text(json.dumps(primary, indent=2))
    log(json.dumps({
        "n": primary["n"],
        "material": primary["primary_material"],
        "increment_prsquared": primary["increment_prsquared"],
        "lr_p": primary["lr_p_chi2_1"],
        "spec_material": primary["specificity_material"],
        "spec_mainpage": primary["specificity_mainpage"],
    }, indent=2))
    log(f"wrote {dest}")


if __name__ == "__main__":
    main()
