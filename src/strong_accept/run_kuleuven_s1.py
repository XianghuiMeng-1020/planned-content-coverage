#!/usr/bin/env python3
"""H18 S1 sensitivity: singleton-only timestamps.

Does not overwrite the primary cascade-root results.
Protocol: if S1 flips the sign of the primary material-reach association,
that is a construct-fragility bound.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_kuleuven_external import (  # noqa: E402
    DATA,
    EXTRACT,
    INFO_COURSE,
    OUT,
    eligible_items,
    env_name,
    fit_logit,
    load_cutoffs,
    log,
    or_block,
)


def cascade_keep_s1(logs: pd.DataFrame) -> pd.DataFrame:
    g = logs.copy()
    g["TIMESTAMP"] = pd.to_datetime(g["TIMESTAMP"], errors="coerce")
    g = g.dropna(subset=["TIMESTAMP", "CONTENT_ID", "USER_ID"])
    cols = ["COURSE_ID", "USER_ID", "CONTENT_ID", "SESSION_ID", "TIMESTAMP"]
    sizes = g.groupby(["COURSE_ID", "USER_ID", "TIMESTAMP"]).CONTENT_ID.transform("nunique")
    return g.loc[sizes == 1, cols].drop_duplicates()


def build_s1(year: str, course: str, cutoffs: dict) -> pd.DataFrame:
    part = pd.read_excel(EXTRACT / f"{year}_course_participation.xlsx")
    cont = pd.read_excel(EXTRACT / f"{year}_course_content.xlsx")
    logs = pd.read_csv(EXTRACT / f"{year}_log_activity.csv")
    part = part[part.COURSE_ID == course].copy()
    cont = cont[cont.COURSE_ID == course].copy()
    logs = logs[logs.COURSE_ID == course].copy()
    info_key = INFO_COURSE[course]
    cutoff = cutoffs[(year, info_key)]
    emat = eligible_items(cont, cutoff, "Course Material")
    kept = cascade_keep_s1(logs)
    kept_pre = kept[kept.TIMESTAMP < cutoff].copy()
    n = len(emat)
    if n == 0:
        part["material_reach"] = 0.0
        part["defined"] = False
    else:
        acc = kept_pre[kept_pre.CONTENT_ID.isin(set(emat.CONTENT_ID))]
        hits = acc.groupby("USER_ID").CONTENT_ID.nunique()
        part["material_reach"] = (hits / n).reindex(part.USER_ID).fillna(0.0).values
        part["defined"] = True
    sess = kept_pre.groupby("USER_ID").SESSION_ID.nunique()
    days = kept_pre.groupby("USER_ID").TIMESTAMP.apply(lambda s: s.dt.normalize().nunique())
    part["environment"] = env_name(year, course)
    part["n_sessions"] = sess.reindex(part.USER_ID).fillna(0).astype(int).values
    part["n_active_days"] = days.reindex(part.USER_ID).fillna(0).astype(int).values
    part["n_eligible_material"] = n
    part["kept_events_s1"] = int(len(kept_pre))
    return part, int(len(kept_pre)), n


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
    env_stats = []
    for year, course in pairs:
        log(f"S1 building {year} {course}")
        frame, n_kept, n_elig = build_s1(year, course, cutoffs)
        frames.append(frame)
        env_stats.append(
            {
                "environment": env_name(year, course),
                "eligible_material": n_elig,
                "kept_events_s1_before_cutoff": n_kept,
                "reach_mean": float(frame.loc[frame.defined, "material_reach"].mean()),
            }
        )
    df = pd.concat(frames, ignore_index=True)
    defined = df[df.defined].copy()
    f1 = (
        "PASSED_FIRST_ATTEMPT ~ material_reach + np.log1p(n_sessions) "
        "+ n_active_days + C(environment)"
    )
    m1 = fit_logit(defined, f1)
    material = or_block(m1, "material_reach")
    primary_sign = 1.0  # frozen primary cascade-root coefficient is positive
    flipped = material["coef"] * primary_sign < 0
    out = {
        "rule": "S1 singleton-only timestamps",
        "n": m1["n"],
        "primary_material": material,
        "sign_flipped_vs_primary": flipped,
        "environments": env_stats,
        "prsquared": m1["prsquared"],
        "converged": m1["converged"],
    }
    dest = OUT / "kuleuven_s1_sensitivity.json"
    dest.write_text(json.dumps(out, indent=2))
    log(json.dumps(out, indent=2))
    log(f"wrote {dest}")


if __name__ == "__main__":
    main()
