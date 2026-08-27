#!/usr/bin/env python3
"""V4 confirmatory study. Refuses to run until the frozen v4 tag exists."""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

_SRC = Path(__file__).resolve().parents[1]
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from common.paths import ROOT  # noqa: E402
from v2.constants import FAMILY_ALL_DOCUMENTED  # noqa: E402
from v2.load import (  # noqa: E402
    load_assessments,
    load_registration,
    load_student_assessment,
    load_student_vle,
    load_vle,
)
from v2.panel import (  # noqa: E402
    attach_prior_score,
    attach_raw_controls,
    attach_tma_outcomes,
    documented_sites,
    map_next_tma,
    map_prev_tma_date,
    registered_learner_sites,
)
from v2.week_panel import build_week_panel  # noqa: E402
from v3.features import annotate_site_states, clicks_by_learner_site_date, occasion_from_sites  # noqa: E402
from v4.constants import FAMILY_RESOURCE, MIN_OPP, V4_CONFIRMATION  # noqa: E402
from v4.run_development import (  # noqa: E402
    dose_bins,
    fit_hierarchy,
    matching,
    resource_identity_placebo,
    residual_risk,
    scored,
    within_learner,
)

OUT = ROOT / "results" / "v4"
TAG = "planned-resource-coverage-v4-frozen"


def _jd(x):
    if isinstance(x, (np.floating,)):
        return float(x)
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, (np.bool_,)):
        return bool(x)
    raise TypeError(type(x))


def assert_frozen() -> str:
    out = subprocess.check_output(["git", "tag", "-l", TAG], cwd=ROOT, text=True).strip()
    if TAG not in out.splitlines():
        raise SystemExit(f"Refusing to inspect confirmatory outcomes: tag {TAG} not found.")
    return subprocess.check_output(["git", "rev-list", "-n", "1", TAG], cwd=ROOT, text=True).strip()


def load_family(sv, vle, reg, assessments, sa, tma_meta, presentations: set, activity_type: str):
    sites0 = documented_sites(vle, presentations, FAMILY_ALL_DOCUMENTED)
    sites0 = sites0[sites0.activity_type == activity_type].copy()
    clicks = clicks_by_learner_site_date(sv, sites0.id_site.tolist())
    week_panel = build_week_panel(sv, presentations)
    panel = registered_learner_sites(reg, sites0)
    panel = map_next_tma(panel, assessments)
    site = annotate_site_states(panel, clicks)
    occ = occasion_from_sites(site)
    occ = attach_tma_outcomes(occ, sa, tma_meta)
    occ = attach_prior_score(occ, sa, assessments)
    occ = attach_raw_controls(occ, week_panel)
    return sites0, site, occ, clicks


def main() -> None:
    sha = assert_frozen()
    print("frozen", TAG, sha, flush=True)
    vle = load_vle()
    reg = load_registration()
    sa = load_student_assessment()
    assessments = load_assessments()
    tma_meta = map_prev_tma_date(assessments)
    sv = load_student_vle()
    sites0, site, occ, clicks = load_family(
        sv, vle, reg, assessments, sa, tma_meta, set(V4_CONFIRMATION), FAMILY_RESOURCE
    )
    del sv
    print(
        "GGG occasions",
        len(occ),
        "scored",
        int(occ.score.notna().sum()),
        "presentations",
        occ.groupby(["code_module", "code_presentation"]).ngroups,
        flush=True,
    )
    hier = fit_hierarchy(occ)
    wl = within_learner(occ)
    match = matching(occ)
    plc = resource_identity_placebo(site, occ, clicks)
    dose = dose_bins(occ)
    resid = residual_risk(occ)
    work = scored(occ)
    rho, _ = stats.spearmanr(work.share_never, work.score) if len(work) > 5 else (np.nan, np.nan)
    n_rep = int((work.groupby(["code_module", "code_presentation", "id_student"]).size() >= 2).sum()) if len(work) else 0
    out = {
        "tag": TAG,
        "tag_sha": sha,
        "family": FAMILY_RESOURCE,
        "presentations": sorted([list(x) for x in V4_CONFIRMATION]),
        "n_occasions": int(len(occ)),
        "n_scored": int(work.score.notna().sum()) if len(work) else 0,
        "n_learners": int(work.id_student.nunique()) if len(work) else 0,
        "n_presentations": int(work.groupby(["code_module", "code_presentation"]).ngroups) if len(work) else 0,
        "n_repeated_learners": n_rep,
        "spearman": float(rho) if np.isfinite(rho) else None,
        "dose": dose,
        "hierarchy": hier,
        "within": wl,
        "matching": match,
        "identity_placebo": plc,
        "residual": resid,
        "c1_directional": bool(np.isfinite(rho) and rho < 0),
        "c2_controlled": bool(hier.get("in_sample", {}).get("M3", {}).get("coef", 0) < 0),
        "c3_within": (
            "NOT_POWERED"
            if (not wl) or wl.get("insufficient") or wl.get("n_learners", 0) < 50
            else ("PASS" if wl.get("within_coef", 0) < 0 and wl.get("within_ci", [1, 1])[1] < 0 else "FAIL")
        ),
        "c4_specificity": (
            "NOT_FEASIBLE"
            if not plc.get("feasible")
            else ("PASS" if plc.get("actual_more_negative_coef") else "FAIL")
        ),
    }
    p = OUT / "confirmatory.json"
    p.write_text(json.dumps(out, indent=2, default=_jd))
    print("wrote", p, flush=True)
    print(
        "C1",
        out["c1_directional"],
        "C2",
        out["c2_controlled"],
        "C3",
        out["c3_within"],
        "C4",
        out["c4_specificity"],
        "M3",
        hier.get("in_sample", {}).get("M3"),
        flush=True,
    )


if __name__ == "__main__":
    main()
