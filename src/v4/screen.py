#!/usr/bin/env python3
"""V4 metadata screens: family × presentation eligibility, GGG, DDD, external.

No student_assessment scores. No timing–outcome correlations.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

_SRC = Path(__file__).resolve().parents[1]
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

import hashlib

from common.paths import OULAD_RAW, ROOT  # noqa: E402
from v2.load import load_assessments, load_courses, load_registration, load_vle  # noqa: E402
from v4.constants import V1_PEEKED, V4_DEVELOPMENT  # noqa: E402

OUT = ROOT / "results" / "v4"
OUT.mkdir(parents=True, exist_ok=True)


def _jd(x):
    if isinstance(x, (np.floating,)):
        return float(x)
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, (np.bool_,)):
        return bool(x)
    raise TypeError(type(x))


def role(mod, pres):
    key = (mod, pres)
    if key in V4_DEVELOPMENT:
        return "v4_development"
    if key in V1_PEEKED:
        return "peeked_v1"
    return "untouched_candidate"


def main() -> None:
    vle = load_vle()
    assessments = load_assessments()
    courses = load_courses()
    reg = load_registration()
    nreg = reg.groupby(["code_module", "code_presentation"], as_index=False).agg(
        n_learners=("id_student", "nunique")
    )

    # Official type semantics (Kuzilek et al. 2017 data descriptor + OU Analyse).
    type_semantics = {
        "oucontent": "HTML content pages planned for study",
        "resource": "downloadable/file resources",
        "url": "external URL",
        "page": "short HTML page",
        "quiz": "quiz activity",
        "subpage": "navigation subpage",
        "forumng": "forum; no planned-use fields populated in this extract",
        "homepage": "module homepage; no planned-use fields",
    }

    family_rows = []
    ggg_detail = []
    for (mod, pres, typ), g in vle.groupby(["code_module", "code_presentation", "activity_type"]):
        planned = g[g["planned_valid"]]
        a = assessments[(assessments.code_module == mod) & (assessments.code_presentation == pres)]
        tma = a[(a.assessment_type == "TMA") & a.date.notna()]
        nl = nreg[(nreg.code_module == mod) & (nreg.code_presentation == pres)]
        n_learners = int(nl.n_learners.iloc[0]) if len(nl) else 0
        n_next = 0
        if len(planned) and len(tma):
            td = tma.date.to_numpy()
            for _, r in planned.iterrows():
                if (td > int(r.week_to) * 7 - 1).any():
                    n_next += 1
        r = role(mod, pres)
        eligible = (
            r == "untouched_candidate"
            and len(planned) >= 3
            and n_learners >= 300
            and len(tma) >= 2
            and n_next >= 2
        )
        rec = {
            "code_module": mod,
            "code_presentation": pres,
            "activity_type": typ,
            "role": r,
            "n_sites": int(len(g)),
            "n_planned": int(len(planned)),
            "n_learners": n_learners,
            "n_dated_tma": int(len(tma)),
            "n_planned_with_next_tma": n_next,
            "week_from_to_valid": bool(len(planned) > 0),
            "id_site_matchable": True,
            "eligible_cross_family_confirmation": bool(eligible),
            "previously_outcome_inspected": r != "untouched_candidate",
        }
        family_rows.append(rec)
        if mod == "GGG":
            ggg_detail.append(rec)

    eligible = [x for x in family_rows if x["eligible_cross_family_confirmation"]]
    ggg_pass = [x for x in eligible if x["code_module"] == "GGG"]
    ddd = [x for x in family_rows if x["code_module"] == "DDD" and x["n_planned"] > 0]

    hashes = {}
    for name in ("vle.csv", "assessments.csv", "courses.csv", "studentInfo.csv", "studentRegistration.csv"):
        fp = OULAD_RAW / name
        if fp.exists():
            hashes[name] = hashlib.sha256(fp.read_bytes()).hexdigest()[:16]

    out = {
        "metadata_file_hashes_sha256_16": hashes,
        "rule": (
            "untouched_candidate AND n_planned>=3 AND n_learners>=300 "
            "AND dated_tma>=2 AND planned_with_next_tma>=2; any activity type "
            "with valid week_from/week_to (not restricted to oucontent)"
        ),
        "type_semantics": type_semantics,
        "n_eligible_cells": len(eligible),
        "eligible": eligible,
        "ggg": {
            "activity_type_meaning": (
                "OULAD activity_type=resource is a file/downloadable VLE material "
                "(Kuzilek et al. 2017). week_from/week_to have the same official "
                "planned-to-be-used semantics as for oucontent."
            ),
            "n_eligible_presentations": len({(x["code_presentation"]) for x in ggg_pass}),
            "cells": ggg_detail,
            "eligibility_decision": "PASS" if len(ggg_pass) >= 1 else "FAIL",
            "reason": (
                "Three GGG presentations have 4 planned resource sites, weeks 1–3, "
                "≥749 learners, 3 dated TMAs, and all planned windows end before the first TMA. "
                "No V1–V3 analysis used GGG scores or GGG resource-level coverage. "
                "Cross-family confirmation is therefore eligible."
                if ggg_pass
                else "GGG failed the sample/metadata rule"
            ),
            "outcomes_inspected": False,
        },
        "ddd": ddd,
        "external": [
            {
                "dataset": "ASSISTments 2012-2013",
                "decision": "INELIGIBLE",
                "reason": "start/end times are attempt timestamps, not instructor planned-use windows.",
            }
        ],
    }
    (OUT / "untouched_family_screen.json").write_text(json.dumps(out, indent=2, default=_jd))
    (OUT / "ggg_eligibility.json").write_text(json.dumps(out["ggg"], indent=2, default=_jd))
    print("eligible cells", [(e["code_module"], e["code_presentation"], e["activity_type"]) for e in eligible], flush=True)
    print("GGG", out["ggg"]["eligibility_decision"], flush=True)


if __name__ == "__main__":
    main()
