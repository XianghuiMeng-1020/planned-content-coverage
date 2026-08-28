"""Shared helpers for breadth-versus-volume and design-reach analyses. Writes only results/v10_value_boundary/."""
from __future__ import annotations

import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

_SRC = Path(__file__).resolve().parents[1]
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from common.paths import OULAD_INTERIM, ROOT  # noqa: E402
from v2.constants import PRIMARY_ASSESSMENT_TYPE  # noqa: E402
from v2.load import load_assessments, load_registration, load_student_vle, load_vle  # noqa: E402
from v2.panel import (  # noqa: E402
    attach_prior_score,
    documented_sites,
    map_next_tma,
    map_prev_tma_date,
    registered_learner_sites,
)
from v3.constants import FAMILY_OUCONTENT  # noqa: E402
from v3.features import clicks_by_learner_site_date  # noqa: E402
from v4.constants import V4_DEVELOPMENT  # noqa: E402
from v5_temporal.run_temporal_safe import collapse_student_assessment  # noqa: E402
from v9_decision.run_decision_utility import (  # noqa: E402
    KEYS,
    LOW_CUT,
    attach_controls_cutoff,
    classify_states,
    coverage_for_cutoff,
)

OUT = ROOT / "results" / "v10_value_boundary"
FORBIDDEN = [
    ROOT / "results" / "v4",
    ROOT / "results" / "v5_temporal_safe",
    ROOT / "results" / "v6_construct_validity",
    ROOT / "results" / "v7_inference",
    ROOT / "results" / "v8_timing",
    ROOT / "results" / "v9_decision_utility",
]
PANEL_PARQUET = OUT / "unit_a_defined_occasions.parquet"
LOG: list[str] = []


def log(msg: str) -> None:
    LOG.append(f"{datetime.now(timezone.utc).isoformat()} {msg}")
    print(msg, flush=True)


def _jd(x):
    if isinstance(x, (np.floating,)):
        return float(x) if np.isfinite(x) else None
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, (np.bool_,)):
        return bool(x)
    raise TypeError(type(x))


def _guard(p: Path) -> Path:
    p = p.resolve()
    assert str(p).startswith(str(OUT.resolve()))
    for bad in FORBIDDEN:
        assert not str(p).startswith(str(bad.resolve()))
    return p


def dump_json(name: str, obj) -> Path:
    OUT.mkdir(parents=True, exist_ok=True)
    p = _guard(OUT / name)
    p.write_text(json.dumps(obj, indent=2, default=_jd))
    log(f"wrote {p.relative_to(ROOT)}")
    return p


def dump_csv(name: str, df: pd.DataFrame) -> Path:
    OUT.mkdir(parents=True, exist_ok=True)
    p = _guard(OUT / name)
    df.to_csv(p, index=False)
    log(f"wrote {p.relative_to(ROOT)} n={len(df)}")
    return p


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def nopp_band(n: int) -> str:
    if n <= 4:
        return "1-4"
    if n <= 8:
        return "5-8"
    return "9+"


def load_sa() -> pd.DataFrame:
    return pd.read_parquet(OULAD_INTERIM / "student_assessment.parquet")


def build_suba_panel(force: bool = False) -> pd.DataFrame:
    """Rebuild Phase-5 SUB-A occasions. Cache to parquet; never writes v9."""
    if PANEL_PARQUET.exists() and not force:
        log(f"reuse {PANEL_PARQUET.relative_to(ROOT)}")
        return pd.read_parquet(PANEL_PARQUET)

    log("rebuild SUB-A panel (Phase-5 logic, v10 cache only)...")
    vle = load_vle()
    assessments = load_assessments()
    reg = load_registration()
    sa, _ = collapse_student_assessment(load_sa())
    sv = load_student_vle()
    tmas = assessments[
        (assessments.assessment_type == PRIMARY_ASSESSMENT_TYPE)
        & assessments.date.notna()
        & assessments.code_module.str.cat(assessments.code_presentation, sep="|").isin(
            {f"{a}|{b}" for a, b in V4_DEVELOPMENT}
        )
    ][["code_module", "code_presentation", "id_assessment", "date"]]
    skel = classify_states(reg, tmas, sa)
    tma_meta = map_prev_tma_date(assessments)
    skel = skel.merge(
        tma_meta.rename(columns={"id_assessment": "next_id_assessment"})[
            ["code_module", "code_presentation", "next_id_assessment", "prev_tma_date"]
        ],
        on=["code_module", "code_presentation", "next_id_assessment"],
        how="left",
    )
    skel["pres"] = skel.code_module + "_" + skel.code_presentation
    primary = skel[skel.state.isin(["submitted", "eligible_nonsubmit"])].copy()
    sites0 = documented_sites(vle, set(V4_DEVELOPMENT), FAMILY_OUCONTENT)
    clicks = clicks_by_learner_site_date(sv, sites0.id_site.tolist())
    panel = registered_learner_sites(reg, sites0)
    panel = map_next_tma(panel, assessments).dropna(subset=["next_id_assessment"])
    panel["next_id_assessment"] = panel.next_id_assessment.astype(int)
    site = panel[KEYS + ["id_site", "window_start_day", "window_end_day", "next_tma_date"]].copy()
    a = primary.copy()
    a["cutoff"] = np.where(a.state == "submitted", a.date_submitted, a.next_tma_date)
    cov = coverage_for_cutoff(site, clicks, a, "cutoff")
    occ = a.merge(cov, on=KEYS, how="left")
    occ["n_opp"] = occ.n_opp.fillna(0).astype(int)
    occ = attach_prior_score(occ, sa, assessments)
    occ = attach_controls_cutoff(occ, sv, "cutoff")
    occ["low_score"] = np.where(occ.score.notna(), (occ.score < LOW_CUT).astype(int), np.nan)
    occ["submitted"] = (occ.state == "submitted").astype(int)
    keep = [
        "code_module",
        "code_presentation",
        "pres",
        "id_student",
        "next_id_assessment",
        "state",
        "submitted",
        "score",
        "low_score",
        "n_opp",
        "never_share",
        "log_clicks",
        "active_rate",
        "prior_score_filled",
        "prior_missing",
        "date_submitted",
        "next_tma_date",
        "cutoff",
    ]
    occ = occ[keep].copy()
    OUT.mkdir(parents=True, exist_ok=True)
    p = _guard(PANEL_PARQUET)
    occ.to_parquet(p, index=False)
    log(f"wrote {p.relative_to(ROOT)} n={len(occ)}")
    return occ
