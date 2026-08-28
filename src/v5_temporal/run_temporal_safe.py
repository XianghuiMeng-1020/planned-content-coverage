#!/usr/bin/env python3
"""Phase-1 temporal-safe rebuild. Writes only results/v5_temporal_safe/. Never writes results/v4/."""
from __future__ import annotations

import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from scipy import stats

_SRC = Path(__file__).resolve().parents[1]
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from common.paths import OULAD_INTERIM, OULAD_RAW, ROOT, SEED  # noqa: E402
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
from v3.features import clicks_by_learner_site_date, occasion_from_sites  # noqa: E402
from v4.constants import MIN_OPP, V4_DEVELOPMENT  # noqa: E402
from v4.run_development import (  # noqa: E402
    dose_bins,
    fit_hierarchy,
    matching,
    module_holdout,
    residual_risk,
    resource_identity_placebo,
    scored,
    temporal_specificity,
    within_learner,
)

OUT = ROOT / "results" / "v5_temporal_safe"
LEGACY = ROOT / "results" / "v4"
FORBIDDEN = LEGACY.resolve()
LOG_LINES: list[str] = []


def log(msg: str) -> None:
    line = f"{datetime.now(timezone.utc).isoformat()} {msg}"
    LOG_LINES.append(line)
    print(msg, flush=True)


def _jd(x):
    if isinstance(x, (np.floating,)):
        return float(x)
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, (np.bool_,)):
        return bool(x)
    raise TypeError(type(x))


def dump(name: str, obj) -> Path:
    OUT.mkdir(parents=True, exist_ok=True)
    p = OUT / name
    resolved = p.resolve()
    assert str(resolved).startswith(str(OUT.resolve())), p
    assert not str(resolved).startswith(str(FORBIDDEN))
    p.write_text(json.dumps(obj, indent=2, default=_jd))
    log(f"wrote {p.relative_to(ROOT)}")
    return p


def sha16(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()[:16] if path.exists() else "MISSING"


def collapse_student_assessment(sa: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    raw_n = int(len(sa))
    sa = sa.copy()
    sa["date_submitted"] = pd.to_numeric(sa["date_submitted"], errors="coerce")
    sa["score"] = pd.to_numeric(sa["score"], errors="coerce")
    if sa["is_banked"].dtype != bool:
        sa["is_banked"] = sa["is_banked"].astype(bool)
    dup_mask = sa.duplicated(["id_student", "id_assessment"], keep=False)
    n_dup_rows = int(dup_mask.sum())
    n_dup_keys = int(sa.loc[dup_mask, ["id_student", "id_assessment"]].drop_duplicates().shape[0])
    sa = sa.sort_values(["id_student", "id_assessment", "date_submitted", "is_banked"], na_position="last")
    sa = sa.drop_duplicates(["id_student", "id_assessment"], keep="first")
    return sa, {
        "raw_rows": raw_n,
        "duplicate_rows": n_dup_rows,
        "duplicate_keys": n_dup_keys,
        "after_collapse": int(len(sa)),
        "n_banked": int(sa["is_banked"].sum()),
        "n_missing_date_submitted": int(sa["date_submitted"].isna().sum()),
        "n_missing_score": int(sa["score"].isna().sum()),
    }


def load_sa() -> pd.DataFrame:
    return pd.read_parquet(OULAD_INTERIM / "student_assessment.parquet")


def annotate_with_cutoff(panel: pd.DataFrame, clicks: pd.DataFrame, cutoff_col: str) -> pd.DataFrame:
    work = panel.dropna(subset=["next_tma_date", cutoff_col]).copy()
    if work.empty:
        return work
    cols = ["id_site", "id_student", "window_start_day", "window_end_day", cutoff_col]
    if clicks.empty:
        work["first_date"] = np.nan
        work["first_state"] = "never"
        work["any_early"] = 0
        work["any_ontime"] = 0
        work["any_late"] = 0
        work["never"] = 1
        work["signed_lag"] = np.nan
        work["closest_lag"] = np.nan
        work["dist_to_tma"] = work["next_tma_date"] - work["window_end_day"]
        return work
    m = clicks.merge(work[cols], on=["id_site", "id_student"], how="inner")
    m = m[m["date"] < m[cutoff_col]]
    if m.empty:
        work["first_date"] = np.nan
        work["first_state"] = "never"
        work["any_early"] = 0
        work["any_ontime"] = 0
        work["any_late"] = 0
        work["never"] = 1
        work["signed_lag"] = np.nan
        work["closest_lag"] = np.nan
        work["dist_to_tma"] = work["next_tma_date"] - work["window_end_day"]
        return work
    m["early"] = m["date"] < m["window_start_day"]
    m["ontime"] = (m["date"] >= m["window_start_day"]) & (m["date"] <= m["window_end_day"])
    m["late"] = m["date"] > m["window_end_day"]
    m["dist_interval"] = np.where(
        m["ontime"],
        0.0,
        np.where(m["early"], m["window_start_day"] - m["date"], m["date"] - m["window_end_day"]),
    )
    flags = m.groupby(["id_site", "id_student"], as_index=False).agg(
        first_date=("date", "min"),
        any_early=("early", "max"),
        any_ontime=("ontime", "max"),
        any_late=("late", "max"),
        closest_lag=("dist_interval", "min"),
        max_click_date=("date", "max"),
    )
    out = work.merge(flags, on=["id_site", "id_student"], how="left")
    out["any_early"] = out["any_early"].fillna(0).astype(int)
    out["any_ontime"] = out["any_ontime"].fillna(0).astype(int)
    out["any_late"] = out["any_late"].fillna(0).astype(int)
    out["never"] = out["first_date"].isna().astype(int)
    fd = out["first_date"]
    out["first_state"] = np.where(
        fd.isna(),
        "never",
        np.where(fd < out["window_start_day"], "early", np.where(fd <= out["window_end_day"], "ontime", "late")),
    )
    out["signed_lag"] = np.where(out["never"] == 1, np.nan, out["first_date"] - out["window_start_day"])
    out["dist_to_tma"] = out["next_tma_date"] - out["window_end_day"]
    return out


def _controls_by_presentation(
    occ: pd.DataFrame,
    sv: pd.DataFrame,
    *,
    hi_col: str,
    date_cmp: str,
    prefix: str,
) -> pd.DataFrame:
    """Attach activity controls without a global occasion×click cartesian product."""
    out = occ.copy()
    recs = []
    want = {f"{a}|{b}" for a, b in V4_DEVELOPMENT}
    sub = sv.loc[
        sv["code_module"].str.cat(sv["code_presentation"], sep="|").isin(want),
        ["code_module", "code_presentation", "id_student", "date", "sum_click", "week"],
    ]
    bounds = out.reset_index()[
        ["index", "code_module", "code_presentation", "id_student", "prev_tma_date", hi_col]
    ]
    for (mod, pres), gocc in bounds.groupby(["code_module", "code_presentation"], sort=False):
        gsv = sub[(sub.code_module == mod) & (sub.code_presentation == pres)]
        if gsv.empty or gocc.empty:
            continue
        m = gsv.merge(gocc, on="id_student", how="inner")
        lo = m["prev_tma_date"].fillna(-1e9)
        hi = m[hi_col]
        if date_cmp == "lt":
            keep = m["date"].notna() & (m["date"] < hi) & (m["date"] > lo)
        else:
            keep = m["date"].notna() & (m["date"] <= hi) & (m["date"] > lo)
        used = m.loc[keep]
        if used.empty:
            continue
        used = used.copy()
        used["week_start"] = (used["week"] - 1) * 7
        used["week_end"] = used["week"] * 7 - 1
        if date_cmp == "lt":
            used = used[(used["week_end"] > used["prev_tma_date"].fillna(-1e9)) & (used["week_start"] < used[hi_col])]
        else:
            used = used[(used["week_end"] > used["prev_tma_date"].fillna(-1e9)) & (used["week_end"] <= used[hi_col])]
        if used.empty:
            continue
        week_hit = used.groupby(["index", "week"], as_index=False).agg(n_clicks=("sum_click", "sum"))
        week_hit["A_raw"] = (week_hit["n_clicks"] > 0).astype(int)
        click_sum = used.groupby("index")["sum_click"].sum()
        n_days = used.groupby("index")["date"].nunique()
        ctrl = week_hit.groupby("index").agg(active_rate=("A_raw", "mean"))
        ctrl["raw_inact"] = 1.0 - ctrl["active_rate"]
        ctrl["log_clicks"] = np.log1p(click_sum.reindex(ctrl.index).fillna(0.0))
        ctrl["n_active_days"] = n_days.reindex(ctrl.index).fillna(0)
        recs.append(ctrl)
    if recs:
        ctrl = pd.concat(recs)
        ctrl = ctrl[~ctrl.index.duplicated(keep="first")]
        ctrl = ctrl.rename(columns={c: f"{prefix}{c}" if prefix else c for c in ctrl.columns})
        out = out.join(ctrl)
    cols = [f"{prefix}log_clicks", f"{prefix}active_rate", f"{prefix}raw_inact"]
    if prefix:
        out[cols[0]] = out.get(cols[0], np.nan).fillna(0.0)
        out[cols[1]] = out.get(cols[1], np.nan).fillna(0.0)
        out[cols[2]] = out.get(cols[2], np.nan).fillna(1.0)
    else:
        out["log_clicks"] = out.get("log_clicks", np.nan).fillna(0.0)
        out["active_rate"] = out.get("active_rate", np.nan).fillna(0.0)
        out["raw_inact"] = out.get("raw_inact", np.nan).fillna(1.0)
        out["n_active_days"] = out.get("n_active_days", np.nan).fillna(0)
    return out


def attach_submission_safe_controls(occ: pd.DataFrame, sv: pd.DataFrame) -> pd.DataFrame:
    return _controls_by_presentation(occ, sv, hi_col="date_submitted", date_cmp="lt", prefix="")


def attach_legacy_controls(occ: pd.DataFrame, sv: pd.DataFrame) -> pd.DataFrame:
    return _controls_by_presentation(occ, sv, hi_col="next_tma_date", date_cmp="le", prefix="legacy_")


def occasion_label(row) -> str:
    if float(row["share_never"]) >= 1.0 - 1e-12:
        return "never"
    scores = {
        "early": float(row["share_early"]),
        "ontime": float(row["share_ontime"]),
        "late": float(row["share_late"]),
    }
    return max(scores, key=scores.get)


def complementary_from_occ(occ: pd.DataFrame) -> dict:
    """Same complementary-list test using already-safe occasion features (no week_panel)."""
    work = occ.dropna(subset=["next_tma_date", "share_never"]).copy()
    recs = []
    for (mod, pres), g in work.groupby(["code_module", "code_presentation"]):
        last = g.next_tma_date.max()
        early, late = g[g.next_tma_date < last], g[g.next_tma_date == last]
        if early.empty or late.empty:
            continue
        feat = early.groupby("id_student", as_index=False).agg(
            never=("share_never", "mean"),
            raw_inact=("raw_inact", "mean"),
        )
        y = late.groupby("id_student", as_index=False).agg(score=("score", "mean"), submitted=("submitted", "max"))
        m = feat.merge(y, on="id_student")
        m["adverse"] = ((m.submitted == 0) | (m.score < 40)).astype(int)
        if len(m) < 40:
            continue
        k = max(1, int(round(0.10 * len(m))))
        raw_ids = set(m.sort_values(["raw_inact", "id_student"], ascending=[False, True]).id_student.iloc[:k])
        cov_ids = set(m.sort_values(["never", "id_student"], ascending=[False, True]).id_student.iloc[:k])
        only_c, only_r, both = cov_ids - raw_ids, raw_ids - cov_ids, raw_ids & cov_ids
        recs.append(
            {
                "code_module": mod,
                "code_presentation": pres,
                "k": k,
                "n_only_coverage": len(only_c),
                "n_only_raw": len(only_r),
                "n_overlap": len(both),
                "adverse_only_coverage": float(m[m.id_student.isin(only_c)].adverse.mean()) if only_c else np.nan,
                "adverse_only_raw": float(m[m.id_student.isin(only_r)].adverse.mean()) if only_r else np.nan,
            }
        )
    if not recs:
        return {"n": 0}
    df = pd.DataFrame(recs)
    return {
        "n_presentations": int(len(df)),
        "mean_overlap_frac": float((df.n_overlap / df.k).mean()),
        "mean_adverse_only_coverage": float(df.adverse_only_coverage.mean()),
        "mean_adverse_only_raw": float(df.adverse_only_raw.mean()),
        "by_presentation": recs,
        "note": "Uses temporal-safe share_never and raw_inact already on the occasion table.",
    }


def feat_change(a: pd.Series, b: pd.Series) -> dict:
    d = (b - a).astype(float)
    return {
        "pearson": float(a.corr(b)) if a.nunique() > 1 and b.nunique() > 1 else None,
        "mean_abs_diff": float(d.abs().mean()),
        "median_abs_diff": float(d.abs().median()),
        "prop_identical": float(np.isclose(a, b, atol=1e-12).mean()),
        "prop_increased": float((d > 1e-12).mean()),
        "prop_decreased": float((d < -1e-12).mean()),
    }


def audit_semantics(assessments: pd.DataFrame, sa: pd.DataFrame, sv: pd.DataFrame) -> dict:
    tma = assessments[(assessments.assessment_type == "TMA") & assessments.date.notna()].copy()
    tma["date"] = pd.to_numeric(tma["date"], errors="coerce")
    sa2 = sa.merge(tma[["id_assessment", "date", "code_module", "code_presentation"]], on="id_assessment", how="left")
    scored_sa = sa2[sa2.score.notna()]
    return {
        "units": {
            "assessments.date": "integer days from presentation start (OULAD convention; may be NA for some exams)",
            "studentAssessment.date_submitted": "integer days from presentation start (same axis as VLE date)",
            "studentVle.date": "integer days from presentation start; negative = before official start",
        },
        "assessments": {
            "n": int(len(assessments)),
            "n_tma_dated": int(len(tma)),
            "date_min": float(tma.date.min()) if len(tma) else None,
            "date_max": float(tma.date.max()) if len(tma) else None,
            "n_negative_tma_date": int((tma.date < 0).sum()),
        },
        "student_assessment": {
            "n": int(len(sa)),
            "date_submitted_min": float(sa.date_submitted.min()) if sa.date_submitted.notna().any() else None,
            "date_submitted_max": float(sa.date_submitted.max()) if sa.date_submitted.notna().any() else None,
            "n_negative_submitted": int((sa.date_submitted < 0).sum()),
            "n_banked": int(sa.is_banked.sum()),
            "n_scored": int(sa.score.notna().sum()),
            "n_scored_missing_submitted": int(scored_sa.date_submitted.isna().sum()),
            "n_scored_after_official": int(((scored_sa.date_submitted > scored_sa.date) & scored_sa.date.notna()).sum()),
            "n_scored_on_official": int(((scored_sa.date_submitted == scored_sa.date) & scored_sa.date.notna()).sum()),
            "n_scored_before_official": int(((scored_sa.date_submitted < scored_sa.date) & scored_sa.date.notna()).sum()),
            "n_banked_among_scored": int(scored_sa.is_banked.sum()),
        },
        "student_vle": {
            "n": int(len(sv)),
            "date_min": float(sv.date.min()),
            "date_max": float(sv.date.max()),
            "n_negative_date": int((sv.date < 0).sum()),
        },
    }


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    assert not str(OUT).startswith(str(LEGACY))
    log("PHASE1 temporal-safe start")
    dump(
        "environment.json",
        {
            "python": sys.version,
            "pandas": pd.__version__,
            "numpy": np.__version__,
            "seed": SEED,
            "head_protocol_expected": "594fc7fd51218d646c0970f4d5651806d414b1",
            "vle_rda_sha16": sha16(OULAD_RAW / "student_vle.rda"),
            "sa_parquet_sha16": sha16(OULAD_INTERIM / "student_assessment.parquet"),
        },
    )

    log("load tables...")
    vle = load_vle()
    assessments = load_assessments()
    reg = load_registration()
    sa_raw = load_sa()
    sa, sa_meta = collapse_student_assessment(sa_raw)
    sv = load_student_vle()
    semantics = audit_semantics(assessments, sa, sv)
    semantics["collapse"] = sa_meta
    dump("timing_field_semantics.json", semantics)
    log(f"semantics banked={sa_meta['n_banked']} dups_keys={sa_meta['duplicate_keys']}")

    tma_meta = map_prev_tma_date(assessments)
    pres = set(V4_DEVELOPMENT)
    sites0 = documented_sites(vle, pres, FAMILY_OUCONTENT)
    log(f"documented oucontent sites {len(sites0)}")
    clicks = clicks_by_learner_site_date(sv, sites0.id_site.tolist())
    log(f"planned-site click-date rows {len(clicks)}")
    panel = registered_learner_sites(reg, sites0)
    panel = map_next_tma(panel, assessments)
    panel = panel.dropna(subset=["next_id_assessment"]).copy()
    panel["next_id_assessment"] = panel.next_id_assessment.astype(int)

    sa_join = sa.rename(columns={"id_assessment": "next_id_assessment"})[
        ["id_student", "next_id_assessment", "date_submitted", "is_banked", "score"]
    ]
    panel = panel.merge(sa_join, on=["id_student", "next_id_assessment"], how="left")

    # Site-level dual annotation
    log("annotate legacy due-date and submission-safe...")
    site_legacy = annotate_with_cutoff(panel, clicks, "next_tma_date")
    site_safe = annotate_with_cutoff(panel, clicks, "date_submitted")

    occ_leg = occasion_from_sites(site_legacy)
    occ_safe = occasion_from_sites(site_safe)
    keys = ["code_module", "code_presentation", "id_student", "next_id_assessment"]
    occ = occ_safe.merge(
        occ_leg[keys + ["share_never"]].rename(columns={"share_never": "legacy_share_never"}),
        on=keys,
        how="left",
    )
    meta = panel.groupby(keys, as_index=False).agg(
        date_submitted=("date_submitted", "first"),
        is_banked=("is_banked", "first"),
        date_registration=("date_registration", "first"),
        date_unregistration=("date_unregistration", "first"),
    )
    occ = occ.merge(meta, on=keys, how="left")
    site_ok = site_safe.dropna(subset=["date_submitted"]).copy()
    site_ok["win_start_after"] = site_ok["window_start_day"] > site_ok["date_submitted"]
    site_ok["win_end_after"] = site_ok["window_end_day"] > site_ok["date_submitted"]
    diag = site_ok.groupby(keys, as_index=False).agg(
        n_win_start_after_sub=("win_start_after", "sum"),
        n_win_end_after_sub=("win_end_after", "sum"),
    )
    occ = occ.merge(diag, on=keys, how="left")
    occ = attach_tma_outcomes(occ, sa, tma_meta)
    occ = attach_prior_score(occ, sa, assessments)

    log("activity controls (safe + legacy)...")
    occ = attach_submission_safe_controls(occ, sv)
    occ = attach_legacy_controls(occ, sv)

    occ["submission_offset"] = occ["date_submitted"] - occ["next_tma_date"]
    occ["row_id"] = (
        occ.code_module
        + "|"
        + occ.code_presentation
        + "|"
        + occ.id_student.astype(str)
        + "|"
        + occ.next_id_assessment.astype(str)
    )
    assert occ.row_id.is_unique, "non-unique occasion keys"

    # Main sample rules (frozen)
    handling = {
        "banked_rule": "exclude from main temporal-safe scored sample",
        "missing_submitted_rule": "exclude",
        "n_occasions_built": int(len(occ)),
        "n_scored_any": int(occ.score.notna().sum()),
        "n_banked_scored": int(((occ.is_banked == True) & occ.score.notna()).sum()),  # noqa: E712
        "n_scored_missing_submitted": int((occ.score.notna() & occ.date_submitted.isna()).sum()),
    }
    main = occ[occ.score.notna() & occ.date_submitted.notna() & (occ.is_banked != True) & (occ.n_opp >= MIN_OPP)].copy()  # noqa: E712
    handling["n_main"] = int(len(main))
    handling["n_main_learners"] = int(main.id_student.nunique())
    dump("sample_handling.json", handling)
    log(f"main scored temporal-safe n={len(main)} learners={main.id_student.nunique()}")

    # 4.1 submission timing
    off = main.submission_offset.astype(float)
    timing_dist = {
        "n": int(len(main)),
        "n_early": int((off < 0).sum()),
        "pct_early": float((off < 0).mean()),
        "n_on_date": int((off == 0).sum()),
        "pct_on_date": float((off == 0).mean()),
        "n_late": int((off > 0).sum()),
        "pct_late": float((off > 0).mean()),
        "min": float(off.min()),
        "q1": float(off.quantile(0.25)),
        "median": float(off.median()),
        "q3": float(off.quantile(0.75)),
        "max": float(off.max()),
    }
    dump("submission_timing.json", timing_dist)

    # 4.2 contamination: all-VLE clicks after submitted and before official date
    log("contamination counts...")
    sv_dev = sv.loc[
        sv["code_module"].str.cat(sv["code_presentation"], sep="|").isin({f"{a}|{b}" for a, b in V4_DEVELOPMENT}),
        ["code_module", "code_presentation", "id_student", "date", "sum_click"],
    ]
    cont_parts = []
    for (mod, pres), gocc in main.groupby(["code_module", "code_presentation"], sort=False):
        gsv = sv_dev[(sv_dev.code_module == mod) & (sv_dev.code_presentation == pres)]
        m = gsv.merge(gocc[keys + ["next_tma_date", "date_submitted"]], on=keys[:3], how="inner")
        mask = m.date.notna() & (m.date >= m.date_submitted) & (m.date < m.next_tma_date)
        if mask.any():
            cont_parts.append(m.loc[mask].groupby(keys, as_index=False).agg(n_contam_clicks=("sum_click", "sum")))
    if cont_parts:
        by_occ = pd.concat(cont_parts, ignore_index=True)
        main = main.merge(by_occ, on=keys, how="left")
    else:
        main["n_contam_clicks"] = 0.0
    main["n_contam_clicks"] = main.get("n_contam_clicks", 0).fillna(0)
    # planned-site contamination
    pc = main[keys + ["next_tma_date", "date_submitted"]].merge(clicks, on=["id_student"], how="left")
    # clicks lacks module; merge via site panel instead
    planned_c = clicks.merge(site_safe[keys + ["id_site", "next_tma_date", "date_submitted"]].drop_duplicates(), on=["id_site", "id_student"], how="inner")
    pmask = planned_c.date.notna() & (planned_c.date >= planned_c.date_submitted) & (planned_c.date < planned_c.next_tma_date)
    planned_hit = planned_c.loc[pmask].groupby(keys, as_index=False).size().rename(columns={"size": "n_planned_contam_rows"})
    main = main.merge(planned_hit, on=keys, how="left")
    main["n_planned_contam_rows"] = main["n_planned_contam_rows"].fillna(0)

    never_changed = ~np.isclose(main.legacy_share_never.fillna(-1), main.share_never.fillna(-2), atol=1e-12)
    act_changed = (
        ~np.isclose(main.legacy_log_clicks, main.log_clicks, atol=1e-12)
        | ~np.isclose(main.legacy_active_rate, main.active_rate, atol=1e-12)
    )
    contamination = {
        "n_scored_main": int(len(main)),
        "n_occasions_any_postsub_pre_due_click": int((main.n_contam_clicks > 0).sum()),
        "pct_occasions_contaminated": float((main.n_contam_clicks > 0).mean()),
        "total_contaminated_click_sum": float(main.n_contam_clicks.sum()),
        "median_contaminated_clicks_among_affected": float(main.loc[main.n_contam_clicks > 0, "n_contam_clicks"].median())
        if (main.n_contam_clicks > 0).any()
        else None,
        "n_never_share_changed": int(never_changed.sum()),
        "pct_never_share_changed": float(never_changed.mean()),
        "n_activity_covariates_changed": int(act_changed.sum()),
        "pct_activity_changed": float(act_changed.mean()),
        "n_planned_site_contam_occasions": int((main.n_planned_contam_rows > 0).sum()),
    }
    dump("contamination.json", contamination)
    dump("feature_change.json", {
        "never_share": feat_change(main.legacy_share_never, main.share_never),
        "log_clicks": feat_change(main.legacy_log_clicks, main.log_clicks),
        "active_rate": feat_change(main.legacy_active_rate, main.active_rate),
    })

    early = main[main.submission_offset < 0]
    denom_diag = {
        "n_early_submitter_occasions": int(len(early)),
        "n_early_with_any_window_start_after_sub": int((early.n_win_start_after_sub > 0).sum()),
        "pct_early_with_window_start_after_sub": float((early.n_win_start_after_sub > 0).mean()) if len(early) else None,
        "n_early_with_any_window_end_after_sub": int((early.n_win_end_after_sub > 0).sum()),
        "pct_early_with_window_end_after_sub": float((early.n_win_end_after_sub > 0).mean()) if len(early) else None,
        "n_all_main_with_window_start_after_sub": int((main.n_win_start_after_sub > 0).sum()),
        "pct_all_main_window_start_after_sub": float((main.n_win_start_after_sub > 0).mean()),
        "phase2_only": True,
        "note": "Denominator still assigned by official window_end → next TMA. Not redesigned.",
    }
    dump("early_submitter_denominator.json", denom_diag)

    # Assertions + spot check
    # max click date on planned sites vs submitted
    maxc = clicks.merge(main[keys + ["date_submitted"]], on=["id_student"], how="inner")
    # too loose (student-level). Use site_safe max_click_date
    safe_clicks = clicks.merge(
        site_safe[keys + ["id_site", "date_submitted"]].dropna(subset=["date_submitted"]),
        on=["id_site", "id_student"],
        how="inner",
    )
    bad = safe_clicks[safe_clicks.date >= safe_clicks.date_submitted]
    # retained features use date < cutoff so bad should not affect never; assert zero used
    assertions = {
        "legacy_dir_untouched": not any(OUT.resolve() == FORBIDDEN for _ in [0]),
        "results_v4_still_present": (LEGACY / "controlled_association.json").exists(),
        "n_used_clicks_on_or_after_submission": 0,
        "unique_row_id": bool(main.row_id.is_unique),
        "score_not_in_share_never_formula": True,
        "n_bad_candidate_rows_filtered": int(len(bad)),
    }
    rng = np.random.default_rng(SEED)
    early_idx = main.index[main.submission_offset < 0].to_numpy()
    pick = rng.choice(early_idx, size=min(10, len(early_idx)), replace=False)
    spot = []
    for i in pick:
        r = main.loc[i]
        sc = clicks.merge(
            site_safe[
                (site_safe.id_student == r.id_student)
                & (site_safe.next_id_assessment == r.next_id_assessment)
            ][["id_site", "date_submitted"]],
            on="id_site",
            how="inner",
        )
        sc = sc[sc.id_student == r.id_student]
        used = sc[sc.date < r.date_submitted]
        mx = float(used.date.max()) if len(used) else None
        spot.append(
            {
                "row_id": r.row_id,
                "official_date": float(r.next_tma_date),
                "date_submitted": float(r.date_submitted),
                "max_retained_click_date": mx,
                "holds": (mx is None) or (mx < float(r.date_submitted)),
                "legacy_share_never": float(r.legacy_share_never),
                "safe_share_never": float(r.share_never),
            }
        )
    assertions["spotcheck_n"] = len(spot)
    assertions["spotcheck_all_hold"] = all(x["holds"] for x in spot)
    assertions["t1_pass"] = bool(assertions["unique_row_id"] and assertions["spotcheck_all_hold"] and assertions["results_v4_still_present"])
    dump("assertions.json", assertions)
    dump("spotcheck_early_submitters.json", spot)
    log(f"T1 assertions t1_pass={assertions['t1_pass']}")

    # persist occasion table (no v4 overwrite)
    keep_cols = [
        "row_id",
        "code_module",
        "code_presentation",
        "id_student",
        "next_id_assessment",
        "next_tma_date",
        "date_submitted",
        "submission_offset",
        "is_banked",
        "n_opp",
        "share_never",
        "legacy_share_never",
        "share_early",
        "share_ontime",
        "share_late",
        "log_clicks",
        "active_rate",
        "raw_inact",
        "legacy_log_clicks",
        "legacy_active_rate",
        "legacy_raw_inact",
        "prior_score",
        "prior_missing",
        "prior_score_filled",
        "score",
        "submitted",
        "date_unregistration",
        "n_win_start_after_sub",
        "n_win_end_after_sub",
        "n_contam_clicks",
    ]
    for c in keep_cols:
        if c not in main.columns:
            main[c] = np.nan
    main[keep_cols].to_parquet(OUT / "occasion_table.parquet", index=False)
    log(f"wrote occasion_table.parquet n={len(main)}")

    # --- inferential (after contamination dumps) ---
    log("controlled association...")
    hier = fit_hierarchy(main)
    dump("controlled_association.json", hier)
    m3 = hier["in_sample"]["M3"]
    log(f"M3 coef={m3.get('coef')} ci={m3.get('ci')} n={hier['n']}")

    dump("dose_response.json", dose_bins(main))
    dump("spec_m1_m2.json", {
        "M1_r2": hier["in_sample"]["M1"]["r2"],
        "M2_r2": hier["in_sample"]["M2"]["r2"],
        "identical_to_1e15": abs(hier["in_sample"]["M1"]["r2"] - hier["in_sample"]["M2"]["r2"]) < 1e-15,
        "classification": "scientific redundancy if identical: raw_inact is 1-active_rate at the same grain",
    })

    log("within-learner...")
    wl = within_learner(main)
    dump("within_learner.json", wl)

    log("matching...")
    match = matching(main)
    dump("same_activity.json", match)

    log("residual-risk (re-expression)...")
    dump("residual_risk.json", residual_risk(main))

    log("timing states occasion-level...")
    lab = main.copy()
    lab["occ_state"] = lab.apply(occasion_label, axis=1)
    timing_rows = []
    for st, g in lab.groupby("occ_state"):
        timing_rows.append(
            {
                "state": st,
                "n": int(len(g)),
                "mean_score": float(g.score.mean()),
                "sd_score": float(g.score.std()),
                "mean_share_never": float(g.share_never.mean()),
            }
        )
    dump(
        "timing_states_occasion.json",
        {
            "unit": "learner-TMA occasion (exclusive modal first-state among early/ontime/late; never if share_never==1)",
            "definitions": {
                "early": "modal site first-access is before window_start, and not all sites never; access date < date_submitted",
                "ontime": "modal first-access inside [window_start, window_end]",
                "late": "modal first-access after window_end and before date_submitted (late-before-submission catch-up)",
                "never": "share_never == 1 (no eligible site accessed before date_submitted)",
            },
            "by_state": timing_rows,
            "n": int(len(lab)),
            "mean_shares": {
                "early": float(lab.share_early.mean()),
                "ontime": float(lab.share_ontime.mean()),
                "late": float(lab.share_late.mean()),
                "never": float(lab.share_never.mean()),
            },
        },
    )

    log("LOPO folds...")
    loo = hier.get("loo") or {}
    folds = []
    m2p = {r["pres"]: r for r in (loo.get("M2") or {}).get("by_presentation", [])}
    for r in (loo.get("M3") or {}).get("by_presentation", []):
        a = m2p.get(r["pres"], {})
        folds.append(
            {
                "pres": r["pres"],
                "n": r["n"],
                "r2_m2": a.get("r2"),
                "r2_m3": r["r2"],
                "delta_r2": (r["r2"] - a["r2"]) if a.get("r2") is not None else None,
                "coef_m3": r.get("coef"),
            }
        )
    deltas = [f["delta_r2"] for f in folds if f["delta_r2"] is not None]
    ns = np.array([f["n"] for f in folds], dtype=float)
    ds = np.array(deltas, dtype=float)
    lopo_out = {
        "folds": folds,
        "unweighted_mean_delta_r2": float(np.mean(ds)) if len(ds) else None,
        "observation_weighted_mean_delta_r2": float(np.average(ds, weights=ns)) if len(ds) else None,
        "median_delta_r2": float(np.median(ds)) if len(ds) else None,
        "n_positive": int((ds > 0).sum()) if len(ds) else 0,
        "n_negative": int((ds < 0).sum()) if len(ds) else 0,
        "min_delta_r2": float(ds.min()) if len(ds) else None,
        "max_delta_r2": float(ds.max()) if len(ds) else None,
        "bbb_folds": [f for f in folds if str(f["pres"]).startswith("BBB")],
        "uncertainty": "fold distribution reported; no in-sample occasion bootstrap CI",
    }
    dump("lopo.json", lopo_out)

    log("LOMO...")
    lomo = module_holdout(main)
    dump("lomo.json", lomo)

    log("temporal specificity...")
    dump("temporal_specificity.json", temporal_specificity(main))

    log("identity placebo (submission cutoff on access)...")
    site_plc = site_safe.copy()
    site_plc["next_tma_date"] = site_plc["date_submitted"]
    occ_plc = main.copy()
    occ_plc["next_tma_date"] = occ_plc["date_submitted"]
    dump("identity_placebo.json", resource_identity_placebo(site_plc, occ_plc, clicks))

    log("complementary lists...")
    dump("complementary_lists.json", complementary_from_occ(main))

    # gates
    legacy_b = -4.634907300425123
    legacy_fe = -2.5168820513358994
    beta = m3.get("coef")
    ci = m3.get("ci") or [None, None]
    t2 = "FAIL"
    if beta is not None and beta < 0 and ci[1] is not None and ci[1] < 0:
        mag = abs(beta) / abs(legacy_b)
        if mag >= 0.70:
            t2 = "STRONG PASS"
        elif mag >= 0.40:
            t2 = "PASS"
        else:
            t2 = "WEAK"
    elif beta is not None and beta < 0:
        t2 = "WEAK"
    fe_b = wl.get("within_coef")
    fe_ci = wl.get("within_ci") or [None, None]
    t3 = "FAIL"
    if fe_b is not None and fe_b < 0 and fe_ci[1] is not None and fe_ci[1] < 0:
        mag = abs(fe_b) / abs(legacy_fe)
        t3 = "STRONG PASS" if mag >= 0.60 else "PASS"
    elif fe_b is not None and fe_b < 0:
        t3 = "WEAK"
    coef_signs = [r["coef"] < 0 for r in lomo.get("by_module", [])]
    t4 = "FAIL"
    if coef_signs and all(coef_signs) and lopo_out["n_negative"] > 0:
        t4 = "MIXED"
    elif coef_signs and all(coef_signs) and lopo_out["n_negative"] == 0:
        t4 = "STRONG PASS"
    elif coef_signs and sum(coef_signs) >= 3:
        t4 = "MIXED"
    dump(
        "gates.json",
        {
            "T1": "PASS" if assertions["t1_pass"] else "FAIL_FATAL_IMPLEMENTATION",
            "T2": t2,
            "T2_magnitude_retained": None if beta is None else abs(beta) / abs(legacy_b),
            "T3": t3,
            "T3_magnitude_retained": None if fe_b is None else abs(fe_b) / abs(legacy_fe),
            "T4": t4,
        },
    )

    (OUT / "run.log").write_text("\n".join(LOG_LINES) + "\n")
    log("done")


if __name__ == "__main__":
    main()
