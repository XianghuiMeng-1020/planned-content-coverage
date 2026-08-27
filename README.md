# Planned-Content Coverage Before Assessed Work

Reproduction code for an observational learning-analytics study of **planned-content coverage** — the share of an instructor's officially planned, pre-assessment learning resources that a student accesses at least once before the next graded task is due — and its association with performance on that task, using the Open University Learning Analytics Dataset (OULAD).

This repository contains the analysis pipeline and the frozen result artifacts behind the reported findings. It does not contain the manuscript, reviewer correspondence, or any author-identifying material.

---

## 1. The construct

Most learning-analytics work on "activity" collapses a course into aggregate counts: total clicks, active days, or time since last login. These measures are blind to *what* a student engaged with relative to *what was planned*. This project instead asks a narrower, content-relative question:

> Of the resources an instructor explicitly scheduled (via documented `week_from` / `week_to` planning windows) to be used before the next tutor-marked assessment, what share did a student never open?

We call the complement of this share **planned-content coverage**. The central variable, `never_share`, is the fraction of a learner's eligible planned resources for a given assessment window that receive zero clicks before that assessment's due date. It is deliberately:

- **content-specific**, not a raw click/day count;
- **plan-anchored**, using only officially documented `week_from`/`week_to` metadata — never inferred from behavior;
- **assessment-relative**, measured in the window that ends at the next tutor-marked assessment (TMA), not a nominal calendar week.

## 2. What this repository reproduces

| Result | Script | Frozen artifact |
|---|---|---|
| Sample and coverage ledger (13 presentations, 5 modules) | `src/v4/run_development.py` | `results/v4/presentation_ledger.json` |
| Dose–response of `never_share` vs. score | `src/v4/run_development.py` | `results/v4/dose_response.json` |
| Controlled association (OLS, cluster-robust SE, presentation fixed effects) | `src/v4/run_development.py` | `results/v4/controlled_association.json` |
| Presentation- and module-level leave-one-out holdout | `src/v4/run_development.py` | `results/v4/module_holdout.json` |
| Within-learner (student fixed-effects) estimate | `src/v4/run_development.py` | `results/v4/within_learner.json` |
| Same-activity-family matched comparison | `src/v4/run_development.py` | `results/v4/same_activity.json` |
| Cross-fitted residual-risk check | `src/v4/run_development.py` | `results/v4/residual_risk.json` |
| Metadata eligibility screen (which module/presentation/activity-type cells qualify) | `src/v4/screen.py` | `results/v4/untouched_family_screen.json`, `results/v4/ggg_eligibility.json` |
| Held-out cross-family confirmatory test (resource family, module GGG) | `src/v4/run_confirmatory.py` | `results/v4/confirmatory.json` |
| Timing-state gradient (early / on-time / late / never) | `src/v3/run_development.py` | `results/v3/timing_gradient.json` |
| External-dataset eligibility screen (why the study stays single-dataset) | *(manual screening protocol; no script)* | `results/v5/external_screen_decision.json` |

Every number below is copied verbatim from the JSON artifacts in `results/`; none is recomputed or rounded beyond display precision.

## 3. Headline findings

**Development sample** (9 primary presentations, `oucontent` family; n = 39,647 scored learner × assessment occasions):

- A dose–response gradient across `never_share` quintile bins: mean scores of 78.1 → 78.5 → 77.3 → 75.1 → 67.9 from lowest to highest never-share bin (`results/v4/dose_response.json`).
- Presentation-fixed-effects OLS, controlling for prior score, click volume, active rate, and raw weekly inactivity: coefficient on `never_share` = **−4.63** points (SE = 0.33, 95% CI [−5.27, −4.00], *p* < 10⁻⁴⁵), partial R² = 0.0055 over the control-only model (`results/v4/controlled_association.json`).
- Leave-one-presentation-out and leave-one-module-out holdout: the coefficient stays negative in 13/13 held-out presentations and 5/5 held-out modules (`results/v4/module_holdout.json`).
- Student fixed-effects (within-learner) estimate on repeat occasions (n = 34,358 occasions, 9,686 learners): **−2.52** points (SE = 0.45, 95% CI [−3.41, −1.63]) (`results/v4/within_learner.json`).
- Same-activity-family matched comparison (learners matched on prior score, click volume, activity rate, and opportunity count within presentation): high-coverage learners score **+4.65** points higher than low-coverage learners on average (95% CI [2.41, 5.42]), positive in 9/9 presentations (`results/v4/same_activity.json`).
- Cross-fitted residual check (out-of-fold residuals from a model excluding `never_share`, then regressed on `never_share`): coefficient **−2.57** (95% CI [−3.07, −2.07]) (`results/v4/residual_risk.json`).

**Held-out cross-family confirmatory test** (module GGG, `resource` activity family — never used in development; n = 2,038 scored occasions): the direction replicates (Spearman ρ = −0.036) but the controlled coefficient flips sign to **+2.49** (SE = 1.00, 95% CI [0.54, 4.44]), and the within-learner check is under-powered (`results/v4/confirmatory.json`). This boundary result is reported as-is; the pipeline does not search for a specification that restores the development-sample sign.

**Timing-state gradient** (auxiliary check, not the primary construct): mean scores by first-access timing relative to the planned window are early = 78.4, on-time = 77.0, late = 76.5, never = **70.6** (`results/v3/timing_gradient.json`).

**External-dataset screen**: 22 candidate public datasets were screened against the construct's metadata requirements (documented planned-use windows, assessment dates, resource-level identifiers); none was eligible for a same-construct cross-dataset confirmatory test (`results/v5/external_screen_decision.json`).

## 4. Repository structure

```
.
├── src/
│   ├── common/paths.py        # shared paths, seed, development-presentation set
│   ├── v2/                    # OULAD loaders, opportunity panel, learner-week controls
│   ├── v3/                    # timing-state features (early/on-time/late/never)
│   └── v4/                    # primary construct: never_share, screening, confirmatory test
├── data/
│   ├── convert_meta.json              # row counts from the raw→parquet conversion
│   └── presentation_feasibility.csv   # per-presentation eligibility scan
├── results/
│   ├── v4/     # primary development + confirmatory artifacts (frozen)
│   ├── v3/     # timing-state gradient (frozen)
│   └── v5/     # external-dataset screening decision (frozen)
├── requirements.txt
└── LICENSE
```

## 5. Data

This repository does **not** redistribute OULAD. Download it from the official release:

> Kuzilek, J., Hlosta, M., & Zdrahal, Z. (2017). Open University Learning Analytics dataset. *Scientific Data*, 4, 170171. https://doi.org/10.1038/sdata.2017.171

The dataset is released under **CC-BY 4.0**. After downloading, convert the raw CSV/RDA tables to the parquet layout the loaders expect (`vle`, `courses`, `assessments`, `student_registration`, `student_assessment` as parquet; `student_vle` as the original `.rda`, read via `pyreadr`), then point the loaders at your local copy:

```bash
export OULAD_RAW=/path/to/oulad/raw          # contains student_vle.rda
export OULAD_INTERIM=/path/to/oulad/parquet  # contains vle.parquet, courses.parquet, ...
```

If the environment variables are unset, the loaders default to `data/raw/oulad` and `data/interim/oulad` inside this repository (not included).

## 6. Installation

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

Tested with Python 3.13. See `requirements.txt` for pinned package versions.

## 7. Running the pipeline

```bash
# 1. Metadata eligibility screen — which module/presentation/activity-type
#    cells qualify for development vs. confirmatory use.
python3 src/v4/screen.py

# 2. Primary development analysis (9 presentations, oucontent family).
#    Writes all files under results/v4/ except confirmatory.json.
python3 src/v4/run_development.py

# 3. Held-out confirmatory test (module GGG, resource family).
#    Refuses to run unless a git tag named "planned-resource-coverage-v4-frozen"
#    exists on the current repository — this is a scientific pre-registration
#    gate, not a technical dependency. Create the tag yourself if you are
#    re-running this pipeline on your own fork.
git tag planned-resource-coverage-v4-frozen
python3 src/v4/run_confirmatory.py

# 4. Timing-state gradient (auxiliary, uses the wider "previously inspected"
#    presentation set).
python3 src/v3/run_development.py
```

Each script writes JSON artifacts to `results/<version>/` and prints a short summary to stdout. Random elements (bootstrap resampling) are seeded (`SEED = 20260826` in `src/common/paths.py`); OLS point estimates and confidence intervals are exact and will match the frozen artifacts bit-for-bit given the same input data.

## 8. Design choices worth knowing before reading the code

- **Undocumented ≠ unplanned.** Resources with missing `week_from`/`week_to` metadata are excluded from the eligible set entirely; they are never treated as "not planned" (`src/v2/load.py`, `planned_valid` flag).
- **No outcome peeking before confirmation.** `src/v4/run_confirmatory.py` hard-fails unless a specific git tag exists, so the confirmatory sample's assessment scores cannot be inspected before the development analysis is frozen.
- **Leakage-safe controls.** Raw activity controls (`raw_inact`, `active_rate`, `log_clicks`) are computed only from the interval strictly between the previous and current assessment, using `pandas.merge_asof` for prior-score attachment (`src/v2/panel.py::attach_prior_score`, `attach_raw_controls`).
- **Cluster-robust inference throughout.** All primary OLS specifications cluster standard errors by student ID.

## 9. License

Code is released under the MIT License (see `LICENSE`). OULAD itself is CC-BY 4.0 and must be attributed separately if you redistribute derived data.

## 10. Citation

If you use this code, please cite the OULAD data descriptor above and the accompanying manuscript (details withheld here pending peer review).
