# When Coverage Is Not a Risk Signal: Planned-Content Reach as a Bounded Design Audit

Reproduction materials for an observational learning-analytics study of **planned-content coverage** on the Open University Learning Analytics Dataset (OULAD).

Planned-content coverage records whether learners digitally reached instructional pages whose documented use window had already elapsed before assessed work. A missing click is documented digital non-reach, not evidence that the pages went unstudied and not a content-quality score. The same construct is associated with learner outcomes, does **not** support incremental student-risk decisions, and can still support a **context-bound** descriptive audit of documented learning-design reach where planned-week slots are identifiable.

## What this repository reproduces

- Learner-level association of non-coverage with submission (SUB-A / SUB-B) and with scores below 40.
- Incremental discrimination (LOPO / LOMO), calibration, and 5/10/20% fixed-budget review.
- Breadth-versus-volume identification (balance failure → unique breadth **INDETERMINATE**).
- Design-slot Gap maps, cross-presentation rank stability, chronological held-out replication, a generic course-position baseline, residual stability, an alignment placebo, and a top-25% inspection shortlist.

It does **not** claim successful student-risk prediction. AAA and BBB are **not identifiable** for held-out slot replication. CCC’s held-out comparison uses **four EARLY** slots.

This repository contains analysis code and compact frozen reference outputs. It does not contain the manuscript, internal audit reports, or raw OULAD files.

## Dataset

Kuzilek, J., Hlosta, M., & Zdrahal, Z. (2017). Open University Learning Analytics dataset. *Scientific Data*, 4, 170171. https://doi.org/10.1038/sdata.2017.171

See `data/README.md`. Do not reconstruct planned windows from clicks.

## Environment

Python 3.11+ recommended.

```bash
python3 -m pip install -r requirements.txt
```

Set `OULAD_RAW` and `OULAD_INTERIM` if the files are not under `data/raw/oulad` and `data/interim/oulad`.

## Reproduction

```bash
python scripts/prepare_data.py
python scripts/run_learner_analyses.py
python scripts/run_design_reach.py
python scripts/check_reference_outputs.py
```

`check_reference_outputs.py` compares frozen artifacts in `results/` with the manuscript headlines and does not rerun the models. The two `run_*.py` scripts recompute those artifacts from prepared OULAD tables; they are deterministic given the same source files and seeds, and they are substantially more expensive than the reference check.

## Expected headline outputs

| Result | Frozen value |
|---|---|
| SUB-A odds ratio | 0.823, 95% CI [0.720, 0.941] |
| Low-score prevalence | 1,951 / 38,662 = 0.0505 |
| LOPO ΔAUROC / ΔAUPRC | +0.00076 / +0.00069 |
| BBB LOMO ΔAUROC | −0.0275 |
| 10% extra low-score cases | −1 |
| Extra non-submissions at 5/10/20% | 0 |
| Breadth versus volume | INDETERMINATE (balance gate failed) |
| Pairwise week-level Gap ρ | 0.807 (310 items) |
| Held-out historical ρ / position ρ / Δρ | 0.811 / 0.405 / 0.413 |
| Residual ρ | 0.756 |
| Top-25% inspection precision | ≈ 0.60 |
| AAA / BBB held-out slot replication | not identifiable |
| CCC held-out slots | 4, all EARLY |

Student-risk operationalization: **not supported**. Design-reach audit: **context-bound**.

## Repository structure

```
scripts/     prepare_data.py, run_learner_analyses.py, run_design_reach.py, check_reference_outputs.py
src/         loaders, construct helpers, and the frozen analysis modules
data/        acquisition instructions only
results/     compact frozen reference outputs used by the checker
```

## Scope and limitations

The learner-level evidence is observational. Unique coverage information beyond activity volume is unidentified. Low-score events are uncommon. No intervention or teacher study is included. Design-level transport is identifiable only on CCC, EEE, and FFF. Gap is documented digital reach, not learning and not content quality.

## Citation

If you use this code, cite the paper:

> When Coverage Is Not a Risk Signal: Planned-Content Reach as a Bounded Design Audit.

and Kuzilek et al. (2017) for OULAD. See `CITATION.cff`.

## License

MIT. OULAD remains under its own CC-BY 4.0 terms.
