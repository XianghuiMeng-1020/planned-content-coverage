# Planned-Content Reach Reproduction

Reproduction code for:

> **Validating Digital Non-Access in Learning Analytics: Planned-Content Reach Before Assessed Work**

The study asks what can and cannot be inferred when a missing digital-access event is read against documented planned use. Planned-content reach is digitally observed access among elapsed planned `oucontent` pages, frozen at the learner’s own tutor-marked assessment. It is not learning, mastery, engagement, or exact instructional alignment.

This repository contains the analysis code needed to reconstruct the eligible planned-resource set, compute submission-safe reach/non-reach, and reproduce the manuscript’s association, interpretation, recurrence/transport, timing, practical-use, and external-probe analyses. It does not redistribute OULAD, the KU Leuven files, the manuscript, or internal audit reports.

## Dataset

Kuzilek, J., Hlosta, M., & Zdrahal, Z. (2017). Open University Learning Analytics dataset. *Scientific Data*, 4, 170171. https://doi.org/10.1038/sdata.2017.171

Tiukhova, E., Van Landuyt, D., Baesens, B., & Snoeck, M. (2026). A multi-year longitudinal study of university student traces. *Scientific Data*. https://doi.org/10.1038/s41597-026-06821-3  
Zenodo record: https://doi.org/10.5281/zenodo.17087849

See `data/README.md`. Do not reconstruct planned-use windows from learner clicks.

## Environment

Python 3.11+ recommended.

```bash
python3 -m pip install -r requirements.txt
```

Set `OULAD_RAW` and `OULAD_INTERIM` if the files are not under `data/raw/oulad` and `data/interim/oulad`.

For the KU Leuven probe, obtain the official Zenodo extract and set `KULEUVEN_DATA` to the folder that contains `official/course_info.json` and `extracted/dataset/`.

## Reproduction

```bash
python scripts/prepare_data.py
python scripts/run_construct.py
python scripts/run_inference.py
python scripts/run_timing.py
python scripts/run_practical.py
python scripts/run_current_later.py
python scripts/run_schedule_perm.py
python scripts/run_kuleuven.py
python scripts/run_specificity_diagnostic.py
python scripts/check_reference_outputs.py
```

`prepare_data.py` converts official OULAD tables to the parquet layout used by the loaders. The `run_*.py` scripts recompute the analyses from those tables. `run_inference.py` includes a learner-cluster bootstrap (B = 1000). `run_schedule_perm.py` uses B = 10,000 and is the most expensive OULAD step. `run_kuleuven.py` requires the separate Zenodo extract. `check_reference_outputs.py` compares compact frozen artifacts in `results/` with the manuscript headlines and does not rerun the models.

Generated intermediates (including `occasion_table.parquet`) are written under `results/` and are gitignored when large.

## Expected headline outputs

| Result | Frozen value |
|---|---|
| Primary adjusted β (non-reach proportion) | −3.47, 95% CI [−4.09, −2.84] |
| Sample | 38,662 occasions; 13,568 learners |
| Partial R² | 0.0031 |
| Learner FE | −1.60, 95% CI [−2.45, −0.75] |
| Activity-matched contrast | +3.64 |
| Current vs later-scheduled (joint) | −4.28 vs −1.25 |
| Current-versus-later Wald contrast | −3.03, *p* = 0.0049 |
| Schedule permutation | B = 10,000; directional *p* = 0.103 |
| Identity placebo | −4.32 vs −4.46 |
| GGG generic-resource boundary | +3.71 |
| Presentation recurrence | 12 of 13 negative |
| Median LOPO ΔAUROC (low score) | +0.0008 |
| 10% review-budget extra low-score cases | −1 |
| KU Leuven material-reach OR | 4.59; Δ McFadden = 0.0019; *n* = 4,292 |
| KU Leuven S1 singleton-only | OR 1.96; sign not flipped |
| KU Leuven joint specificity | conditional contrast (PASS B); material OR 12.59 in the joint model |

The OULAD association is small. Exact schedule partition and exact page identity are not established. The same construction reverses on a generic resource family. Predictive transport is heterogeneous. The operational increment is null. The second ecology uses a different, metadata-faithful construct and is not a copy of planned-content reach.

## Repository structure

```
scripts/             prepare_data.py and analysis entry points
src/                 loaders, construct helpers, and analysis modules
src/strong_accept/   Wald contrast, B=10,000 permutation, KU Leuven probe
data/                acquisition instructions only
results/             compact frozen reference outputs used by the checker
```

## Scope and limitations

The evidence is observational. Resource-to-TMA assignment is analytical, not semantic. Offline, downloaded, shared, and external study remain unobserved. No teacher-decision or intervention study is included.

## Citation

If you use this code, cite the paper:

> Validating Digital Non-Access in Learning Analytics: Planned-Content Reach Before Assessed Work.

and Kuzilek et al. (2017) for OULAD. For the external probe, cite Tiukhova et al. (2026) and the Zenodo record. See `CITATION.cff`.

## License

MIT. OULAD and the KU Leuven files remain under their own CC-BY 4.0 terms.
