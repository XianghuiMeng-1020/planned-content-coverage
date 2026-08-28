# What Non-Access Can Mean: Planned-Content Reach Reproduction

Reproduction code for:

> **What Non-Access Can Mean: A Validity Study of Planned-Content Reach Before Assessed Work**

The study asks what can and cannot be inferred when digital resource traces are interpreted relative to documented planned use at the learner–assessment occasion. Planned-content reach is digitally observed access among elapsed planned `oucontent` pages, frozen at the learner’s own TMA submission. It is not learning, mastery, or exact instructional alignment.

This repository contains the analysis code needed to reconstruct the eligible planned-resource set, compute submission-safe reach/non-reach, and reproduce the manuscript’s association, interpretation, recurrence/transport, timing, and practical-use analyses. It does not redistribute OULAD, the manuscript, or internal audit reports.

## Dataset

Kuzilek, J., Hlosta, M., & Zdrahal, Z. (2017). Open University Learning Analytics dataset. *Scientific Data*, 4, 170171. https://doi.org/10.1038/sdata.2017.171

See `data/README.md`. Do not reconstruct planned-use windows from learner clicks.

## Environment

Python 3.11+ recommended.

```bash
python3 -m pip install -r requirements.txt
```

Set `OULAD_RAW` and `OULAD_INTERIM` if the files are not under `data/raw/oulad` and `data/interim/oulad`.

## Reproduction

```bash
python scripts/prepare_data.py
python scripts/run_construct.py
python scripts/run_inference.py
python scripts/run_timing.py
python scripts/run_practical.py
python scripts/check_reference_outputs.py
```

`prepare_data.py` converts official OULAD tables to the parquet layout used by the loaders. The `run_*.py` scripts recompute the analyses from those tables. `run_inference.py` includes a learner-cluster bootstrap (B = 1000) and is the most expensive step. `check_reference_outputs.py` compares compact frozen artifacts in `results/` with the manuscript headlines and does not rerun the models.

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
| Schedule permutation directional *p* | ≈ 0.15 |
| Identity placebo | −4.32 vs −4.46 |
| GGG generic-resource boundary | +3.71 |
| Presentation recurrence | 12 of 13 negative |
| Median LOPO ΔAUROC (low score) | +0.0008 |
| 10% review-budget extra low-score cases | −1 |

The association is small. Exact schedule partition and exact page identity are not established. The same construction reverses on a generic resource family. Predictive transport is heterogeneous. The operational increment is null.

## Repository structure

```
scripts/   prepare_data.py and analysis entry points
src/       loaders, construct helpers, and analysis modules
data/      acquisition instructions only
results/   compact frozen reference outputs used by the checker
```

## Scope and limitations

The evidence is observational. Resource-to-TMA assignment is analytical, not semantic. Offline, downloaded, shared, and external study remain unobserved. No teacher-decision or intervention study is included.

## Citation

If you use this code, cite the paper:

> What Non-Access Can Mean: A Validity Study of Planned-Content Reach Before Assessed Work.

and Kuzilek et al. (2017) for OULAD. See `CITATION.cff`.

## License

MIT. OULAD remains under its own CC-BY 4.0 terms.
