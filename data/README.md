# Data

This repository does **not** redistribute the Open University Learning Analytics Dataset (OULAD).

## Source

Kuzilek, J., Hlosta, M., & Zdrahal, Z. (2017). Open University Learning Analytics dataset. *Scientific Data*, 4, 170171. https://doi.org/10.1038/sdata.2017.171

Official landing page: https://analyse.kmi.open.ac.uk/open_dataset

License: CC-BY 4.0. Obtain the dataset from the official release.

## Expected raw files

Place the official tables in `data/raw/oulad/` (or set `OULAD_RAW`):

| Official file | Role |
|---|---|
| `vle.csv` | Resource metadata, including `week_from` / `week_to` |
| `courses.csv` | Presentation length |
| `assessments.csv` | Assessment dates and types |
| `studentRegistration.csv` | Registration / unregistration |
| `studentAssessment.csv` | Scores and submission dates |
| `studentVle.csv` (or `student_vle.rda`) | Daily site-level click summaries |

Do not reconstruct planned-use windows from learner clicks. Only documented `week_from` / `week_to` values define opportunity.

## Preparation

```bash
python scripts/prepare_data.py
```

This writes parquet tables to `data/interim/oulad/`. Those files are local and gitignored.

## KU Leuven external probe (optional)

This repository does **not** redistribute the KU Leuven Blackboard files.

Source: Tiukhova, E., Van Landuyt, D., Baesens, B., & Snoeck, M. (2026). *Scientific Data*. https://doi.org/10.1038/s41597-026-06821-3

Zenodo: https://doi.org/10.5281/zenodo.17087849 (CC BY 4.0)

After download, set `KULEUVEN_DATA` to a folder with:

| Path | Role |
|---|---|
| `official/course_info.json` | Exam-week calendar used for the end-of-instruction cutoff |
| `extracted/dataset/{1819,1920,2021}_course_participation.xlsx` | Enrolment and first-attempt outcomes |
| `extracted/dataset/{1819,1920,2021}_course_content.xlsx` | Content hierarchy and availability |
| `extracted/dataset/{1819,1920,2021}_log_activity.csv` | Fine-grained access events |

The probe construct is **end-of-instruction course-material reach**, not planned-content reach. Do not treat the two as interchangeable.
