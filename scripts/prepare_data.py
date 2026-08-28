#!/usr/bin/env python3
"""Convert official OULAD tables into the parquet layout used by the analyses."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

_SRC = Path(__file__).resolve().parents[1] / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from common.paths import OULAD_INTERIM, OULAD_RAW  # noqa: E402

META = {
    "vle.parquet": ("vle.csv", "vle.csv"),
    "courses.parquet": ("courses.csv", "courses.csv"),
    "assessments.parquet": ("assessments.csv", "assessments.csv"),
    "student_registration.parquet": ("studentRegistration.csv", "student_registration.csv"),
    "student_assessment.parquet": ("studentAssessment.csv", "student_assessment.csv"),
}
CLICK_NAMES = ("studentVle.csv", "student_vle.csv", "student_vle.rda")


def _find(raw: Path, names: tuple[str, ...]) -> Path | None:
    for name in names:
        p = raw / name
        if p.exists():
            return p
    return None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw", type=Path, default=OULAD_RAW)
    parser.add_argument("--interim", type=Path, default=OULAD_INTERIM)
    args = parser.parse_args()
    raw = args.raw
    interim = args.interim
    interim.mkdir(parents=True, exist_ok=True)

    missing = []
    for out_name, candidates in META.items():
        src = _find(raw, candidates)
        if src is None:
            missing.append(out_name)
            continue
        df = pd.read_csv(src)
        dest = interim / out_name
        df.to_parquet(dest, index=False)
        print(f"wrote {dest} from {src.name} n={len(df)}")

    click = _find(raw, CLICK_NAMES)
    if click is None:
        missing.append("student_vle")
    elif click.suffix == ".csv":
        dest = interim / "student_vle.parquet"
        print(f"converting {click} -> {dest} (large; one-time)")
        pd.read_csv(click).to_parquet(dest, index=False)
        print(f"wrote {dest}")
    else:
        print(f"click stream left as {click}; loaders will read it directly")

    if missing:
        raise SystemExit(
            "Missing official OULAD files for: "
            + ", ".join(missing)
            + f". Download Kuzilek et al. (2017) and place tables in {raw}."
        )
    print("preparation complete")


if __name__ == "__main__":
    main()
