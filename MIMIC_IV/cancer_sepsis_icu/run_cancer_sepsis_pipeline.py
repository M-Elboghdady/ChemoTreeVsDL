from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parent

steps = [
    "build_sepsis3_duckdb.py",
    "build_cancer_sepsis_cohort.py",
    "extract_cancer_sepsis_icu24h_labs.py",
    "create_cancer_sepsis_folds.py",
]

for step in steps:
    path = ROOT / step
    print(f"\n{'='*60}")
    print(f"Running: {step}")
    print(f"{'='*60}\n")

    subprocess.run(
        [sys.executable, str(path)],
        check=True,
    )

print("\nCancer-sepsis pipeline completed successfully.")
