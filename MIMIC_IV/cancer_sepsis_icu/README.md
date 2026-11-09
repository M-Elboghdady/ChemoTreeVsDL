## Build the cancer-sepsis dataset

Run the complete pipeline:

python MIMIC_IV/cancer_sepsis_icu/run_cancer_sepsis_pipeline.py

This performs:

1. Build Sepsis-3 definitions
2. Build the cancer-sepsis ICU cohort
3. Extract first-24-hour longitudinal laboratory events
4. Create the 5 subject-level CV folds

No additional preprocessing step is required before EHRMamba-Lab training.
