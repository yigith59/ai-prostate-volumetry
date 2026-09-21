# AI-Derived Prostate Volumetry for PSA Density–Based Detection of Clinically Significant Prostate Cancer

Analysis code accompanying the manuscript **“AI-Derived Prostate Volumetry for PSA Density–Based Detection of Clinically Significant Prostate Cancer”** submitted to **Urology**.

This repository contains the executable analysis pipeline used to generate the numerical results, tables, and figures reported in the manuscript and supplementary materials.

## Overview

This study evaluates whether prostate-specific antigen density (PSAD) calculated using **AI-derived whole-gland prostate volume** improves discrimination of clinically significant prostate cancer (csPCa) compared with PSAD calculated using **clinically reported prostate volume**, using the public PI-CAI dataset.

Two main analytic cohorts are used:

* **Cohort A**: single-marker comparison of calculated manual PSAD versus AI-PSAD
  **N = 1418**, all three participating centres.

* **Cohort B**: multivariable model comparison of Model A versus Model A′
  **N = 1096**, Centres 1 and 2 only, by prespecified design to allow bidirectional centre-based validation.

The scripts reproduce the main analyses, including cohort derivation, volume-agreement analyses, AUC estimation, bootstrap confidence intervals, likelihood-ratio tests, calibration metrics, centre-based validation, threshold analyses, decision-curve analysis, reference-standard sensitivity analysis, and supplementary exploratory analyses.

No manuscript result is intended to be hand-entered or manually estimated.

## Data availability

This repository contains **code only**. It does **not** redistribute the PI-CAI dataset, medical images, segmentation masks, labels, clinical marksheets, or any patient-level data.

The PI-CAI Public Training and Development Dataset is publicly available from Zenodo under a CC BY-NC 4.0 license:

> Saha A, Bosma JS, Twilt JJ, et al. *Artificial intelligence and radiologists in prostate cancer detection on MRI (PI-CAI): an international, paired, non-inferiority, confirmatory study.* Lancet Oncology. 2024.

Dataset version used in this manuscript:

> **PI-CAI Public Training and Development Dataset, version 2.0.**
> **Zenodo DOI:** `INSERT EXACT DOI HERE`

Before running the scripts, download the dataset from the official PI-CAI/Zenodo source and place it according to the paths configured in the scripts. The dataset itself is governed by its own license and is not included in this repository.

## Repository structure

```text
.
├── README.md
├── LICENSE
├── requirements.txt
└── scripts/
    ├── phase1_cohort_derivation.py
    ├── phase1b_centre_eligibility_audit.py
    ├── phase2_functional_form_diagnostics.py
    ├── phase3_primary_analyses.py
    ├── phase4_validation.py
    ├── phase5_clinical_performance.py
    ├── phase6_reference_standard_sensitivity.py
    ├── phase7_tables_and_figures.py
    ├── qc_visual_review.py
    ├── qc_format_table_s1.py
    ├── exploratory_adc_lesion_radiomics.py
    ├── exploratory_lesion_radiomics_validation.py
    ├── verify_volume_agreement.py
    └── find_exploratory_radiomics_files.py
```

### Script descriptions

| Script                                       | Purpose                                                                                                  |
| -------------------------------------------- | -------------------------------------------------------------------------------------------------------- |
| `phase1_cohort_derivation.py`                | Derives Cohort A and Cohort B                                                                            |
| `phase1b_centre_eligibility_audit.py`        | Audits Centre 3 eligibility and reported-vs-calculated PSAD discordance                                  |
| `phase2_functional_form_diagnostics.py`      | Assesses log-linearity, interaction terms, and collinearity diagnostics                                  |
| `phase3_primary_analyses.py`                 | Runs primary AUC analyses, multivariable models, coefficient-ratio bootstrap, and likelihood-ratio tests |
| `phase4_validation.py`                       | Runs centre-based validation, ProstateX-exclusion sensitivity, and three-centre sensitivity analysis     |
| `phase5_clinical_performance.py`             | Runs threshold analyses, reclassification analyses, and decision-curve analysis                          |
| `phase6_reference_standard_sensitivity.py`   | Runs histology-verified reference-standard sensitivity analysis                                          |
| `phase7_tables_and_figures.py`               | Generates main-text and supplementary tables and figures                                                 |
| `qc_visual_review.py`                        | Supports visual quality control of 25 pre-specified segmentation cases                                   |
| `qc_format_table_s1.py`                      | Formats visual QC output for Supplementary Table S1                                                      |
| `exploratory_adc_lesion_radiomics.py`        | Runs exploratory ADC-heterogeneity and lesion-level radiomics analyses                                   |
| `exploratory_lesion_radiomics_validation.py` | Runs internal validation for the exploratory lesion-level analysis                                       |
| `verify_volume_agreement.py`                 | Independently verifies prostate-volume agreement statistics                                              |
| `find_exploratory_radiomics_files.py`        | Locates output files from exploratory imaging-feature analyses                                           |

Earlier superseded analysis scripts based on pre-revision cohort definitions are intentionally excluded from this release. The scripts listed above correspond to the final revised manuscript analyses.

## Environment

Python 3.10 or later is recommended.

Install dependencies with:

```bash
pip install -r requirements.txt
```

The main analysis uses:

* Python
* NumPy
* pandas
* SciPy
* scikit-learn
* statsmodels
* SimpleITK
* matplotlib

Exact package versions used in the manuscript are reported in the Methods section.

## Configuration

Each script auto-detects whether it is running in Google Colab using:

```python
ON_COLAB = os.path.exists("/content/drive")
```

Paths are then selected accordingly.

Outside Colab, dataset paths default to a local dataset directory. If your folder structure differs, edit the `DATASET_ROOT` variable near the top of each script.

Some scripts import functions from earlier phase scripts. This works automatically as long as all scripts remain together in the `scripts/` folder.

Scripts can be run either from the repository root:

```bash
python scripts/phase1_cohort_derivation.py
```

or from inside the scripts folder:

```bash
cd scripts
python phase1_cohort_derivation.py
```

## How to reproduce the manuscript analyses

Run the scripts in the following order. Phases 1–6 write intermediate outputs to `QA_outputs/revision_outputs/`. Phase 7 writes final tables and figures to `final_tables/` and `final_figures/`.

| Order | Script                                                                                         | Output                                                                                          |
| ----: | ---------------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------- |
|     1 | `phase1_cohort_derivation.py`                                                                  | Cohort A and Cohort B definitions                                                               |
|     2 | `phase1b_centre_eligibility_audit.py`                                                          | Centre 3 eligibility audit and PSAD discordance analysis                                        |
|     3 | `phase2_functional_form_diagnostics.py`                                                        | Spline, interaction, and collinearity diagnostics                                               |
|     4 | `phase3_primary_analyses.py`                                                                   | Primary AUCs, model coefficients, coefficient-ratio bootstrap, and likelihood-ratio tests       |
|     5 | `phase4_validation.py`                                                                         | Centre-based validation, ProstateX-exclusion sensitivity, and three-centre sensitivity analysis |
|     6 | `phase5_clinical_performance.py`                                                               | Threshold performance, reclassification, and decision-curve analysis                            |
|     7 | `phase6_reference_standard_sensitivity.py`                                                     | Histology-verified sensitivity analysis excluding MRI-only negative cases                       |
|     8 | `qc_visual_review.py` followed by `qc_format_table_s1.py`                                      | Visual quality-control summary for Supplementary Table S1                                       |
|     9 | `exploratory_adc_lesion_radiomics.py` followed by `exploratory_lesion_radiomics_validation.py` | Exploratory ADC-heterogeneity and lesion-level imaging-feature analyses                         |
|    10 | `phase7_tables_and_figures.py`                                                                 | Final main-text and supplementary tables and figures                                            |

Optional utility scripts:

| Script                                | Purpose                                                                                   |
| ------------------------------------- | ----------------------------------------------------------------------------------------- |
| `verify_volume_agreement.py`          | Independent check of prostate-volume agreement statistics reported in Results section 3.2 |
| `find_exploratory_radiomics_files.py` | Utility for locating exploratory-analysis output files                                    |

Exploratory ADC and lesion-level analyses are not part of the primary revised-cohort inference and are reported only as hypothesis-generating supplementary analyses in the manuscript.

## Main analyses reproduced by this repository

The pipeline reproduces:

* Cohort A and Cohort B derivation
* Reported-PSAD availability and discordance audit
* AI-derived versus clinically reported prostate-volume agreement
* Algorithm 1 versus Algorithm 2 inter-algorithm consistency analysis
* Calculated manual PSAD versus AI-PSAD discrimination
* Model A versus Model A′ multivariable logistic regression
* Bootstrap confidence intervals for AUC differences
* Bootstrap confidence intervals for the volume-to-PSA coefficient ratio
* Restricted versus freely estimated PSA-volume likelihood-ratio tests
* Bootstrap optimism correction
* Repeated 5-fold × 20-repeat cross-validation
* Bidirectional centre-based validation
* ProstateX-exclusion sensitivity analysis
* Three-centre sensitivity analysis
* Threshold-level performance and reclassification analysis
* Decision-curve analysis using out-of-fold predictions
* Histology-verified reference-standard sensitivity analysis
* Supplementary exploratory ADC and lesion-level imaging-feature analyses
* Final table and figure generation

## Reproducibility notes

* A fixed random seed (`RANDOM_SEED = 42`) is used for bootstrap resampling and cross-validation.
* Bootstrap confidence intervals for ΔAUC use patient-level resampling of the paired statistic.
* Model-based bootstrap procedures refit the relevant models within each resample.
* Bootstrap optimism correction is used for optimism-corrected performance estimates and is not used as the reported ΔAUC confidence interval.
* Repeated 5-fold × 20-repeat cross-validation results are reported as empirical performance ranges, not as confidence intervals.
* Centre identities are anonymised in the manuscript as Centre 1, Centre 2, and Centre 3. The scripts may use original PI-CAI centre codes internally but map them to anonymised labels in manuscript-ready outputs.
* The repository is designed to regenerate manuscript tables and figures from the downloaded dataset and analysis scripts.

## Expected outputs

The final script, `phase7_tables_and_figures.py`, generates the manuscript-ready outputs, including:

* Table 1: Baseline characteristics
* Table 2: Discrimination, internal validation, and calibration
* Table 3: Multivariable logistic regression and coefficient-ratio analysis
* Supplementary Table S1: Cohort derivation, centre distribution, reported-PSAD audit, verification composition, and visual QC
* Supplementary Table S2: Threshold performance, reclassification, and decision-curve analysis
* Supplementary Table S3: Centre-based validation, sensitivity analyses, diagnostics, and exploratory analyses
* Figure 1: Participant flow diagram
* Figure 2: ROC curves
* Supplementary Figure S1: Bland-Altman prostate-volume agreement plots
* Supplementary Figure S2: Calibration and decision-curve plots

## Limitations

This repository does not include:

* PI-CAI images
* segmentation masks
* clinical marksheets
* patient-level data
* trained segmentation models
* proprietary software
* manuscript source files

The user must obtain the PI-CAI dataset independently from the official source and comply with its license terms.

## License

The code in this repository is released under the MIT License. See `LICENSE`.

The PI-CAI dataset is not part of this repository and is governed by its own CC BY-NC 4.0 license.

## Citation

If you use this code, please cite the manuscript after publication:

> Arı YH, et al. **AI-Derived Prostate Volumetry for PSA Density–Based Detection of Clinically Significant Prostate Cancer.** Urology. Forthcoming.

Please also cite the PI-CAI dataset and associated PI-CAI publication:

> Saha A, Bosma JS, Twilt JJ, et al. **Artificial intelligence and radiologists in prostate cancer detection on MRI (PI-CAI): an international, paired, non-inferiority, confirmatory study.** Lancet Oncology. 2024.
