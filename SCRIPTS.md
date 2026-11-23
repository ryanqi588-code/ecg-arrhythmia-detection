# Script map

Which script produces each result, for reviewers. The paper numbers its
tests 1–5; the scripts keep the project's own numbers (9, 11, 12, 17, 18),
because the registration commits and [`check_registrations.py`](check_registrations.py)
refer to them. Nothing has been moved or renamed, for the same reason.
Results go to [`results/`](results/), figures to [`figures/`](figures/). Those are the committed
outputs, so a re-run writes to `rerun/` instead (e.g. `rerun/results/afib_cpsc_test.csv`, to compare
with the committed file); pass `--force` to overwrite the committed one. The same goes for
`models/cnn_model.keras` (the registered network), `tests/golden/` and `registration_evidence/`.
Setup and data: README, *Running it*.

## The paper

| Paper | Script | Output |
|---|---|---|
| Test 1 (project test 9): AF detection, SPH | [`afib_sph_test.py`](afib_sph_test.py) | `results/afib_sph_test.csv`, `results/afib_sph_test_recordings.csv`, `figures/afib_sph_test_roc.png` |
| Test 2 (11): against deep learning and doctors | [`afib_comparison_test.py`](afib_comparison_test.py), after [`afib_cnn.py`](afib_cnn.py) and [`afib_compare_dnn.py`](afib_compare_dnn.py) (TensorFlow environment) | `results/afib_comparison_test.csv`, `results/afib_comparison_diffs.csv`, `results/afib_comparison_doctors.csv`, `results/afib_comparison_sph_recordings.csv` (and `_ningbo`, `_code15`, `_codetest`), `results/compare_probs_sph.csv` (and the other sets) |
| Test 2 figure | [`afib_comparison_figure.py`](afib_comparison_figure.py) | `figures/afib_comparison.png` |
| Test 3 (12): unmeasurable P waves | [`afib_unmeasured_test.py`](afib_unmeasured_test.py) | `results/afib_unmeasured_test.csv`, `results/afib_unmeasured_test_recordings.csv` |
| Test 4 (17): "needs review" | [`afib_abstain_test.py`](afib_abstain_test.py) | `results/afib_abstain_test.csv`, `results/afib_abstain_test_recordings.csv`, `results/afib_abstain_preview.csv` |
| Test 5 (18): AF vs PAC, registered | [`afib_afpac_test.py`](afib_afpac_test.py), with `afib_compare_dnn.py mimic18` (TensorFlow environment) | `results/afib_afpac_test.csv`, `results/afib_afpac_test_diffs.csv`, `results/afib_afpac_test_recordings.csv`, `results/compare_probs_mimic18.csv`, `results/mimic18_selected.txt` |
| Figure: two ECGs with feature contributions | [`afib_example_figure.py`](afib_example_figure.py) | `figures/afib_examples.png`, `figures/Figure_1.pdf`, `figures/Figure_1.tiff`, `results/afib_example_scores.csv` |
| After test 2: AF-vs-PAC ranking | [`afib_comparison_afpac.py`](afib_comparison_afpac.py) | `results/afib_comparison_afpac.csv` |
| After test 2: PAC split by ventricular ectopy | [`afib_comparison_ectopy.py`](afib_comparison_ectopy.py) | `results/afib_comparison_ectopy.csv` |
| After test 2: CODE-15% labels and regular AF | [`afib_code15_labels.py`](afib_code15_labels.py) | `results/afib_code15_labels.csv` |
| Subgroups and demographics (TRIPOD+AI) | [`afib_subgroups.py`](afib_subgroups.py) | `results/afib_subgroups.csv`, `results/demographics.csv` |
| Limitations: MIMIC label check | [`afib_mimic_labels.py`](afib_mimic_labels.py) | `results/afib_mimic_labels.csv`, `results/afib_mimic_labels_diffs.csv` |
| Every number the paper quotes, checked against the README | [`paper_numbers.py`](paper_numbers.py) (`--check`) | (prints each number and its source) |

## The model, its development, and tools

| What | Script | Output |
|---|---|---|
| Signal processing and beat detection | [`ecg.py`](ecg.py) | (library) |
| Features, fitting, cutoffs | [`af.py`](af.py) | (library) |
| Score one ECG | [`predict.py`](predict.py) | (prints the call and contributions) |
| The adopted model as a table | [`export_final_model.py`](export_final_model.py) | `results/final_af_model.csv` |
| Confidence intervals (for new code) | [`stats.py`](stats.py) | (library) |
| Where outputs are written: committed files are never overwritten unless `--force` | [`outputs.py`](outputs.py) | (library) |
| 12-lead model development | [`afib_pac_dev.py`](afib_pac_dev.py); still imported: [`afib_clinical_dev.py`](afib_clinical_dev.py), [`afib_pwave_dev.py`](afib_pwave_dev.py) | `results/afib_pac_dev.csv`, `results/afib_clinical_dev.csv`, `results/afib_pwave_dev.csv` |
| Test 3's fix, development | [`afib_fa_dev.py`](afib_fa_dev.py) | `results/afib_fa_dev.csv` |
| Same-data CNN | [`afib_cnn.py`](afib_cnn.py) | `models/cnn_model.keras`, `results/afib_cnn_dev.csv` |
| Both networks' outputs | [`afib_compare_dnn.py`](afib_compare_dnn.py) | `results/compare_probs_*.csv` |
| MIMIC-IV-ECG selection (tests 14–16, 18) | [`afib_mimic_select.py`](afib_mimic_select.py), [`afib_slowaf_dev.py`](afib_slowaf_dev.py) (`select_dev`) | `results/mimic*_selected.txt` |
| Registration check | [`check_registrations.py`](check_registrations.py) | `results/registration_check.csv` |
| Registration evidence for archiving | [`export_registration_evidence.py`](export_registration_evidence.py) | `registration_evidence/` |
| Data checksums | [`data_manifest.py`](data_manifest.py) | `results/data_sha256sums.txt.gz`, `results/data_manifest_summary.csv`, `results/data_manifest_problems.csv` |
| Old vs new "unmeasurable" rule | [`check_unmeasured_definition.py`](check_unmeasured_definition.py) | `results/unmeasured_definition_check.csv` |
| Golden test data | [`tests/golden/build_golden.py`](tests/golden/build_golden.py) | `tests/golden/` |

## The supplement

| Supplement | Script | Output |
|---|---|---|
| Test 10: atrial flutter | [`afib_flutter_dev.py`](afib_flutter_dev.py), [`afib_flutter_test.py`](afib_flutter_test.py) | `results/afib_flutter_dev.csv`, `results/afib_flutter_test.csv` |
| S6: calibration | [`afib_calibration.py`](afib_calibration.py) | `results/afib_calibration.csv`, `figures/afib_calibration.png` |
| S8: hand-made PAC features | [`afib_ppresence_dev.py`](afib_ppresence_dev.py) (still imported), [`afib_pconc_dev.py`](afib_pconc_dev.py) | `results/afib_ppresence_dev.csv`, `results/afib_pconc_dev.csv` |
| Test 13: hybrid | [`afib_hybrid_dev.py`](afib_hybrid_dev.py), [`afib_hybrid_test.py`](afib_hybrid_test.py) | `results/afib_hybrid_dev.csv`, `results/afib_hybrid_test.csv` |
| Test 14: hybrid on MIMIC-IV-ECG | [`afib_mimic_test.py`](afib_mimic_test.py) | `results/afib_mimic_test.csv` |
| Test 15: hybrid with override | [`afib_hybrid2_dev.py`](afib_hybrid2_dev.py), [`afib_mimic15_test.py`](afib_mimic15_test.py) | `results/afib_hybrid2_dev.csv`, `results/afib_mimic15_test.csv` |
| Why MIMIC AF is missed | [`afib_mimic_errors.py`](afib_mimic_errors.py) | `results/afib_mimic_errors_af.csv`, `figures/mimic_missed_af.png` |
| Test 16: slow-AF fix | [`afib_slowaf_dev.py`](afib_slowaf_dev.py), [`afib_slowaf_test.py`](afib_slowaf_test.py) | `results/afib_slowaf_dev.csv`, `results/afib_slowaf_test.csv` |

## Archived

Early-stage scripts: project tests 1–8 (not the paper's tests 1–5), the beat detector, and development that led
nowhere. Nothing in the paper or the current model depends on them. They
are kept unchanged, not moved, because the registration commits refer to
them by name. Their committed outputs are the results as reported. They
aren't maintained, and several run their whole test on import, so don't
import them. Each row gives the supplement section that reports it.

| Archived script | What it was | Output | Supplement |
|---|---|---|---|
| [`qrs_benchmark.py`](qrs_benchmark.py) | beat detection on MIT-BIH | `results/qrs_benchmark.csv` | S2 |
| [`afib.py`](afib.py) | the original MIT-BIH AF Database Holter model | `results/afib_classifiers.csv`, `figures/afib_roc.png` | S3 |
| [`afib_test.py`](afib_test.py), [`ltaf_errors.py`](ltaf_errors.py) | project test 1: Holter model on AFDB later hours and LTAFDB | `results/ltaf_errors.csv`, `figures/afib_ltaf_test_roc.png` | S3 |
| [`afib_svdb_test.py`](afib_svdb_test.py) | project test 2: SVDB atrial ectopy | `results/afib_svdb_test.csv` | S4 |
| [`afib_cinc_test.py`](afib_cinc_test.py), [`afib_handheld_cv.py`](afib_handheld_cv.py) | project test 3: Challenge 2017 handheld | `results/afib_cinc_test.csv`, `results/afib_handheld_cv.csv` | S4 |
| [`afib_cpsc_test.py`](afib_cpsc_test.py) | project test 4: CPSC 2018, lead I | `results/afib_cpsc_test.csv` | S4 |
| [`afib_cpscx_test.py`](afib_cpscx_test.py) | project test 5: CPSC 2018 extra | `results/afib_cpscx_test.csv` | S4 |
| [`afib_georgia_test.py`](afib_georgia_test.py) | project test 6: Georgia | `results/afib_georgia_test.csv`, `figures/georgia_missed_af.png` | S5 |
| [`afib_ptbxl_test.py`](afib_ptbxl_test.py) | project test 7: PTB-XL | `results/afib_ptbxl_test.csv` | S5 |
| [`afib_chapman_test.py`](afib_chapman_test.py) | project test 8: Chapman-Shaoxing | `results/afib_chapman_test.csv` | S5 |
| [`afib_ectopy_dev.py`](afib_ectopy_dev.py) | removing atrial ectopy (no gain) | `results/afib_ectopy_dev.csv` | S5 |
