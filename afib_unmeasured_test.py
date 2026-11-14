"""Test 12: does handling unmeasurable P waves honestly cut false alarms? (CODE-15% part 1)

Test 11 showed that recordings with fewer than 3 usable regular beats get
p_match = 0 ("no P waves") and are pushed to AF with near certainty. The
fix (afib_fa_dev.py): an indicator p_unmeasured, with p_match_regular and
p_match_mean filled with the training median of measured recordings
instead of 0. Everything else is the final model: CLINICAL_FEATURES_12,
trained on the original development units (CPSC crops, PTB-XL, Chapman),
strictest per-source 95%-sensitivity cutoff on those three sources.
In development (5-fold CV on all seven used sources) it lifted AUC from
0.985 to 0.993 with sensitivity unchanged, cut CODE-15% part 0 false
alarms from 4.2% to 3.6% and CODE test set false alarms from 3.7% to
3.1%, and left PAC false alarms unchanged (16.2% vs 16.1%).

Data: CODE-15% part 1 (chosen before download, never opened), every exam,
AF label from exams.csv (cardiologist report agreeing with automatic
analysis and an irregular pulse, so sensitivity is optimistic for both
models). Recordings either model can't score are counted.

Primary, fixed before the data are opened, paired by patient (2000 bootstrap
resamples):
  1. non-AF false alarm rate, fixed - final: 95% interval entirely below 0
  2. AF sensitivity, fixed - final: lower bound of the 95% interval above
     -0.02 (non-inferiority margin of 2 percentage points)
The fix is adopted only if both hold. Run once.
"""

import os

import h5py
import numpy as np
import pandas as pd

from af import CLINICAL_FEATURES_12, fit_with_source_cutoffs, model_inputs, auc
from afib_compare_dnn import strip_padding
from afib_comparison_test import boot_ci, rate_str
from afib_fa_dev import fit, with_unmeasured_fix
from afib_pac_dev import dev_units, prepare, unit_features

PART = "exams_part1.hdf5"
OUR_CUTOFF_TEST9 = 0.473559896031711
FIXED_FEATURES = CLINICAL_FEATURES_12 + ["p_unmeasured"]


def code15_features(part=PART, directory="data/code15"):
    exams = pd.read_csv(f"{directory}/exams.csv")
    exams = exams[exams.trace_file == part].set_index("exam_id")
    rows, skipped = [], {}
    with h5py.File(f"{directory}/{part}", "r") as f:
        ids = f["exam_id"][()]
        for i, eid in enumerate(ids):
            if eid not in exams.index:
                continue
            if i % 2000 == 0:
                print(f"test {i} ...", flush=True)
            label = "A" if exams.AF[eid] else "notAF"
            row = unit_features(strip_padding(f["tracings"][i]), 400, label == "A", "multilead12")
            if row is None:
                skipped[label] = skipped.get(label, 0) + 1
            else:
                rows.append({**row, "record": int(eid), "patient": int(exams.patient_id[eid]), "label": label})
    return pd.DataFrame(rows), skipped, len(exams)


if __name__ == "__main__":
    units = prepare(dev_units("multilead12"))
    m_final, cuts = fit_with_source_cutoffs(units, CLINICAL_FEATURES_12)
    c_final = min(cuts.values())
    assert abs(c_final - OUR_CUTOFF_TEST9) < 1e-9, c_final
    train = units.assign(patient=units.record, p_unmeasured=(units.p_match_mean == 0).astype(float))
    m_fixed, c_fixed = fit(train, FIXED_FEATURES, True)
    fills = {c: units[c].median() for c in ["log_f_wave", "f_ratio_II", "f_ratio_V1"]}
    print(f"cutoffs: final {c_final:.4f}, fixed {c_fixed:.4f}")

    t, skipped, n_selected = code15_features()
    for c, v in fills.items():
        t[c] = t[c].fillna(v)
    t["p_unmeasured"] = (t.p_match_mean == 0).astype(float)
    _, t_fixed = with_unmeasured_fix(train, t)
    t["prob_final"] = m_final.predict_proba(model_inputs(t, CLINICAL_FEATURES_12))[:, 1]
    t["prob_fixed"] = m_fixed.predict_proba(model_inputs(t_fixed, FIXED_FEATURES))[:, 1]
    y = (t.label == "A").values
    g = t.patient.astype(str).values
    pf, px = t.prob_final.values >= c_final, t.prob_fixed.values >= c_fixed
    print(f"\nCODE-15% part 1: {n_selected} exams, {len(t)} scored ({y.sum()} AF); unscorable: {skipped}; "
          f"p_unmeasured {t.p_unmeasured.mean():.1%}")
    rows = []
    for name, pred, col in [("final", pf, "prob_final"), ("fixed", px, "prob_fixed")]:
        rows.append({"model": name, "auc": auc(t[col].values, y), "AF caught": rate_str(pred[y]),
                     "non-AF called AF": rate_str(pred[~y]),
                     "non-AF with unmeasured P called AF": rate_str(pred[~y & (t.p_unmeasured == 1).values]),
                     "AF with unmeasured P caught": rate_str(pred[y & (t.p_unmeasured == 1).values])})
    res = pd.DataFrame(rows)
    pd.set_option("display.width", 250, "display.max_colwidth", 45)
    print(res.to_string(index=False))
    fa_d = px[~y].mean() - pf[~y].mean()
    fa_ci = boot_ci(lambda i: px[i][~y[i]].mean() - pf[i][~y[i]].mean(), g)
    se_d = px[y].mean() - pf[y].mean()
    se_ci = boot_ci(lambda i: px[i][y[i]].mean() - pf[i][y[i]].mean(), g)
    auc_d = auc(t.prob_fixed.values, y) - auc(t.prob_final.values, y)
    auc_ci = boot_ci(lambda i: auc(t.prob_fixed.values[i], y[i]) - auc(t.prob_final.values[i], y[i]), g)
    ok1, ok2 = fa_ci[1] < 0, se_ci[0] > -0.02
    print(f"\nPRIMARY 1 false alarms, fixed - final: {fa_d:+.4f} (95% CI {fa_ci[0]:+.4f} to {fa_ci[1]:+.4f}) -> "
          f"{'lower' if ok1 else 'not shown lower'}")
    print(f"PRIMARY 2 sensitivity, fixed - final: {se_d:+.4f} (95% CI {se_ci[0]:+.4f} to {se_ci[1]:+.4f}) -> "
          f"{'non-inferior' if ok2 else 'NOT shown non-inferior'}")
    print(f"AUC, fixed - final: {auc_d:+.4f} (95% CI {auc_ci[0]:+.4f} to {auc_ci[1]:+.4f})")
    print(f"VERDICT: {'adopt the fix' if ok1 and ok2 else 'keep the final model'}")
    res.assign(fa_diff=fa_d, fa_diff_lo=fa_ci[0], fa_diff_hi=fa_ci[1], sens_diff=se_d, sens_diff_lo=se_ci[0],
               sens_diff_hi=se_ci[1], auc_diff=auc_d, auc_diff_lo=auc_ci[0], auc_diff_hi=auc_ci[1],
               adopt=ok1 and ok2).to_csv("results/afib_unmeasured_test.csv", index=False)
    t[["record", "patient", "label", "p_unmeasured", "prob_final", "prob_fixed"]].to_csv(
        "results/afib_unmeasured_test_recordings.csv", index=False)
