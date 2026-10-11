"""Re-test of the clinical AF models on PTB-XL, whose labels describe each recording.

The Georgia test (afib_georgia_test.py) suggested its AF labels often
don't describe the rhythm in the 10 s strip: many "AF" recordings were
regular with consistent P waves. PTB-XL's labels are cardiologists'
statements about each ECG itself, so this checks how much of Georgia's
shortfall was the labels and how much the models.

Data: PTB-XL 1.0.3, 10 s 12-lead recordings at 500 Hz, human-validated
reports only (validated_by_human), chosen from the label table alone:
  A   - AFIB rhythm statement, no atrial flutter         (all 984)
  PAC - PAC statement, no AF or flutter                  (all 391)
  N   - SR, no AF, flutter, ectopy or pacing; 1500 drawn at random (seed 0)
The models are exactly those of the Georgia test (nothing retrained or
re-tuned). The plan was committed before the test was run; it is run once.
Several recordings can come from one patient (984 AF from 854 patients).
"""

import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve

from af import (RECORDS, FEATURES_5, CLINICAL_FEATURES, load_af_record, load_12lead, clinical_features,
                clean_windows, fit_with_cutoff, cutoff_for_sensitivity, model_inputs, p_wave_override,
                auc, summarise)
from afib_clinical_dev import dev_crops
from ecg import detect_peaks_adaptive

PTBXL_DIR = "data/ptbxl"


def wilson(k, n, z=1.96):
    p = k / n
    centre = (p + z * z / (2 * n)) / (1 + z * z / n)
    half = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return centre - half, centre + half


os.makedirs("results", exist_ok=True)
os.makedirs("figures", exist_ok=True)

# ------------------------------- the models, built as for the Georgia test

dev = dev_crops()
afdb = []
for rec_id in RECORDS:
    print(f"train {rec_id} ...", flush=True)
    sig, fs, mask, _ = load_af_record(rec_id, 0, 2)
    afdb += [{**w, "record": rec_id} for w in clean_windows(detect_peaks_adaptive, sig, fs, mask, 60)]
m_current, _ = fit_with_cutoff(pd.DataFrame(afdb), FEATURES_5)
c_current = cutoff_for_sensitivity(m_current.predict_proba(model_inputs(dev, FEATURES_5))[:, 1],
                                   dev.af.values, dev.record.values)
m_clinical, c_clinical = fit_with_cutoff(dev, CLINICAL_FEATURES)
fill_f = dev.log_f_wave.median()
print(f"cutoffs: current {c_current:.3f}, clinical {c_clinical:.3f}")

# ----------------------------------------------------------------- test

rows, skipped = [], {}
for i, line in enumerate(open(os.path.join(PTBXL_DIR, "SELECTED")).read().splitlines()):
    path, label = line.split()
    if i % 500 == 0:
        print(f"test {i} ...", flush=True)
    sig, fs = load_12lead(os.path.join(PTBXL_DIR, path))
    row = clinical_features(sig, fs, label == "A")
    if row is None:
        skipped[label] = skipped.get(label, 0) + 1
        continue
    rows.append({**row, "record": path, "label": label})
test = pd.DataFrame(rows)
test["log_f_wave"] = test.log_f_wave.fillna(fill_f)
is_af = (test.label == "A").values

p_cur = m_current.predict_proba(model_inputs(test, FEATURES_5))[:, 1]
p_cli = m_clinical.predict_proba(model_inputs(test, CLINICAL_FEATURES))[:, 1]
preds = {
    "current (AFDB model, clinical cutoff)": (p_cur >= c_current, p_cur),
    "current + P-wave override": (p_wave_override(p_cur >= c_current, test), None),
    "clinical model (primary)": (p_cli >= c_clinical, p_cli),
    "clinical model + P-wave override": (p_wave_override(p_cli >= c_clinical, test), None),
}

print(f"\nPTB-XL: {len(test)} recordings scored ({is_af.sum()} AF, {(test.label == 'PAC').sum()} PAC, "
      f"{(test.label == 'N').sum()} sinus); skipped for too few beats: {skipped}")
out = []
for name, (pred, prob) in preds.items():
    s = summarise(name, is_af, pred, prob)
    lo, hi = wilson(int((pred & is_af).sum()), int(is_af.sum()))
    s.update({"sens_ci_low": lo, "sens_ci_high": hi,
              "false_alarm_rate_PAC": pred[(test.label == "PAC").values].mean(),
              "false_alarm_rate_N": pred[(test.label == "N").values].mean()})
    out.append(s)
results = pd.DataFrame(out)
results.to_csv("results/afib_ptbxl_test.csv", index=False)
test.assign(prob_current=p_cur, prob_clinical=p_cli).to_csv("results/afib_ptbxl_test_recordings.csv", index=False)
print(results.set_index("classifier")[["sensitivity", "sens_ci_low", "sens_ci_high", "false_alarm_rate_PAC",
                                       "false_alarm_rate_N", "false_alarms", "missed_af", "auc"]].round(3).to_string())

missed = test[is_af & ~preds["clinical model (primary)"][0]]
print(f"\nMissed AF (clinical model): {len(missed)}; perfectly regular (CV < 0.05) with consistent "
      f"P waves: {((missed.cv < 0.05) & (missed.p_match_regular >= 0.99)).sum()}  "
      f"(Georgia: 67 of 179)")

plt.figure(figsize=(5.5, 5))
for (name, (pred, prob)), color in zip(list(preds.items())[::2], ["tab:red", "tab:blue"]):
    fpr, tpr_, _ = roc_curve(is_af, prob)
    plt.plot(fpr, tpr_, color=color, label=f"{name.split(' (')[0]} (AUC {auc(prob, is_af):.3f})")
for (name, (pred, _)), color in zip(preds.items(), ["tab:red", "tab:red", "tab:blue", "tab:blue"]):
    plt.plot(pred[~is_af].mean(), pred[is_af].mean(), "s" if "override" in name else "o", color=color)
plt.plot([0, 1], [0, 1], color="0.7", linestyle=":", linewidth=0.8)
plt.xlabel("False alarm rate (PAC + sinus)")
plt.ylabel("Sensitivity")
plt.title("PTB-XL, 10 s 12-lead (circles: cutoff; squares: + override)")
plt.legend(loc="lower right", fontsize=8)
plt.tight_layout()
plt.savefig("figures/afib_ptbxl_test_roc.png", dpi=120)
plt.show()
