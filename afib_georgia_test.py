"""Test of the clinical 12-lead AF model on the Georgia 12-lead ECG Challenge data.

2942 10 s clinical recordings (500 Hz) from the PhysioNet/CinC Challenge 2020
Georgia set, chosen from their headers only:
  A   - AF (164889003) among the diagnoses             (570)
  PAC - PAC (284470004), and no AF or atrial flutter   (620)
  N   - sinus rhythm (426783006) as the only diagnosis (1752)
None of it was used in development; this plan was committed before the
test was run, and it is run once. Recordings with fewer than
CLINICAL_MIN_RR usable RR intervals are skipped and counted.

Models, each with and without the P-wave override:
  current  - AFDB-trained model (5 features, ventricular ectopics removed),
             cutoff for 95% sensitivity on the CPSC development crops
  clinical - logistic regression on CLINICAL_FEATURES (RR features, lead II
             P waves, V1 fibrillatory waves) trained on the CPSC development
             crops (afib_clinical_dev.py), cutoff for 95% sensitivity there.
             Its primary result is without the override, since P waves are
             already in the model.
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

GEORGIA_DIR = "data/georgia"


def wilson(k, n, z=1.96):
    p = k / n
    centre = (p + z * z / (2 * n)) / (1 + z * z / n)
    half = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return centre - half, centre + half


os.makedirs("results", exist_ok=True)
os.makedirs("figures", exist_ok=True)

# ------------------------------------------------------------ the models

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
for i, line in enumerate(open(os.path.join(GEORGIA_DIR, "SELECTED")).read().splitlines()):
    rec_id, label = line.split()
    if i % 500 == 0:
        print(f"test {i} ...", flush=True)
    sig, fs = load_12lead(os.path.join(GEORGIA_DIR, rec_id))
    row = clinical_features(sig, fs, label == "A")
    if row is None:
        skipped[label] = skipped.get(label, 0) + 1
        continue
    rows.append({**row, "record": rec_id, "label": label})
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

print(f"\nGeorgia: {len(test)} recordings scored ({is_af.sum()} AF, {(test.label == 'PAC').sum()} PAC, "
      f"{(test.label == 'N').sum()} normal); skipped for too few beats: {skipped}")
out = []
for name, (pred, prob) in preds.items():
    s = summarise(name, is_af, pred, prob)
    lo, hi = wilson(int((pred & is_af).sum()), int(is_af.sum()))
    s.update({"sens_ci_low": lo, "sens_ci_high": hi,
              "false_alarm_rate_PAC": pred[(test.label == "PAC").values].mean(),
              "false_alarm_rate_N": pred[(test.label == "N").values].mean()})
    out.append(s)
results = pd.DataFrame(out)
results.to_csv("results/afib_georgia_test.csv", index=False)
test.assign(prob_current=p_cur, prob_clinical=p_cli).to_csv("results/afib_georgia_test_recordings.csv", index=False)
print(results.set_index("classifier")[["sensitivity", "sens_ci_low", "sens_ci_high", "false_alarm_rate_PAC",
                                       "false_alarm_rate_N", "false_alarms", "missed_af", "auc"]].round(3).to_string())

plt.figure(figsize=(5.5, 5))
for (name, (pred, prob)), color in zip(list(preds.items())[::2], ["tab:red", "tab:blue"]):
    fpr, tpr_, _ = roc_curve(is_af, prob)
    plt.plot(fpr, tpr_, color=color, label=f"{name.split(' (')[0]} (AUC {auc(prob, is_af):.3f})")
for (name, (pred, _)), color in zip(preds.items(), ["tab:red", "tab:red", "tab:blue", "tab:blue"]):
    plt.plot(pred[~is_af].mean(), pred[is_af].mean(), "s" if "override" in name else "o", color=color)
plt.plot([0, 1], [0, 1], color="0.7", linestyle=":", linewidth=0.8)
plt.xlabel("False alarm rate (PAC + normal)")
plt.ylabel("Sensitivity")
plt.title("Georgia, 10 s 12-lead (circles: cutoff; squares: + override)")
plt.legend(loc="lower right", fontsize=8)
plt.tight_layout()
plt.savefig("figures/afib_georgia_test_roc.png", dpi=120)
plt.show()
