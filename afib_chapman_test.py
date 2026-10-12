"""Test of the retrained clinical AF model on Chapman-Shaoxing (new data, run once).

Development (afib_pwave_dev.py) found that a P-wave measure built to resist
short AF recordings (QRS-T subtraction, leave-one-out template) didn't
help, but that training on more varied clinical data did: with PTB-XL
added to the CPSC crops, cross-validated PTB-XL sensitivity rose from
87.4% (the CPSC-only model's PTB-XL test) to 93.1%, and to 94.6% when the
cutoff is the strictest of the per-source 95% cutoffs (CPSC 98.5%; false
alarms 16.9% on PAC, 1.8% on sinus).

Models (features: CLINICAL_FEATURES, unchanged):
  retrained  - fitted on CPSC crops + PTB-XL, cutoff = strictest per-source
               95% cutoff (registered primary)
  retrained, shared cutoff - same model, one cutoff for 95% on the pool
  previous   - the clinical model of the Georgia and PTB-XL tests (CPSC crops)

Data: Chapman-Shaoxing, from the PhysioNet/CinC Challenge 2021: 10 s
12-lead recordings at 500 Hz, rhythm labelled per ECG by physicians.
Chosen from headers only, using every recording that qualifies:
  A   - AF (164889003), no atrial flutter (164890007)           (1780)
  PAC - PAC (284470004), no AF or flutter                       (258)
  N   - sinus rhythm (426783006), no AF, flutter, PAC, PVC,
        ventricular or supraventricular ectopics, or pacing     (1777)
None of it was used before. This plan is committed before the test
signals are downloaded; the test is run once.
"""

import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve

from af import (CLINICAL_FEATURES, load_12lead, clinical_features, fit_with_cutoff, fit_with_source_cutoffs,
                model_inputs, auc, summarise)
from afib_clinical_dev import dev_crops
from afib_pwave_dev import dev_units, prepare

CHAPMAN_DIR = "data/chapman"


def wilson(k, n, z=1.96):
    p = k / n
    centre = (p + z * z / (2 * n)) / (1 + z * z / n)
    half = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return centre - half, centre + half


os.makedirs("results", exist_ok=True)
os.makedirs("figures", exist_ok=True)

# ------------------------------------------------------------ the models

units = prepare(dev_units())
m_new, cutoffs = fit_with_source_cutoffs(units, CLINICAL_FEATURES)
c_strict = min(cutoffs.values())
_, c_shared = fit_with_cutoff(units, CLINICAL_FEATURES)
m_old, c_old = fit_with_cutoff(dev_crops(), CLINICAL_FEATURES)
fill_f = units.log_f_wave.median()
print(f"cutoffs: per source {cutoffs}, strictest {c_strict:.3f}, shared {c_shared:.3f}, previous {c_old:.3f}")

# ----------------------------------------------------------------- test

rows, skipped = [], {}
for i, line in enumerate(open(os.path.join(CHAPMAN_DIR, "SELECTED")).read().splitlines()):
    path, label = line.split()
    rec_id = os.path.basename(path)
    if i % 500 == 0:
        print(f"test {i} ...", flush=True)
    sig, fs = load_12lead(os.path.join(CHAPMAN_DIR, rec_id))
    row = clinical_features(sig, fs, label == "A")
    if row is None:
        skipped[label] = skipped.get(label, 0) + 1
        continue
    rows.append({**row, "record": rec_id, "label": label})
test = pd.DataFrame(rows)
test["log_f_wave"] = test.log_f_wave.fillna(fill_f)
is_af = (test.label == "A").values

p_new = m_new.predict_proba(model_inputs(test, CLINICAL_FEATURES))[:, 1]
p_old = m_old.predict_proba(model_inputs(test, CLINICAL_FEATURES))[:, 1]
preds = {
    "retrained, strictest-source cutoff (primary)": (p_new >= c_strict, p_new),
    "retrained, shared cutoff": (p_new >= c_shared, None),
    "previous clinical model": (p_old >= c_old, p_old),
}

print(f"\nChapman-Shaoxing: {len(test)} recordings scored ({is_af.sum()} AF, {(test.label == 'PAC').sum()} PAC, "
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
results.to_csv("results/afib_chapman_test.csv", index=False)
test.assign(prob_retrained=p_new, prob_previous=p_old).to_csv("results/afib_chapman_test_recordings.csv",
                                                               index=False)
print(results.set_index("classifier")[["sensitivity", "sens_ci_low", "sens_ci_high", "false_alarm_rate_PAC",
                                       "false_alarm_rate_N", "false_alarms", "missed_af", "auc"]].round(3).to_string())

plt.figure(figsize=(5.5, 5))
for name, color in [("retrained, strictest-source cutoff (primary)", "tab:blue"),
                    ("previous clinical model", "tab:gray")]:
    pred, prob = preds[name]
    fpr, tpr_, _ = roc_curve(is_af, prob)
    plt.plot(fpr, tpr_, color=color, label=f"{name.split(',')[0]} (AUC {auc(prob, is_af):.3f})")
for (name, (pred, _)), (color, marker) in zip(preds.items(), [("tab:blue", "o"), ("tab:blue", "s"),
                                                              ("tab:gray", "o")]):
    plt.plot(pred[~is_af].mean(), pred[is_af].mean(), marker, color=color)
plt.plot([0, 1], [0, 1], color="0.7", linestyle=":", linewidth=0.8)
plt.xlabel("False alarm rate (PAC + sinus)")
plt.ylabel("Sensitivity")
plt.title("Chapman-Shaoxing, 10 s 12-lead\n(circles: strictest/own cutoff; square: shared cutoff)")
plt.legend(loc="lower right", fontsize=8)
plt.tight_layout()
plt.savefig("figures/afib_chapman_test_roc.png", dpi=120)
plt.show()
