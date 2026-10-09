"""Final test of the AF classifiers on data never used during development.

Development (afib.py) used only hours 0-2 of each AFDB record. Here every
classifier is trained on ALL of hours 0-2, its cutoff is set on that same
training data to reach 95% sensitivity, and it is then evaluated once on
hours 2-end (about 8 hours per record).

The classifiers and the 95%-sensitivity rule were fixed before this test
was first run, and should not be changed in response to its results.

Caveat: the test windows come from the same 23 patients as the training
windows, just later in their recordings. This tests generalisation over
time, not to new patients (afib.py's leave-one-record-out results do that).
"""

import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve

from af import (RECORDS, FEATURES_5, TARGET_SENS, load_af_record, window_features,
                fit_thresholds, model_inputs, make_logreg, cutoff_for_sensitivity,
                auc, summarise)
from ecg import detect_peaks_adaptive


def build_windows(start_h, end_h):
    rows = []
    for rec_id in RECORDS:
        print(f"record {rec_id} ...", flush=True)
        sig, fs, is_af, _ = load_af_record(rec_id, start_h, end_h)
        for row in window_features(detect_peaks_adaptive(sig, fs), fs, len(sig), is_af):
            row["record"] = rec_id
            rows.append(row)
    return pd.DataFrame(rows)


os.makedirs("results", exist_ok=True)
os.makedirs("figures", exist_ok=True)

train = build_windows(0, 2)
test = build_windows(2, None)
print(f"\ntrain (hours 0-2): {len(train)} windows, {train.af.sum()} AF")
print(f"test  (hours 2-end): {len(test)} windows, {test.af.sum()} AF, "
      f"{test.record[test.af].nunique()} records with AF")

af = test.af.values
rows, probs, preds = [], {}, {}

t_cv, t_nr = fit_thresholds(train, min_sens=TARGET_SENS)
rule_pred = (test.cv.values > t_cv) & (test.nrmssd.values > t_nr)
rows.append(summarise("threshold rule (CV, nRMSSD)", af, rule_pred))
print(f"threshold rule: CV > {t_cv:.3f} and nRMSSD > {t_nr:.3f}")

for name, features in [("logistic regression (5 features)", FEATURES_5),
                       ("logistic regression (pNN50 only)", ["pnn50"])]:
    model = make_logreg().fit(model_inputs(train, features), train.af)
    cutoff = cutoff_for_sensitivity(
        model.predict_proba(model_inputs(train, features))[:, 1], train.af.values)
    prob = model.predict_proba(model_inputs(test, features))[:, 1]
    probs[name], preds[name] = prob, prob >= cutoff
    rows.append(summarise(name, af, preds[name], prob))
    print(f"{name}: cutoff {cutoff:.3f}")

results = pd.DataFrame(rows)
results.to_csv("results/afib_test.csv", index=False)
print(f"\nTest set, cutoffs set on training data for {TARGET_SENS:.0%} sensitivity:")
print(results.set_index("classifier").round(3).to_string())

test["prob_logreg5"] = probs["logistic regression (5 features)"]
test["prob_pnn50"] = probs["logistic regression (pNN50 only)"]
test["pred_logreg5"] = preds["logistic regression (5 features)"]
test.to_csv("results/afib_test_windows.csv", index=False)

per_rec = test.groupby("record").apply(
    lambda g: pd.Series({"windows": len(g), "af_windows": g.af.sum(),
                         "missed_af": (g.af & ~g.pred_logreg5).sum(),
                         "false_alarms": (~g.af & g.pred_logreg5).sum()}),
    include_groups=False).astype(int)
print("\n5-feature model, per record:")
print(per_rec.to_string())

plt.figure(figsize=(5.5, 5))
colors = {"logistic regression (5 features)": "tab:orange",   # same as afib_roc.png
          "logistic regression (pNN50 only)": "tab:green"}
for name, prob in probs.items():
    fpr, tpr_, _ = roc_curve(af, prob)
    plt.plot(fpr, tpr_, color=colors[name],
             label=f"{name.replace('logistic regression', 'logistic')} (AUC {auc(prob, af):.3f})")
plt.plot(1 - rows[0]["specificity"], rows[0]["sensitivity"], "ks", label="threshold rule")
plt.plot([0, 1], [0, 1], color="0.7", linestyle=":", linewidth=0.8)
plt.xlabel("False alarm rate (1 - specificity)")
plt.ylabel("Sensitivity")
plt.title("AF detection, held-out hours 2-end")
plt.legend(loc="lower right", fontsize=8)
plt.tight_layout()
plt.savefig("figures/afib_test_roc.png", dpi=120)
plt.show()
