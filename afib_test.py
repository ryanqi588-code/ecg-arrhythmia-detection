"""Final tests of the AF classifiers on data not used during development.

Every classifier is trained on hours 0-2 of all 23 AFDB records (the
development data) with its cutoff set for 95% sensitivity (out-of-fold,
records weighted equally; see af.fit_with_cutoff), then scored on:

1. AFDB hours 2-end (~8 h per record). Same 23 patients, later in their
   recordings. This set was first used to test an earlier cutoff method
   (in-sample, per window), which undershot its target (92.5%). The cutoff
   method was then changed using development data only, so this run is a
   SECOND look at this set and is somewhat optimistic.
2. The Long-Term AF Database, first 2 hours of each of 84 records:
   different patients, a different recorder and 128 Hz sampling. Its test
   plan was fixed before any of its data was looked at; this is the clean
   test, and the only one of generalisation to new patients.

Classifiers, all registered before the Long-Term AF Database test was run:
the threshold rule; logistic regression on 5 features; on pNN50 alone; and
on 5 features after removing RR intervals around likely ectopic beats.
"""

import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve

from af import (RECORDS, LTAF_RECORDS, FEATURES_5, TARGET_SENS, load_af_record,
                load_ltaf_record, window_features, fit_thresholds, model_inputs,
                fit_with_cutoff, auc, summarise)
from ecg import detect_peaks_adaptive, flag_ectopic

# (name, features, plot colour, uses windows with ectopic beats removed)
CLASSIFIERS = [("logistic regression (5 features)", FEATURES_5, "tab:orange", False),
               ("logistic regression (pNN50 only)", ["pnn50"], "tab:green", False),
               ("logistic regression (5 features, ectopic beats removed)", FEATURES_5, "tab:red", True)]


def windows_from(records, load):
    """Windows from each record, plus the same windows with ectopic beats removed."""
    rows, ect_rows = [], []
    for rec_id in records:
        print(f"record {rec_id} ...", flush=True)
        sig, fs, is_af = load(rec_id)
        peaks = detect_peaks_adaptive(sig, fs)
        for out, ectopic in [(rows, None), (ect_rows, flag_ectopic(sig, fs, peaks))]:
            for row in window_features(peaks, fs, len(sig), is_af, ectopic):
                row["record"] = rec_id
                out.append(row)
    df, ect = pd.DataFrame(rows), pd.DataFrame(ect_rows)
    assert len(df) == len(ect) and (df.af.values == ect.af.values).all()
    return df, ect


def record_avg_sensitivity(test, pred):
    """Mean over records (with >= 5 AF windows) of the fraction of their AF windows caught."""
    g = test[test.af].assign(caught=pred[test.af.values]).groupby("record").caught
    return g.mean()[g.size() >= 5].mean()


def evaluate(tests, label, name):
    test, test_ect = tests
    af = test.af.values
    rows, probs = [], {}
    rule_pred = (test.cv.values > t_cv) & (test.nrmssd.values > t_nr)
    rows.append(summarise("threshold rule (CV, nRMSSD)", af, rule_pred)
                | {"record_avg_sensitivity": record_avg_sensitivity(test, rule_pred)})
    for clf, features, _, ectopic in CLASSIFIERS:
        data = test_ect if ectopic else test
        prob = models[clf][0].predict_proba(model_inputs(data, features))[:, 1]
        pred = prob >= models[clf][1]
        probs[clf] = prob
        rows.append(summarise(clf, af, pred, prob)
                    | {"record_avg_sensitivity": record_avg_sensitivity(test, pred)})
        short = "logreg5_ectopic_removed" if ectopic else "logreg5" if len(features) > 1 else "pnn50"
        test[f"prob_{short}"], test[f"pred_{short}"] = prob, pred
    test["ectopic_frac"] = test_ect.ectopic_frac.values

    results = pd.DataFrame(rows)
    results.to_csv(f"results/{name}.csv", index=False)
    test.to_csv(f"results/{name}_windows.csv", index=False)
    print(f"\n{label}: {len(test)} windows, {af.sum()} AF, "
          f"{test.record[test.af].nunique()} of {test.record.nunique()} records with AF")
    print(results.set_index("classifier").round(3).to_string())

    per_rec = test.groupby("record").apply(
        lambda g: pd.Series({"af_windows": g.af.sum(),
                             "missed_af": (g.af & ~g.pred_logreg5).sum(),
                             "false_alarms": (~g.af & g.pred_logreg5).sum(),
                             "missed_af_ectopic_removed": (g.af & ~g.pred_logreg5_ectopic_removed).sum(),
                             "false_alarms_ectopic_removed": (~g.af & g.pred_logreg5_ectopic_removed).sum()}),
        include_groups=False).astype(int)
    print("5-feature models, records with the most errors (without ectopic removal):")
    print(per_rec.assign(errors=per_rec.missed_af + per_rec.false_alarms)
          .sort_values("errors", ascending=False).head(6).to_string())

    plt.figure(figsize=(5.5, 5))
    for clf, _, color, _ in CLASSIFIERS:
        fpr, tpr_, _ = roc_curve(af, probs[clf])
        plt.plot(fpr, tpr_, color=color,
                 label=f"{clf.replace('logistic regression', 'logistic')} (AUC {auc(probs[clf], af):.3f})")
    plt.plot(1 - rows[0]["specificity"], rows[0]["sensitivity"], "ks", label="threshold rule")
    for i, color in [(1, "tab:orange"), (3, "tab:red")]:
        plt.plot(1 - rows[i]["specificity"], rows[i]["sensitivity"], "o", color=color)
    plt.plot([0, 1], [0, 1], color="0.7", linestyle=":", linewidth=0.8)
    plt.xlabel("False alarm rate (1 - specificity)")
    plt.ylabel("Sensitivity")
    plt.title(label)
    plt.legend(loc="lower right", fontsize=8)
    plt.tight_layout()
    plt.savefig(f"figures/{name}_roc.png", dpi=120)
    plt.show()


os.makedirs("results", exist_ok=True)
os.makedirs("figures", exist_ok=True)

train, train_ect = windows_from(RECORDS, lambda r: load_af_record(r, 0, 2)[:3])
print(f"\ntrain (AFDB hours 0-2): {len(train)} windows, {train.af.sum()} AF")

t_cv, t_nr = fit_thresholds(train, min_sens=TARGET_SENS)
print(f"threshold rule: CV > {t_cv:.3f} and nRMSSD > {t_nr:.3f}")
models = {}
for clf, features, _, ectopic in CLASSIFIERS:
    models[clf] = fit_with_cutoff(train_ect if ectopic else train, features)
    print(f"{clf}: cutoff {models[clf][1]:.3f}")

afdb_later = windows_from(RECORDS, lambda r: load_af_record(r, 2, None)[:3])
ltaf = windows_from(LTAF_RECORDS, load_ltaf_record)

evaluate(afdb_later, "AFDB hours 2-end (second look)", "afib_test")
evaluate(ltaf, "Long-Term AF Database, new patients", "afib_ltaf_test")
