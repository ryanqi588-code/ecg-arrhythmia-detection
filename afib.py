"""Atrial fibrillation check on the MIT-BIH Atrial Fibrillation Database (development).

AF makes the heartbeat irregularly irregular, so we detect R-peaks with our
adaptive detector and measure RR-interval irregularity in 60 s windows:
  CV       - std(RR) / mean(RR)
  nRMSSD   - RMS of successive RR differences / mean(RR)
  entropy  - Shannon entropy of the 16-bin RR histogram, scaled to 0-1
  TPR      - turning points ratio: fraction of RR intervals that are a local peak/trough
  pNN50    - fraction of successive RR differences over 50 ms

A window is labelled AF if at least half of it is annotated (AFIB; atrial
flutter (AFL) and junctional rhythm (J) count as non-AF.

Classifiers compared, all by leave-one-record-out cross-validation (each
record is classified by a model fitted on the other 22 records only):
  - threshold rule: AF when both CV and nRMSSD exceed grid-searched thresholds
  - logistic regression on CV + nRMSSD, on all 5 features, and on pNN50 alone
Each is reported at two operating points: its default (balanced accuracy /
probability 0.5) and a cutoff chosen on the training folds to reach 95%
sensitivity.

This script uses only the first 2 hours of each record. Hours 2-10 are kept
as an untouched test set for afib_test.py. Signals are cached in data/afdb/.
"""

import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import NullFormatter
from sklearn.metrics import roc_curve

from af import (RECORDS, FEATURES_5, TARGET_SENS, load_af_record, window_features,
                fit_thresholds, model_inputs, make_logreg, cutoff_for_sensitivity,
                auc, summarise)
from ecg import bandpass, detect_peaks_adaptive, score


def rule_leave_one_record_out(df, min_sens=None):
    """Out-of-fold rule predictions; returns (pred, list of fitted thresholds)."""
    pred = np.zeros(len(df), dtype=bool)
    fitted = []
    for rec_id in df.record.unique():
        test = (df.record == rec_id).values
        t_cv, t_nr = fit_thresholds(df[~test], min_sens)
        fitted.append((t_cv, t_nr))
        pred[test] = (df.cv[test] > t_cv) & (df.nrmssd[test] > t_nr)
    return pred, fitted


def logreg_leave_one_record_out(df, features):
    """Out-of-fold results for each window.

    Returns (probability, AF prediction at a cutoff giving 95% sensitivity
    on the training folds, each fold's coefficients).
    """
    prob = np.zeros(len(df))
    pred_sens = np.zeros(len(df), dtype=bool)
    coefs = []
    for rec_id in df.record.unique():
        test = (df.record == rec_id).values
        model = make_logreg().fit(model_inputs(df[~test], features), df.af[~test])
        train_prob = model.predict_proba(model_inputs(df[~test], features))[:, 1]
        cutoff = cutoff_for_sensitivity(train_prob, df.af[~test].values)
        prob[test] = model.predict_proba(model_inputs(df[test], features))[:, 1]
        pred_sens[test] = prob[test] >= cutoff
        coefs.append(model[-1].coef_[0])
    return prob, pred_sens, np.array(coefs)


# ----------------------------------------- detect beats, build windows

os.makedirs("results", exist_ok=True)
os.makedirs("figures", exist_ok=True)

windows, ref_windows, det_rows, examples = [], [], [], {}
for rec_id in RECORDS:
    print(f"record {rec_id} ...", flush=True)
    sig, fs, is_af, qrs = load_af_record(rec_id, 0, 2)
    peaks = detect_peaks_adaptive(sig, fs)

    tp, fp, fn, sens, ppv = score(qrs, peaks, int(0.15 * fs))
    det_rows.append((rec_id, tp, fp, fn, sens, ppv))

    # Same windows built from PhysioNet's beat positions, to separate
    # beat-detector errors from limits of the RR-irregularity method.
    for out, beats in [(windows, peaks), (ref_windows, np.asarray(qrs))]:
        for row in window_features(beats, fs, len(sig), is_af):
            row["record"] = rec_id
            out.append(row)
    examples[rec_id] = (peaks, fs, is_af)

df = pd.DataFrame(windows)
det = pd.DataFrame(det_rows, columns=["record", "TP", "FP", "FN", "sensitivity", "precision"])
det.to_csv("results/afib_beat_detection.csv", index=False)

# The qrs annotations are machine-generated and unaudited. On record 07162
# they mostly fall between beats (checked by plotting), so it is also
# reported without that record.
print("\nBeat detection vs PhysioNet's (unaudited) qrs annotations:")
for label, d in [("all records", det), ("excluding 07162", det[det.record != "07162"])]:
    tot = d[["TP", "FP", "FN"]].sum()
    print(f"  {label:<16} sensitivity={tot.TP / (tot.TP + tot.FN):.4f}  "
          f"precision={tot.TP / (tot.TP + tot.FP):.4f}")
print(f"{len(df)} windows used ({df.af.sum()} AF, {(~df.af).sum()} non-AF) from {df.record.nunique()} records")
af = df.af.values


# -------------------------------------- leave-one-record-out evaluation

df["pred"], fitted = rule_leave_one_record_out(df)
t_cv_med, t_nr_med = np.median(fitted, axis=0)
print(f"\nThreshold rule, median thresholds over folds: CV > {t_cv_med:.3f} and nRMSSD > {t_nr_med:.3f}")
print(f"ROC AUC of each feature alone (no fitting needed): " + "  ".join(
    f"{f}={auc(df[f].values, af):.3f}" for f in FEATURES_5))

ref_df = pd.DataFrame(ref_windows)
ref_pred, _ = rule_leave_one_record_out(ref_df)
r_af = ref_df.af.values
print(f"Threshold rule on PhysioNet's qrs beat positions instead of ours: "
      f"sensitivity {ref_pred[r_af].mean():.3f}  specificity {(~ref_pred[~r_af]).mean():.3f}")

prob2, pred2_sens, _ = logreg_leave_one_record_out(df, ["cv", "nrmssd"])
prob5, pred5_sens, fold_coefs = logreg_leave_one_record_out(df, FEATURES_5)
# pNN50 alone was added after seeing that its single-feature AUC
# matched the 5-feature model's.
prob1, pred1_sens, _ = logreg_leave_one_record_out(df, ["pnn50"])
rule_sens, _ = rule_leave_one_record_out(df, min_sens=TARGET_SENS)

df["prob_logreg2"], df["prob_logreg5"], df["prob_pnn50"] = prob2, prob5, prob1
df.to_csv("results/afib_windows.csv", index=False)

comparison = pd.DataFrame([
    summarise("threshold rule (CV, nRMSSD)", af, df.pred.values),
    summarise("logistic regression (CV, nRMSSD)", af, prob2 >= 0.5, prob2),
    summarise("logistic regression (5 features)", af, prob5 >= 0.5, prob5),
    summarise("logistic regression (pNN50 only)", af, prob1 >= 0.5, prob1),
])
comparison.to_csv("results/afib_classifiers.csv", index=False)
print("\nDefault operating points (rule: best balanced accuracy; logistic: probability >= 0.5):")
print(comparison.set_index("classifier").round(3).to_string())

at_sens = pd.DataFrame([
    summarise("threshold rule (CV, nRMSSD)", af, rule_sens),
    summarise("logistic regression (CV, nRMSSD)", af, pred2_sens, prob2),
    summarise("logistic regression (5 features)", af, pred5_sens, prob5),
    summarise("logistic regression (pNN50 only)", af, pred1_sens, prob1),
])
at_sens.to_csv("results/afib_classifiers_95sens.csv", index=False)
print(f"\nCutoffs chosen on the training folds for {TARGET_SENS:.0%} sensitivity:")
print(at_sens.set_index("classifier").round(3).to_string())

# Coefficients of a model fitted on all windows (standardised inputs, so
# each is the change in log-odds of AF per standard deviation), with the
# range across the leave-one-out folds to show how stable they are.
full = make_logreg().fit(model_inputs(df, FEATURES_5), df.af)
coefs = pd.DataFrame({"coefficient": full[-1].coef_[0],
                      "fold_min": fold_coefs.min(axis=0),
                      "fold_max": fold_coefs.max(axis=0)},
                     index=["log CV", "log nRMSSD", "entropy", "TPR", "pNN50"])
print("\nLogistic regression coefficients (per standard deviation of each feature):")
print(coefs.round(2).to_string())
print("Feature correlations:")
print(pd.DataFrame(model_inputs(df, FEATURES_5), columns=coefs.index).corr().round(2).to_string())

df["pred_logreg5"] = prob5 >= 0.5
per_rec = df.groupby("record").apply(
    lambda g: pd.Series({"windows": len(g), "af_windows": g.af.sum(),
                         "rule_accuracy": (g.af == g.pred).mean(),
                         "logreg5_accuracy": (g.af == g.pred_logreg5).mean()}),
    include_groups=False)
print("\nPer-record accuracy (worst 5 under the threshold rule):")
print(per_rec.sort_values("rule_accuracy").head(5).round(3).to_string())


# ------------------------------------------------------------- figures

plt.figure(figsize=(6, 5))
plt.scatter(df.cv[~df.af], df.nrmssd[~df.af], s=6, alpha=0.4, label="non-AF")
plt.scatter(df.cv[df.af], df.nrmssd[df.af], s=6, alpha=0.4, label="AF")
plt.axvline(t_cv_med, color="k", linestyle="--", linewidth=0.8)
plt.axhline(t_nr_med, color="k", linestyle="--", linewidth=0.8)
plt.xlabel("CV of RR intervals")
plt.ylabel("nRMSSD")
plt.xscale("log")
plt.yscale("log")
ticks = [0.01, 0.02, 0.05, 0.1, 0.2, 0.5]
for axis in (plt.gca().xaxis, plt.gca().yaxis):
    axis.set_minor_formatter(NullFormatter())
    axis.set_ticks(ticks, labels=[str(v) for v in ticks])
plt.title("60 s windows: RR irregularity (dashed = median thresholds)")
plt.legend()
plt.tight_layout()
plt.savefig("figures/afib_scatter.png", dpi=120)
plt.show()

# ROC curves: every operating point of each model, vs the rule's points
plt.figure(figsize=(5.5, 5))
for name, prob in [("logistic, CV + nRMSSD", prob2),
                   ("logistic, 5 features", prob5),
                   ("logistic, pNN50 only", prob1)]:
    fpr, tpr_, _ = roc_curve(af, prob)
    plt.plot(fpr, tpr_, label=f"{name} (AUC {auc(prob, af):.3f})")
for row, marker, label in [(comparison.iloc[0], "ko", "threshold rule, balanced"),
                           (at_sens.iloc[0], "ks", "threshold rule, 95% sens. target")]:
    plt.plot(1 - row.specificity, row.sensitivity, marker, label=label)
plt.plot([0, 1], [0, 1], color="0.7", linestyle=":", linewidth=0.8)
plt.xlabel("False alarm rate (1 - specificity)")
plt.ylabel("Sensitivity")
plt.title("AF detection, leave-one-record-out")
plt.legend(loc="lower right", fontsize=8)
plt.tight_layout()
plt.savefig("figures/afib_roc.png", dpi=120)
plt.show()

# RR tachogram for the record with the most even mix of AF and non-AF
mix = per_rec[per_rec.windows > 0].assign(frac=lambda d: d.af_windows / d.windows)
rec_id = (mix.frac - 0.5).abs().idxmin()
peaks, fs, is_af = examples[rec_id]
rr = np.diff(peaks) / fs
t_min = peaks[1:] / fs / 60
plt.figure(figsize=(12, 3.5))
plt.plot(t_min, rr, ".", markersize=1.5, color="0.3")
plt.fill_between(np.arange(len(is_af))[::fs] / fs / 60, 0, 2.5 * is_af[::fs],
                 color="tab:orange", alpha=0.2, step="post", label="annotated AF")
plt.ylim(0, 2.5)
plt.xlabel("Time (min)")
plt.ylabel("RR interval (s)")
plt.title(f"AFDB record {rec_id}: RR intervals from our detector")
plt.legend(loc="upper right")
plt.tight_layout()
plt.savefig("figures/afib_tachogram.png", dpi=120)
plt.show()

# Record 07162: our detections vs the unaudited qrs annotations
sig, fs, _, qrs = load_af_record("07162", 0, 2)
peaks = examples["07162"][0]
filt = bandpass(sig, fs)
i0, i1 = 600 * fs, 608 * fs
p = peaks[(peaks >= i0) & (peaks < i1)]
q = qrs[(qrs >= i0) & (qrs < i1)]
plt.figure(figsize=(12, 3.5))
plt.plot(np.arange(i0, i1) / fs, filt[i0:i1], linewidth=0.8)
plt.plot(p / fs, filt[p], "go", label="our detections")
plt.plot(q / fs, filt[q], "rv", label="PhysioNet qrs annotations")
plt.xlabel("Time (s)")
plt.ylabel("Amplitude (mV)")
plt.title("AFDB record 07162: the reference annotations miss the QRS complexes")
plt.legend(loc="upper left", bbox_to_anchor=(1.01, 1))
plt.tight_layout()
plt.savefig("figures/afib_07162_reference.png", dpi=120)
plt.show()
