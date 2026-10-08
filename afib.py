"""Atrial fibrillation check on the MIT-BIH Atrial Fibrillation Database.

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

Only the first 2 hours of each record are used, because downloading the
full database (~635 MB) from PhysioNet is slow. Signals are cached in
data/afdb/ after the first run.
"""

import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import NullFormatter
import wfdb
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_curve
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from ecg import bandpass, detect_peaks_adaptive, score

NO_SIGNAL = ["00735", "03665"]   # these two records ship without ECG signals
RECORDS = [r for r in wfdb.get_record_list("afdb") if r not in NO_SIGNAL]
HOURS = 2
WINDOW_SEC = 60
MIN_RR = 20                      # skip windows with fewer valid RR intervals (noise/dropout)
CACHE_DIR = "data/afdb"


def load_af_record(rec_id):
    """Return (ECG1 signal, fs, rhythm annotation, qrs annotation sample indices).

    The signal is cached as .npy because PhysioNet downloads are slow.
    """
    os.makedirs(CACHE_DIR, exist_ok=True)
    cache = os.path.join(CACHE_DIR, f"{rec_id}_{HOURS}h.npy")
    fs = 250
    n = HOURS * 3600 * fs
    if os.path.exists(cache):
        sig = np.load(cache)
    else:
        rec = wfdb.rdrecord(rec_id, pn_dir="afdb", channels=[0], sampto=n)
        sig, fs = rec.p_signal[:, 0], rec.fs
        np.save(cache, sig)
    rhythm = wfdb.rdann(rec_id, "atr", pn_dir="afdb", sampto=n)
    qrs = wfdb.rdann(rec_id, "qrs", pn_dir="afdb", sampto=n).sample
    return sig, fs, rhythm, qrs


def af_mask(rhythm, n):
    """Boolean array, True where the rhythm annotation is (AFIB."""
    mask = np.zeros(n, dtype=bool)
    bounds = list(rhythm.sample) + [n]
    for start, end, label in zip(bounds[:-1], bounds[1:], rhythm.aux_note):
        if label.strip("\x00 ") == "(AFIB":
            mask[start:end] = True
    return mask


def shannon_entropy(rr, bins=16):
    """Entropy of the RR histogram (bins span the window's own range), scaled to 0-1."""
    counts, _ = np.histogram(rr, bins=bins)
    p = counts[counts > 0] / len(rr)
    return -np.sum(p * np.log(p)) / np.log(bins)


def turning_points_ratio(rr):
    """Fraction of interior RR intervals that are a local peak or trough.

    About 2/3 for a random series; near 0 for a smooth trend.
    """
    d = np.diff(rr)
    return np.mean(d[:-1] * d[1:] < 0)


def window_features(peaks, fs, n, is_af):
    """One row per 60 s window: AF label and RR-irregularity features."""
    rows = []
    win = WINDOW_SEC * fs
    for start in range(0, n - win + 1, win):
        p = peaks[(peaks >= start) & (peaks < start + win)]
        rr = np.diff(p) / fs
        rr = rr[(rr > 0.25) & (rr < 2.5)]          # 24-240 BPM
        if len(rr) < MIN_RR:
            continue
        mean = rr.mean()
        rows.append({
            "start_min": start / fs / 60,
            "af": is_af[start:start + win].mean() >= 0.5,
            "cv": rr.std() / mean,
            "nrmssd": np.sqrt(np.mean(np.diff(rr) ** 2)) / mean,
            "entropy": shannon_entropy(rr),
            "tpr": turning_points_ratio(rr),
            "pnn50": np.mean(np.abs(np.diff(rr)) > 0.05),
        })
    return rows


def fit_thresholds(df):
    """Grid-search (cv, nrmssd) thresholds maximising balanced accuracy.

    A window is called AF when both features exceed their thresholds.
    """
    best = (-1, None, None)
    af, cv, nr = df.af.values, df.cv.values, df.nrmssd.values
    for t_cv in np.arange(0.02, 0.30, 0.005):
        for t_nr in np.arange(0.02, 0.40, 0.005):
            pred = (cv > t_cv) & (nr > t_nr)
            sens = pred[af].mean()
            spec = (~pred[~af]).mean()
            bal = (sens + spec) / 2
            if bal > best[0]:
                best = (bal, t_cv, t_nr)
    return best[1], best[2]


def auc(score_, label):
    """ROC AUC via the Mann-Whitney rank statistic."""
    ranks = pd.Series(score_).rank().values
    n_pos, n_neg = label.sum(), (~label).sum()
    return (ranks[label].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)


def leave_one_record_out(df):
    """Add a 'pred' column; each record uses thresholds fitted on the others.

    Returns the list of fitted (cv, nrmssd) thresholds.
    """
    df["pred"] = False
    fitted = []
    for rec_id in df.record.unique():
        test = df.record == rec_id
        t_cv, t_nr = fit_thresholds(df[~test])
        fitted.append((t_cv, t_nr))
        df.loc[test, "pred"] = (df.cv[test] > t_cv) & (df.nrmssd[test] > t_nr)
    return fitted


def model_inputs(df, features):
    """Feature matrix; CV and nRMSSD are log-transformed (they span two decades)."""
    X = df[features].copy()
    for col in ("cv", "nrmssd"):
        if col in X:
            X[col] = np.log(X[col])
    return X.values


def make_logreg():
    # Standardised inputs make coefficients comparable; balanced class
    # weights match the threshold rule's balanced-accuracy objective.
    return make_pipeline(StandardScaler(),
                         LogisticRegression(class_weight="balanced", max_iter=1000))


def logreg_leave_one_record_out(df, features):
    """Out-of-fold AF probability for each window, plus each fold's coefficients."""
    prob = np.zeros(len(df))
    coefs = []
    for rec_id in df.record.unique():
        test = (df.record == rec_id).values
        model = make_logreg().fit(model_inputs(df[~test], features), df.af[~test])
        prob[test] = model.predict_proba(model_inputs(df[test], features))[:, 1]
        coefs.append(model[-1].coef_[0])
    return prob, np.array(coefs)


def summarise(name, af, pred, prob=None):
    tp, fn = (pred & af).sum(), (~pred & af).sum()
    tn, fp = (~pred & ~af).sum(), (pred & ~af).sum()
    return {"classifier": name,
            "sensitivity": tp / (tp + fn), "specificity": tn / (tn + fp),
            "precision": tp / (tp + fp), "balanced_accuracy": (tp / (tp + fn) + tn / (tn + fp)) / 2,
            "auc": auc(prob, af) if prob is not None else np.nan,
            "false_alarms": fp, "missed_af": fn}


# ----------------------------------------- detect beats, build windows

os.makedirs("results", exist_ok=True)
os.makedirs("figures", exist_ok=True)

windows, ref_windows, det_rows, examples = [], [], [], {}
for rec_id in RECORDS:
    print(f"record {rec_id} ...", flush=True)
    sig, fs, rhythm, qrs = load_af_record(rec_id)
    peaks = detect_peaks_adaptive(sig, fs)
    is_af = af_mask(rhythm, len(sig))

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


# -------------------------------------- leave-one-record-out evaluation

fitted = leave_one_record_out(df)

af, pred = df.af.values, df.pred.values
tp, fn = (pred & af).sum(), (~pred & af).sum()
tn, fp = (~pred & ~af).sum(), (pred & ~af).sum()
t_cv_med, t_nr_med = np.median(fitted, axis=0)

print("\nAF detection, 60 s windows, leave-one-record-out:")
print(f"  sensitivity  {tp / (tp + fn):.3f}   ({tp}/{tp + fn} AF windows caught)")
print(f"  specificity  {tn / (tn + fp):.3f}   ({tn}/{tn + fp} non-AF windows correctly passed)")
print(f"  precision    {tp / (tp + fp):.3f}")
print(f"  accuracy     {(tp + tn) / len(df):.3f}")
print(f"  thresholds (median over folds): CV > {t_cv_med:.3f} and nRMSSD > {t_nr_med:.3f}")
print(f"\nROC AUC of each feature alone (no threshold needed): "
      f"CV={auc(df.cv.values, af):.3f}  nRMSSD={auc(df.nrmssd.values, af):.3f}")

ref_df = pd.DataFrame(ref_windows)
leave_one_record_out(ref_df)
r_af, r_pred = ref_df.af.values, ref_df.pred.values
print(f"\nSame method on PhysioNet's qrs beat positions instead of ours: "
      f"sensitivity {r_pred[r_af].mean():.3f}  specificity {(~r_pred[~r_af]).mean():.3f}")


# ------------------------------------ logistic regression, more features

FEATURES_2 = ["cv", "nrmssd"]
FEATURES_5 = ["cv", "nrmssd", "entropy", "tpr", "pnn50"]

prob2, _ = logreg_leave_one_record_out(df, FEATURES_2)
prob5, fold_coefs = logreg_leave_one_record_out(df, FEATURES_5)
# pNN50 alone was added after seeing that its single-feature AUC
# matched the 5-feature model's.
prob1, _ = logreg_leave_one_record_out(df, ["pnn50"])
df["prob_logreg2"], df["prob_logreg5"], df["prob_pnn50"] = prob2, prob5, prob1
df.to_csv("results/afib_windows.csv", index=False)

comparison = pd.DataFrame([
    summarise("threshold rule (CV, nRMSSD)", af, df.pred.values),
    summarise("logistic regression (CV, nRMSSD)", af, prob2 >= 0.5, prob2),
    summarise("logistic regression (5 features)", af, prob5 >= 0.5, prob5),
    summarise("logistic regression (pNN50 only)", af, prob1 >= 0.5, prob1),
])
comparison.to_csv("results/afib_classifiers.csv", index=False)
print("\nClassifier comparison, leave-one-record-out:")
print(comparison.set_index("classifier").round(3).to_string())

print("\nROC AUC of each new feature alone: " + "  ".join(
    f"{f}={auc(df[f].values, af):.3f}" for f in ["entropy", "tpr", "pnn50"]))

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

# ROC curves: every operating point of each model, vs the rule's single point
plt.figure(figsize=(5.5, 5))
for name, prob in [("logistic, CV + nRMSSD", prob2),
                   ("logistic, 5 features", prob5),
                   ("logistic, pNN50 only", prob1)]:
    fpr, tpr_, _ = roc_curve(af, prob)
    plt.plot(fpr, tpr_, label=f"{name} (AUC {auc(prob, af):.3f})")
rule = comparison.iloc[0]
plt.plot(1 - rule.specificity, rule.sensitivity, "ko", label="threshold rule")
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
sig, fs, _, qrs = load_af_record("07162")
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
