"""Shared AF helpers: loading AFDB records, RR features, classifiers, metrics."""

import os
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd
import wfdb
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

NO_SIGNAL = ["00735", "03665"]   # these two records ship without ECG signals
RECORDS = [r for r in wfdb.get_record_list("afdb") if r not in NO_SIGNAL]
FS = 250
WINDOW_SEC = 60
MIN_RR = 20                      # skip windows with fewer valid RR intervals (noise/dropout)
CACHE_DIR = "data/afdb"
TARGET_SENS = 0.95               # operating point for choosing cutoffs

FEATURES_5 = ["cv", "nrmssd", "entropy", "tpr", "pnn50"]


# ------------------------------------------------------------ loading

def load_af_record(rec_id, start_h, end_h=None):
    """Return (ECG1 signal, fs, rhythm annotation, qrs sample indices) for [start_h, end_h).

    Sample indices are relative to the start of the segment. end_h=None
    means the end of the record. Signals are cached as .npy because
    PhysioNet downloads are slow.
    """
    os.makedirs(CACHE_DIR, exist_ok=True)
    tag = f"{start_h}-{end_h}h" if end_h is not None else f"{start_h}h-end"
    cache = os.path.join(CACHE_DIR, f"{rec_id}_{tag}.npy")
    sampfrom = int(start_h * 3600 * FS)
    sampto = int(end_h * 3600 * FS) if end_h is not None else None
    if os.path.exists(cache):
        sig = np.load(cache)
    else:
        rec = wfdb.rdrecord(rec_id, pn_dir="afdb", channels=[0],
                            sampfrom=sampfrom, sampto=sampto)
        sig = rec.p_signal[:, 0]
        np.save(cache, sig)
    end = sampfrom + len(sig)
    rhythm = wfdb.rdann(rec_id, "atr", pn_dir="afdb")
    qrs = wfdb.rdann(rec_id, "qrs", pn_dir="afdb").sample
    qrs = qrs[(qrs >= sampfrom) & (qrs < end)] - sampfrom
    return sig, FS, af_mask(rhythm, sampfrom, end), qrs


def prefetch(start_h, end_h=None, workers=4):
    """Download and cache every record's segment, several at a time."""
    with ThreadPoolExecutor(workers) as pool:
        for rec_id in pool.map(lambda r: (load_af_record(r, start_h, end_h), r)[1], RECORDS):
            print(f"cached {rec_id}", flush=True)


def af_mask(rhythm, start, end):
    """Boolean array over samples [start, end), True where the rhythm is (AFIB.

    Uses the full-record rhythm annotation so a segment that begins
    mid-episode still gets the right label.
    """
    mask = np.zeros(end - start, dtype=bool)
    bounds = list(rhythm.sample) + [np.iinfo(np.int64).max]
    for s, e, label in zip(bounds[:-1], bounds[1:], rhythm.aux_note):
        if label.strip("\x00 ") == "(AFIB":
            lo, hi = max(s, start), min(e, end)
            if lo < hi:
                mask[lo - start:hi - start] = True
    return mask


# ----------------------------------------------------------- features

def shannon_entropy(rr, bins=16):
    """Entropy of the RR histogram, scaled to 0-1.

    The bins span the window's 5th-95th percentile range; intervals outside
    it are dropped first (as in Dash et al. 2009). Without that, one missed
    or extra beat stretches the range and squeezes the rest into a few
    bins: on record 07859 this pulled AF windows' entropy from ~0.93 to ~0.69.
    """
    lo, hi = np.percentile(rr, [5, 95])
    rr = rr[(rr >= lo) & (rr <= hi)]
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


# -------------------------------------------------------- classifiers

def fit_thresholds(df, min_sens=None):
    """Grid-search (cv, nrmssd) thresholds for the rule "AF if both exceed".

    With min_sens=None, maximise balanced accuracy. Otherwise maximise
    specificity among thresholds whose sensitivity is at least min_sens.
    """
    best = (-1, None, None)
    af, cv, nr = df.af.values, df.cv.values, df.nrmssd.values
    for t_cv in np.arange(0.02, 0.30, 0.005):
        for t_nr in np.arange(0.02, 0.40, 0.005):
            pred = (cv > t_cv) & (nr > t_nr)
            sens = pred[af].mean()
            spec = (~pred[~af]).mean()
            if min_sens is None:
                objective = (sens + spec) / 2
            elif sens >= min_sens:
                objective = spec
            else:
                continue
            if objective > best[0]:
                best = (objective, t_cv, t_nr)
    return best[1], best[2]


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


def cutoff_for_sensitivity(prob, af, target=TARGET_SENS):
    """Highest probability cutoff that still catches `target` of the AF windows."""
    return np.quantile(prob[af], 1 - target)


# ------------------------------------------------------------ metrics

def auc(score_, label):
    """ROC AUC via the Mann-Whitney rank statistic."""
    ranks = pd.Series(score_).rank().values
    n_pos, n_neg = label.sum(), (~label).sum()
    return (ranks[label].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)


def summarise(name, af, pred, prob=None):
    tp, fn = (pred & af).sum(), (~pred & af).sum()
    tn, fp = (~pred & ~af).sum(), (pred & ~af).sum()
    return {"classifier": name,
            "sensitivity": tp / (tp + fn), "specificity": tn / (tn + fp),
            "precision": tp / (tp + fp), "balanced_accuracy": (tp / (tp + fn) + tn / (tn + fp)) / 2,
            "auc": auc(prob, af) if prob is not None else np.nan,
            "false_alarms": fp, "missed_af": fn}
