"""Shared AF helpers: loading AFDB records, RR features, classifiers, metrics."""

import os
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd
import wfdb
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

# The 23 AFDB records with ECG signals (00735 and 03665 ship without them).
# Listed here rather than fetched so importing this module needs no network.
RECORDS = ["04015", "04043", "04048", "04126", "04746", "04908", "04936", "05091",
           "05121", "05261", "06426", "06453", "06995", "07162", "07859", "07879",
           "07910", "08215", "08219", "08378", "08405", "08434", "08455"]
# Long-Term AF Database: 84 different patients, used only as a final test set.
LTAF_RECORDS = ["00", "01", "03", "05", "06", "07", "08", "10", "100", "101", "102", "103",
                "104", "105", "11", "110", "111", "112", "113", "114", "115", "116", "117",
                "118", "119", "12", "120", "121", "122", "13", "15", "16", "17", "18", "19",
                "20", "200", "201", "202", "203", "204", "205", "206", "207", "208", "21",
                "22", "23", "24", "25", "26", "28", "30", "32", "33", "34", "35", "37", "38",
                "39", "42", "43", "44", "45", "47", "48", "49", "51", "53", "54", "55", "56",
                "58", "60", "62", "64", "65", "68", "69", "70", "71", "72", "74", "75"]
# MIT-BIH Supraventricular Arrhythmia Database: 78 patients with frequent
# supraventricular ectopy, used only as the final test of atrial-ectopy handling.
SVDB_RECORDS = [str(r) for r in [*range(800, 813), *range(820, 830), *range(840, 895)]]
SVDB_CACHE_DIR = "data/svdb"
FS = 250
WINDOW_SEC = 60
MIN_RR = 20                      # skip windows with fewer valid RR intervals (noise/dropout)
CACHE_DIR = "data/afdb"
LTAF_CACHE_DIR = "data/ltafdb"
TARGET_SENS = 0.95               # operating point for choosing cutoffs

FEATURES_5 = ["cv", "nrmssd", "entropy", "tpr", "pnn50"]
# p_match is the one P-wave feature used. p_snr is still computed for reference
# but left out: on development data it is 85% correlated with p_match, adds
# nothing (AUC 0.984 with both vs 0.985 with p_match alone), and its weight
# came out with the wrong sign.
FEATURES_6 = FEATURES_5 + ["p_match"]


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


def load_ltaf_record(rec_id, hours=2):
    """First `hours` of a Long-Term AF Database record, used as a new-patient test set.

    Returns (ECG channel 0, fs, AF mask). LTAFDB is sampled at 128 Hz.
    """
    os.makedirs(LTAF_CACHE_DIR, exist_ok=True)
    cache = os.path.join(LTAF_CACHE_DIR, f"{rec_id}_0-{hours}h.npy")
    fs = 128
    n = int(hours * 3600 * fs)
    if os.path.exists(cache):
        sig = np.load(cache)
    else:
        rec = wfdb.rdrecord(rec_id, pn_dir="ltafdb", channels=[0], sampto=n)
        sig, fs = rec.p_signal[:, 0], rec.fs
        np.save(cache, sig)
    ann = wfdb.rdann(rec_id, "atr", pn_dir="ltafdb", sampto=n)
    return sig, fs, af_mask(ann, 0, len(sig))


def load_svdb_record(rec_id):
    """A full 30-minute MIT-BIH Supraventricular Arrhythmia Database record.

    Returns (ECG1, fs, AF mask, beat annotation samples, beat symbols).
    Used only as the final test set for atrial-ectopy handling.
    """
    os.makedirs(SVDB_CACHE_DIR, exist_ok=True)
    cache = os.path.join(SVDB_CACHE_DIR, f"{rec_id}.npy")
    fs = 128
    if os.path.exists(cache):
        sig = np.load(cache)
    else:
        rec = wfdb.rdrecord(rec_id, pn_dir="svdb", channels=[0])
        sig, fs = rec.p_signal[:, 0], rec.fs
        np.save(cache, sig)
    ann = wfdb.rdann(rec_id, "atr", pn_dir="svdb")
    return sig, fs, af_mask(ann, 0, len(sig)), np.asarray(ann.sample), np.asarray(ann.symbol)


def prefetch_svdb(workers=4):
    with ThreadPoolExecutor(workers) as pool:
        for rec_id in pool.map(lambda r: (load_svdb_record(r), r)[1], SVDB_RECORDS):
            print(f"cached {rec_id}", flush=True)


def prefetch_ltaf(hours=2, workers=4):
    with ThreadPoolExecutor(workers) as pool:
        for rec_id in pool.map(lambda r: (load_ltaf_record(r, hours), r)[1], LTAF_RECORDS):
            print(f"cached {rec_id}", flush=True)


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
    # Keep only rhythm-change annotations ("+"); LTAFDB mixes beat labels in.
    keep = np.array(rhythm.symbol) == "+"
    samples = np.asarray(rhythm.sample)[keep]
    notes = np.asarray(rhythm.aux_note, dtype=object)[keep]
    mask = np.zeros(end - start, dtype=bool)
    bounds = list(samples) + [np.iinfo(np.int64).max]
    for s, e, label in zip(bounds[:-1], bounds[1:], notes):
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


def p_wave_features(segs):
    """P-wave presence in one window, from its beats' pre-QRS stretches (ecg.pr_segments).

    p_snr   - peak-to-peak size of the window's median pre-QRS stretch,
              relative to how much individual beats deviate from it
    p_match - fraction of beats whose stretch correlates > 0.5 with that median

    In sinus rhythm, even with ectopic beats, most beats share a P wave, so
    both are high. In AF there is no P wave; fibrillatory waves average out,
    so both are low.
    """
    template = np.median(segs, axis=0)
    resid = (segs - template).std()
    corr = np.array([np.corrcoef(s, template)[0, 1] if s.std() > 0 and template.std() > 0 else 0.0
                     for s in segs])
    return {"p_snr": np.ptp(template) / resid if resid > 0 else 0.0,
            "p_match": np.mean(corr > 0.5)}


def window_features(peaks, fs, n, is_af, ectopic=None, pr_segs=None):
    """One row per 60 s window: AF label and RR-irregularity features.

    With `pr_segs` (ecg.pr_segments for all peaks), the P-wave presence
    features p_snr and p_match are added, computed over every beat in the
    window.

    With `ectopic` (one flag per peak, from ecg.flag_ectopic), RR intervals
    that start or end at a flagged beat are dropped first. The intervals
    either side of a dropped beat are both normal, so joining them doesn't
    create a large artificial jump. If that leaves too few intervals, the
    window falls back to its unfiltered intervals rather than being
    skipped, so every window still gets a decision.
    """
    rows = []
    win = WINDOW_SEC * fs
    ectopic = np.zeros(len(peaks), dtype=bool) if ectopic is None else np.asarray(ectopic)
    for start in range(0, n - win + 1, win):
        in_win = (peaks >= start) & (peaks < start + win)
        p, e = peaks[in_win], ectopic[in_win]
        rr = np.diff(p) / fs
        plausible = (rr > 0.25) & (rr < 2.5)                    # 24-240 BPM
        filtered = plausible & ~e[:-1] & ~e[1:]                  # neither end ectopic
        rr = rr[filtered] if filtered.sum() >= MIN_RR else rr[plausible]
        if len(rr) < MIN_RR:
            continue
        mean = rr.mean()
        rows.append({
            "start_min": start / fs / 60,
            "af": is_af[start:start + win].mean() >= 0.5,
            "ectopic_frac": e.mean() if len(e) else 0.0,
            "cv": rr.std() / mean,
            "nrmssd": np.sqrt(np.mean(np.diff(rr) ** 2)) / mean,
            "entropy": shannon_entropy(rr),
            "tpr": turning_points_ratio(rr),
            "pnn50": np.mean(np.abs(np.diff(rr)) > 0.05),
        })
        if pr_segs is not None:
            rows[-1].update(p_wave_features(pr_segs[in_win]))
    return rows


# -------------------------------------------------------- classifiers

def record_weights(records):
    """Weight each window by 1 / (its record's window count), so records count equally."""
    records = pd.Series(np.asarray(records))
    return (1.0 / records.map(records.value_counts())).values


def fit_thresholds(df, min_sens=None):
    """Grid-search (cv, nrmssd) thresholds for the rule "AF if both exceed".

    With min_sens=None, maximise balanced accuracy. Otherwise maximise
    specificity among thresholds whose sensitivity is at least min_sens,
    with sensitivity averaged over records (each record's AF windows count
    equally, as in cutoff_for_sensitivity).
    """
    best = (-1, None, None)
    af, cv, nr = df.af.values, df.cv.values, df.nrmssd.values
    w_af = record_weights(df.record.values[af])
    for t_cv in np.arange(0.02, 0.30, 0.005):
        for t_nr in np.arange(0.02, 0.40, 0.005):
            pred = (cv > t_cv) & (nr > t_nr)
            sens = pred[af].mean() if min_sens is None else np.sum(w_af * pred[af]) / w_af.sum()
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
    """Feature matrix; CV, nRMSSD and p_snr are log-transformed (each spans a decade or more)."""
    X = df[features].copy()
    for col in ("cv", "nrmssd", "p_snr"):
        if col in X:
            X[col] = np.log(np.maximum(X[col], 1e-3))
    return X.values


def make_logreg():
    # Standardised inputs make coefficients comparable; balanced class
    # weights match the threshold rule's balanced-accuracy objective.
    return make_pipeline(StandardScaler(),
                         LogisticRegression(class_weight="balanced", max_iter=1000))


def cutoff_for_sensitivity(prob, af, records=None, target=TARGET_SENS):
    """Highest probability cutoff that still catches `target` of the AF windows.

    With `records`, each record's AF windows are weighted equally, so a few
    records with long AF episodes can't set the cutoff on their own.
    """
    p = prob[af]
    w = np.ones(len(p)) if records is None else record_weights(np.asarray(records)[af])
    order = np.argsort(p)
    cum = np.cumsum(w[order]) / w.sum()
    return p[order][np.searchsorted(cum, 1 - target)]


def fit_with_cutoff(train, features, target=TARGET_SENS):
    """Fit logistic regression on `train` and choose its cutoff for `target` sensitivity.

    The cutoff is set on out-of-fold probabilities (each training record
    scored by a model that didn't see it), not in-sample ones, which are
    overconfident, and with records weighted equally. Both fixes were chosen
    on development data after cutoffs set the old way (in-sample, per
    window) undershot their target on new data.
    """
    oof = np.zeros(len(train))
    for rec_id in train.record.unique():
        held = (train.record == rec_id).values
        inner = make_logreg().fit(model_inputs(train[~held], features), train.af[~held])
        oof[held] = inner.predict_proba(model_inputs(train[held], features))[:, 1]
    model = make_logreg().fit(model_inputs(train, features), train.af)
    return model, cutoff_for_sensitivity(oof, train.af.values, train.record.values, target)


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
