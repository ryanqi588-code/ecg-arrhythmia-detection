"""Shared AF helpers: loading AFDB records, RR features, classifiers, metrics."""

import functools
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
# PhysioNet/CinC Challenge 2017 training set, downloaded and unzipped by hand
# (see README); used only as the final test of the P-wave override.
CINC_DIR = "data/cinc2017"
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
FEATURES_6R = FEATURES_5 + ["p_match_regular"]
# Same 5 features with entropy normalised for the number of beats, for
# models that mix 60 s Holter windows with short handheld recordings.
FEATURES_5N = ["cv", "nrmssd", "entropy_norm", "tpr", "pnn50"]


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


def cinc_labels():
    """PhysioNet/CinC Challenge 2017 labels: record -> N (normal), A (AF), O (other rhythm), ~ (noisy)."""
    ref = pd.read_csv(os.path.join(CINC_DIR, "REFERENCE-v3.csv"), header=None, names=["record", "label"])
    return dict(zip(ref.record, ref.label))


def load_cinc_record(rec_id):
    """One single-lead Challenge 2017 recording (300 Hz, 9-61 s). Returns (signal, fs)."""
    rec = wfdb.rdrecord(os.path.join(CINC_DIR, "training2017", rec_id))
    return rec.p_signal[:, 0], rec.fs


def clean_windows(peaks_fn, sig, fs, is_af, window_sec):
    """Windows with ventricular ectopics removed, plus P-wave features (as the override needs)."""
    from ecg import bandpass, flag_ectopic, pr_segments   # ecg imports nothing from here
    peaks = peaks_fn(sig, fs)
    return window_features(peaks, fs, len(sig), is_af, flag_ectopic(sig, fs, peaks),
                           pr_segments(bandpass(sig, fs), peaks, fs), regular_beats(peaks, fs),
                           window_sec=window_sec)


def cinc_windows(cache="data/cinc2017/windows.pkl"):
    """One clean window per Challenge 2017 recording (noisy ones excluded), cached.

    Columns as window_features, plus record, label (N / A / O) and seconds.
    """
    if os.path.exists(cache):
        return pd.read_pickle(cache)
    from ecg import detect_peaks_adaptive
    rows = []
    for rec_id, label in sorted(cinc_labels().items()):
        if label == "~":
            continue
        sig, fs = load_cinc_record(rec_id)
        w = clean_windows(detect_peaks_adaptive, sig, fs, np.full(len(sig), label == "A"), len(sig) / fs)
        if w:
            rows.append({**w[0], "record": rec_id, "label": label, "seconds": len(sig) / fs})
    df = pd.DataFrame(rows)
    df.to_pickle(cache)
    return df


CPSC_DIR = "data/cpsc2018"
CPSC_CODES = {"164889003": "A", "284470004": "PAC", "426783006": "N"}   # SNOMED CT


def cpsc_selected():
    """The registered CPSC 2018 test recordings: record -> label (A / PAC / N).

    Single-diagnosis AF, PAC or normal sinus rhythm recordings at least 15 s
    long, chosen from the headers before any signal was looked at.
    """
    out = {}
    for rec_id in open(os.path.join(CPSC_DIR, "SELECTED")).read().split():
        lines = open(os.path.join(CPSC_DIR, rec_id + ".hea")).read().splitlines()
        dx = next(l for l in lines if l.startswith("# Dx")).split(":")[1].strip()
        out[rec_id] = CPSC_CODES[dx]
    return out


def load_cpsc_record(rec_id, directory=CPSC_DIR):
    """Lead I of a CPSC 2018 12-lead recording (500 Hz). Returns (signal, fs)."""
    rec = wfdb.rdrecord(os.path.join(directory, rec_id), channels=[0])
    assert rec.sig_name[0] == "I"
    return rec.p_signal[:, 0], rec.fs


CLINICAL_FEATURES = FEATURES_5N + ["p_match_regular", "log_f_wave"]
RR_PATTERN_FEATURES = ["rr_mode_frac", "rr_robust_cv", "cv_no_early", "rr_lag1"]
# With all 12 leads (atrial_features): P-wave consistency averaged over the
# 12 leads, and the share of beat-free activity at AF's 4-9 Hz in II and V1.
# Chosen on CPSC + PTB-XL + Chapman (afib_pac_dev.py): at the same
# sensitivity target, false alarms on PAC 20.1% -> 11.6%.
CLINICAL_FEATURES_12 = CLINICAL_FEATURES + ["p_match_mean", "f_ratio_II", "f_ratio_V1"]
CLINICAL_MIN_RR = 6        # 10 s clinical recordings give ~8-15 RR intervals
# Fewest regular beats for p_match_regular in a 10 s recording (60 s windows
# use 5). With 5, many 10 s normal recordings had too few and were scored as
# P-wave-free; chosen on the CPSC development crops.
CLINICAL_MIN_REGULAR = 3


def load_12lead(path):
    """Leads I, II and V1 of a Challenge 2020 12-lead recording. Returns (n x 3 array, fs)."""
    rec = wfdb.rdrecord(path, channels=[0, 1, 6])
    assert rec.sig_name == ["I", "II", "V1"], rec.sig_name
    return rec.p_signal, rec.fs


LEADS_12 = ["I", "II", "III", "AVR", "AVL", "AVF", "V1", "V2", "V3", "V4", "V5", "V6"]


def load_12lead_all(path):
    """All 12 leads of a clinical recording, in the standard order. Returns (n x 12 array, fs)."""
    rec = wfdb.rdrecord(path)
    assert [n.upper() for n in rec.sig_name] == LEADS_12, rec.sig_name
    return rec.p_signal, rec.fs


def atrial_features(sig12, fs, peaks, regular, not_v, min_regular=CLINICAL_MIN_REGULAR):
    """Atrial-activity features of a 12-lead recording, given its beats.

    p_match_<lead> - p_match over regular, non-ventricular beats in each
                     lead (as p_match_regular is for lead II); 0 with fewer
                     than `min_regular` such beats
    p_match_mean   - their mean over the 12 leads
    f_ratio_<lead> - ecg.f_wave_ratio in leads II and V1
    """
    from ecg import bandpass, f_wave_ratio, pr_segments
    use = np.asarray(regular) & np.asarray(not_v)
    out = {}
    for k, name in enumerate(LEADS_12):
        if use.sum() >= min_regular:
            segs = pr_segments(bandpass(sig12[:, k], fs), peaks, fs)[use]
            out[f"p_match_{name}"] = p_wave_features(segs)["p_match"]
        else:
            out[f"p_match_{name}"] = 0.0
    out["p_match_mean"] = np.mean([out[f"p_match_{n}"] for n in LEADS_12])
    for name in ["II", "V1"]:
        out[f"f_ratio_{name}"] = f_wave_ratio(sig12[:, LEADS_12.index(name)], fs, peaks)
    return out


def clinical_features(leads3, fs, is_af, detector="lead2", sig12=None):
    """Features of one short clinical recording (leads I, II, V1), or None if too few beats.

    R peaks are found on lead II. RR features (FEATURES_5N) after removing
    ventricular ectopics, with a minimum of CLINICAL_MIN_RR intervals;
    p_match_regular from lead II, where P waves are clearest; and the log of
    fibrillatory-wave activity in V1 (ecg.f_wave_activity).
    `detector`: "lead2" (detect_peaks_adaptive on lead II) or "multilead"
    (detect_peaks_multilead on all three leads). With `sig12` (all 12
    leads), atrial_features are added.
    """
    from ecg import (bandpass, detect_peaks_adaptive, detect_peaks_multilead, f_wave_activity, flag_ectopic,
                     pr_segments)
    lead2 = leads3[:, 1]
    peaks = (detect_peaks_multilead(leads3, fs) if detector == "multilead"
             else detect_peaks_adaptive(lead2, fs))
    if len(peaks) < CLINICAL_MIN_RR + 1:
        return None
    ectopic = flag_ectopic(lead2, fs, peaks)
    regular = regular_beats(peaks, fs)
    rows = window_features(peaks, fs, len(lead2), np.full(len(lead2), is_af), ectopic,
                           pr_segments(bandpass(lead2, fs), peaks, fs), regular,
                           window_sec=len(lead2) / fs, min_rr=CLINICAL_MIN_RR,
                           min_regular=CLINICAL_MIN_REGULAR)
    if not rows:
        return None
    f = f_wave_activity(leads3[:, 2], fs, peaks)
    row = {**rows[0], "log_f_wave": np.log(max(f, 1e-4)) if np.isfinite(f) else np.nan}
    if sig12 is not None:
        row.update(atrial_features(sig12, fs, peaks, regular, ~ectopic))
    return row


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


@functools.lru_cache(maxsize=None)
def random_entropy(n, bins=16, trials=400):
    """Mean shannon_entropy of n uniformly random RR intervals (fixed seed).

    A completely random series of n intervals can't fill 16 bins evenly when
    n is small, so its entropy is below 1 and depends on n. This is the
    baseline that entropy_norm divides by.
    """
    rng = np.random.default_rng(n)
    return float(np.mean([shannon_entropy(rng.uniform(0.4, 1.0, n), bins) for _ in range(trials)]))


def rr_pattern_features(rr, early=0.85, near=0.08):
    """RR features that tell premature beats from AF.

    A PAC recording is regular apart from a few early beats, each usually
    followed by a longer pause; AF is irregular throughout.
      rr_mode_frac - fraction of intervals within `near` of the median
      rr_robust_cv - median absolute deviation / median (x1.4826, so it
                     matches CV for normally spread intervals); a few
                     outliers barely move it
      cv_no_early  - CV after dropping each early interval (< `early` x
                     median) and the one after it; plain CV if fewer than 3
                     intervals remain
      rr_lag1      - correlation of successive intervals: strongly negative
                     for short-long pairs, near 0 for AF
    """
    rr = np.asarray(rr, dtype=float)
    med = np.median(rr)
    is_early = rr < early * med
    drop = is_early | np.r_[False, is_early[:-1]]
    kept = rr[~drop] if (~drop).sum() >= 3 else rr
    a, b = rr[:-1] - rr.mean(), rr[1:] - rr.mean()
    denom = np.sqrt((a ** 2).sum() * (b ** 2).sum())
    return {"rr_mode_frac": np.mean(np.abs(rr - med) < near * med),
            "rr_robust_cv": 1.4826 * np.median(np.abs(rr - med)) / med,
            "cv_no_early": kept.std() / kept.mean(),
            "rr_lag1": (a * b).sum() / denom if denom > 0 else 0.0}


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


def regular_beats(peaks, fs, early=0.9):
    """True for beats that aren't early: RR before them >= `early` x the local median RR.

    Atrial ectopics are early; the beats between them are ordinary sinus
    beats that share one P wave. In AF no beats share a P wave, early or not.
    """
    peaks = np.asarray(peaks)
    rr = np.diff(peaks) / fs
    regular = np.zeros(len(peaks), dtype=bool)
    for i in range(1, len(peaks)):
        regular[i] = rr[i - 1] >= early * np.median(rr[max(0, i - 8):i + 8])
    return regular


P_OVERRIDE = 0.95


def p_wave_override(pred, df, threshold=P_OVERRIDE):
    """Overrule AF calls in windows whose regular beats share a P wave.

    AF means no P waves. If at least `threshold` of a window's regular beats
    (not early, not ventricular ectopic) match the window's typical P wave,
    the window isn't called AF, however irregular its RR intervals. This is
    a rule rather than a model feature because the AFDB training data has
    almost no irregular-but-P-waves windows to learn it from.

    Chosen on development data: on AFDB (leave-one-record-out) it overrides
    0.5% of AF windows, keeping sensitivity at 96.1%, and cuts false alarms
    100 -> 71; on SVDB (a second look) 390 -> 277. Lower thresholds remove
    more false alarms but drop sensitivity below 95%.
    """
    return np.asarray(pred) & (df.p_match_regular.values < threshold)


def window_features(peaks, fs, n, is_af, ectopic=None, pr_segs=None, regular=None,
                    window_sec=WINDOW_SEC, min_rr=MIN_RR, min_regular=5):
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
    win = int(window_sec * fs)
    ectopic = np.zeros(len(peaks), dtype=bool) if ectopic is None else np.asarray(ectopic)
    for start in range(0, n - win + 1, win):
        in_win = (peaks >= start) & (peaks < start + win)
        p, e = peaks[in_win], ectopic[in_win]
        rr = np.diff(p) / fs
        plausible = (rr > 0.25) & (rr < 2.5)                    # 24-240 BPM
        filtered = plausible & ~e[:-1] & ~e[1:]                  # neither end ectopic
        rr = rr[filtered] if filtered.sum() >= min_rr else rr[plausible]
        if len(rr) < min_rr:
            continue
        mean = rr.mean()
        rows.append({
            "start_min": start / fs / 60,
            "af": is_af[start:start + win].mean() >= 0.5,
            "ectopic_frac": e.mean() if len(e) else 0.0,
            "cv": rr.std() / mean,
            "nrmssd": np.sqrt(np.mean(np.diff(rr) ** 2)) / mean,
            "entropy": shannon_entropy(rr),
            # Entropy relative to a random series of the same length: ~1 for
            # AF however many beats there are (entropy alone falls with fewer).
            "entropy_norm": shannon_entropy(rr) / random_entropy(len(rr)),
            "tpr": turning_points_ratio(rr),
            "pnn50": np.mean(np.abs(np.diff(rr)) > 0.05),
            **rr_pattern_features(rr),
        })
        if pr_segs is not None:
            rows[-1].update(p_wave_features(pr_segs[in_win]))
            if regular is not None:
                # Same, over regular beats only (not early, not ventricular ectopic).
                use = regular[in_win] & ~e
                if use.sum() >= min_regular:
                    pw = p_wave_features(pr_segs[in_win][use])
                    rows[-1]["p_match_regular"] = pw["p_match"]
                else:
                    rows[-1]["p_match_regular"] = 0.0      # too few regular beats: AF-like
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
    records = train.record.unique()
    # Leave one record out; with many short single-window records (Challenge
    # 2017) that would mean thousands of fits, so use 10 folds of records.
    folds = ([[r] for r in records] if len(records) <= 30
             else np.array_split(np.random.default_rng(0).permutation(records), 10))
    for fold in folds:
        held = train.record.isin(fold).values
        inner = make_logreg().fit(model_inputs(train[~held], features), train.af[~held])
        oof[held] = inner.predict_proba(model_inputs(train[held], features))[:, 1]
    model = make_logreg().fit(model_inputs(train, features), train.af)
    return model, cutoff_for_sensitivity(oof, train.af.values, train.record.values, target)


def fit_with_source_cutoffs(train, features, target=TARGET_SENS):
    """One model fitted on several kinds of recording, with a separate cutoff for each.

    `train` needs a `source` column (e.g. "holter", "short"). Out-of-fold
    probabilities come from 10 folds of records; each source's cutoff is the
    one giving `target` sensitivity on that source's windows, records weighted
    equally (as fit_with_cutoff).

    Chosen on development data: training on AFDB + Challenge 2017 with one
    shared cutoff let short recordings drag the cutoff down and cut AFDB
    specificity from 95% to 83-86%; per-source cutoffs restored it to ~91%
    while keeping handheld sensitivity at 95%.
    """
    oof = np.zeros(len(train))
    for fold in np.array_split(np.random.default_rng(0).permutation(train.record.unique()), 10):
        held = train.record.isin(fold).values
        inner = make_logreg().fit(model_inputs(train[~held], features), train.af[~held])
        oof[held] = inner.predict_proba(model_inputs(train[held], features))[:, 1]
    model = make_logreg().fit(model_inputs(train, features), train.af)
    cutoffs = {}
    for src in train.source.unique():
        s = (train.source == src).values
        cutoffs[src] = cutoff_for_sensitivity(oof[s], train.af.values[s], train.record.values[s], target)
    return model, cutoffs


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
