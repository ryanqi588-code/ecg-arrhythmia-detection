"""Shared ECG helpers: loading MIT-BIH records, filtering, R-peak detection, scoring."""

import numpy as np
from scipy.signal import butter, filtfilt, find_peaks
from scipy.ndimage import maximum_filter1d
import wfdb

BEAT_SYMS = set("NLRBAaJSVrFejnE/fQ?")   # annotation symbols that are heartbeats


def load_record(rec_id, sampto=None):
    """Load an MIT-BIH Arrhythmia record.

    Returns (MLII signal, fs, beat annotation sample indices, excluded spans).
    Excluded spans are ventricular flutter/fibrillation episodes, marked
    "[" ... "]" in the annotations. There are no discrete beats inside them,
    so (per ANSI/AAMI EC57) they are left out of beat-detection scoring.
    """
    rec = wfdb.rdrecord(rec_id, pn_dir="mitdb", sampto=sampto)
    ann = wfdb.rdann(rec_id, "atr", pn_dir="mitdb", sampto=sampto)
    sig = rec.p_signal[:, rec.sig_name.index("MLII")]
    ref = np.array([s for s, sym in zip(ann.sample, ann.symbol)
                    if sym in BEAT_SYMS and s < len(sig)])
    excluded, start = [], None
    for s, sym in zip(ann.sample, ann.symbol):
        if sym == "[":
            start = s
        elif sym == "]":
            excluded.append((0 if start is None else start, s))
            start = None
    if start is not None:                 # episode still running at sampto
        excluded.append((start, len(sig)))
    return sig, rec.fs, ref, excluded


def bandpass(sig, fs):
    """0.5-40 Hz band-pass: removes baseline wander and high-frequency noise."""
    b, a = butter(3, [0.5 / (fs / 2), 40 / (fs / 2)], btype="band")
    return filtfilt(b, a, sig)


def detect_peaks_fixed(sig, fs):
    """Peaks above half the global maximum, at least 0.25 s apart."""
    filt = bandpass(sig, fs)
    peaks, _ = find_peaks(filt, height=0.5 * np.max(filt), distance=int(0.25 * fs))
    return peaks


def detect_peaks_adaptive(sig, fs, window_sec=2.0, frac=0.4,
                          t_wave_sec=0.36, slope_frac=0.5):
    """Peaks above a fraction of the local maximum of |signal|.

    Using |signal| catches beats with inverted QRS complexes (e.g. PVCs).
    The floor stops the threshold collapsing onto noise during pauses.
    T-wave rejection (from Pan-Tompkins): a candidate within t_wave_sec of
    the previous beat whose steepest slope is under slope_frac of that
    beat's is a T wave, not a QRS, and is dropped.
    """
    filt = bandpass(sig, fs)
    mag = np.abs(filt)
    local_max = maximum_filter1d(mag, size=int(window_sec * fs))
    floor = 0.3 * np.median(local_max)
    thr = np.maximum(frac * local_max, floor)
    cand, _ = find_peaks(mag, height=thr, distance=int(0.25 * fs))

    slope = np.abs(np.gradient(filt))
    half = int(0.075 * fs)                # look +/-75 ms around each peak
    peaks, peak_slopes = [], []
    for c in cand:
        s = slope[max(0, c - half):c + half].max()
        if peaks and c - peaks[-1] < t_wave_sec * fs and s < slope_frac * peak_slopes[-1]:
            continue
        peaks.append(c)
        peak_slopes.append(s)
    return np.array(peaks, dtype=int)


def pr_segments(filt, peaks, fs, start=0.30, end=0.06):
    """The stretch from `start` to `end` seconds before each peak, where the P wave sits.

    One row per peak, mean removed; zeros for peaks too close to the start.
    """
    a, b = int(start * fs), int(end * fs)
    segs = np.zeros((len(peaks), a - b))
    for i, c in enumerate(peaks):
        if c - a >= 0:
            segs[i] = filt[c - a:c - b] - filt[c - a:c - b].mean()
    return segs


def _corr(x, y):
    return np.corrcoef(x, y)[0, 1] if x.std() > 0 and y.std() > 0 else 0.0


def flag_atrial_ectopic(sig, fs, peaks, early=0.85, p_corr=0.8):
    """Flag likely atrial (supraventricular) ectopic beats: early, with an unusual P wave.

    A beat is flagged if its RR is under `early` x the local median and its
    pre-QRS stretch correlates under `p_corr` with the median of its
    neighbours' (+/-15 beats, itself excluded).

    This works poorly on a single lead. On the full 30 minutes of 46 MIT-BIH
    records (excluding record 232, whose long pauses distort timing) it
    catches 55% of supraventricular ectopics but also flags 2.3% of normal
    beats, so only ~10% of flagged beats are truly atrial ectopics. The P-wave
    check adds little to timing alone (58% caught, 2.75% of normal beats).
    """
    peaks = np.asarray(peaks)
    flags = np.zeros(len(peaks), dtype=bool)
    if len(peaks) < 3:
        return flags
    segs = pr_segments(bandpass(sig, fs), peaks, fs)
    rr = np.diff(peaks) / fs
    for i in range(1, len(peaks)):
        med = np.median(rr[max(0, i - 8):i + 8])
        if rr[i - 1] >= early * med:
            continue
        neighbours = np.r_[max(0, i - 15):i, i + 1:min(len(peaks), i + 16)]
        flags[i] = _corr(segs[i], np.median(segs[neighbours], axis=0)) < p_corr
    return flags


def f_wave_activity(sig, fs, peaks, pre=0.25, post=0.55):
    """Atrial (fibrillatory) activity not locked to the beats, relative to beat size.

    Subtract the median beat (P wave, QRS and T, from `pre` s before to
    `post` s after each peak) from the band-passed signal, band-pass what's
    left to 4-9 Hz (the range of AF's fibrillatory waves) and return its RMS
    divided by the median beat's peak-to-peak size. Best in lead V1. Returns
    NaN with fewer than 3 complete beats.
    """
    filt = bandpass(sig, fs)
    a, b = int(pre * fs), int(post * fs)
    ok = [p for p in peaks if p - a >= 0 and p + b <= len(filt)]
    if len(ok) < 3:
        return np.nan
    template = np.median([filt[p - a:p + b] for p in ok], axis=0)
    resid = filt.copy()
    for p in ok:
        resid[p - a:p + b] -= template
    bb, aa = butter(3, [4 / (fs / 2), 9 / (fs / 2)], btype="band")
    f = filtfilt(bb, aa, resid)[ok[0] - a:ok[-1] + b]
    return np.sqrt(np.mean(f ** 2)) / np.ptp(template)


def p_wave_consistency(sig, fs, peaks, use=None, remove_qrst=True, leave_one_out=True,
                       qrst=(0.06, 0.5), guard=0.25, match=0.5):
    """Fraction of beats whose pre-QRS stretch matches the other beats' typical one.

    Unlike af.p_wave_features over regular beats, this
      - first subtracts the median QRS-T complex (`qrst` s before/after each
        peak) at every beat, so the previous beat's T wave, which sits in
        the pre-QRS stretch at fast rates, can't pass for a P wave; and
      - compares each beat with the median of the *other* beats, so with
        few beats a beat isn't matched against (nearly) itself.
    Each subtraction stops `guard` s before the next peak, so at slow rates
    the template's tail doesn't erase the next beat's P wave.
    `use` selects the beats to compare (e.g. not ventricular ectopic). NaN
    with fewer than 3.
    """
    peaks = np.asarray(peaks)
    use = np.ones(len(peaks), dtype=bool) if use is None else np.asarray(use)
    filt = bandpass(sig, fs)
    if remove_qrst:
        a, b = int(qrst[0] * fs), int(qrst[1] * fs)
        ok = [p for p in peaks if p - a >= 0 and p + b <= len(filt)]
        if len(ok) >= 3:
            template = np.median([filt[p - a:p + b] for p in ok], axis=0)
            filt = filt.copy()
            nxt = np.r_[peaks[1:], len(filt) + b + int(guard * fs)]
            for p, q in zip(peaks, nxt):
                lo, hi = max(0, p - a), min(len(filt), p + b, q - int(guard * fs))
                if hi <= lo:
                    continue
                filt[lo:hi] -= template[lo - (p - a):hi - (p - a)]
    segs = pr_segments(filt, peaks, fs)
    idx = np.flatnonzero(use & (peaks >= int(0.30 * fs)))
    if len(idx) < 3:
        return np.nan
    segs = segs[idx]
    hits = []
    for i in range(len(segs)):
        ref = np.median(np.delete(segs, i, axis=0) if leave_one_out else segs, axis=0)
        hits.append(_corr(segs[i], ref) > match)
    return float(np.mean(hits))


def flag_ectopic(sig, fs, peaks, shape_corr=0.9, early=0.8, pause=1.15):
    """Flag likely ectopic beats among detected peaks.

    A beat is flagged if either
      - its shape is unlike its neighbours': correlation with the median of
        the surrounding +/-15 beats (+/-80 ms around each peak) is below
        shape_corr. This catches ventricular ectopics, whose QRS is wide and
        differently shaped. AF beats keep their normal shape, so this cue
        doesn't remove AF irregularity.
      - its timing is premature-then-pause: the RR before it is under
        `early` x the local median RR and the RR after it is over `pause` x.

    On the first 5 minutes of 46 MIT-BIH records this catches 97.6% of
    ventricular ectopic beats and flags 3.5% of normal beats, but only 5% of
    supraventricular ectopics, which look normal and are often only slightly
    early. (Thresholds were chosen on that same data.)
    """
    peaks = np.asarray(peaks)
    flags = np.zeros(len(peaks), dtype=bool)
    if len(peaks) < 3:
        return flags
    filt = bandpass(sig, fs)
    h = int(0.08 * fs)
    segs = np.zeros((len(peaks), 2 * h))
    for i, c in enumerate(peaks):
        if h <= c <= len(filt) - h:
            segs[i] = filt[c - h:c + h]
    rr = np.diff(peaks) / fs
    for i in range(len(peaks)):
        template = np.median(segs[max(0, i - 15):i + 16], axis=0)
        if segs[i].std() > 0 and template.std() > 0:
            flags[i] |= np.corrcoef(segs[i], template)[0, 1] < shape_corr
        if 1 <= i <= len(rr) - 1:
            med = np.median(rr[max(0, i - 8):i + 8])
            flags[i] |= rr[i - 1] < early * med and rr[i] > pause * med
    return flags


def score(ref, det, tol, excluded=()):
    """Greedy one-to-one match of reference beats to detections within tol samples.

    Detections inside any (start, end) span in `excluded` are ignored.
    """
    ref = np.asarray(ref)
    det = np.asarray(det)
    for start, end in excluded:
        det = det[(det < start) | (det > end)]
    used = np.zeros(len(det), dtype=bool)
    tp = 0
    for r in ref:
        if len(det) == 0:
            break
        d = np.abs(det - r).astype(float)
        d[used] = np.inf
        j = np.argmin(d)
        if d[j] <= tol:
            tp += 1
            used[j] = True
    fn = len(ref) - tp
    fp = len(det) - tp
    sens = tp / (tp + fn) if len(ref) else float("nan")
    ppv = tp / (tp + fp) if len(det) else float("nan")
    return tp, fp, fn, sens, ppv
