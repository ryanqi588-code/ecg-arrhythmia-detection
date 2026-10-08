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
