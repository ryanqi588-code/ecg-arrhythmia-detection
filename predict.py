"""Score one 12-lead ECG with the adopted AF model: AF, not AF, needs review, or unscored, with each feature's contribution.

    python predict.py path/to/record            # WFDB record (.hea/.dat; leads matched by name)
    python predict.py ecg.csv --fs 500          # CSV, one column per lead (header with lead names,
                                                #   or no header and the standard order I..V6), in mV
    python predict.py ecg.csv --fs 500 --units uV   # a CSV in microvolts (or V)
    python predict.py path/to/record --json
    python predict.py path/to/record --resample # score a recording sampled outside 400-500 Hz

The model is read from results/final_af_model.csv (written by
export_final_model.py), so no training data is needed. Features come from
af.clinical_features, as in the tests: beats found on leads I, II and V1,
rhythm features after removing ventricular ectopics, and P-wave and
fibrillatory-wave features from all 12 leads. Flat stretches at the start
and end (the zero padding of CODE-format ECGs) are stripped first, as the
test scripts do, and then only the first 10 s are used; the model was
developed and tested on 7-10 s recordings sampled at 400-500 Hz.

Each feature's contribution is its coefficient times its standardized
input, coef_per_sd * (input - mean) / sd, in log-odds. They add up, with
the intercept (the log-odds of a recording at the training average), to
the score z, and the probability of AF is 1 / (1 + e^-z). Positive
contributions push towards AF. The probability is not a calibrated risk.

"Needs review" means the P waves couldn't be measured (fewer than 3
regular, non-ectopic beats): the model can't tell AF from non-AF there,
so it makes no call (test 17); the probability and contributions are
still shown. "Unscored" means the recording can't be scored, and the
reason says why:
  - sampling rate outside 400-500 Hz, unless --resample (resample=True):
    then 250-2000 Hz is resampled to 500 Hz first. On 1500 SPH recordings
    taken to 250-2000 Hz and back, 99.7-99.9% of calls were unchanged
    (check_resampling.py);
  - amplitudes that can't be millivolts: the median lead's amplitude
    range under 0.05 or over 100 (real ECGs: 0.28-36 mV), as from a signal
    in volts or microvolts. Signals must be in mV: WFDB records are
    converted from the units in their header; for a CSV, give --units;
  - shorter than 7 s once flat edges are stripped;
  - missing samples;
  - a lead flat or disconnected: its amplitude range (0.5th to 99.5th
    percentile) under 0.05 mV;
  - a lead off for part of the recording: constant for more than 1 s;
  - a lead clipped: flattened at its extreme value for 10 ms or more on
    at least 3 beats at the same level, at least 20% of the way out to
    the lead's tallest beats, with a steep approach (>= 0.1 mV in the 6 ms either side; a
    smooth wave top flattened by coarse quantization isn't flagged);
  - too few beats to measure the rhythm (fewer than 7 detected).
The checks run on the signal as recorded, at its own rate, before any
resampling (which would blur the exact flatness they look for). They were
set on real recordings (2,000 from seven sources: none flagged except
leads that really were flat, off or noise-only; with every lead clipped
at 70% of its peak height, 99-100% caught per source, with or without an
artifact spike) and on the golden ECGs clipped artificially.
The model is a research tool, not a medical device.
"""

import argparse
import json
import os
import sys

import numpy as np
import pandas as pd

from af import LEADS_12, clinical_features, p_unmeasured as unmeasured

MODEL = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results", "final_af_model.csv")
I_II_V1 = [0, 1, 6]
SECONDS = 10
MIN_SECONDS = 7              # shortest standard length in the test sets (CODE-15%'s 7.3 s recordings)
FS_RANGE = (400, 500)        # sampling rates the model was developed and tested at
FS_RESAMPLE = 500            # with resample=True, other rates are resampled to this
RESAMPLE_FROM = (250, 2000)  # ... but only from this range (check_resampling.py: calls 99.7-99.9% unchanged)
FLAT_MV = 0.05               # amplitude range below this: lead flat or disconnected
SCALE_MV = (0.05, 100)       # median lead range outside this isn't millivolts (real ECGs: 0.28-36 mV)
TO_MV = {"mv": 1.0, "uv": 1e-3, "µv": 1e-3, "μv": 1e-3, "v": 1e3, "nv": 1e-6}
DROPOUT_SECONDS = 1.0        # constant for longer than this: lead off for part of the recording
CLIP_MS = 10                 # clipping: plateau at the extreme value at least this long,
CLIP_EDGE_MS = 6             #   rising/falling at least CLIP_EDGE_MV within CLIP_EDGE_MS either side,
CLIP_EDGE_MV = 0.1
CLIP_MIN_BEATS = 3           #   on at least this many beats,
CLIP_MIN_HEIGHT = 0.2        #   at least this fraction of the lead's tallest beats from its median,
CLIP_MAX_BEYOND = 0.005      #   and with at most this share of samples beyond it (an artifact spike).
CLIP_Q_REF = 0.005           # Coarser quantization than this (mV) lengthens the minimum plateau by sqrt(step / this):
                             #   a natural peak stays within one step for a time that grows with the step's square root


def load_model(path=MODEL):
    t = pd.read_csv(path).set_index("feature")
    return t.drop(["intercept", "cutoff"]), float(t.loc["intercept", "coef_per_sd"]), float(t.loc["cutoff", "value"])


def to_mv(units):
    """Factor converting `units` (mV, uV, V, ...) to millivolts."""
    k = str(units).strip().lower()
    if k not in TO_MV:
        sys.exit(f"unknown units {units!r}; expected one of mV, uV, V")
    return TO_MV[k]


def read_ecg(path, fs=None, units="mV"):
    """(n x 12 signal in mV in the standard lead order, sampling rate).

    WFDB records are converted from the units in their header; a CSV is
    taken to be in `units`.
    """
    if path.lower().endswith((".csv", ".txt")):
        if fs is None:
            sys.exit("--fs is needed for a CSV file")
        d = pd.read_csv(path)
        names = [str(c).strip().upper() for c in d.columns]
        if set(LEADS_12) <= set(names):
            d.columns = names
            return d[LEADS_12].to_numpy(float) * to_mv(units), fs
        d = pd.read_csv(path, header=None)
        if d.shape[1] != 12:
            sys.exit(f"expected 12 columns (I, II, III, aVR, aVL, aVF, V1-V6), found {d.shape[1]}")
        return d.to_numpy(float) * to_mv(units), fs
    import wfdb
    rec = wfdb.rdrecord(os.path.splitext(path)[0] if path.endswith((".hea", ".dat")) else path)
    names = [n.strip().upper() for n in rec.sig_name]
    missing = [n for n in LEADS_12 if n not in names]
    if missing:
        sys.exit(f"all 12 leads are needed; missing {missing} (found {rec.sig_name})")
    idx = [names.index(n) for n in LEADS_12]
    scale = np.array([to_mv(rec.units[i]) for i in idx])
    return rec.p_signal[:, idx] * scale, rec.fs


def score_features(row, model=None):
    """Probability, call and contributions from a clinical_features row (dict)."""
    feats, intercept, cutoff = model or load_model()
    p_unmeasured = unmeasured(row)
    contributions = []
    for name, f in feats.iterrows():
        value = p_unmeasured if name == "p_unmeasured" else row.get(name, np.nan)
        filled = False
        fill_when = str(f.fill_when) if pd.notna(f.fill_when) else ""
        if pd.notna(f.fill) and (("p_unmeasured" in fill_when and p_unmeasured) or not np.isfinite(value)):
            value, filled = float(f.fill), True
        x = np.log(max(value, 1e-3)) if f.input.startswith("log(") else value
        z = (x - f["mean"]) / f.sd
        contributions.append({"feature": name, "value": float(value), "filled": filled,
                              "standardized": float(z), "contribution": float(f.coef_per_sd * z)})
    z = intercept + sum(c["contribution"] for c in contributions)
    prob = 1 / (1 + np.exp(-z))
    contributions.sort(key=lambda c: -abs(c["contribution"]))
    result = "needs review" if p_unmeasured else "AF" if prob >= cutoff else "not AF"
    return {"result": result, "probability": float(prob), "cutoff": cutoff,
            "score": float(z), "intercept": intercept, "contributions": contributions}


def strip_flat_edges(sig):
    """Drop flat stretches at the start and end: rows where all 12 leads equal the first (or last) row.

    CODE-format ECGs are zero-padded on both sides (afib_compare_dnn.strip_padding,
    which the test scripts apply); left in, the padding changes the filtered
    signal near the edges and the beats and P waves found there. This also
    catches padding with a constant non-zero value. A signal that is flat
    throughout comes back empty.
    """
    if len(sig) == 0:
        return sig
    differs_from_first = np.flatnonzero((sig != sig[0]).any(axis=1))
    if len(differs_from_first) == 0:
        return sig[:0]
    differs_from_last = np.flatnonzero((sig != sig[-1]).any(axis=1))
    start = differs_from_first[0]                   # rows before this equal row 0
    end = differs_from_last[-1] + 1                 # rows from here on equal the last row
    # A stretch is two or more rows; a lone edge sample is signal, not padding.
    return sig[start if start >= 2 else 0: end if len(sig) - end >= 2 else len(sig)]


def _runs(mask):
    """(start, end) indices of the runs of True in a 1-D boolean array."""
    d = np.diff(np.r_[0, mask.astype(np.int8), 0])
    return np.flatnonzero(d == 1), np.flatnonzero(d == -1)


def _clipped_beats(x, fs):
    """Number of clipped peaks in one lead: the most steep-edged plateaus at any one level.

    A clipping rail flattens every beat that reaches it at the same value,
    and nothing goes beyond it but the odd artifact. So plateaus (runs of
    identical samples, CLIP_MS or longer) are grouped by value, and a
    level counts only if at most CLIP_MAX_BEYOND of the samples lie beyond
    it: a lone artifact spike past the rail doesn't hide it (as it would
    if only the lead's maximum were checked), while a level many beats go
    past (coarsely quantized S waves bottoming out at the same value) is
    no rail. With coarse quantization a natural peak sits on one value
    longer, so the minimum plateau grows with the step's square root
    (CLIP_Q_REF): 10 ms at 5 µV steps or finer, 20 ms at 20 µV.
    A plateau counts if it is at least CLIP_MIN_HEIGHT of the way out to
    the lead's tallest beats (the larger of the 99.5th and 0.5th
    percentiles' distance from the median, so a lone spike doesn't set the
    scale and one-sided clipping counts however deep the other side goes):
    a flat baseline isn't clipping. And the signal must come towards the
    baseline by CLIP_EDGE_MV within CLIP_EDGE_MS on both sides: a smooth
    wave top flattened by coarse quantization (MIMIC-IV, 0.02 mV steps)
    rises to it slowly.
    """
    k = max(1, round(CLIP_EDGE_MS * fs / 1000))
    med = np.median(x)
    lo, hi = np.percentile(x, [0.5, 99.5])
    tallest = max(hi - med, med - lo)
    if tallest <= 0:
        return 0
    steps = np.diff(np.unique(x))
    step = steps.min() if len(steps) else 0.0         # the lead's quantization step
    min_ms = CLIP_MS * max(1.0, np.sqrt(step / CLIP_Q_REF))
    counts = {}
    for a, e in zip(*_runs(np.diff(x) == 0)):      # samples a..e equal (e inclusive)
        if (e - a + 1) * 1000 / fs < min_ms or a < k or e + k >= len(x):
            continue
        v = x[a]
        side = np.sign(v - med)
        if abs(v - med) < CLIP_MIN_HEIGHT * tallest or np.mean(side * (x - v) > 0) > CLIP_MAX_BEYOND:
            continue
        if side * (v - x[a - k]) >= CLIP_EDGE_MV and side * (v - x[e + k]) >= CLIP_EDGE_MV:
            counts[v] = counts.get(v, 0) + 1
    return max(counts.values(), default=0)


def scale_problem(sig):
    """A message if the signal's amplitudes can't be millivolts (as from volts or microvolts), else None."""
    lo, hi = np.percentile(sig, [0.5, 99.5], axis=0)
    r = float(np.median(hi - lo))
    if r < SCALE_MV[0]:
        return (f"amplitudes far too small for millivolts (median lead range {r:.3g}): no signal, or "
                f"in volts (convert to mV; predict.py --units V)")
    if r > SCALE_MV[1]:
        return (f"amplitudes far too large for millivolts (median lead range {r:.3g}): "
                f"in microvolts? (convert to mV; predict.py --units uV)")
    return None


def lead_problems(sig, fs):
    """Problems with individual leads of an n x 12 signal (finite values), as messages; empty if none."""
    lo, hi = np.percentile(sig, [0.5, 99.5], axis=0)
    flat, off, clipped = [], [], []
    for j, x in enumerate(sig.T):
        if hi[j] - lo[j] < FLAT_MV:
            flat.append(LEADS_12[j])
            continue
        start, end = _runs(np.diff(x) == 0)
        if len(start) and (end - start).max() + 1 > DROPOUT_SECONDS * fs:
            off.append(LEADS_12[j])
        elif _clipped_beats(x, fs) >= CLIP_MIN_BEATS:
            clipped.append(LEADS_12[j])
    names = lambda leads: ("lead " if len(leads) == 1 else "leads ") + ", ".join(n.replace("AV", "aV") for n in leads)
    out = []
    if flat:
        out.append(f"{names(flat)} flat or disconnected (amplitude under {FLAT_MV} mV)")
    if off:
        out.append(f"{names(off)} flat for over {DROPOUT_SECONDS:g} s (lead off for part of the recording)")
    if clipped:
        out.append(f"{names(clipped)} clipped (peaks cut flat)")
    return out


def resample_to(sig, fs, target=FS_RESAMPLE):
    """Polyphase resampling (anti-aliased), as afib_compare_dnn.to_ribeiro does."""
    from fractions import Fraction
    from scipy.signal import resample_poly
    r = Fraction(target) / Fraction(fs).limit_denominator(1000)
    return resample_poly(sig, r.numerator, r.denominator, axis=0), target


def predict(sig, fs, model=None, checks=True, resample=False):
    """Score an n x 12 signal in millivolts. Returns a dict with "result" and, unless unscored, the score.

    checks=False runs the model exactly as it was tested and as the
    published results were made: the first 10 s as given, no padding
    stripped and no input checks beyond missing samples and too few beats.
    The test scripts use it, so they reproduce their saved results; use
    the default for scoring new recordings.

    resample=True scores a recording sampled outside 400-500 Hz (but
    within RESAMPLE_FROM) by resampling it to 500 Hz, after stripping
    padding and after the input checks, which run on the signal as
    recorded; the result then has "resampled_from". Without it such
    recordings are unscored.
    """
    sig = np.asarray(sig, float)
    if sig.ndim != 2 or sig.shape[1] != 12:
        raise ValueError("expected an n x 12 signal")
    resampled_from, to_resample = None, False
    if checks:
        # Strip padding before taking the first 10 s, so a padded recording keeps all its signal.
        sig = strip_flat_edges(sig)
        if not FS_RANGE[0] <= fs <= FS_RANGE[1]:
            if not resample:
                return {"result": "unscored", "reason": f"sampling rate {fs:g} Hz; the model was tested at "
                        f"{FS_RANGE[0]}-{FS_RANGE[1]} Hz (resample first, or predict.py --resample)"}
            if not RESAMPLE_FROM[0] <= fs <= RESAMPLE_FROM[1]:
                return {"result": "unscored", "reason": f"sampling rate {fs:g} Hz; resampling is only "
                        f"supported from {RESAMPLE_FROM[0]}-{RESAMPLE_FROM[1]} Hz"}
            to_resample = True
        if len(sig) == 0:
            return {"result": "unscored", "reason": "flat signal"}
        if len(sig) < MIN_SECONDS * fs:
            return {"result": "unscored", "reason": f"recording too short: {len(sig) / fs:.1f} s of signal "
                    f"(after stripping flat edges); at least {MIN_SECONDS} s needed"}
    sig = sig[: int(round(SECONDS * fs))]
    if not np.isfinite(sig).all():
        return {"result": "unscored", "reason": "missing samples"}
    if checks:
        scale = scale_problem(sig)
        if scale:
            return {"result": "unscored", "reason": scale}
        problems = lead_problems(sig, fs)
        if problems:
            return {"result": "unscored", "reason": "; ".join(problems)}
        # Resample only now: the checks above look for exact flatness (dead leads, clipped
        # plateaus), which resampling's interpolation would blur, so they run on the signal
        # as recorded, at its own rate.
        if to_resample:
            resampled_from = fs
            sig, fs = resample_to(sig, fs)
    row = clinical_features(sig[:, I_II_V1], fs, False, "multilead", sig)
    if row is None:
        return {"result": "unscored", "reason": "too few beats to measure the rhythm"}
    out = score_features(row, model)
    if resampled_from is not None:
        out["resampled_from"] = resampled_from
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("record", help="WFDB record path (without extension) or a CSV file")
    ap.add_argument("--fs", type=float, help="sampling rate in Hz (CSV only)")
    ap.add_argument("--units", default="mV", help="amplitude units of a CSV: mV (default), uV or V; "
                    "WFDB records use their header's")
    ap.add_argument("--resample", action="store_true",
                    help="resample a recording sampled outside 400-500 Hz to 500 Hz instead of leaving it unscored")
    ap.add_argument("--json", action="store_true", help="print JSON")
    a = ap.parse_args()
    sig, fs = read_ecg(a.record, a.fs, a.units)
    out = predict(sig, fs, resample=a.resample)
    if a.json:
        print(json.dumps(out, indent=1))
        return
    if out["result"] == "unscored":
        print(f"unscored: {out['reason']}")
        return
    note = "; P waves can't be measured, so no call" if out["result"] == "needs review" else ""
    if "resampled_from" in out:
        note += f"; resampled from {out['resampled_from']:g} Hz to {FS_RESAMPLE} Hz"
    print(f"{out['result']}  (probability {out['probability']:.3f}; AF if >= {out['cutoff']:.4f}{note})")
    print(f"\n{'feature':<16}{'value':>10}{'standardized':>14}{'contribution':>14}")
    for c in out["contributions"]:
        print(f"{c['feature']:<16}{c['value']:>10.4g}{c['standardized']:>14.2f}{c['contribution']:>+14.3f}"
              f"{'  (filled)' if c['filled'] else ''}")
    print(f"{'intercept':<40}{out['intercept']:>+14.3f}")
    print(f"{'score (log-odds of AF)':<40}{out['score']:>+14.3f}")


if __name__ == "__main__":
    main()
