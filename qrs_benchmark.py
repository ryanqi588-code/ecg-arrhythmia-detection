"""R-peak detection on the MIT-BIH Arrhythmia Database.

Compares three detectors on the first 5 minutes of each record:
  fixed     - threshold at half the global maximum
  adaptive  - sliding-window threshold with T-wave rejection (ours)
  xqrs      - wfdb's XQRS detector, as an established baseline

The adaptive detector's settings were tuned while looking at the 10 DEV
records, so its fair score is the one on the HELD_OUT records.
"""

import time

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.ndimage import maximum_filter1d
import wfdb
from wfdb import processing

from ecg import load_record, bandpass, detect_peaks_fixed, detect_peaks_adaptive, score

DEV = ["100", "101", "103", "105", "108", "200", "203", "207", "210", "215"]
NO_MLII = ["102", "104"]   # these two records have no MLII lead
HELD_OUT = [r for r in wfdb.get_record_list("mitdb") if r not in DEV + NO_MLII]
N5_SEC = 5 * 60   # evaluate the first 5 minutes of each record (all MIT-BIH records are 360 Hz)


def detect_peaks_xqrs(sig, fs):
    return processing.xqrs_detect(sig, fs=fs, verbose=False)


# ---------------------------------------------------------- record 100 demo

signal, FS, ref, excluded = load_record("100")
filtered = bandpass(signal, FS)
t = np.arange(len(signal)) / FS
peaks = detect_peaks_adaptive(signal, FS)

print(f"Detected {len(peaks)} beats, annotated {len(ref)}")

plt.figure(figsize=(12, 4))
plt.plot(t, filtered, label="filtered")
plt.plot(peaks / FS, filtered[peaks], "go", label="detected")
plt.plot(ref / FS, filtered[ref], "rv", label="annotated")
plt.xlim(0, 10)
plt.xlabel("Time (s)")
plt.ylabel("Amplitude (mV)")
plt.title("Record 100: filtered signal with detected vs annotated beats")
plt.legend()
plt.tight_layout()
plt.savefig("figures/record100_beats.png", dpi=120)
plt.show()

tp, fp, fn, sens, ppv = score(ref, peaks, tol=int(0.15 * FS), excluded=excluded)
print(f"TP={tp}  FP={fp}  FN={fn}")
print(f"Sensitivity={sens:.3f}  Precision={ppv:.3f}")

# Heart rate: drop physiologically impossible RR intervals (30-220 BPM)
# so a single false/missed detection doesn't create a fake spike.
rr = np.diff(peaks) / FS          # seconds between beats
hr = 60 / rr                      # beats per minute
valid = (hr >= 30) & (hr <= 220)
print(f"Average heart rate: {np.mean(hr[valid]):.1f} BPM "
      f"({np.sum(~valid)} implausible intervals dropped)")
print(f"Min: {np.min(hr[valid]):.1f}  Max: {np.max(hr[valid]):.1f}")

plt.figure(figsize=(12, 3))
plt.plot(peaks[1:][valid] / FS, hr[valid], "o-", markersize=2)
plt.xlabel("Time (s)")
plt.ylabel("Heart rate (BPM)")
plt.title("Record 100: heart rate over time")
plt.tight_layout()
plt.savefig("figures/record100_heart_rate.png", dpi=120)
plt.show()


# ------------------------------------------ fixed vs adaptive vs xqrs

detectors = {"fixed": detect_peaks_fixed,
             "adaptive": detect_peaks_adaptive,
             "xqrs": detect_peaks_xqrs}
rows = []

for split, records in [("dev", DEV), ("held-out", HELD_OUT)]:
    for rec_id in records:
        sig, fs, ref, excluded = load_record(rec_id, sampto=N5_SEC * 360)
        for name, detect in detectors.items():
            start = time.perf_counter()
            det = detect(sig, fs)
            elapsed = time.perf_counter() - start
            tp, fp, fn, sens, ppv = score(ref, det, int(0.15 * fs), excluded)
            rows.append((split, rec_id, name, tp, fp, fn, sens, ppv, elapsed))

df = pd.DataFrame(rows, columns=["split", "record", "detector", "TP", "FP", "FN",
                                 "sensitivity", "precision", "seconds"])
df.to_csv("results/qrs_benchmark.csv", index=False)

print("\nDev records (detector tuned on these):")
side_by_side = df[df.split == "dev"].pivot(index="record", columns="detector",
                                           values=["sensitivity", "precision"])
print(side_by_side.round(3).to_string())

for split in ["dev", "held-out"]:
    n = df[df.split == split].record.nunique()
    print(f"\nTotals, {split} ({n} records):")
    for name in detectors:
        g = df[(df.split == split) & (df.detector == name)]
        tot = g[["TP", "FP", "FN"]].sum()
        print(f"{name:>8}: sensitivity={tot['TP'] / (tot['TP'] + tot['FN']):.4f}  "
              f"precision={tot['TP'] / (tot['TP'] + tot['FP']):.4f}  "
              f"(FP={tot['FP']}, FN={tot['FN']}, {g.seconds.sum():.1f} s total)")
print("(detections inside ventricular flutter/fibrillation episodes not scored)")


# ----------------------------- record 215: why the fixed threshold fails

sig, fs, ref, _ = load_record("215", sampto=N5_SEC * 360)
filt = bandpass(sig, fs)
local_max = maximum_filter1d(np.abs(filt), size=int(2.0 * fs))
adaptive_thr = np.maximum(0.4 * local_max, 0.3 * np.median(local_max))
fixed_thr = 0.5 * np.max(filt)
print("\nRecord 215 fixed threshold:", round(fixed_thr, 2), "mV")

tt = np.arange(len(filt)) / fs
plt.figure(figsize=(12, 4))
plt.plot(tt, filt, label="filtered", linewidth=0.8)
plt.axhline(fixed_thr, color="r", linestyle="--", label="fixed threshold")
plt.plot(tt, adaptive_thr, color="g", label="adaptive threshold (on |signal|)")
plt.xlim(0, 20)
plt.xlabel("Time (s)")
plt.ylabel("Amplitude (mV)")
plt.title("Record 215: fixed vs adaptive threshold")
plt.legend()
plt.tight_layout()
plt.savefig("figures/record215_thresholds.png", dpi=120)
plt.show()
