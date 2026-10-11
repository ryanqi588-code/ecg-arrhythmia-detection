"""Final test of the P-wave override on the PhysioNet/CinC Challenge 2017 data.

8528 single-lead recordings (300 Hz, 9-61 s, handheld AliveCor device),
each labelled by experts as Normal (N), AF (A), Other rhythm (O) or Noisy (~).
None of it was used in development; the plan below was committed before
the test was run, and it is run once.

- Noisy recordings are excluded.
- Each recording is scored as one window (they are too short for 60 s
  windows). Recordings with fewer than 20 usable RR intervals are excluded
  and counted.
- AF = label A. Non-AF = N and O; false alarms are also reported for each
  separately, since O (other rhythms, including ectopy) is the hard case.
- Models, all trained on AFDB hours 0-2 (60 s windows) with cutoffs set
  for 95% sensitivity (af.fit_with_cutoff):
    1. threshold rule (CV, nRMSSD)
    2. logistic regression, 5 RR features
    3. logistic regression, 5 RR features, ventricular ectopic beats removed
    4. model 3 + P-wave override (af.p_wave_override, threshold 0.95)
"""

import os

import numpy as np
import pandas as pd

from af import (RECORDS, FEATURES_5, TARGET_SENS, load_af_record, load_cinc_record, cinc_labels,
                window_features, regular_beats, fit_thresholds, fit_with_cutoff,
                model_inputs, p_wave_override, auc, summarise)
from ecg import bandpass, detect_peaks_adaptive, flag_ectopic, pr_segments


def features(sig, fs, is_af, window_sec):
    """(plain windows, windows with ventricular ectopics removed + P-wave features)."""
    peaks = detect_peaks_adaptive(sig, fs)
    plain = window_features(peaks, fs, len(sig), is_af, window_sec=window_sec)
    clean = window_features(peaks, fs, len(sig), is_af, flag_ectopic(sig, fs, peaks),
                            pr_segments(bandpass(sig, fs), peaks, fs), regular_beats(peaks, fs),
                            window_sec=window_sec)
    return plain, clean


os.makedirs("results", exist_ok=True)

# ------------------------------------------------------------- training

train_plain, train_clean = [], []
for rec_id in RECORDS:
    print(f"train {rec_id} ...", flush=True)
    sig, fs, is_af, _ = load_af_record(rec_id, 0, 2)
    plain, clean = features(sig, fs, is_af, 60)
    for row in plain + clean:
        row["record"] = rec_id
    train_plain += plain
    train_clean += clean
train_plain, train_clean = pd.DataFrame(train_plain), pd.DataFrame(train_clean)

t_cv, t_nr = fit_thresholds(train_plain, min_sens=TARGET_SENS)
m_plain, c_plain = fit_with_cutoff(train_plain, FEATURES_5)
m_clean, c_clean = fit_with_cutoff(train_clean, FEATURES_5)

# ----------------------------------------------------------------- test

labels = cinc_labels()
rows, skipped = [], {}
for i, (rec_id, label) in enumerate(sorted(labels.items())):
    if label == "~":
        continue
    if i % 500 == 0:
        print(f"test {i} / {len(labels)} ...", flush=True)
    sig, fs = load_cinc_record(rec_id)
    is_af = np.full(len(sig), label == "A")
    plain, clean = features(sig, fs, is_af, len(sig) / fs)
    if not plain or not clean:
        skipped[label] = skipped.get(label, 0) + 1
        continue
    rows.append({**{f"plain_{k}": v for k, v in plain[0].items()},
                 **clean[0], "record": rec_id, "label": label})
test = pd.DataFrame(rows)
plain_cols = test[[c for c in test.columns if c.startswith("plain_")]]
test_plain = plain_cols.rename(columns=lambda c: c.removeprefix("plain_"))

af = (test.label == "A").values
p_plain = m_plain.predict_proba(model_inputs(test_plain, FEATURES_5))[:, 1]
p_clean = m_clean.predict_proba(model_inputs(test, FEATURES_5))[:, 1]
preds = {
    "threshold rule (CV, nRMSSD)": ((test_plain.cv.values > t_cv) & (test_plain.nrmssd.values > t_nr), None),
    "logistic regression (5 features)": (p_plain >= c_plain, p_plain),
    "logistic regression (5 features, ectopic beats removed)": (p_clean >= c_clean, p_clean),
    "... + P-wave override": (p_wave_override(p_clean >= c_clean, test), None),
}

print(f"\nChallenge 2017: {len(test)} recordings scored "
      f"({(test.label == 'N').sum()} N, {af.sum()} A, {(test.label == 'O').sum()} O); "
      f"skipped for too few beats: {skipped}; noisy excluded: {sum(v == '~' for v in labels.values())}")

out = []
for name, (pred, prob) in preds.items():
    s = summarise(name, af, pred, prob)
    s["false_alarm_rate_N"] = pred[(test.label == "N").values].mean()
    s["false_alarm_rate_O"] = pred[(test.label == "O").values].mean()
    out.append(s)
    test[f"pred_{len(out)}"] = pred
results = pd.DataFrame(out)
results.to_csv("results/afib_cinc_test.csv", index=False)
test.drop(columns=plain_cols.columns).to_csv("results/afib_cinc_test_recordings.csv", index=False)
print(results.set_index("classifier")[["sensitivity", "specificity", "precision", "auc",
                                       "false_alarm_rate_N", "false_alarm_rate_O",
                                       "false_alarms", "missed_af"]].round(3).to_string())
