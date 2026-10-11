"""Test of the clinical-device cutoff on CPSC 2018 "extra" recordings.

After the CPSC 2018 test, the fix was to choose models and cutoffs by
recording *device* rather than length. For clinical 12-lead recordings that
means: keep the AFDB-trained model (it ranked CPSC best, AUC 0.952) but set
its cutoff on clinical data, the 777 already-used CPSC recordings, to reach
95% sensitivity there; it reached only 92.3% with the Holter cutoff.

This tests that on the CPSC 2018 "extra" set (PhysioNet/CinC Challenge
2020): clinical 12-lead recordings, 500 Hz, lead I, none used before and
none shared with the CPSC set. Here AF only occurs alongside other
diagnoses, so:
  AF     = AF (164889003) among the diagnoses
  non-AF = neither AF nor atrial flutter (164890007) among them
Recordings of at least 15 s, chosen from headers only; one window each;
fewer than 20 usable RR intervals -> skipped and counted. The plan was
committed before the test was run, and it is run once. There are only ~51
AF recordings, so sensitivity gets a 95% Wilson interval.

Models (5 features, ventricular ectopics removed), with and without the
P-wave override:
  Holter cutoff   - AFDB model, cutoff from AFDB (the previous best)
  clinical cutoff - same model, cutoff for 95% sensitivity on CPSC 2018
"""

import os

import numpy as np
import pandas as pd

from af import (RECORDS, FEATURES_5, load_af_record, load_cpsc_record, cpsc_selected, clean_windows,
                fit_with_cutoff, cutoff_for_sensitivity, model_inputs, p_wave_override, summarise)
from ecg import detect_peaks_adaptive

AF_CODE, AFL_CODE, PAC_CODE = "164889003", "164890007", "284470004"
EXTRA_DIR = "data/cpsc2018_extra"


def wilson(k, n, z=1.96):
    p = k / n
    centre = (p + z * z / (2 * n)) / (1 + z * z / n)
    half = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return centre - half, centre + half


def one_window(sig, fs, is_af):
    w = clean_windows(detect_peaks_adaptive, sig, fs, np.full(len(sig), is_af), len(sig) / fs)
    return w[0] if w else None


os.makedirs("results", exist_ok=True)

# ---------------------------------------------- AFDB model, both cutoffs

rows = []
for rec_id in RECORDS:
    print(f"train {rec_id} ...", flush=True)
    sig, fs, is_af, _ = load_af_record(rec_id, 0, 2)
    for w in clean_windows(detect_peaks_adaptive, sig, fs, is_af, 60):
        rows.append({**w, "record": rec_id})
model, holter_cutoff = fit_with_cutoff(pd.DataFrame(rows), FEATURES_5)

calib = []
for rec_id, label in sorted(cpsc_selected().items()):
    sig, fs = load_cpsc_record(rec_id)
    w = one_window(sig, fs, label == "A")
    if w:
        calib.append(w)
calib = pd.DataFrame(calib)
calib_prob = model.predict_proba(model_inputs(calib, FEATURES_5))[:, 1]
clinical_cutoff = cutoff_for_sensitivity(calib_prob, calib.af.values)
print(f"Holter cutoff {holter_cutoff:.3f}; clinical cutoff {clinical_cutoff:.3f} "
      f"(calibrated on {len(calib)} CPSC 2018 recordings)")

# ----------------------------------------------------------------- test

rows, skipped = [], {}
for rec_id in open(os.path.join(EXTRA_DIR, "SELECTED")).read().split():
    lines = open(os.path.join(EXTRA_DIR, rec_id + ".hea")).read().splitlines()
    dx = next(l for l in lines if l.startswith("# Dx")).split(":")[1].strip().split(",")
    label = "A" if AF_CODE in dx else "PAC" if PAC_CODE in dx else "other"
    sig, fs = load_cpsc_record(rec_id, EXTRA_DIR)
    w = one_window(sig, fs, label == "A")
    if w is None:
        skipped[label] = skipped.get(label, 0) + 1
        continue
    rows.append({**w, "record": rec_id, "label": label, "dx": " ".join(dx)})
test = pd.DataFrame(rows)
is_af = (test.label == "A").values
prob = model.predict_proba(model_inputs(test, FEATURES_5))[:, 1]

print(f"\nCPSC 2018 extra: {len(test)} recordings scored ({is_af.sum()} AF, "
      f"{(test.label == 'PAC').sum()} with PAC, {(test.label == 'other').sum()} other non-AF); "
      f"skipped for too few beats: {skipped}")
out = []
for name, cutoff in [("Holter cutoff", holter_cutoff), ("clinical cutoff", clinical_cutoff)]:
    for override in (False, True):
        pred = prob >= cutoff
        if override:
            pred = p_wave_override(pred, test)
        s = summarise(f"{name}{' + P-wave override' if override else ''}", is_af, pred,
                      None if override else prob)
        lo, hi = wilson(int((pred & is_af).sum()), int(is_af.sum()))
        s.update({"sens_ci_low": lo, "sens_ci_high": hi,
                  "false_alarm_rate_PAC": pred[(test.label == "PAC").values].mean(),
                  "false_alarm_rate_other": pred[(test.label == "other").values].mean()})
        out.append(s)
results = pd.DataFrame(out)
results.to_csv("results/afib_cpscx_test.csv", index=False)
test.assign(prob=prob).to_csv("results/afib_cpscx_test_recordings.csv", index=False)
print(results.set_index("classifier")[["sensitivity", "sens_ci_low", "sens_ci_high", "specificity",
                                       "false_alarm_rate_PAC", "false_alarm_rate_other",
                                       "false_alarms", "missed_af", "auc"]].round(3).to_string())
