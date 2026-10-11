"""Final test of handheld-aware training on the CPSC 2018 data (PhysioNet/CinC Challenge 2020).

931 short clinical 12-lead recordings (500 Hz, >= 15 s) with a single
expert diagnosis: AF, PAC (atrial premature contractions) or normal sinus
rhythm. Lead I only, one window per recording. None of it was used in
development; the recordings were chosen from their headers alone, and this
plan was committed before the test was run. Run once.

Recordings with fewer than 20 usable RR intervals are skipped and counted.
Reported: sensitivity on AF, false-alarm rates on PAC and on normal.

Models (all on windows with ventricular ectopics removed), each with and
without the P-wave override (af.p_wave_override):
  previous - trained on AFDB hours 0-2 only, 5 features, one cutoff
             (af.fit_with_cutoff): the model tested on Challenge 2017
  new      - trained on AFDB hours 0-2 + all Challenge 2017 recordings,
             entropy normalised for beat count (FEATURES_5N), with the
             cutoff for short recordings (af.fit_with_source_cutoffs)
"""

import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve

from af import (RECORDS, FEATURES_5, FEATURES_5N, load_af_record, load_cpsc_record, cpsc_selected,
                cinc_windows, clean_windows, fit_with_cutoff, fit_with_source_cutoffs,
                model_inputs, p_wave_override, auc, summarise)
from ecg import detect_peaks_adaptive

os.makedirs("results", exist_ok=True)
os.makedirs("figures", exist_ok=True)

# ------------------------------------------------------------- training

rows = []
for rec_id in RECORDS:
    print(f"train {rec_id} ...", flush=True)
    sig, fs, is_af, _ = load_af_record(rec_id, 0, 2)
    for w in clean_windows(detect_peaks_adaptive, sig, fs, is_af, 60):
        rows.append({**w, "record": rec_id, "source": "holter"})
afdb = pd.DataFrame(rows)
cinc = cinc_windows().assign(record=lambda d: "c" + d.record, source="short")

m_prev, c_prev = fit_with_cutoff(afdb, FEATURES_5)
m_new, c_new = fit_with_source_cutoffs(pd.concat([afdb, cinc], ignore_index=True), FEATURES_5N)
print(f"previous cutoff {c_prev:.3f}; new cutoffs {({k: round(v, 3) for k, v in c_new.items()})}")

# ----------------------------------------------------------------- test

labels = cpsc_selected()
rows, skipped = [], {}
for i, (rec_id, label) in enumerate(sorted(labels.items())):
    if i % 200 == 0:
        print(f"test {i} / {len(labels)} ...", flush=True)
    sig, fs = load_cpsc_record(rec_id)
    w = clean_windows(detect_peaks_adaptive, sig, fs, np.full(len(sig), label == "A"), len(sig) / fs)
    if not w:
        skipped[label] = skipped.get(label, 0) + 1
        continue
    rows.append({**w[0], "record": rec_id, "label": label, "seconds": len(sig) / fs})
test = pd.DataFrame(rows)
af = (test.label == "A").values

p_prev = m_prev.predict_proba(model_inputs(test, FEATURES_5))[:, 1]
p_new = m_new.predict_proba(model_inputs(test, FEATURES_5N))[:, 1]
preds = {
    "previous (AFDB only, 5 features)": (p_prev >= c_prev, p_prev),
    "previous + P-wave override": (p_wave_override(p_prev >= c_prev, test), None),
    "new (AFDB + Challenge 2017, entropy_norm, short-recording cutoff)": (p_new >= c_new["short"], p_new),
    "new + P-wave override": (p_wave_override(p_new >= c_new["short"], test), None),
}

print(f"\nCPSC 2018: {len(test)} recordings scored ({af.sum()} AF, {(test.label == 'PAC').sum()} PAC, "
      f"{(test.label == 'N').sum()} normal); skipped for too few beats: {skipped}")
out = []
for name, (pred, prob) in preds.items():
    s = summarise(name, af, pred, prob)
    s["false_alarm_rate_PAC"] = pred[(test.label == "PAC").values].mean()
    s["false_alarm_rate_N"] = pred[(test.label == "N").values].mean()
    out.append(s)
results = pd.DataFrame(out)
results.to_csv("results/afib_cpsc_test.csv", index=False)
test.assign(prob_previous=p_prev, prob_new=p_new).to_csv("results/afib_cpsc_test_recordings.csv", index=False)
print(results.set_index("classifier")[["sensitivity", "false_alarm_rate_PAC", "false_alarm_rate_N",
                                       "auc", "false_alarms", "missed_af"]].round(3).to_string())

plt.figure(figsize=(5.5, 5))
for (name, (pred, prob)), color in zip(list(preds.items())[::2], ["tab:red", "tab:blue"]):
    fpr, tpr_, _ = roc_curve(af, prob)
    plt.plot(fpr, tpr_, color=color, label=f"{name.split(' (')[0]} (AUC {auc(prob, af):.3f})")
for (name, (pred, _)), color in zip(preds.items(), ["tab:red", "tab:red", "tab:blue", "tab:blue"]):
    plt.plot(pred[~af].mean(), pred[af].mean(), "s" if "override" in name else "o", color=color)
plt.plot([0, 1], [0, 1], color="0.7", linestyle=":", linewidth=0.8)
plt.xlabel("False alarm rate (PAC + normal)")
plt.ylabel("Sensitivity")
plt.title("CPSC 2018, lead I (circles: cutoff; squares: + override)")
plt.legend(loc="lower right", fontsize=8)
plt.tight_layout()
plt.savefig("figures/afib_cpsc_test_roc.png", dpi=120)
plt.show()
