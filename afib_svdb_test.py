"""Final test of atrial-ectopy handling on the MIT-BIH Supraventricular Arrhythmia Database.

78 patients chosen for frequent supraventricular (atrial) ectopy, full 30
minutes each, 128 Hz. None of this data was used in development, and the
test plan below was fixed before any of it was looked at. Run once.

Models, all trained on AFDB hours 0-2 with cutoffs set for 95% sensitivity
(af.fit_with_cutoff):
  1. threshold rule (CV, nRMSSD)
  2. logistic regression, 5 RR features
  3. logistic regression, 5 RR features, ventricular ectopic beats removed
  4. logistic regression, 5 RR features + p_match, ventricular ectopic beats removed

Also reported, as a description only (no model uses it): how well the beat-level
ectopic detectors match the database's beat labels (S = supraventricular
ectopic, V = ventricular ectopic). Records with no rhythm label count as
non-AF, as in every other test here.
"""

import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve

from af import (RECORDS, SVDB_RECORDS, FEATURES_5, FEATURES_6, TARGET_SENS, load_af_record,
                load_svdb_record, window_features, fit_thresholds, fit_with_cutoff,
                model_inputs, auc, summarise)
from ecg import (BEAT_SYMS, bandpass, detect_peaks_adaptive, flag_ectopic,
                 flag_atrial_ectopic, pr_segments)

# (name, features, uses windows with ventricular ectopic beats removed, plot colour)
CLASSIFIERS = [("logistic regression (5 features)", FEATURES_5, False, "tab:orange"),
               ("logistic regression (5 features, ectopic beats removed)", FEATURES_5, True, "tab:red"),
               ("logistic regression (5 features + p_match, ectopic beats removed)", FEATURES_6, True, "tab:purple")]


def windows_for(rec_id, sig, fs, is_af):
    peaks = detect_peaks_adaptive(sig, fs)
    plain = window_features(peaks, fs, len(sig), is_af)
    clean = window_features(peaks, fs, len(sig), is_af, flag_ectopic(sig, fs, peaks),
                            pr_segments(bandpass(sig, fs), peaks, fs))
    for row in plain + clean:
        row["record"] = rec_id
    return plain, clean, peaks


def beat_level(rec_id, sig, fs, peaks, ann_samples, ann_symbols):
    """Match detections to labelled beats; count how often each detector flags each beat type."""
    v_flag = flag_ectopic(sig, fs, peaks)
    a_flag = flag_atrial_ectopic(sig, fs, peaks)
    keep = np.isin(ann_symbols, list(BEAT_SYMS))
    samples, symbols = ann_samples[keep], ann_symbols[keep]
    rows = []
    for i, p in enumerate(peaks):
        j = np.argmin(np.abs(samples - p))
        if abs(samples[j] - p) <= 0.15 * fs:
            cls = "S" if symbols[j] in "AaJS" else "V" if symbols[j] in "VE" else "N"
            rows.append((rec_id, cls, v_flag[i], a_flag[i]))
    return rows


os.makedirs("results", exist_ok=True)
os.makedirs("figures", exist_ok=True)

# ------------------------------------------------------------- training

train_plain, train_clean = [], []
for rec_id in RECORDS:
    print(f"train {rec_id} ...", flush=True)
    sig, fs, is_af, _ = load_af_record(rec_id, 0, 2)
    plain, clean, _ = windows_for(rec_id, sig, fs, is_af)
    train_plain += plain
    train_clean += clean
train_plain, train_clean = pd.DataFrame(train_plain), pd.DataFrame(train_clean)

t_cv, t_nr = fit_thresholds(train_plain, min_sens=TARGET_SENS)
models = {name: fit_with_cutoff(train_clean if clean else train_plain, features)
          for name, features, clean, _ in CLASSIFIERS}

# ----------------------------------------------------------------- test

test_plain, test_clean, beats = [], [], []
for rec_id in SVDB_RECORDS:
    print(f"test {rec_id} ...", flush=True)
    sig, fs, is_af, ann_samples, ann_symbols = load_svdb_record(rec_id)
    plain, clean, peaks = windows_for(rec_id, sig, fs, is_af)
    test_plain += plain
    test_clean += clean
    beats += beat_level(rec_id, sig, fs, peaks, ann_samples, ann_symbols)
test_plain, test_clean = pd.DataFrame(test_plain), pd.DataFrame(test_clean)
assert len(test_plain) == len(test_clean) and (test_plain.af.values == test_clean.af.values).all()
af = test_plain.af.values

print(f"\nSVDB: {len(test_plain)} windows, {af.sum()} AF, from "
      f"{test_plain.record.nunique()} records ({test_plain.record[af].nunique()} with AF)")

rows, probs = [], {}
rule_pred = (test_plain.cv.values > t_cv) & (test_plain.nrmssd.values > t_nr)
rows.append(summarise("threshold rule (CV, nRMSSD)", af, rule_pred))
for name, features, clean, _ in CLASSIFIERS:
    data = test_clean if clean else test_plain
    prob = models[name][0].predict_proba(model_inputs(data, features))[:, 1]
    probs[name] = prob
    test_plain[f"prob_{len(features)}{'_clean' if clean else ''}"] = prob
    test_plain[f"pred_{len(features)}{'_clean' if clean else ''}"] = prob >= models[name][1]
    rows.append(summarise(name, af, prob >= models[name][1], prob))

results = pd.DataFrame(rows)
results.to_csv("results/afib_svdb_test.csv", index=False)
test_plain["p_match"] = test_clean.p_match.values
test_plain.to_csv("results/afib_svdb_test_windows.csv", index=False)
print(results.set_index("classifier").round(3).to_string())

# ------------------------------------------------- beat-level detectors

b = pd.DataFrame(beats, columns=["record", "cls", "v_flag", "a_flag"])
b.to_csv("results/svdb_beat_flags.csv", index=False)
print("\nBeat-level ectopic detectors vs SVDB beat labels:")
for flag, label in [("v_flag", "ventricular detector (flag_ectopic)"),
                    ("a_flag", "atrial detector (flag_atrial_ectopic)")]:
    caught = b.groupby("cls")[flag].mean()
    flagged = b[b[flag]]
    print(f"  {label}: catches {caught.get('V', np.nan):.1%} of V beats and {caught.get('S', np.nan):.1%} "
          f"of S beats; flags {caught['N']:.1%} of normal beats; {(flagged.cls == 'S').mean():.1%} of its "
          f"flags are S beats, {(flagged.cls == 'V').mean():.1%} V beats")
print(f"  beats matched to labels: {len(b)} ({(b.cls == 'S').sum()} S, {(b.cls == 'V').sum()} V)")

# ------------------------------------------- where the false alarms are

s_frac = b.groupby("record").cls.apply(lambda c: (c == "S").mean())
fa = test_plain.groupby("record").apply(
    lambda g: pd.Series({"windows": len(g), "af_windows": g.af.sum(),
                         "fa_5": (~g.af & g.pred_5).sum(),
                         "fa_5_clean": (~g.af & g.pred_5_clean).sum(),
                         "fa_6_clean": (~g.af & g.pred_6_clean).sum()}),
    include_groups=False).astype(int).assign(s_beat_frac=s_frac.round(3))
fa.to_csv("results/svdb_records.csv")
print("\nRecords with the most false alarms (5-feature model):")
print(fa.sort_values("fa_5", ascending=False).head(8).to_string())

# ----------------------------------------------------------------- plot

if af.any():
    plt.figure(figsize=(5.5, 5))
    for name, _, _, color in CLASSIFIERS:
        fpr, tpr_, _ = roc_curve(af, probs[name])
        plt.plot(fpr, tpr_, color=color,
                 label=f"{name.replace('logistic regression', 'logistic')} (AUC {auc(probs[name], af):.3f})")
    plt.plot(1 - rows[0]["specificity"], rows[0]["sensitivity"], "ks", label="threshold rule")
    plt.plot([0, 1], [0, 1], color="0.7", linestyle=":", linewidth=0.8)
    plt.xlabel("False alarm rate (1 - specificity)")
    plt.ylabel("Sensitivity")
    plt.title("AF detection, SVDB (78 new patients, atrial ectopy)")
    plt.legend(loc="lower right", fontsize=7)
    plt.tight_layout()
    plt.savefig("figures/afib_svdb_test_roc.png", dpi=120)
    plt.show()
