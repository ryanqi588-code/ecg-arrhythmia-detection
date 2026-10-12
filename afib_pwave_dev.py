"""Development of a P-wave measure that holds up in short AF recordings.

On PTB-XL (afib_ptbxl_test.py) the clinical model's missed AF was clearly
irregular but scored high on p_match_regular. Two suspected causes:
  - with only 3-4 regular beats, each beat is compared with a median that
    is (nearly) itself, so the match is high by construction;
  - at fast rates the previous beat's T wave sits in the pre-QRS stretch,
    and after two similar RR intervals the T waves line up like a P wave.
ecg.p_wave_consistency addresses both (QRS-T subtraction, leave-one-out
template). Its variants are compared here.

Development data, all already used as tests: 10 s crops of CPSC 2018 and
CPSC-extra (as afib_clinical_dev.py), and the 2867 PTB-XL recordings.
Cross-validation is grouped by recording (CPSC) or patient (PTB-XL).
"""

import os

import numpy as np
import pandas as pd

from af import (FEATURES_5N, cpsc_selected, load_12lead, clinical_features,
                regular_beats, fit_with_cutoff, fit_with_source_cutoffs, model_inputs, auc)
from afib_clinical_dev import dev_recordings
from ecg import detect_peaks_adaptive, flag_ectopic, p_wave_consistency

CACHE = "data/pwave_dev.pkl"
LEADS = {"I": 0, "II": 1, "V1": 2}

# name -> (lead, beats, keyword arguments of p_wave_consistency)
VARIANTS = {
    "pw_plain_reg": ("II", "regular", dict(remove_qrst=False, leave_one_out=False)),
    "pw_loo_reg": ("II", "regular", dict(remove_qrst=False)),
    "pw_qrst_reg": ("II", "regular", dict(leave_one_out=False)),
    "pw_both_reg": ("II", "regular", dict()),
    "pw_loo_all": ("II", "all", dict(remove_qrst=False)),
    "pw_both_all": ("II", "all", dict()),
    "pw_both_all_g20": ("II", "all", dict(guard=0.20)),
    "pw_both_all_g30": ("II", "all", dict(guard=0.30)),
    "pw_both_all_I": ("I", "all", dict()),
    "pw_both_all_V1": ("V1", "all", dict()),
}


def unit_features(leads3, fs, is_af):
    """clinical_features plus every P-wave variant, or None if too few beats."""
    row = clinical_features(leads3, fs, is_af)
    if row is None:
        return None
    lead2 = leads3[:, 1]
    peaks = detect_peaks_adaptive(lead2, fs)
    not_v = ~flag_ectopic(lead2, fs, peaks)
    regular = regular_beats(peaks, fs) & not_v
    for name, (lead, beats, kw) in VARIANTS.items():
        row[name] = p_wave_consistency(leads3[:, LEADS[lead]], fs, peaks,
                                       regular if beats == "regular" else not_v, **kw)
    return row


def dev_units():
    try:
        return pd.read_pickle(CACHE)
    except FileNotFoundError:
        pass
    rows = []
    cpsc_labels = {f"data/cpsc2018/{r}": lab for r, lab in cpsc_selected().items()}
    for i, (path, is_af) in enumerate(dev_recordings()):
        if i % 300 == 0:
            print(f"cpsc {i} ...", flush=True)
        sig, fs = load_12lead(path)
        n10 = 10 * fs
        for k in range(min(3, len(sig) // n10)):
            row = unit_features(sig[k * n10:(k + 1) * n10], fs, is_af)
            if row:
                label = cpsc_labels.get(path, "A" if is_af else "other")
                rows.append({**row, "source": "cpsc", "record": path, "label": label})
    db = pd.read_csv("data/ptbxl/ptbxl_database.csv", index_col="filename_hr")
    for i, line in enumerate(open("data/ptbxl/SELECTED").read().splitlines()):
        if i % 500 == 0:
            print(f"ptbxl {i} ...", flush=True)
        path, label = line.split()
        sig, fs = load_12lead(os.path.join("data/ptbxl", path))
        row = unit_features(sig, fs, label == "A")
        if row:
            rows.append({**row, "source": "ptbxl", "record": f"ptbxl-{int(db.patient_id[path])}",
                         "label": label})
    df = pd.DataFrame(rows)
    df.to_pickle(CACHE)
    return df


def prepare(df):
    df = df.copy()
    df["log_f_wave"] = df.log_f_wave.fillna(df.log_f_wave.median())
    for name in VARIANTS:
        df[name] = df[name].fillna(0.0)          # too few beats to compare: AF-like
    return df


def cross_validate(d, features, folds, strictest=False):
    """Out-of-fold probabilities and calls. With `strictest`, the cutoff is the
    lowest of the per-source 95% cutoffs, so each source gets at least 95%."""
    prob, pred = np.zeros(len(d)), np.zeros(len(d), dtype=bool)
    for fold in folds:
        held = d.record.isin(fold).values
        if strictest:
            model, cutoffs = fit_with_source_cutoffs(d[~held], features)
            cutoff = min(cutoffs.values())
        else:
            model, cutoff = fit_with_cutoff(d[~held], features)
        prob[held] = model.predict_proba(model_inputs(d[held], features))[:, 1]
        pred[held] = prob[held] >= cutoff
    return prob, pred


if __name__ == "__main__":
    d = prepare(dev_units())
    is_af = d.af.values
    print(f"{len(d)} units ({(d.source == 'cpsc').sum()} CPSC crops, {(d.source == 'ptbxl').sum()} PTB-XL), "
          f"{is_af.sum()} AF")

    # Each measure alone: AF vs everything, and AF vs PAC (where P waves matter).
    pac = (d.label == "PAC").values
    alone = pd.DataFrame({name: {"AUC, AF vs rest": 1 - auc(d[name].values, is_af),
                                 "AUC, AF vs PAC": 1 - auc(d[name].values[is_af | pac], is_af[is_af | pac]),
                                 "median AF": d[name][is_af].median(),
                                 "median PAC": d[name][pac].median()}
                          for name in ["p_match_regular", *VARIANTS]}).T
    print("\nEach P-wave measure alone (low = AF):")
    print(alone.round(3).to_string())

    folds = np.array_split(np.random.default_rng(2).permutation(d.record.unique()), 5)
    rows = []
    runs = [(name, False) for name in ["p_match_regular", *VARIANTS]]
    runs += [("p_match_regular", True), ("pw_loo_all", True)]
    for name, strictest in runs:
        features = FEATURES_5N + [name, "log_f_wave"]
        prob, pred = cross_validate(d, features, folds, strictest)
        r = {"P-wave measure": name, "cutoff": "strictest source" if strictest else "shared",
             "auc": auc(prob, is_af)}
        for src in ["cpsc", "ptbxl"]:
            s = (d.source == src).values
            r[f"sens_{src}"] = pred[s & is_af].mean()
            r[f"fa_{src}"] = pred[s & ~is_af].mean()
        r["fa_PAC"] = pred[pac].mean()
        r["fa_N"] = pred[(d.label == "N").values].mean()
        rows.append(r)
    results = pd.DataFrame(rows)
    results.to_csv("results/afib_pwave_dev.csv", index=False)
    print("\nModel RR (5N) + P-wave measure + f-waves, 5-fold CV grouped by recording/patient:")
    print(results.round(3).to_string(index=False))
