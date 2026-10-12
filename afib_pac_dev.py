"""Development of RR-pattern features that separate AF from atrial premature beats.

The model retrained on CPSC + PTB-XL met its sensitivity target on
Chapman-Shaoxing (afib_chapman_test.py) only by lowering its cutoff, which
called 24% of PAC recordings AF. A PAC recording is regular apart from a
few early beats, each usually followed by a longer pause; AF is irregular
throughout, with no short-long pattern. af.rr_pattern_features measures
that. Its features are compared here, added to CLINICAL_FEATURES.

Development data, all already used as tests: 10 s crops of CPSC 2018 and
CPSC-extra, PTB-XL (2867) and Chapman-Shaoxing (3813). Cross-validation is
grouped by recording (CPSC, Chapman) or patient (PTB-XL); the cutoff is
the strictest per-source 95% cutoff, as in the Chapman test.
"""

import os

import numpy as np
import pandas as pd

from af import (CLINICAL_FEATURES, CLINICAL_FEATURES_12, RR_PATTERN_FEATURES, LEADS_12, cpsc_selected, load_12lead_all,
                clinical_features, fit_with_source_cutoffs, model_inputs, auc)
from afib_clinical_dev import dev_recordings

DETECTORS = ["lead2", "multilead", "multilead12"]
I_II_V1 = [0, 1, 6]


def unit_features(sig12, fs, is_af, detector):
    """clinical_features from leads I, II, V1; "multilead12" also adds the 12-lead atrial features."""
    if detector == "multilead12":
        return clinical_features(sig12[:, I_II_V1], fs, is_af, "multilead", sig12)
    return clinical_features(sig12[:, I_II_V1], fs, is_af, detector)


def dev_units(detector="lead2"):
    """Clinical features (with the RR-pattern ones) of every development unit (cached)."""
    cache = f"data/pac_dev_{detector}.pkl"
    try:
        return pd.read_pickle(cache)
    except FileNotFoundError:
        pass
    rows = []
    cpsc_labels = {f"data/cpsc2018/{r}": lab for r, lab in cpsc_selected().items()}
    for i, (path, is_af) in enumerate(dev_recordings()):
        if i % 300 == 0:
            print(f"cpsc {i} ...", flush=True)
        sig, fs = load_12lead_all(path)
        n10 = 10 * fs
        for k in range(min(3, len(sig) // n10)):
            row = unit_features(sig[k * n10:(k + 1) * n10], fs, is_af, detector)
            if row:
                rows.append({**row, "source": "cpsc", "record": path,
                             "label": cpsc_labels.get(path, "A" if is_af else "other")})
    db = pd.read_csv("data/ptbxl/ptbxl_database.csv", index_col="filename_hr")
    for i, line in enumerate(open("data/ptbxl/SELECTED").read().splitlines()):
        if i % 500 == 0:
            print(f"ptbxl {i} ...", flush=True)
        path, label = line.split()
        row = unit_features(*load_12lead_all(os.path.join("data/ptbxl", path)), label == "A", detector)
        if row:
            rows.append({**row, "source": "ptbxl", "record": f"ptbxl-{int(db.patient_id[path])}", "label": label})
    for i, line in enumerate(open("data/chapman/SELECTED").read().splitlines()):
        if i % 500 == 0:
            print(f"chapman {i} ...", flush=True)
        path, label = line.split()
        rec_id = os.path.basename(path)
        row = unit_features(*load_12lead_all(os.path.join("data/chapman", rec_id)), label == "A", detector)
        if row:
            rows.append({**row, "source": "chapman", "record": f"chapman-{rec_id}", "label": label})
    df = pd.DataFrame(rows)
    df.to_pickle(cache)
    return df


def prepare(df):
    df = df.copy()
    df["log_f_wave"] = df.log_f_wave.fillna(df.log_f_wave.median())
    for col in ["f_ratio_II", "f_ratio_V1"]:
        if col in df:
            df[col] = df[col].fillna(df[col].median())
    return df


def cross_validate(d, features, folds):
    """Out-of-fold calls with the strictest per-source 95% cutoff."""
    prob, pred = np.zeros(len(d)), np.zeros(len(d), dtype=bool)
    for fold in folds:
        held = d.record.isin(fold).values
        model, cutoffs = fit_with_source_cutoffs(d[~held], features)
        prob[held] = model.predict_proba(model_inputs(d[held], features))[:, 1]
        pred[held] = prob[held] >= min(cutoffs.values())
    return prob, pred


if __name__ == "__main__":
    rows = []
    for detector in DETECTORS:
        d = prepare(dev_units(detector))
        is_af, pac = d.af.values, (d.label == "PAC").values
        both = is_af | pac
        print(f"\n[{detector}] {len(d)} units, {is_af.sum()} AF, {pac.sum()} PAC: "
              f"{d.groupby('source').size().to_dict()}; median sinus CV {d.cv[(d.label == 'N').values].median():.3f}")
        if detector == "multilead12":
            alone = pd.DataFrame({f: {"AUC, AF vs PAC": auc(d[f].values[both], is_af[both]),
                                      "median AF": d[f][is_af].median(), "median PAC": d[f][pac].median(),
                                      "median sinus": d[f][(d.label == "N").values].median()}
                                  for f in ["cv", "pnn50", "entropy_norm", "p_match_regular", "log_f_wave",
                                            *RR_PATTERN_FEATURES, "p_match_mean", "f_ratio_II", "f_ratio_V1"]}).T
            print("Each feature alone (AUC < 0.5: lower in AF):")
            print(alone.round(3).to_string())
        folds = np.array_split(np.random.default_rng(2).permutation(d.record.unique()), 5)
        candidates = [("clinical", CLINICAL_FEATURES)]
        if detector == "multilead12":
            p12 = [f"p_match_{lead}" for lead in LEADS_12]
            candidates += [("+ p_match_mean", CLINICAL_FEATURES + ["p_match_mean"]),
                           ("+ f_ratio_V1", CLINICAL_FEATURES + ["f_ratio_V1"]),
                           ("+ p_match_mean + f_ratio_V1", CLINICAL_FEATURES + ["p_match_mean", "f_ratio_V1"]),
                           ("CLINICAL_FEATURES_12 (+ p_match_mean, f_ratio_II, f_ratio_V1)", CLINICAL_FEATURES_12),
                           ("+ 12 p_match", CLINICAL_FEATURES + p12),
                           ("+ 12 p_match + f_ratios", CLINICAL_FEATURES + p12 + ["f_ratio_II", "f_ratio_V1"]),
                           ("CLINICAL_FEATURES_12 + RR pattern", CLINICAL_FEATURES_12 + RR_PATTERN_FEATURES)]
        else:
            candidates += [(f"+ {f}", CLINICAL_FEATURES + [f]) for f in RR_PATTERN_FEATURES]
            candidates += [("+ all RR-pattern features", CLINICAL_FEATURES + RR_PATTERN_FEATURES)]
        for name, features in candidates:
            prob, pred = cross_validate(d, features, folds)
            r = {"detector": detector, "model": name, "auc": auc(prob, is_af),
                 "auc_AF_vs_PAC": auc(prob[both], is_af[both])}
            for src in ["cpsc", "ptbxl", "chapman"]:
                r[f"sens_{src}"] = pred[(d.source == src).values & is_af].mean()
            r["fa_PAC"] = pred[pac].mean()
            r["fa_N"] = pred[(d.label == "N").values].mean()
            r["fa_other"] = pred[(d.label == "other").values].mean()
            rows.append(r)
    results = pd.DataFrame(rows)
    results.to_csv("results/afib_pac_dev.csv", index=False)
    print("\n5-fold CV, strictest per-source cutoff:")
    print(results.round(3).to_string(index=False))
