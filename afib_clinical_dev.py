"""Development of a clinical 12-lead AF model on 10 s crops of the (already used) CPSC data.

The CPSC 2018 and CPSC 2018 extra recordings have been used as tests, so
here they are development data. Each recording is cut into up to three
non-overlapping 10 s crops (the length of the Georgia test recordings) and
described by af.clinical_features: RR features from lead II, P-wave
consistency over regular beats in lead II, and fibrillatory-wave activity
in V1. Models are compared by 5-fold cross-validation grouped by recording,
each with its cutoff set on the training folds for 95% sensitivity.

AF: CPSC label A, or AF among CPSC-extra diagnoses. Non-AF: everything else
selected (PAC and normal from CPSC; no AF or flutter in CPSC-extra).
"""

import numpy as np
import pandas as pd

from af import (RECORDS, FEATURES_5, FEATURES_5N, CLINICAL_FEATURES, cpsc_selected, load_12lead,
                clinical_features, load_af_record, clean_windows, fit_with_cutoff, cutoff_for_sensitivity,
                model_inputs, p_wave_override, auc)
from ecg import detect_peaks_adaptive

AF_CODE = "164889003"
CACHE = "data/clinical_dev_crops.pkl"


def dev_recordings():
    out = [(f"data/cpsc2018/{r}", lab == "A") for r, lab in cpsc_selected().items()]
    for r in open("data/cpsc2018_extra/SELECTED").read().split():
        dx = next(l for l in open(f"data/cpsc2018_extra/{r}.hea").read().splitlines() if l.startswith("# Dx"))
        out.append((f"data/cpsc2018_extra/{r}", AF_CODE in dx.split(":")[1].strip().split(",")))
    return out


def dev_crops():
    """Clinical features of up to three 10 s crops per development recording (cached)."""
    try:
        return pd.read_pickle(CACHE)
    except FileNotFoundError:
        pass
    rows = []
    for path, is_af in dev_recordings():
        sig, fs = load_12lead(path)
        n10 = 10 * fs
        for k in range(min(3, len(sig) // n10)):
            row = clinical_features(sig[k * n10:(k + 1) * n10], fs, is_af)
            if row:
                rows.append({**row, "record": path, "crop": k})
    df = pd.DataFrame(rows)
    df["log_f_wave"] = df.log_f_wave.fillna(df.log_f_wave.median())
    df.to_pickle(CACHE)
    return df


if __name__ == "__main__":
    d = dev_crops()
    is_af = d.af.values
    print(f"{len(d)} crops from {d.record.nunique()} recordings, {is_af.sum()} AF")

    afdb = []
    for r in RECORDS:
        sig, fs, mask, _ = load_af_record(r, 0, 2)
        afdb += [{**w, "record": r} for w in clean_windows(detect_peaks_adaptive, sig, fs, mask, 60)]
    afdb_model, _ = fit_with_cutoff(pd.DataFrame(afdb), FEATURES_5)

    folds = np.array_split(np.random.default_rng(2).permutation(d.record.unique()), 5)
    rows = []
    for name, features, use_afdb in [
            ("AFDB model (5 features), clinical cutoff", FEATURES_5, True),
            ("clinical model, RR (5N)", FEATURES_5N, False),
            ("clinical model, RR + P waves (II)", FEATURES_5N + ["p_match_regular"], False),
            ("clinical model, RR + P waves (II) + f-waves (V1)", CLINICAL_FEATURES, False),
            ("clinical model, RR + f-waves (V1)", FEATURES_5N + ["log_f_wave"], False)]:
        prob, pred = np.zeros(len(d)), np.zeros(len(d), dtype=bool)
        for fold in folds:
            held = d.record.isin(fold).values
            train = d[~held]
            if use_afdb:
                model = afdb_model
                cutoff = cutoff_for_sensitivity(model.predict_proba(model_inputs(train, features))[:, 1],
                                                train.af.values, train.record.values)
            else:
                model, cutoff = fit_with_cutoff(train, features)
            prob[held] = model.predict_proba(model_inputs(d[held], features))[:, 1]
            pred[held] = prob[held] >= cutoff
        ov = p_wave_override(pred, d)
        rows.append({"model": name, "auc": auc(prob, is_af),
                     "sensitivity": pred[is_af].mean(), "false_alarm_rate": pred[~is_af].mean(),
                     "override_sensitivity": ov[is_af].mean(), "override_false_alarm_rate": ov[~is_af].mean()})
    results = pd.DataFrame(rows)
    results.to_csv("results/afib_clinical_dev.csv", index=False)
    print(results.round(3).to_string(index=False))
