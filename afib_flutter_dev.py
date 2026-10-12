"""Development of an atrial flutter detector for 10 s 12-lead ECGs.

So far flutter was left out of every test. In flutter the atria beat at
250-350 a minute in a regular sawtooth, and the ventricles follow every
2nd, 3rd or 4th wave, or a varying number: with fixed 2:1 block the pulse
is regular at about 150 bpm, like sinus tachycardia or SVT; with variable
block it is irregular, like AF.

Development data, all from sources already used:
  - the AF development units (afib_pac_dev.py): CPSC crops, PTB-XL and
    Chapman AF / PAC / sinus recordings
  - flutter and its look-alikes, never loaded before:
      Chapman: flutter without AF (445), SVT (587), atrial tachycardia
        (121), sinus tachycardia (600 of 1568 at random, seed 0)
      PTB-XL (validated reports): flutter without AF (44), SVT/PSVT
        without flutter or AF (30), sinus tachycardia (300 at random, seed 0)
    (SELECTED_FLUTTER_DEV in data/chapman and data/ptbxl)
Cross-validation is grouped by recording (CPSC, Chapman) or patient (PTB-XL).
"""

import os

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier

from af import (CLINICAL_FEATURES_12, FLUTTER_LEADS, load_12lead_all, fit_with_source_cutoffs, make_logreg,
                model_inputs, cutoff_for_sensitivity, auc)
from afib_pac_dev import dev_units as af_dev_units, prepare as af_prepare, unit_features

CACHE = "data/flutter_dev.pkl"
FL = [f"{k}_{lead}" for lead in FLUTTER_LEADS for k in ["fl_autocorr", "fl_organization", "fl_freq", "fl_baseline"]]
FLUTTER_FEATURES = (["hr", "cv", "pnn50", "entropy_norm", "rr_mode_frac", "p_match_mean", "f_ratio_II",
                     "f_ratio_V1", "log_f_wave"] + FL)


def flutter_units():
    """The AF development units plus the flutter / look-alike recordings, all with flutter features."""
    try:
        return pd.read_pickle(CACHE)
    except FileNotFoundError:
        pass
    old = af_dev_units("multilead12")
    rows = []
    # Recompute every unit, since the cached AF units predate the flutter features.
    from afib_clinical_dev import dev_recordings
    from af import cpsc_selected
    cpsc_labels = {f"data/cpsc2018/{r}": lab for r, lab in cpsc_selected().items()}
    for i, (path, is_af) in enumerate(dev_recordings()):
        if i % 300 == 0:
            print(f"cpsc {i} ...", flush=True)
        sig, fs = load_12lead_all(path)
        n10 = 10 * fs
        for k in range(min(3, len(sig) // n10)):
            row = unit_features(sig[k * n10:(k + 1) * n10], fs, is_af, "multilead12")
            if row:
                rows.append({**row, "source": "cpsc", "record": path,
                             "label": cpsc_labels.get(path, "A" if is_af else "other")})
    db = pd.read_csv("data/ptbxl/ptbxl_database.csv", index_col="filename_hr")
    for sel in ["SELECTED", "SELECTED_FLUTTER_DEV"]:
        for i, line in enumerate(open(f"data/ptbxl/{sel}").read().splitlines()):
            if i % 500 == 0:
                print(f"ptbxl {sel} {i} ...", flush=True)
            path, label = line.split()
            row = unit_features(*load_12lead_all(os.path.join("data/ptbxl", path)), label == "A", "multilead12")
            if row:
                rows.append({**row, "source": "ptbxl", "record": f"ptbxl-{int(db.patient_id[path])}",
                             "label": label})
    for sel in ["SELECTED", "SELECTED_FLUTTER_DEV"]:
        for i, line in enumerate(open(f"data/chapman/{sel}").read().splitlines()):
            if i % 500 == 0:
                print(f"chapman {sel} {i} ...", flush=True)
            path, label = line.split()
            rec_id = os.path.basename(path)
            row = unit_features(*load_12lead_all(os.path.join("data/chapman", rec_id)), label == "A", "multilead12")
            if row:
                rows.append({**row, "source": "chapman", "record": f"chapman-{rec_id}", "label": label})
    df = pd.DataFrame(rows)
    assert len(df) >= len(old)
    df.to_pickle(CACHE)
    return df


LOOKALIKES = ["ST", "SVT"]       # regular tachycardias that 2:1 flutter resembles
LOOKALIKE_BUDGET = 0.10          # at most this share of them may be called flutter


def make_flutter_model():
    """Gradient-boosted trees: unlike logistic regression, they can learn that a fast
    regular rate means flutter only together with continuous atrial activity."""
    return HistGradientBoostingClassifier(max_depth=4, learning_rate=0.05, max_iter=400,
                                          class_weight="balanced", random_state=0)


def budget_cutoff(prob, labels, budget=LOOKALIKE_BUDGET):
    """Lowest cutoff that calls at most `budget` of the look-alike recordings flutter.

    2:1 flutter and SVT can look the same in 10 s, so the operating point is
    set by how many look-alikes may be flagged, not by a sensitivity target.
    """
    return np.quantile(prob[np.isin(labels, LOOKALIKES)], 1 - budget)


def fit_flutter(d):
    """Flutter model fitted on all of `d`, with its cutoff from 10-fold out-of-fold probabilities."""
    oof = np.zeros(len(d))
    for fold in np.array_split(np.random.default_rng(0).permutation(d.record.unique()), 10):
        held = d.record.isin(fold).values
        m = make_flutter_model().fit(model_inputs(d[~held], FLUTTER_FEATURES), d.flutter[~held])
        oof[held] = m.predict_proba(model_inputs(d[held], FLUTTER_FEATURES))[:, 1]
    model = make_flutter_model().fit(model_inputs(d, FLUTTER_FEATURES), d.flutter)
    return model, budget_cutoff(oof, d.label.values)


def prepare(df):
    df = af_prepare(df)
    df["flutter"] = df.label == "AFL"
    for col in FLUTTER_FEATURES:
        df[col] = df[col].fillna(df[col].median())
    return df


if __name__ == "__main__":
    d = prepare(flutter_units())
    fl = d.flutter.values
    print(f"{len(d)} units; labels: {d.label.value_counts().to_dict()}")

    # 1. What the final AF model does with flutter (flutter was never in its training).
    af_units = af_prepare(af_dev_units("multilead12"))
    m_af, cut_af = fit_with_source_cutoffs(af_units, CLINICAL_FEATURES_12)
    called_af = m_af.predict_proba(model_inputs(d, CLINICAL_FEATURES_12))[:, 1] >= min(cut_af.values())
    print("\nFinal AF model: share of each rhythm called AF")
    print(d.assign(called_af=called_af).groupby("label").called_af.agg(["mean", "size"]).round(3).to_string())

    # 2. Each flutter feature alone.
    groups = {"vs all": ~fl, "vs AF": (d.label == "A").values, "vs sinus tach": (d.label == "ST").values,
              "vs SVT/AT": d.label.isin(["SVT", "AT"]).values, "vs sinus": (d.label == "N").values}
    alone = pd.DataFrame({f: {g: auc(d[f].values[fl | m], fl[fl | m]) for g, m in groups.items()}
                          for f in FLUTTER_FEATURES}).T
    alone["median flutter"] = d[FLUTTER_FEATURES][fl].median()
    alone["median AF"] = d[FLUTTER_FEATURES][(d.label == "A").values].median()
    alone["median sinus tach"] = d[FLUTTER_FEATURES][(d.label == "ST").values].median()
    print("\nEach feature alone, AUC for flutter (< 0.5: lower in flutter):")
    print(alone.round(3).to_string())

    # 3. Flutter models, 5-fold CV grouped by recording / patient. Cutoffs: 80% and 90%
    # sensitivity, and the look-alike budget (at most 10% of sinus tachycardia + SVT).
    folds = np.array_split(np.random.default_rng(2).permutation(d.record.unique()), 5)
    new = d.label.isin(["AFL", "ST", "SVT", "AT"]).values
    af_call = np.zeros(len(d), dtype=bool)          # AF model: out-of-fold for its own units
    for fold in folds:
        held = d.record.isin(fold).values
        m, cs = fit_with_source_cutoffs(d[~held & ~new], CLINICAL_FEATURES_12)
        af_call[held] = m.predict_proba(model_inputs(d[held], CLINICAL_FEATURES_12))[:, 1] >= min(cs.values())
    rr_only = ["hr", "cv", "pnn50", "entropy_norm", "rr_mode_frac"]
    candidates = {
        "logistic, rate + RR only": (make_logreg, rr_only),
        "logistic, + AF atrial features": (make_logreg, rr_only + ["p_match_mean", "f_ratio_II", "f_ratio_V1",
                                                                    "log_f_wave"]),
        "logistic, + flutter features": (make_logreg, FLUTTER_FEATURES),
        "boosted trees, + flutter features": (make_flutter_model, FLUTTER_FEATURES),
    }
    rows = []
    for name, (make, features) in candidates.items():
        prob = np.zeros(len(d))
        for fold in folds:
            held = d.record.isin(fold).values
            m = make().fit(model_inputs(d[~held], features), fl[~held])
            prob[held] = m.predict_proba(model_inputs(d[held], features))[:, 1]
        cutoffs = {"80% sensitivity": cutoff_for_sensitivity(prob, fl, d.record.values, 0.80),
                   "90% sensitivity": cutoff_for_sensitivity(prob, fl, d.record.values, 0.90),
                   "look-alike budget": budget_cutoff(prob, d.label.values)}
        for point, c in cutoffs.items():
            named = prob >= c
            either = named | af_call
            r = {"model": name, "cutoff": point, "auc": auc(prob, fl),
                 "named_flutter": named[fl].mean(), "AF_named_flutter": named[(d.label == "A").values].mean()}
            for lab in ["AFL", "A", "ST", "SVT", "AT", "PAC", "N", "other"]:
                r[f"af_or_flutter_{lab}"] = either[(d.label == lab).values].mean()
            rows.append(r)
    base = {"model": "AF model alone", "cutoff": "-", "named_flutter": np.nan, "AF_named_flutter": np.nan}
    for lab in ["AFL", "A", "ST", "SVT", "AT", "PAC", "N", "other"]:
        base[f"af_or_flutter_{lab}"] = af_call[(d.label == lab).values].mean()
    results = pd.DataFrame([base] + rows)
    results.to_csv("results/afib_flutter_dev.csv", index=False)
    pd.set_option("display.width", 250)
    print("\nFlutter models (CV). named_flutter: share of flutter called flutter; af_or_flutter_X: share of X "
          "called AF or flutter (AF model or flutter model)")
    print(results.round(3).to_string(index=False))
