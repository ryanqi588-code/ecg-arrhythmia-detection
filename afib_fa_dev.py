"""Development of fixes for the 12-lead AF model's false alarms, after test 11.

Test 11 found two causes:
  - recordings with fewer than 3 usable regular beats get p_match = 0
    ("no P waves"), so noisy, very fast and bundle-branch-block recordings
    are pushed to AF with near certainty;
  - the networks made 40-70% as many PAC false alarms, and PAC false
    alarms varied by hospital (SPH 9.5%, Ningbo 24.5%).
Candidate fixes:
  - unmeasured P waves: an indicator `p_unmeasured` (p_match_mean == 0,
    which in development always means all 12 leads had too few regular
    beats), with p_match_regular and p_match_mean then filled with the
    training median instead of 0;
  - more varied training data: the sets already used in tests 9-11 join
    development (SPH, Ningbo PAC and sinus, CODE-15% part 0, CODE test set).

All candidates are compared on the same expanded development set by 5-fold
cross-validation grouped by patient (recording where no patient id), each
with the strictest per-source 95%-sensitivity cutoff over CPSC, PTB-XL and
Chapman (as for the final model). Features come from the saved test outputs (the same
af.clinical_features pipeline), so nothing is recomputed.
"""

from outputs import write_path      # committed outputs go to rerun/ unless --force
import numpy as np
import pandas as pd

from af import (CLINICAL_FEATURES_12, RR_PATTERN_FEATURES, make_logreg, model_inputs, cutoff_for_sensitivity,
                record_weights, auc)
from afib_pac_dev import dev_units, prepare

ORIGINAL = ["cpsc", "ptbxl", "chapman"]
# Cutoffs come from the original three sources, as for the final model. With
# all sources the strictest-source rule is set by CODE-15%, the hardest, and
# its cutoff collapses (first run: PAC false alarms 16% -> 41% at AUC 0.986).
CUTOFF_SOURCES = ORIGINAL
P_COLS = ["p_match_regular", "p_match_mean"]


def expanded_units():
    d = prepare(dev_units("multilead12"))
    d["patient"] = d.record
    parts = [d]
    sph = pd.read_csv("results/afib_sph_test_recordings.csv")
    meta = pd.read_csv("data/sph/metadata.csv")[["ECG_ID", "Patient_ID"]]
    sph = sph.merge(meta, left_on="record", right_on="ECG_ID")
    parts.append(sph.assign(source="sph", patient="sph-" + sph.Patient_ID.astype(str)))
    for name in ["ningbo", "code15", "codetest"]:
        t = pd.read_csv(f"results/afib_comparison_{name}_recordings.csv")
        t["label"] = t.label.replace({"notAF": f"notAF_{name}"})
        parts.append(t.assign(source=name, patient=f"{name}-" + t.patient.astype(str)))
    keep = ["source", "patient", "record", "label"] + CLINICAL_FEATURES_12 + RR_PATTERN_FEATURES
    u = pd.concat([p[keep] for p in parts], ignore_index=True)
    u["af"] = u.label == "A"
    for col in ["log_f_wave", "f_ratio_II", "f_ratio_V1"] + RR_PATTERN_FEATURES:
        u[col] = u[col].fillna(u[col].median())
    u["p_unmeasured"] = (u.p_match_mean == 0).astype(float)
    return u


def with_unmeasured_fix(train, test):
    """p_match columns NaN where unmeasured, filled with the training median of the measured."""
    train, test = train.copy(), test.copy()
    for col in P_COLS:
        med = train.loc[train.p_unmeasured == 0, col].median()
        for df in (train, test):
            df.loc[df.p_unmeasured == 1, col] = med
    return train, test


def strictest_cutoff(prob, d):
    cuts = []
    for s in CUTOFF_SOURCES:
        m = (d.source == s).values
        if m.any() and d.af.values[m].sum() >= 50:
            cuts.append(cutoff_for_sensitivity(prob[m], d.af.values[m], d.patient.values[m]))
    return min(cuts)


def fit(train, features, fix):
    """Logistic model and its cutoff, from 10-fold out-of-fold probabilities on `train`."""
    oof = np.zeros(len(train))
    for fold in np.array_split(np.random.default_rng(0).permutation(train.patient.unique()), 10):
        held = train.patient.isin(fold).values
        tr, te = (with_unmeasured_fix(train[~held], train[held]) if fix else (train[~held], train[held]))
        oof[held] = make_logreg().fit(model_inputs(tr, features), tr.af).predict_proba(model_inputs(te, features))[:, 1]
    tr = with_unmeasured_fix(train, train)[0] if fix else train
    return make_logreg().fit(model_inputs(tr, features), tr.af), strictest_cutoff(oof, train)


CANDIDATES = {
    "A final model (original 3 sources)": (CLINICAL_FEATURES_12, False, ORIGINAL),
    "A2 final model + unmeasured-P fix (original 3 sources)": (CLINICAL_FEATURES_12 + ["p_unmeasured"], True, ORIGINAL),
    "B + all 7 sources": (CLINICAL_FEATURES_12, False, None),
    "C + unmeasured-P fix": (CLINICAL_FEATURES_12 + ["p_unmeasured"], True, None),
    "D + RR-pattern features": (CLINICAL_FEATURES_12 + ["p_unmeasured"] + RR_PATTERN_FEATURES, True, None),
}

if __name__ == "__main__":
    u = expanded_units()
    print(f"{len(u)} units; by source and label:")
    print(u.groupby(["source", "label"]).size().unstack(fill_value=0).to_string())
    print("p_unmeasured share by source:", u.groupby("source").p_unmeasured.mean().round(3).to_dict())
    folds = np.array_split(np.random.default_rng(2).permutation(u.patient.unique()), 5)
    rows = []
    for name, (features, fix, sources) in CANDIDATES.items():
        prob, pred = np.zeros(len(u)), np.zeros(len(u), dtype=bool)
        for fold in folds:
            held = u.patient.isin(fold).values
            train = u[~held] if sources is None else u[~held & u.source.isin(sources).values]
            model, cutoff = fit(train, features, fix)
            test = with_unmeasured_fix(train, u[held])[1] if fix else u[held]
            prob[held] = model.predict_proba(model_inputs(test, features))[:, 1]
            pred[held] = prob[held] >= cutoff
        pac_or_af = u.af.values | (u.label == "PAC").values
        r = {"model": name, "auc": auc(prob, u.af.values),
             "auc_AF_vs_PAC": auc(prob[pac_or_af], u.af.values[pac_or_af])}
        for s in ["cpsc", "ptbxl", "chapman", "sph", "code15", "codetest"]:
            m = (u.source == s).values
            r[f"sens_{s}"] = pred[m & u.af.values].mean()
        for s in ["cpsc", "ptbxl", "chapman", "sph", "ningbo"]:
            r[f"PAC_fa_{s}"] = pred[(u.source == s).values & (u.label == "PAC").values].mean()
        r["PAC_fa_all"] = pred[(u.label == "PAC").values].mean()
        r["sinus_fa_all"] = pred[(u.label == "N").values].mean()
        r["fa_code15"] = pred[(u.label == "notAF_code15").values].mean()
        r["fa_codetest"] = pred[(u.label == "notAF_codetest").values].mean()
        r["fa_cpsc_other"] = pred[(u.label == "other").values].mean()
        r["fa_p_unmeasured_nonAF"] = pred[(u.p_unmeasured == 1).values & ~u.af.values].mean()
        r["sens_p_unmeasured_AF"] = pred[(u.p_unmeasured == 1).values & u.af.values].mean()
        rows.append(r)
        print(pd.DataFrame([r]).round(3).T.to_string(header=False), "\n", flush=True)
    pd.DataFrame(rows).to_csv(write_path("results/afib_fa_dev.csv"), index=False)
