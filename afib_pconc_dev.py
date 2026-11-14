"""Development of P-wave concentration features against PAC false alarms.

No change so far has reduced the 12-lead model's false alarms on atrial
premature beats (afib_ectopy_dev.py, afib_fa_dev.py), while the deep
networks of test 11 made 40-70% as many. Those features ask whether beats
share one P wave; a PAC's own P wave is a different shape, so it reads as
"no P wave". ecg.p_concentration asks only whether each beat's atrial
activity is concentrated just before its QRS (a P wave of any shape) or
spread across the interval (fibrillation), over all 12 leads, for all beats
(pconc_all) and for the early beats alone (pconc_early, af.atrial_premature).

Development data: the AF development units (CPSC crops, PTB-XL, Chapman)
plus the already-used SPH and Ningbo recordings, which add 1577 PAC;
5-fold CV grouped by patient; the adopted model (test 12) as baseline,
cutoff rule as afib_fa_dev.py (strictest per-source cutoff on CPSC, PTB-XL
and Chapman).
"""

import os

import h5py
import numpy as np
import pandas as pd

from af import CLINICAL_FEATURES_12, CLINICAL_MIN_RR, load_12lead_all, atrial_premature, auc
from afib_clinical_dev import dev_recordings
from afib_fa_dev import fit, with_unmeasured_fix
from afib_pac_dev import I_II_V1
from ecg import detect_peaks_multilead, flag_ectopic, p_concentration
from af import model_inputs

CACHE = "data/pconc_dev.pkl"
ADOPTED = CLINICAL_FEATURES_12 + ["p_unmeasured"]


def pconc(sig12, fs):
    """(pconc_all, pconc_early), or None where af.clinical_features would find too few beats."""
    peaks = detect_peaks_multilead(sig12[:, I_II_V1], fs)
    rr = np.diff(peaks) / fs
    if len(peaks) < CLINICAL_MIN_RR + 1 or ((rr > 0.25) & (rr < 2.5)).sum() < CLINICAL_MIN_RR:
        return None
    early = atrial_premature(peaks, fs) & ~flag_ectopic(sig12[:, 1], fs, peaks)
    return p_concentration(sig12, fs, peaks, early)


def pconc_units():
    """pconc for every development unit, keyed like the saved feature tables (cached)."""
    try:
        return pd.read_pickle(CACHE)
    except FileNotFoundError:
        pass
    rows = []
    for i, (path, _) in enumerate(dev_recordings()):
        if i % 300 == 0:
            print(f"cpsc {i} ...", flush=True)
        sig, fs = load_12lead_all(path)
        n10 = 10 * fs
        for k in range(min(3, len(sig) // n10)):
            v = pconc(sig[k * n10:(k + 1) * n10], fs)
            if v is not None:
                rows.append(("cpsc", path, k, *v))
    for line in open("data/ptbxl/SELECTED").read().splitlines():
        path = line.split()[0]
        v = pconc(*load_12lead_all(os.path.join("data/ptbxl", path)))
        if v is not None:
            rows.append(("ptbxl", path, 0, *v))
    for line in open("data/chapman/SELECTED").read().splitlines():
        rec = os.path.basename(line.split()[0])
        v = pconc(*load_12lead_all(f"data/chapman/{rec}"))
        if v is not None:
            rows.append(("chapman", rec, 0, *v))
    sph = pd.read_csv("results/afib_sph_test_recordings.csv", usecols=["record"])
    for i, rec in enumerate(sph.record):
        if i % 2000 == 0:
            print(f"sph {i} ...", flush=True)
        with h5py.File(f"data/sph/records/{rec}.h5", "r") as f:
            sig = f["ecg"][()][:, :5000].T
        v = pconc(sig, 500)
        rows.append(("sph", rec, 0, *(v if v is not None else (np.nan, np.nan))))
    for line in open("data/ningbo/SELECTED").read().splitlines():
        rec = os.path.basename(line.split()[0])
        v = pconc(*load_12lead_all(f"data/ningbo/rec/{rec}"))
        rows.append(("ningbo", rec, 0, *(v if v is not None else (np.nan, np.nan))))
    df = pd.DataFrame(rows, columns=["source", "key", "crop", "pconc_all", "pconc_early"])
    df.to_pickle(CACHE)
    return df


def units_with_pconc():
    """The adopted model's development units (afib_fa_dev) restricted to CPSC, PTB-XL, Chapman,
    SPH and Ningbo, with the P-concentration features joined on."""
    from afib_fa_dev import expanded_units
    u = expanded_units()
    u = u[u.source.isin(["cpsc", "ptbxl", "chapman", "sph", "ningbo"])].reset_index(drop=True)
    pc = pconc_units()
    out = []
    for src in ["cpsc", "ptbxl", "chapman", "sph", "ningbo"]:
        us, ps = u[u.source == src], pc[pc.source == src]
        if src == "cpsc":
            # Both tables keep only scorable crops, in order: match them within each recording.
            merged = []
            for rec, grp in us.groupby("record", sort=False):
                cand = ps[ps.key == rec]
                assert len(cand) == len(grp), (rec, len(cand), len(grp))
                merged.append(cand.iloc[:len(grp)][["pconc_all", "pconc_early"]].set_index(grp.index))
            out.append(us.join(pd.concat(merged)))
        elif src == "ptbxl":
            ptb = [l.split()[0] for l in open("data/ptbxl/SELECTED").read().splitlines()]
            db = pd.read_csv("data/ptbxl/ptbxl_database.csv", index_col="filename_hr")
            ps = ps.assign(record=[f"ptbxl-{int(db.patient_id[k])}" for k in ps.key])
            # several recordings per patient: match in order within patient
            merged = []
            for rec, grp in us.groupby("record", sort=False):
                cand = ps[ps.record == rec]
                assert len(cand) == len(grp), (rec, len(cand), len(grp))
                merged.append(cand.iloc[:len(grp)][["pconc_all", "pconc_early"]].set_index(grp.index))
            out.append(us.join(pd.concat(merged)))
        else:
            key = us.record.astype(str).str.replace("chapman-", "", regex=False)
            m = ps.set_index(ps.key.astype(str))[["pconc_all", "pconc_early"]]
            out.append(us.join(m.reindex(key.values).set_index(us.index)))
    u = pd.concat(out).sort_index()
    return u


if __name__ == "__main__":
    u = units_with_pconc()
    af, pac = u.af.values, (u.label == "PAC").values
    print(f"{len(u)} units, {af.sum()} AF, {pac.sum()} PAC; pconc missing: all {u.pconc_all.isna().mean():.3f}, "
          f"early {u.pconc_early.isna().mean():.3f}")
    for f in ["pconc_all", "pconc_early"]:
        ok = u[f].notna().values
        print(f"{f}: AF-vs-PAC AUC {auc(u[f].values[ok & (af | pac)], af[ok & (af | pac)]):.3f}; median AF "
              f"{u[f][af].median():.2f}, PAC {u[f][pac].median():.2f}, sinus {u[f][(u.label == 'N').values].median():.2f}")
    u["pconc_early_missing"] = u.pconc_early.isna().astype(float)
    for f in ["pconc_all", "pconc_early"]:
        u[f] = u[f].fillna(u[f].median())
    folds = np.array_split(np.random.default_rng(2).permutation(u.patient.unique()), 5)
    rows = []
    for name, feats in [("adopted model (test 12)", ADOPTED),
                        ("+ pconc_all", ADOPTED + ["pconc_all"]),
                        ("+ pconc_all + pconc_early", ADOPTED + ["pconc_all", "pconc_early", "pconc_early_missing"])]:
        prob, pred = np.zeros(len(u)), np.zeros(len(u), bool)
        for fold in folds:
            held = u.patient.isin(fold).values
            train = u[~held & u.source.isin(["cpsc", "ptbxl", "chapman"]).values]
            model, cutoff = fit(train, feats, True)
            test = with_unmeasured_fix(train, u[held])[1]
            prob[held] = model.predict_proba(model_inputs(test, feats))[:, 1]
            pred[held] = prob[held] >= cutoff
        r = {"model": name, "auc": auc(prob, af), "auc_AF_vs_PAC": auc(prob[af | pac], af[af | pac])}
        for s in ["cpsc", "ptbxl", "chapman", "sph"]:
            r[f"sens_{s}"] = pred[af & (u.source == s).values].mean()
        for s in ["cpsc", "ptbxl", "chapman", "sph", "ningbo"]:
            r[f"PAC_fa_{s}"] = pred[pac & (u.source == s).values].mean()
        r["PAC_fa_all"] = pred[pac].mean()
        r["sinus_fa"] = pred[(u.label == "N").values].mean()
        r["other_fa"] = pred[(u.label == "other").values].mean()
        rows.append(r)
        print(pd.DataFrame([r]).round(3).T.to_string(header=False), "\n", flush=True)
    pd.DataFrame(rows).to_csv("results/afib_pconc_dev.csv", index=False)
