"""Development of per-beat P-wave presence features against PAC false alarms.

The 12-lead model's false alarms on atrial premature beats haven't moved
with any change so far (afib_ectopy_dev.py, afib_fa_dev.py,
afib_pconc_dev.py). Plots of PAC recordings it calls AF while the CNN of
test 11 doesn't show sinus rhythm with a P wave before nearly every beat.
ecg.p_presence makes the per-beat check a cardiologist makes: in a fixed
lead, does a wave just before the QRS stand out from the quiet part of the
same interval? pp_II and pp_V1 are the shares of scored beats where it
does (fixed leads, since choosing each recording's best lead lets real AF
score high by chance). Beats without a quiet stretch (fast rates, the
premature beats themselves) are unscored; recordings with none scored get
the median and an indicator.

Development data, CV and cutoff rule as afib_pconc_dev.py (CPSC crops,
PTB-XL, Chapman, SPH, Ningbo; 5-fold CV grouped by patient; training on
the original three sources; adopted model of test 12 as baseline).
"""

from outputs import write_path      # committed outputs go to rerun/ unless --force
import os

import h5py
import numpy as np
import pandas as pd

from af import CLINICAL_FEATURES_12, CLINICAL_MIN_RR, load_12lead_all, atrial_premature, auc
from afib_clinical_dev import dev_recordings
from afib_fa_dev import fit, with_unmeasured_fix
from afib_pac_dev import I_II_V1
from ecg import detect_peaks_multilead, flag_ectopic, p_presence
from af import model_inputs

CACHE = "data/ppresence_dev_bc.pkl"
CACHE_A = "data/ppresence_dev.pkl"         # setting A: window (0.25, 0.05), quiet part from 0.45 s
# Window settings: (pre-QRS window, quiet part starts this long after the previous peak).
# The first run used ((0.25, 0.05), 0.45): only 41% of recordings had a scorable beat.
SETTINGS = {"B": ((0.20, 0.05), 0.40), "C": ((0.16, 0.04), 0.35)}
COLS = [f"pp_{lead}_{k}" for k in SETTINGS for lead in ["II", "V1"]]
ADOPTED = CLINICAL_FEATURES_12 + ["p_unmeasured"]


def pp(sig12, fs):
    """(pp_II, pp_V1), or None where af.clinical_features would find too few beats."""
    peaks = detect_peaks_multilead(sig12[:, I_II_V1], fs)
    rr = np.diff(peaks) / fs
    if len(peaks) < CLINICAL_MIN_RR + 1 or ((rr > 0.25) & (rr < 2.5)).sum() < CLINICAL_MIN_RR:
        return None
    early = atrial_premature(peaks, fs) & ~flag_ectopic(sig12[:, 1], fs, peaks)
    return tuple(p_presence(sig12[:, k], fs, peaks, early, window=w, t_end=te)[0]
                 for w, te in SETTINGS.values() for k in (1, 6))


def pp_units(cache=CACHE):
    """pconc for every development unit, keyed like the saved feature tables (cached)."""
    try:
        return pd.read_pickle(cache)
    except FileNotFoundError:
        pass
    rows = []
    for i, (path, _) in enumerate(dev_recordings()):
        if i % 300 == 0:
            print(f"cpsc {i} ...", flush=True)
        sig, fs = load_12lead_all(path)
        n10 = 10 * fs
        for k in range(min(3, len(sig) // n10)):
            v = pp(sig[k * n10:(k + 1) * n10], fs)
            if v is not None:
                rows.append(("cpsc", path, k, *v))
    for line in open("results/ptbxl_selected.txt").read().splitlines():
        path = line.split()[0]
        v = pp(*load_12lead_all(os.path.join("data/ptbxl", path)))
        if v is not None:
            rows.append(("ptbxl", path, 0, *v))
    for line in open("results/chapman_selected.txt").read().splitlines():
        rec = os.path.basename(line.split()[0])
        v = pp(*load_12lead_all(f"data/chapman/{rec}"))
        if v is not None:
            rows.append(("chapman", rec, 0, *v))
    sph = pd.read_csv("results/afib_sph_test_recordings.csv", usecols=["record"])
    for i, rec in enumerate(sph.record):
        if i % 2000 == 0:
            print(f"sph {i} ...", flush=True)
        with h5py.File(f"data/sph/records/{rec}.h5", "r") as f:
            sig = f["ecg"][()][:, :5000].T
        v = pp(sig, 500)
        rows.append(("sph", rec, 0, *(v if v is not None else (np.nan,) * len(COLS))))
    for line in open("results/ningbo_selected.txt").read().splitlines():
        rec = os.path.basename(line.split()[0])
        v = pp(*load_12lead_all(f"data/ningbo/rec/{rec}"))
        rows.append(("ningbo", rec, 0, *(v if v is not None else (np.nan,) * len(COLS))))
    df = pd.DataFrame(rows, columns=["source", "key", "crop", *COLS])
    df.to_pickle(CACHE)
    return df


def units_with_pp(cache=CACHE, cols=None):
    """The adopted model's development units (afib_fa_dev) restricted to CPSC, PTB-XL, Chapman,
    SPH and Ningbo, with the P-concentration features joined on."""
    from afib_fa_dev import expanded_units
    u = expanded_units()
    u = u[u.source.isin(["cpsc", "ptbxl", "chapman", "sph", "ningbo"])].reset_index(drop=True)
    cols = COLS if cols is None else cols
    pc = pp_units(cache)
    out = []
    for src in ["cpsc", "ptbxl", "chapman", "sph", "ningbo"]:
        us, ps = u[u.source == src], pc[pc.source == src]
        if src == "cpsc":
            # Both tables keep only scorable crops, in order: match them within each recording.
            merged = []
            for rec, grp in us.groupby("record", sort=False):
                cand = ps[ps.key == rec]
                assert len(cand) == len(grp), (rec, len(cand), len(grp))
                merged.append(cand.iloc[:len(grp)][cols].set_index(grp.index))
            out.append(us.join(pd.concat(merged)))
        elif src == "ptbxl":
            ptb = [l.split()[0] for l in open("results/ptbxl_selected.txt").read().splitlines()]
            db = pd.read_csv("data/ptbxl/ptbxl_database.csv", index_col="filename_hr")
            ps = ps.assign(record=[f"ptbxl-{int(db.patient_id[k])}" for k in ps.key])
            # several recordings per patient: match in order within patient
            merged = []
            for rec, grp in us.groupby("record", sort=False):
                cand = ps[ps.record == rec]
                assert len(cand) == len(grp), (rec, len(cand), len(grp))
                merged.append(cand.iloc[:len(grp)][cols].set_index(grp.index))
            out.append(us.join(pd.concat(merged)))
        else:
            key = us.record.astype(str).str.replace("chapman-", "", regex=False)
            m = ps.set_index(ps.key.astype(str))[cols]
            out.append(us.join(m.reindex(key.values).set_index(us.index)))
    u = pd.concat(out).sort_index()
    return u


if __name__ == "__main__":
    u = units_with_pp()
    ua = units_with_pp(CACHE_A, ["pp_II", "pp_V1"])
    u["pp_II_A"], u["pp_V1_A"] = ua.pp_II.values, ua.pp_V1.values
    COLS = ["pp_II_A", "pp_V1_A"] + COLS
    SETTINGS = {"A": ((0.25, 0.05), 0.45), **SETTINGS}
    af, pac = u.af.values, (u.label == "PAC").values
    print(f"{len(u)} units, {af.sum()} AF, {pac.sum()} PAC")
    for f in COLS:
        ok = u[f].notna().values
        print(f"{f}: scored {ok.mean():.3f}; AF-vs-PAC AUC {1 - auc(u[f].values[ok & (af | pac)], af[ok & (af | pac)]):.3f}; "
              f"median AF {u[f][af].median():.2f}, PAC {u[f][pac].median():.2f}, sinus {u[f][(u.label == 'N').values].median():.2f}")
    for k in SETTINGS:
        u[f"pp_missing_{k}"] = u[f"pp_II_{k}"].isna().astype(float)
    for f in COLS:
        u[f] = u[f].fillna(u[f].median())
    folds = np.array_split(np.random.default_rng(2).permutation(u.patient.unique()), 5)
    rows = []
    cands = [("adopted model (test 12)", ADOPTED)]
    cands += [(f"+ pp II, V1 (setting {k})", ADOPTED + [f"pp_II_{k}", f"pp_V1_{k}", f"pp_missing_{k}"]) for k in SETTINGS]
    probs = {}
    for name, feats in cands:
        prob, pred = np.zeros(len(u)), np.zeros(len(u), bool)
        for fold in folds:
            held = u.patient.isin(fold).values
            train = u[~held & u.source.isin(["cpsc", "ptbxl", "chapman"]).values]
            model, cutoff = fit(train, feats, True)
            test = with_unmeasured_fix(train, u[held])[1]
            prob[held] = model.predict_proba(model_inputs(test, feats))[:, 1]
            pred[held] = prob[held] >= cutoff
        r = {"model": name, "auc": auc(prob, af), "auc_AF_vs_PAC": auc(prob[af | pac], af[af | pac])}
        for s_ in ["cpsc", "ptbxl", "chapman", "sph"]:
            r[f"sens_{s_}"] = pred[af & (u.source == s_).values].mean()
        for s_ in ["cpsc", "ptbxl", "chapman", "sph", "ningbo"]:
            r[f"PAC_fa_{s_}"] = pred[pac & (u.source == s_).values].mean()
        r["PAC_fa_all"] = pred[pac].mean()
        r["sinus_fa"] = pred[(u.label == "N").values].mean()
        r["other_fa"] = pred[(u.label == "other").values].mean()
        rows.append(r)
        probs[name] = prob
        print(pd.DataFrame([r]).round(3).T.to_string(header=False), "\n", flush=True)
    pd.DataFrame(rows).to_csv(write_path("results/afib_ppresence_dev.csv"), index=False)
    # The cutoff rule lands at different sensitivities for different models, so compare
    # PAC false alarms at matched pooled sensitivity too.
    matched = []
    for sens in [0.95, 0.97, 0.98]:
        for name, p in probs.items():
            thr = np.quantile(p[af], 1 - sens)
            matched.append({"sensitivity": sens, "model": name, "PAC_fa_all": (p[pac] >= thr).mean(),
                            "PAC_fa_sph": (p[pac & (u.source == "sph").values] >= thr).mean(),
                            "PAC_fa_ningbo": (p[pac & (u.source == "ningbo").values] >= thr).mean()})
    matched = pd.DataFrame(matched)
    matched.to_csv(write_path("results/afib_ppresence_dev_matched.csv"), index=False)
    print(matched.round(3).to_string(index=False))
