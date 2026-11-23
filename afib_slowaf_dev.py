"""Development: catching slow AF whose quiet baseline passes for consistent P waves.

The MIMIC-IV-ECG error analysis (afib_mimic_errors.py) found that most AF
the model misses there is slow and less irregular, and that its P-wave
consistency features score it like sinus rhythm (median p_match_mean 0.92):
at slow rates the stretch before each QRS is mostly flat baseline, which
matches from beat to beat. ecg.p_presence asks a different question, beat
by beat: does a wave before the QRS stand out from the quiet part of the
same interval? On test 14-15's (used) recordings it scores the missed AF
0.11 and the PAC recordings the model correctly clears 0.60. And it's
scorable exactly where the problem is, since slow beats leave a quiet part.

The development data (CPSC, PTB-XL, Chapman) has little slow AF with
fooled P-wave features (133 of 3416 AF units), so a model can't learn
much there. So this script adds a MIMIC development set from patients in
no test: 1500 AF, 1500 PAC and 1500 sinus-rhythm patients, chosen from
the machine reports by afib_mimic_select's rules (sinus: a sinus rhythm
statement and none of the AF, PAC or PAC-exclusion words), one recording
per patient, seed 2 (results/mimic_dev_selected.txt).

Candidates, compared by 5-fold CV grouped by patient over all four
sources, with cutoffs set by the usual rule on CPSC, PTB-XL and Chapman:
  0 adopted model (test 12), trained on the original three sources
  1 + p_presence in II and V1 (setting B of afib_ppresence_dev, with a
      missing indicator), original three sources
  2 adopted features, trained with MIMIC development data added
  3 + p_presence, trained with MIMIC development data added
  4 as 3, with MIMIC also a cutoff source (95% sensitivity there too)
  5 gated P-wave match: p_match_mean and p_match_regular multiplied by
      min(1, mean p_presence / 0.5) (unchanged where p_presence is unscored),
      original three sources
  6 as 5, + MIMIC training
  7 + p_presence + the gated p_match_mean as an extra feature (lets the
      model learn "a P-wave match counts only if the P waves stand out"),
      + MIMIC training
  8 as 7, original three sources
The cutoff rule lands at different points, so candidates are also
compared at matched MIMIC PAC false alarms (10% and 15%).
Reported: sensitivity per source, MIMIC sensitivity by machine heart
rate, false alarms on PAC (per source) and sinus/normal/other rhythms.
Then a preview (not a test) on test 14-15's MIMIC recordings and on SPH.
"""

from outputs import write_path      # committed outputs go to rerun/ unless --force
import numpy as np
import pandas as pd

import afib_fa_dev
from af import auc, model_inputs
from afib_fa_dev import fit, with_unmeasured_fix
from afib_hybrid_dev import ADOPTED
from afib_mimic_select import (AF, AF_EXCLUDE, MIMIC_DIR, PAC, PAC_EXCLUDE, load_mimic, read_selected,
                               report_text)
from afib_pac_dev import unit_features
from afib_ppresence_dev import pp, units_with_pp

PP = ["pp_II_B", "pp_V1_B", "pp_missing_B"]
GATED = [f for f in ADOPTED if f not in ("p_match_mean", "p_match_regular")] + \
    ["p_match_mean_g", "p_match_regular_g", "pp_missing_B"]
SLOWAF = ADOPTED + PP + ["p_match_mean_g"]          # candidate 7


def add_gate(df):
    """Gated P-wave match: scaled down where P waves don't stand out from the baseline."""
    df = df.copy()
    g = np.where(df.pp_missing_B == 1, 1.0, np.clip((df.pp_II_B + df.pp_V1_B) / 2 / 0.5, 0, 1))
    df["p_match_mean_g"] = df.p_match_mean * g
    df["p_match_regular_g"] = df.p_match_regular * g
    return df
N_EACH = 1500
ORIGINAL = ["cpsc", "ptbxl", "chapman"]


def select_dev(out="mimic_dev_selected", seed=2, n=None,
               exclude=("mimic_selected", "mimic15_selected")):
    """One recording per patient from the machine reports; patients in `exclude` files left out."""
    n = n or {"A": N_EACH, "PAC": N_EACH, "N": N_EACH}
    m = pd.read_csv(f"{MIMIC_DIR}/machine_measurements.csv", low_memory=False)
    paths = pd.read_csv(f"{MIMIC_DIR}/record_list.csv").set_index("study_id").path
    used = {int(s) for f in exclude for _, s, _ in read_selected(f)}
    m = m[~m.subject_id.isin(used)]
    t = report_text(m)
    groups = {"A": t.str.contains(AF) & ~t.str.contains(AF_EXCLUDE),
              "PAC": t.str.contains(PAC) & t.str.contains("sinus") & ~t.str.contains(PAC_EXCLUDE),
              "N": t.str.contains("sinus") & ~t.str.contains(PAC) & ~t.str.contains(PAC_EXCLUDE)
                   & ~t.str.contains(AF) & ~t.str.contains("possible|probable|consider|arrhythmia undetermined")}
    rng, taken, rows_out = np.random.default_rng(seed), set(), []
    for lab, k in groups.items():
        rows = m[k & ~m.subject_id.isin(taken)]
        pick = rows.groupby("subject_id").sample(1, random_state=seed)
        pick = pick.iloc[rng.permutation(len(pick))[:n[lab]]]
        taken |= set(pick.subject_id)
        rows_out += [(paths[s], subj, lab) for s, subj in zip(pick.study_id, pick.subject_id)]
        print(f"{lab}: {len(rows)} recordings from {rows.subject_id.nunique()} patients; {len(pick)} chosen")
    with open(write_path(f"results/{out}.txt"), "w") as f:
        f.writelines(f"{p} {s} {lab}\n" for p, s, lab in rows_out)


def mimic_units(name, source, cache):
    """Clinical features + p_presence for a MIMIC selection (cached); machine heart rate attached."""
    try:
        return pd.read_pickle(cache)
    except FileNotFoundError:
        pass
    mm = pd.read_csv(f"{MIMIC_DIR}/machine_measurements.csv", low_memory=False,
                     usecols=["study_id", "rr_interval"]).set_index("study_id").rr_interval
    rows = []
    for i, (path, subject, label) in enumerate(read_selected(name)):
        if i % 500 == 0:
            print(f"{source} {i} ...", flush=True)
        sig, fs = load_mimic(f"{MIMIC_DIR}/{path}")
        if np.isnan(sig).any():
            continue
        row = unit_features(sig, fs, label == "A", "multilead12")
        if row is None:
            continue
        v = pp(sig, fs)
        study = int(path.split("/")[-1])
        rows.append({**row, "pp_II_B": v[0] if v else np.nan, "pp_V1_B": v[1] if v else np.nan,
                     "source": source, "record": str(study), "patient": f"mimic-{subject}", "label": label,
                     "machine_hr": 60000 / mm[study] if mm[study] > 0 else np.nan})
    df = pd.DataFrame(rows)
    df.to_pickle(cache)
    return df


def prepare(df, medians):
    df = df.copy()
    df["p_unmeasured"] = (df.p_match_mean == 0).astype(float)
    df["pp_missing_B"] = df.pp_II_B.isna().astype(float)
    for c, v in medians.items():
        df[c] = df[c].fillna(v)
    return add_gate(df)


def summarise(name, pred, prob, u):
    af, pac = u.af.values, (u.label == "PAC").values
    r = {"model": name, "auc": auc(prob, af)}
    for s in ORIGINAL + ["mimicdev"]:
        r[f"sens_{s}"] = pred[af & (u.source == s).values].mean()
    hr = u.machine_hr.values
    r["sens_mimic_hr<80"] = pred[af & (u.source == "mimicdev").values & (hr < 80)].mean()
    r["sens_mimic_hr>=80"] = pred[af & (u.source == "mimicdev").values & (hr >= 80)].mean()
    for s in ORIGINAL + ["mimicdev"]:
        r[f"PAC_fa_{s}"] = pred[pac & (u.source == s).values].mean()
    r["sinus_fa_orig"] = pred[(u.label == "N").values & u.source.isin(ORIGINAL).values].mean()
    r["sinus_fa_mimic"] = pred[(u.label == "N").values & (u.source == "mimicdev").values].mean()
    r["other_fa"] = pred[(u.label == "other").values].mean()
    mm = (u.source == "mimicdev").values
    r["auc_mimic_AF_vs_PAC"] = auc(prob[mm & (af | pac)], af[mm & (af | pac)])
    for fa in [0.10, 0.15]:
        thr = np.quantile(prob[pac & mm], 1 - fa)
        r[f"sens_mimic_at_PAC{int(fa * 100)}"] = (prob[af & mm] >= thr).mean()
        r[f"sens_mimic_hr<80_at_PAC{int(fa * 100)}"] = (prob[af & mm & (hr < 80)] >= thr).mean()
    return r


def build_dev():
    """All development units (original three sources + MIMIC development) and the fill medians."""
    base = units_with_pp(cols=["pp_II_B", "pp_V1_B"])
    base = base[base.source.isin(ORIGINAL)].copy()
    base["machine_hr"] = np.nan
    md = mimic_units("mimic_dev_selected", "mimicdev", "data/mimic_dev_units.pkl")
    cols = ["log_f_wave", "f_ratio_II", "f_ratio_V1"]
    base["pp_missing_B"] = base.pp_II_B.isna().astype(float)
    medians = {c: base[c].median() for c in cols + ["pp_II_B", "pp_V1_B"]}
    base = add_gate(base.fillna({c: medians[c] for c in ["pp_II_B", "pp_V1_B"]}))
    md = prepare(md, medians)
    md["af"] = md.label == "A"
    u = pd.concat([base, md[[c for c in base.columns if c in md.columns]]], ignore_index=True)
    u["af"] = u.af.astype(bool)
    return u, medians


if __name__ == "__main__":
    import sys
    if "select" in sys.argv:
        select_dev()
        sys.exit()
    u, medians = build_dev()
    print(u.groupby(["source", "label"]).size().unstack(fill_value=0).to_string())

    cands = {"0 adopted (3 sources)": (ADOPTED, ORIGINAL, ORIGINAL),
             "1 + p_presence (3 sources)": (ADOPTED + PP, ORIGINAL, ORIGINAL),
             "2 adopted, + MIMIC training": (ADOPTED, ORIGINAL + ["mimicdev"], ORIGINAL),
             "3 + p_presence, + MIMIC training": (ADOPTED + PP, ORIGINAL + ["mimicdev"], ORIGINAL),
             "4 as 3, MIMIC also a cutoff source": (ADOPTED + PP, ORIGINAL + ["mimicdev"], ORIGINAL + ["mimicdev"]),
             "5 gated P-wave match (3 sources)": (GATED, ORIGINAL, ORIGINAL),
             "6 gated P-wave match, + MIMIC training": (GATED, ORIGINAL + ["mimicdev"], ORIGINAL),
             "7 + p_presence + gated match, + MIMIC training": (SLOWAF, ORIGINAL + ["mimicdev"], ORIGINAL),
             "8 + p_presence + gated match (3 sources)": (SLOWAF, ORIGINAL, ORIGINAL)}
    folds = np.array_split(np.random.default_rng(2).permutation(u.patient.unique()), 5)
    rows = []
    for name, (feats, train_src, cut_src) in cands.items():
        afib_fa_dev.CUTOFF_SOURCES = cut_src
        prob, pred = np.zeros(len(u)), np.zeros(len(u), bool)
        for fold in folds:
            held = u.patient.isin(fold).values
            train = u[~held & u.source.isin(train_src).values]
            model, cutoff = fit(train, feats, True)
            test = with_unmeasured_fix(train, u[held])[1]
            prob[held] = model.predict_proba(model_inputs(test, feats))[:, 1]
            pred[held] = prob[held] >= cutoff
        rows.append(summarise(name, pred, prob, u))
        print(pd.DataFrame([rows[-1]]).round(3).T.to_string(header=False), "\n", flush=True)
    afib_fa_dev.CUTOFF_SOURCES = ORIGINAL
    pd.DataFrame(rows).to_csv(write_path("results/afib_slowaf_dev.csv"), index=False)

    # Preview on used sets: models fitted on all development data of their sources.
    used = pd.concat([mimic_units("mimic_selected", "mimic14", "data/mimic14_units.pkl"),
                      mimic_units("mimic15_selected", "mimic15", "data/mimic15_units.pkl")])
    used = prepare(used, medians)
    prev = []
    for name, (feats, train_src, cut_src) in cands.items():
        afib_fa_dev.CUTOFF_SOURCES = cut_src
        train = u[u.source.isin(train_src).values]
        model, cutoff = fit(train, feats, True)
        t = with_unmeasured_fix(train, used)[1]
        called = model.predict_proba(model_inputs(t, feats))[:, 1] >= cutoff
        y, hr = (t.label == "A").values, t.machine_hr.values
        prev.append({"model": name, "cutoff": cutoff, "AF caught": called[y].mean(),
                     "AF caught, hr<80": called[y & (hr < 80)].mean(),
                     "PAC called AF": called[(t.label == "PAC").values].mean()})
    afib_fa_dev.CUTOFF_SOURCES = ORIGINAL
    prev = pd.DataFrame(prev)
    print("Preview on test 14-15's MIMIC recordings (used; not a test):")
    print(prev.round(3).to_string(index=False))
    prev.to_csv(write_path("results/afib_slowaf_preview.csv"), index=False)

    # Chosen: candidate 7 (best AF-vs-PAC separation and sensitivity at matched PAC false alarms).
    # Preview on SPH and Ningbo (used in tests 9 and 11), where the adopted model already works.
    sn = units_with_pp(cols=["pp_II_B", "pp_V1_B"])
    sn = sn[sn.source.isin(["sph", "ningbo"])].copy()
    sn["pp_missing_B"] = sn.pp_II_B.isna().astype(float)
    sn = add_gate(sn.fillna({c: medians[c] for c in ["pp_II_B", "pp_V1_B"]}))
    rows = []
    for name in ["0 adopted (3 sources)", "7 + p_presence + gated match, + MIMIC training"]:
        feats, train_src, cut_src = cands[name]
        train = u[u.source.isin(train_src).values]
        model, cutoff = fit(train, feats, True)
        t = with_unmeasured_fix(train, sn)[1]
        called = model.predict_proba(model_inputs(t, feats))[:, 1] >= cutoff
        for src in ["sph", "ningbo"]:
            k = (t.source == src).values
            rows.append({"model": name, "set": src, "cutoff": cutoff,
                         "AF caught": called[k & t.af.values].mean() if (k & t.af.values).any() else np.nan,
                         "PAC called AF": called[k & (t.label == "PAC").values].mean(),
                         "normal/sinus called AF": called[k & (t.label == "N").values].mean()})
    sp = pd.DataFrame(rows)
    print("Preview on SPH and Ningbo (used; not a test):")
    print(sp.round(3).to_string(index=False))
    sp.to_csv(write_path("results/afib_slowaf_preview_sph_ningbo.csv"), index=False)
