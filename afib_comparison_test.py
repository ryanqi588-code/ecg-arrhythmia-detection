"""Test 11: the interpretable AF model against deep learning and against doctors.

Comparators
  ribeiro - the published deep neural network of Ribeiro et al. (2020),
            trained on 2.3 million Brazilian 12-lead ECGs (afib_compare_dnn.py)
  cnn     - a 1D residual CNN trained on exactly our development data with
            our cutoff rule (afib_cnn.py; architecture fixed before training)
  doctors - on the CODE test set (827 ECGs, gold standard from two
            cardiologists plus a senior specialist), the AF calls of two
            4th-year cardiology residents, two 3rd-year emergency residents and
            two 5th-year medical students, as released with that dataset
Every model uses its own strictest per-source 95%-sensitivity cutoff from
the development data (CPSC, PTB-XL, Chapman); ours is the final 12-lead
model exactly as registered for test 9 (cutoff 0.4736).

Sets, each chosen from labels or headers only:
  sph      - the 15,113 recordings of test 9 (671 AF, 537 PAC, 13,905
             normal). Our model's numbers there are test 9's; the networks
             haven't seen SPH.
  ningbo   - Challenge 2021 Ningbo, never opened: every PAC recording without
             AF or flutter codes (1041) and 1500 sinus-rhythm recordings with
             no AF, flutter, PAC, ventricular or other ectopy or pacing
             (random, seed 0). False alarms only: Ningbo's AF/flutter codes
             can't be trusted (README, test 9).
  code15   - CODE-15% part 0 (fixed before download), never opened: every
             exam, AF label from exams.csv. Brazil, mostly primary care,
             2010-2016, 400 Hz. Its AF labels required both a cardiologist's
             report and an irregular RR (SDNN above a threshold), so AF with a
             regular pulse is under-represented and sensitivity will be
             optimistic for every model. Ribeiro's network is excluded here:
             CODE-15% is a sample of its training data.
  codetest - the CODE test set (827 ECGs, 13 AF by the gold standard).
Comparisons are paired: every model is scored on the recordings our model
can score (unscored recordings are counted and reported).

Primary question, fixed before any of these sets was opened: is the
interpretable model close to each black box in ranking AF? Non-inferiority
margin 0.01 in ROC AUC on SPH: the lower bound of the 95% paired bootstrap
interval of (our AUC - comparator AUC) must be above -0.01.
Secondary: sensitivity and false alarms at each model's cutoff on every set,
with paired bootstrap intervals for the differences (2000 resamples by
patient where known); on codetest, sensitivity and specificity against the
doctors. Run once.
"""

from outputs import write_path      # committed outputs go to rerun/ unless --force
import os

import h5py
import numpy as np
import pandas as pd

from af import CLINICAL_FEATURES_12, fit_with_source_cutoffs, cutoff_for_sensitivity, model_inputs, auc
from afib_pac_dev import dev_units, prepare, unit_features
from afib_sph_test import wilson
from afib_compare_dnn import strip_padding
from afib_cnn import load_wfdb_12

OUR_CUTOFF_TEST9 = 0.473559896031711
B = 2000


def our_features(name):
    """Our model's features for a comparison set; None rows (unscorable) dropped and counted."""
    rows, skipped = [], {}

    def add(rec, label, sig, fs, patient=None):
        row = unit_features(sig, fs, label == "A", "multilead12")
        if row is None:
            skipped[label] = skipped.get(label, 0) + 1
        else:
            rows.append({**row, "record": rec, "label": label, "patient": patient if patient is not None else rec})

    if name == "ningbo":
        for line in open("results/ningbo_selected.txt").read().splitlines():
            path, label = line.split()
            rec = os.path.basename(path)
            add(rec, label, *load_wfdb_12(f"data/ningbo/rec/{rec}"))
    elif name == "code15":
        exams = pd.read_csv("data/code15/exams.csv")
        exams = exams[exams.trace_file == "exams_part0.hdf5"].set_index("exam_id")
        with h5py.File("data/code15/exams_part0.hdf5", "r") as f:
            ids = f["exam_id"][()]
            for i, eid in enumerate(ids):
                if eid in exams.index:
                    add(int(eid), "A" if exams.AF[eid] else "notAF", strip_padding(f["tracings"][i]), 400,
                        int(exams.patient_id[eid]))
    elif name == "codetest":
        gold = pd.read_csv("data/ribeiro/data/annotations/gold_standard.csv")
        with h5py.File("data/ribeiro/data/ecg_tracings.hdf5", "r") as f:
            tr = f["tracings"][()]
        for i in range(len(tr)):
            add(i, "A" if gold.AF[i] == 1 else "notAF", strip_padding(tr[i]), 400)
    return pd.DataFrame(rows), skipped


def boot_ci(stat, groups, n=B, seed=0):
    """Paired bootstrap 95% interval of stat(index array), resampling groups (patients)."""
    rng = np.random.default_rng(seed)
    uniq, inv = np.unique(groups, return_inverse=True)
    members = [np.flatnonzero(inv == g) for g in range(len(uniq))]
    vals = []
    for _ in range(n):
        pick = rng.integers(0, len(uniq), len(uniq))
        vals.append(stat(np.concatenate([members[g] for g in pick])))
    return np.nanpercentile(vals, [2.5, 97.5])


def rate_str(hits):
    hits = np.asarray(hits, bool)
    if not len(hits):
        return "–"
    lo, hi = wilson(int(hits.sum()), len(hits))
    return f"{hits.mean():.1%} ({lo:.1%}–{hi:.1%}) [{int(hits.sum())}/{len(hits)}]"


if __name__ == "__main__":
    os.makedirs("results", exist_ok=True)
    units = prepare(dev_units("multilead12"))
    m_af, cut_af = fit_with_source_cutoffs(units, CLINICAL_FEATURES_12)
    c_ours = min(cut_af.values())
    assert abs(c_ours - OUR_CUTOFF_TEST9) < 1e-9, c_ours
    fills = {c: units[c].median() for c in ["log_f_wave", "f_ratio_II", "f_ratio_V1"]}

    dev = pd.read_csv("results/compare_probs_dev.csv")
    c_rib = min(cutoff_for_sensitivity(dev.prob_ribeiro.values[(dev.source == s).values],
                                       (dev.label == "A").values[(dev.source == s).values],
                                       dev.record.values[(dev.source == s).values]) for s in ["cpsc", "ptbxl", "chapman"])
    c_cnn = float(pd.read_csv("results/afib_cnn_dev.csv").cutoff[0])
    print(f"cutoffs: ours {c_ours:.4f}, Ribeiro {c_rib:.4f}, CNN {c_cnn:.4f}")

    summary, diffs = [], []
    for name in ["sph", "ningbo", "code15", "codetest"]:
        probs = pd.read_csv(f"results/compare_probs_{name}.csv")
        if name == "sph":
            t = pd.read_csv("results/afib_sph_test_recordings.csv")[["record", "label", "prob_new"]]
            meta = pd.read_csv("data/sph/metadata.csv")[["ECG_ID", "Patient_ID"]]
            t = t.merge(meta, left_on="record", right_on="ECG_ID").rename(
                columns={"prob_new": "prob_ours", "Patient_ID": "patient"})
            skipped = {}
        else:
            t, skipped = our_features(name)
            for c, v in fills.items():
                t[c] = t[c].fillna(v)
            t["prob_ours"] = m_af.predict_proba(model_inputs(t, CLINICAL_FEATURES_12))[:, 1]
        probs["record"] = probs.record.astype(str)
        t["record"] = t.record.astype(str)
        t = t.merge(probs.drop(columns=["label", "source"]), on="record", how="left")
        assert len(t) and t.prob_cnn.notna().all()
        models = {"ours": ("prob_ours", c_ours), "cnn": ("prob_cnn", c_cnn)}
        if "prob_ribeiro" in t:
            models["ribeiro"] = ("prob_ribeiro", c_rib)
        y = (t.label == "A").values
        groups = t.patient.astype(str).values
        print(f"\n{name}: {len(t)} scored {t.label.value_counts().to_dict()}; unscorable by our model: {skipped}")
        for m, (col, c) in models.items():
            p = t[col].values
            pred = p >= c
            r = {"set": name, "model": m, "n": len(t), "unscored_by_ours": sum(skipped.values()),
                 "auc": auc(p, y) if 0 < y.sum() < len(y) else np.nan,
                 "AF caught": rate_str(pred[y]) if y.any() else "–"}
            for lab in sorted(set(t.label) - {"A"}):
                r[f"{lab} called AF"] = rate_str(pred[(t.label == lab).values])
            summary.append(r)
            print({k: (round(v, 4) if isinstance(v, float) else v) for k, v in r.items()})
        for m in [k for k in models if k != "ours"]:
            po, pc = t.prob_ours.values, t[models[m][0]].values
            do, dc = po >= c_ours, pc >= models[m][1]
            d = {"set": name, "comparison": f"ours - {m}"}
            if 0 < y.sum() < len(y):
                d["auc_diff"] = auc(po, y) - auc(pc, y)
                d["auc_diff_ci"] = boot_ci(lambda i: auc(po[i], y[i]) - auc(pc[i], y[i]), groups)
                d["sens_diff"] = do[y].mean() - dc[y].mean()
                d["sens_diff_ci"] = boot_ci(lambda i: do[i][y[i]].mean() - dc[i][y[i]].mean(), groups)
            for lab in sorted(set(t.label) - {"A"}):
                k = (t.label == lab).values
                d[f"{lab}_fa_diff"] = do[k].mean() - dc[k].mean()
                d[f"{lab}_fa_diff_ci"] = boot_ci(lambda i: do[i][k[i]].mean() - dc[i][k[i]].mean(), groups)
            diffs.append(d)
            print(d)
        if name == "codetest":
            ann = "data/ribeiro/data/annotations"
            gold = pd.read_csv(f"{ann}/gold_standard.csv").AF.values == 1
            keep = t.record.astype(int).values
            rows = []
            for who in ["cardiologist1", "cardiologist2", "cardiology_residents", "emergency_residents",
                        "medical_students", "dnn"]:
                call = pd.read_csv(f"{ann}/{who}.csv").AF.values == 1
                for subset, idx in [("all 827", np.arange(len(gold))), ("scored by ours", keep)]:
                    rows.append({"reader": who, "on": subset, "sensitivity": rate_str(call[idx][gold[idx]]),
                                 "specificity": rate_str(~call[idx][~gold[idx]])})
            for m, (col, c) in models.items():
                pred = t[col].values >= c
                rows.append({"reader": f"model: {m}", "on": "scored by ours", "sensitivity": rate_str(pred[y]),
                             "specificity": rate_str(~pred[~y])})
            readers = pd.DataFrame(rows)
            readers.to_csv(write_path("results/afib_comparison_doctors.csv"), index=False)
            print(readers.to_string(index=False))
        t.to_csv(write_path(f"results/afib_comparison_{name}_recordings.csv"), index=False)
    pd.DataFrame(summary).to_csv(write_path("results/afib_comparison_test.csv"), index=False)
    pd.DataFrame(diffs).to_csv(write_path("results/afib_comparison_diffs.csv"), index=False)
    sph = [d for d in diffs if d["set"] == "sph"]
    for d in sph:
        lo, hi = d["auc_diff_ci"]
        print(f"\nPRIMARY {d['comparison']}: AUC difference {d['auc_diff']:+.4f} (95% CI {lo:+.4f} to {hi:+.4f}); "
              f"{'close (non-inferior at -0.01)' if lo > -0.01 else 'NOT shown to be within 0.01'}")
