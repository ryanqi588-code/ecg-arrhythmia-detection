"""Test 18: is the interpretable model worse than both networks at telling AF from PACs? (registered)

Test 11's headline, that the interpretable model ranks AF against atrial
premature beats (PACs) worse than both deep networks (SPH AF-vs-PAC AUC
0.988 against 0.996 for both; about three times the ranking errors), was
calculated after that test. This registers it as a prediction and tests
it once on patients no test has used.

Models, all unchanged:
  interpretable - the adopted model, from results/final_af_model.csv via
                  predict.py (cutoff 0.4309)
  CNN           - models/cnn_model.keras (same-data CNN, cutoff 0.06515)
  Ribeiro       - data/ribeiro/model/model.hdf5 (Ribeiro et al. 2020;
                  cutoff 0.1199 by the same rule, as in test 11)
The networks are run by `afib_compare_dnn.py mimic18` (TensorFlow
environment), the interpretable model here.

Data: MIMIC-IV-ECG v1.0 (Boston, USA), chosen from the machine reports
only (signals not opened) by the rules of tests 14 and 16
(afib_slowaf_dev.select_dev: AF = an "atrial fibrillation" statement
without possible/probable/consider, flutter, pacing, sinus, artifact or
undetermined; PAC = a PAC statement with sinus rhythm and none of
fibrillation, flutter, pacing, ectopic/junctional rhythms, SVT,
ventricular premature beats, irregular/undetermined rhythm, artifact).
One recording per patient, seed 4: 1000 AF and 2000 PAC patients,
leaving out every patient of tests 14, 15 and 16 and of the MIMIC
development set (results/mimic18_selected.txt, committed with this plan).
Labels are the ECG machine's. All three models are scored on the
recordings the interpretable model can score (as in test 11).

Primary, fixed before the signals are downloaded: AF-vs-PAC ROC AUC,
interpretable minus each network, paired bootstrap by patient (2000
resamples). Prediction: the interpretable model is worse, with the 95%
interval's upper bound below 0, against both networks. Confirmed only if
it holds against both.
Secondary: PAC recordings called AF and AF caught at each model's
cutoff; pairs ranked wrong (1 - AUC) and their ratio; the same AUC
comparison on AF confirmed by another AF ECG of the patient within 24 h
(afib_mimic_labels.neighbour_groups) against all PAC; the interpretable
model's "needs review" count. Run once.

Previews on used MIMIC data (tests 14 and 15's recordings; Ribeiro's
network run on them for this preview only), AUC interpretable / CNN /
Ribeiro and interpretable minus each network (95% CI):
  test 14: 0.904 / 0.933 / 0.923; -0.029 (-0.040 to -0.018) vs CNN,
           -0.019 (-0.030 to -0.008) vs Ribeiro
  test 15: 0.895 / 0.927 / 0.918; -0.032 (-0.044 to -0.021) vs CNN,
           -0.024 (-0.035 to -0.014) vs Ribeiro
So the used data already point the predicted way; this test checks it
on new patients.

    python afib_afpac_test.py select      # writes results/mimic18_selected.txt (reports only)
    python afib_afpac_test.py download    # .hea/.dat of the selected recordings into data/mimic/
    python afib_compare_dnn.py mimic18    # TensorFlow environment: both networks
    python afib_afpac_test.py             # the test
"""

import os
import sys
import urllib.request
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from afib_comparison_test import boot_ci, rate_str
from afib_mimic_labels import load_reports, neighbour_groups
from afib_mimic_select import MIMIC_DIR, URL, load_mimic, read_selected
from afib_slowaf_dev import select_dev
from predict import load_model, predict

NAME = "mimic18_selected"
EXCLUDE = ("mimic_selected", "mimic15_selected", "mimic_dev_selected", "mimic16_selected")
CUT = {"interpretable": 0.4309, "CNN": 0.06515, "Ribeiro": 0.11988194}
COL = {"interpretable": "prob_interp", "CNN": "prob_cnn", "Ribeiro": "prob_ribeiro"}


def download():
    def get(path):
        for ext in (".hea", ".dat"):
            dest = f"{MIMIC_DIR}/{path}{ext}"
            if not os.path.exists(dest):
                os.makedirs(os.path.dirname(dest), exist_ok=True)
                urllib.request.urlretrieve(f"{URL}{path}{ext}", dest)
    paths = [p for p, _, _ in read_selected(NAME)]
    with ThreadPoolExecutor(8) as pool:
        list(pool.map(get, paths))
    print(f"{len(paths)} recordings downloaded")


def interpretable(name=NAME):
    model, rows = load_model(), []
    for path, subject, label in read_selected(name):
        sig, fs = load_mimic(f"{MIMIC_DIR}/{path}")
        o = predict(sig, fs, model)
        rows.append({"record": int(path.split("/")[-1]), "patient": f"mimic-{subject}", "label": label,
                     "result": o["result"], "prob_interp": o.get("probability", np.nan)})
    return pd.DataFrame(rows)


def auc_diff(y, a, b, g):
    v = roc_auc_score(y, a) - roc_auc_score(y, b)
    return v, *boot_ci(lambda i: roc_auc_score(y[i], a[i]) - roc_auc_score(y[i], b[i]), g)


def run(name=NAME, nets_path="results/compare_probs_mimic18.csv", out="results/afib_afpac_test"):
    t = interpretable(name)
    nets = pd.read_csv(nets_path)[["record", "prob_cnn", "prob_ribeiro"]]
    t = t.merge(nets.assign(record=nets.record.astype(int)), on="record")
    print(f"selected {len(read_selected(name))}; interpretable result: {t.result.value_counts().to_dict()}")
    t = t[t.prob_interp.notna()].reset_index(drop=True)        # recordings the interpretable model scores
    y, g = (t.label == "A").values, t.patient.values
    print(f"scored: {y.sum()} AF, {(~y).sum()} PAC")

    rows = []
    for m, c in COL.items():
        p = t[c].values
        rows.append({"model": m, "AF vs PAC AUC": roc_auc_score(y, p), "pairs ranked wrong": 1 - roc_auc_score(y, p),
                     "AF caught": rate_str(p[y] >= CUT[m]), "PAC called AF": rate_str(p[~y] >= CUT[m])})
    diffs, ok = {}, []
    for net in ["CNN", "Ribeiro"]:
        d = auc_diff(y, t.prob_interp.values, t[COL[net]].values, g)
        diffs[f"interpretable - {net}"] = d
        ok.append(d[2] < 0)
    rpt = pd.DataFrame(rows)
    for net in ["CNN", "Ribeiro"]:
        rpt.loc[rpt.model == net, "error ratio, interpretable / network"] = (
            rpt.loc[rpt.model == "interpretable", "pairs ranked wrong"].iloc[0]
            / rpt.loc[rpt.model == net, "pairs ranked wrong"].iloc[0])
    print(rpt.round(4).to_string(index=False))

    groups = neighbour_groups(t, load_reports())
    conf = t.record.map(groups.set_index("record").group).eq("confirmed").values
    sel = (conf & y) | ~y
    for net in ["CNN", "Ribeiro"]:
        d = auc_diff(y[sel], t.prob_interp.values[sel], t[COL[net]].values[sel], g[sel])
        diffs[f"confirmed AF ({int((conf & y).sum())}) vs PAC: interpretable - {net}"] = d
    for k, (v, lo, hi) in diffs.items():
        print(f"{k}: {v:+.4f} (95% CI {lo:+.4f} to {hi:+.4f})")
    print(f"needs review (interpretable): {(t.result == 'needs review').sum()}")
    verdict = "confirmed" if all(ok) else "not confirmed"
    print(f"PRIMARY: worse than CNN {'yes' if ok[0] else 'NO'}; worse than Ribeiro {'yes' if ok[1] else 'NO'}; "
          f"VERDICT: {verdict}")
    rpt.to_csv(f"{out}.csv", index=False)
    pd.DataFrame([{"comparison": k, "diff": v, "lo": lo, "hi": hi} for k, (v, lo, hi) in diffs.items()]
                 ).assign(verdict=verdict).to_csv(f"{out}_diffs.csv", index=False)
    t.assign(confirmed_af=conf & y).to_csv(f"{out}_recordings.csv", index=False)


if __name__ == "__main__":
    if "select" in sys.argv:
        select_dev(NAME, seed=4, n={"A": 1000, "PAC": 2000, "N": 0}, exclude=EXCLUDE)
        sys.exit()
    if "download" in sys.argv:
        download()
        sys.exit()
    run()
