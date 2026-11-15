"""Test 11 on SPH, split by what else the PAC recordings contain (worked out after the test; not registered).

SPH's PAC group (afib_sph_test.sph_selected) is every recording with
atrial premature complexes and no AF or flutter; other findings are
allowed. Split, from the AHA statements (modifiers ignored):
  PAC only   - atrial premature complexes and nothing else
  PAC+other  - other findings, but no ventricular premature complexes
  PAC+VPC    - ventricular premature complexes too (code 60, the only
               ventricular-ectopy code)
For each group and each of test 11's three models (at their test 11
cutoffs): PAC recordings called AF, and AF-vs-PAC ROC AUC (all SPH AF
against that group). Paired bootstrap by patient (2000 resamples) for this
model minus each network. Uses saved test 11 probabilities only.
"""

import csv

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from af import cutoff_for_sensitivity
from afib_comparison_test import OUR_CUTOFF_TEST9, boot_ci, rate_str

MODELS = {"this model": "prob_ours", "Ribeiro's network": "prob_ribeiro", "same-data CNN": "prob_cnn"}
GROUPS = ["PAC only", "PAC+other", "PAC+VPC"]


def statements():
    codes = list(csv.DictReader(open("data/sph/code.csv")))
    mods = {r["Code"] for r in codes if r["Category"] == "Modifier"}
    return {r["ECG_ID"]: set(r["AHA_Code"].replace(";", "+").split("+")) - mods
            for r in csv.DictReader(open("data/sph/metadata.csv"))}


def group(c):
    return "PAC+VPC" if "60" in c else "PAC only" if c == {"30"} else "PAC+other"


if __name__ == "__main__":
    dev = pd.read_csv("results/compare_probs_dev.csv")
    cut = {"prob_ours": OUR_CUTOFF_TEST9, "prob_cnn": float(pd.read_csv("results/afib_cnn_dev.csv").cutoff[0]),
           "prob_ribeiro": min(cutoff_for_sensitivity(dev.prob_ribeiro.values[(dev.source == s).values],
                                                      (dev.label == "A").values[(dev.source == s).values],
                                                      dev.record.values[(dev.source == s).values])
                               for s in ["cpsc", "ptbxl", "chapman"])}
    meta = statements()
    d = pd.read_csv("results/afib_comparison_sph_recordings.csv")
    d = d[d.label.isin(["A", "PAC"])].reset_index(drop=True)
    d["group"] = np.where(d.label == "A", "AF", d.record.map(meta).map(group))
    y, g = (d.label == "A").values, d.patient.values
    called = {c: d[c].values >= cut[c] for c in cut}

    rows = []
    for G in GROUPS:
        m = (d.group == G).values
        sel = y | m
        for name, c in MODELS.items():
            r = {"group": G, "PAC recordings": int(m.sum()), "model": name, "PAC called AF": rate_str(called[c][m]),
                 "AF vs PAC AUC": roc_auc_score(y[sel], d[c].values[sel]),
                 "pairs ranked wrong": 1 - roc_auc_score(y[sel], d[c].values[sel])}
            if c != "prob_ours":
                ours = d.prob_ours.values
                fa = called["prob_ours"][m].mean() - called[c][m].mean()
                lo, hi = boot_ci(lambda i: called["prob_ours"][i][m[i]].mean() - called[c][i][m[i]].mean(), g)
                r["false alarms, this model - network"] = f"{fa:+.3f} ({lo:+.3f} to {hi:+.3f})"
                da = roc_auc_score(y[sel], d[c].values[sel]) - roc_auc_score(y[sel], ours[sel])
                lo, hi = boot_ci(lambda i: roc_auc_score(y[i][sel[i]], d[c].values[i][sel[i]])
                                 - roc_auc_score(y[i][sel[i]], ours[i][sel[i]]), g)
                r["AUC, network - this model"] = f"{da:+.4f} ({lo:+.4f} to {hi:+.4f})"
                r["error ratio, this model / network"] = (1 - roc_auc_score(y[sel], ours[sel])) / max(
                    1 - roc_auc_score(y[sel], d[c].values[sel]), 1e-12)
            rows.append(r)
    out = pd.DataFrame(rows)
    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 20)
    print(out.round(4).to_string(index=False))
    out.to_csv("results/afib_comparison_ectopy.csv", index=False)
