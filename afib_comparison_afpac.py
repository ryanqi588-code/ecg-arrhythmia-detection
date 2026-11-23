"""AF-vs-PAC ranking on SPH for test 11's three models (worked out after the test; not registered).

Test 11's registered primary compared ROC AUC over all SPH recordings, where
13,905 normal ECGs are easy for every model. This asks the harder question
directly: given one AF and one PAC recording, how often does each model
score the AF one higher? 1 - AUC is the share of AF-PAC pairs ranked the
wrong way round. Uses test 11's saved per-recording probabilities only.
Paired bootstrap by patient (2000 resamples).
"""

from outputs import write_path      # committed outputs go to rerun/ unless --force
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from afib_comparison_test import boot_ci

if __name__ == "__main__":
    d = pd.read_csv("results/afib_comparison_sph_recordings.csv")
    s = d[d.label.isin(["A", "PAC"])].dropna(subset=["prob_ours", "prob_ribeiro", "prob_cnn"]).reset_index(drop=True)
    y, g = (s.label == "A").values, s.patient.values
    ours = s.prob_ours.values
    rows = [{"model": "this model", "AF vs PAC AUC": roc_auc_score(y, ours), "pairs ranked wrong": 1 - roc_auc_score(y, ours)}]
    for name, col in [("Ribeiro's network", "prob_ribeiro"), ("same-data CNN", "prob_cnn")]:
        p = s[col].values
        a = roc_auc_score(y, p)
        lo, hi = boot_ci(lambda i: roc_auc_score(y[i], p[i]) - roc_auc_score(y[i], ours[i]), g)
        rlo, rhi = boot_ci(lambda i: (1 - roc_auc_score(y[i], ours[i])) / (1 - roc_auc_score(y[i], p[i])), g)
        rows.append({"model": name, "AF vs PAC AUC": a, "pairs ranked wrong": 1 - a,
                     "AUC, network - this model": a - roc_auc_score(y, ours), "diff lo": lo, "diff hi": hi,
                     "error ratio, this model / network": (1 - roc_auc_score(y, ours)) / (1 - a),
                     "ratio lo": rlo, "ratio hi": rhi})
    out = pd.DataFrame(rows)
    print(f"SPH: {y.sum()} AF, {(~y).sum()} PAC")
    print(out.round(4).to_string(index=False))
    out.to_csv(write_path("results/afib_comparison_afpac.csv"), index=False)
