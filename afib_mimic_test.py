"""Test 14: the hybrid against the interpretable model alone, on PACs from 2000 patients.

Test 13's PAC result came from INCART, where 2 of 14 patients supplied 410 of
the 447 PAC crops. This test repeats the comparison, with nothing changed, on
many more people: MIMIC-IV-ECG (Boston, USA; a new hospital, country and ECG
machine for this project), one 10 s 12-lead recording per patient, chosen
from outputs import write_path      # committed outputs go to rerun/ unless --force
from the machine's report only (afib_mimic_select.py,
results/mimic_selected.txt): 2000 patients whose report says sinus rhythm
with PACs and 1000 other patients whose report says atrial fibrillation.

Models, unchanged from test 13:
  alone  - the adopted interpretable model (test 12), cutoff 0.4309
  hybrid - interpretable probability >= 0.0567 and CNN probability >= 0.06308
Both are scored on the recordings the interpretable model can score;
recordings with missing samples (NaN) are counted as unscorable.

Primary, fixed before the signals are opened (paired bootstrap by patient,
2000 resamples):
  1. PAC recordings called AF, hybrid - alone: 95% interval entirely below 0
  2. AF sensitivity, hybrid - alone: lower bound above -0.02
Test 13's conclusion (the hybrid offered as an alternative) is confirmed on
a broad population only if both hold; otherwise the README says it was not.
Secondary: each model's rates with Wilson intervals; AUC of the
interpretable model; the hybrid's PAC false alarms against the interpretable
model alone at a cutoff giving the hybrid's sensitivity on this set (to see
whether the CNN does more than a stricter cutoff would).

Caveat fixed in advance: the labels are the ECG machine's statements, not a
cardiologist's; both models are compared on the same recordings.
Run once.
"""

import numpy as np
import pandas as pd

from af import auc
from afib_comparison_test import boot_ci, rate_str
from afib_fa_dev import fit, with_unmeasured_fix
from afib_hybrid_dev import ADOPTED, aligned_dev
from afib_hybrid_test import ADOPTED_CUTOFF
from afib_mimic_select import MIMIC_DIR, load_mimic, read_selected
from afib_pac_dev import unit_features
from af import model_inputs


def mimic_features(name="mimic_selected"):
    rows, skipped = [], {}
    for i, (path, subject, label) in enumerate(read_selected(name)):
        if i % 500 == 0:
            print(f"mimic {i} ...", flush=True)
        sig, fs = load_mimic(f"{MIMIC_DIR}/{path}")
        row = None if np.isnan(sig).any() else unit_features(sig, fs, label == "A", "multilead12")
        if row is None:
            skipped[label] = skipped.get(label, 0) + 1
        else:
            rows.append({**row, "record": path.split("/")[-1], "patient": f"mimic-{subject}", "label": label})
    return pd.DataFrame(rows), skipped


if __name__ == "__main__":
    d = aligned_dev()
    model, c_alone = fit(d, ADOPTED, True)
    assert abs(c_alone - ADOPTED_CUTOFF) < 1e-9, c_alone
    th = pd.read_csv("results/afib_hybrid_thresholds.csv").iloc[0]
    t_lr, t_cnn = float(th.t_lr), float(th.t_cnn)
    print(f"alone cutoff {c_alone:.4f}; hybrid: interpretable >= {t_lr:.4f} and CNN >= {t_cnn:.4g}")

    t, skipped = mimic_features()
    for c in ["log_f_wave", "f_ratio_II", "f_ratio_V1"]:
        t[c] = t[c].fillna(d[c].median())
    t["p_unmeasured"] = (t.p_match_mean == 0).astype(float)
    t["prob_lr"] = model.predict_proba(model_inputs(with_unmeasured_fix(d, t)[1], ADOPTED))[:, 1]
    cnn = pd.read_csv("results/compare_probs_mimic.csv")
    cnn["record"] = cnn.record.astype(str)
    t = t.merge(cnn[["record", "prob_cnn"]], on="record", how="left")
    assert t.prob_cnn.notna().all()
    alone = t.prob_lr.values >= c_alone
    hybrid = (t.prob_lr.values >= t_lr) & (t.prob_cnn.values >= t_cnn)
    y, k, g = (t.label == "A").values, (t.label == "PAC").values, t.patient.values
    print(f"\n{len(t)} scored {t.label.value_counts().to_dict()}; unscorable: {skipped}")
    print(f"interpretable AUC (AF vs PAC) {auc(t.prob_lr.values, y):.4f}")

    out = []
    for name, called in [("alone", alone), ("hybrid", hybrid)]:
        r = {"model": name, "AF caught": rate_str(called[y]), "PAC called AF": rate_str(called[k])}
        out.append(r)
        print(r)
    # Secondary: the interpretable model alone at a cutoff matching the hybrid's sensitivity here.
    c_match = np.quantile(t.prob_lr[y], 1 - hybrid[y].mean())
    matched = t.prob_lr.values >= c_match
    r = {"model": f"alone, cutoff {c_match:.3f} (hybrid's sensitivity)", "AF caught": rate_str(matched[y]),
         "PAC called AF": rate_str(matched[k])}
    out.append(r)
    print(r)

    diffs = {}
    diffs["pac_fa"] = (hybrid[k].mean() - alone[k].mean(),
                       *boot_ci(lambda i: hybrid[i][k[i]].mean() - alone[i][k[i]].mean(), g))
    diffs["sens"] = (hybrid[y].mean() - alone[y].mean(),
                     *boot_ci(lambda i: hybrid[i][y[i]].mean() - alone[i][y[i]].mean(), g))
    diffs["pac_fa_vs_matched"] = (hybrid[k].mean() - matched[k].mean(),
                                  *boot_ci(lambda i: hybrid[i][k[i]].mean() - matched[i][k[i]].mean(), g))
    ok1 = diffs["pac_fa"][2] < 0
    ok2 = diffs["sens"][1] > -0.02
    for name, (v, lo, hi) in diffs.items():
        print(f"{name}: hybrid - comparator {v:+.4f} (95% CI {lo:+.4f} to {hi:+.4f})")
    print(f"PRIMARY 1 (PAC false alarms lower): {'yes' if ok1 else 'NO'}; "
          f"PRIMARY 2 (sensitivity non-inferior): {'yes' if ok2 else 'NO'}; "
          f"VERDICT: {'test 13 confirmed on 2000 patients' if ok1 and ok2 else 'test 13 not confirmed'}")
    pd.DataFrame(out).to_csv(write_path("results/afib_mimic_test.csv"), index=False)
    pd.DataFrame([{"comparison": n, "diff": v, "lo": lo, "hi": hi} for n, (v, lo, hi) in diffs.items()]
                 ).assign(primary1=ok1, primary2=ok2).to_csv(write_path("results/afib_mimic_test_diffs.csv"), index=False)
    t.assign(called_alone=alone, called_hybrid=hybrid)[["record", "patient", "label", "prob_lr", "prob_cnn",
                                                        "called_alone", "called_hybrid"]].to_csv(
        write_path("results/afib_mimic_test_recordings.csv"), index=False)
