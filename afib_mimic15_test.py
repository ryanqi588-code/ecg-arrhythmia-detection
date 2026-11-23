"""Test 15: the hybrid with a confident-call override, on 2000 new PAC and 1000 new AF patients.

Test 14 (MIMIC-IV-ECG): test 13's hybrid cut PAC false alarms (14.3% ->
8.8%) but lost 0.7 points of sensitivity (95% CI -2.5 to +1.1), failing the
-2 point bar. Fix:

  AF if (interpretable >= 0.0567 and CNN >= 0.06308)   [test 13's hybrid]
     or interpretable >= 0.9                             [confident: no CNN veto]

How 0.9 was chosen, in full: the rule fixed first in afib_hybrid2_dev.py
(the highest override threshold keeping development sensitivity per source
at least the interpretable model's) gave 0.46, which undoes nearly all of
the PAC gain (preview: MIMIC PAC 15.7%). Looking at the trade-off on test
14's recordings (already used) and the development grid, 0.9 kept most of
the PAC gain (dev PAC 7.0% vs 11.8% alone; test 14 preview: sensitivity
87.1% vs 85.3%, PAC 10.5% vs 14.3%). Since test 14's data informed the
choice, the test uses only new patients.

Data: MIMIC-IV-ECG, chosen from the machine reports only by test 14's rules
(afib_mimic_select.py test15; results/mimic15_selected.txt), one recording
per patient, seed 1, no patient from test 14: 2000 PAC, 1000 AF patients.

Primary, against the interpretable model alone (cutoff 0.4309), paired
bootstrap by patient (2000 resamples):
  1. PAC recordings called AF, override hybrid - alone: 95% interval below 0
  2. AF sensitivity, override hybrid - alone: lower bound above -0.02
The override hybrid replaces test 13's hybrid as the offered alternative
only if both hold. Secondary: test 13's hybrid on the same recordings.
Run once.
"""

from outputs import write_path      # committed outputs go to rerun/ unless --force
import pandas as pd

from af import auc
from afib_comparison_test import boot_ci, rate_str
from afib_fa_dev import fit, with_unmeasured_fix
from afib_hybrid_dev import ADOPTED, aligned_dev
from afib_hybrid_test import ADOPTED_CUTOFF
from afib_mimic_test import mimic_features
from af import model_inputs

T_LR, T_CNN, T_HIGH = 0.0567, 0.06308, 0.9

if __name__ == "__main__":
    d = aligned_dev()
    model, c_alone = fit(d, ADOPTED, True)
    assert abs(c_alone - ADOPTED_CUTOFF) < 1e-9, c_alone
    th = pd.read_csv("results/afib_hybrid_thresholds.csv").iloc[0]
    assert abs(th.t_lr - T_LR) < 1e-4 and abs(th.t_cnn - T_CNN) < 1e-5
    t_lr, t_cnn = float(th.t_lr), float(th.t_cnn)

    t, skipped = mimic_features("mimic15_selected")
    for c in ["log_f_wave", "f_ratio_II", "f_ratio_V1"]:
        t[c] = t[c].fillna(d[c].median())
    t["p_unmeasured"] = (t.p_match_mean == 0).astype(float)
    t["prob_lr"] = model.predict_proba(model_inputs(with_unmeasured_fix(d, t)[1], ADOPTED))[:, 1]
    cnn = pd.read_csv("results/compare_probs_mimic15.csv")
    cnn["record"] = cnn.record.astype(str)
    t = t.merge(cnn[["record", "prob_cnn"]], on="record", how="left")
    assert t.prob_cnn.notna().all() and t.prob_lr.notna().all()
    alone = t.prob_lr.values >= c_alone
    hybrid = (t.prob_lr.values >= t_lr) & (t.prob_cnn.values >= t_cnn)
    override = hybrid | (t.prob_lr.values >= T_HIGH)
    y, k, g = (t.label == "A").values, (t.label == "PAC").values, t.patient.values
    print(f"{len(t)} scored {t.label.value_counts().to_dict()}; unscorable: {skipped}")
    print(f"interpretable AUC (AF vs PAC) {auc(t.prob_lr.values, y):.4f}")

    out = []
    for name, called in [("alone", alone), ("hybrid (test 13)", hybrid), ("hybrid + override", override)]:
        r = {"model": name, "AF caught": rate_str(called[y]), "PAC called AF": rate_str(called[k])}
        out.append(r)
        print(r)
    diffs = {}
    for name, a, b in [("override_pac_fa", override, alone), ("override_sens", override, alone),
                       ("hybrid_pac_fa", hybrid, alone), ("hybrid_sens", hybrid, alone)]:
        m = k if name.endswith("pac_fa") else y
        diffs[name] = (a[m].mean() - b[m].mean(), *boot_ci(lambda i: a[i][m[i]].mean() - b[i][m[i]].mean(), g))
    ok1 = diffs["override_pac_fa"][2] < 0
    ok2 = diffs["override_sens"][1] > -0.02
    for name, (v, lo, hi) in diffs.items():
        print(f"{name}: minus alone {v:+.4f} (95% CI {lo:+.4f} to {hi:+.4f})")
    print(f"PRIMARY 1 (PAC false alarms lower): {'yes' if ok1 else 'NO'}; "
          f"PRIMARY 2 (sensitivity non-inferior): {'yes' if ok2 else 'NO'}; "
          f"VERDICT: {'override hybrid offered as the alternative' if ok1 and ok2 else 'override hybrid not confirmed'}")
    pd.DataFrame(out).to_csv(write_path("results/afib_mimic15_test.csv"), index=False)
    pd.DataFrame([{"comparison": n, "diff": v, "lo": lo, "hi": hi} for n, (v, lo, hi) in diffs.items()]
                 ).assign(primary1=ok1, primary2=ok2).to_csv(write_path("results/afib_mimic15_test_diffs.csv"), index=False)
    t.assign(called_alone=alone, called_hybrid=hybrid, called_override=override)[
        ["record", "patient", "label", "prob_lr", "prob_cnn", "called_alone", "called_hybrid", "called_override"]
    ].to_csv(write_path("results/afib_mimic15_test_recordings.csv"), index=False)
