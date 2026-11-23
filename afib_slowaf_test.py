"""Test 16: does P-wave presence fix the missed slow AF on new MIMIC-IV-ECG patients?

The fix (afib_slowaf_dev.py, candidate 7): the adopted model's features plus
per-beat P-wave presence in II and V1 (ecg.p_presence, with a missing
indicator) and a gated P-wave match (p_match_mean scaled down where P
waves don't stand out from the baseline), trained on the original three
development sources plus 4439 MIMIC development recordings from patients
in no test. The cutoff rule is unchanged: the strictest per-source
95%-sensitivity cutoff on CPSC, PTB-XL and Chapman.

In development (CV) on MIMIC it caught 86.5% of AF against 84.1% (slow
AF under 80 a minute: 79.5% against 74.0%), with PAC false alarms 15.1%
against 14.3%; at matched PAC false alarms it caught about 3 points more
AF. Previews on used data: SPH PAC false alarms 8.9% -> 8.2%, Ningbo
25.2% -> 27.2%, Chapman (CV) 18.2% -> 21.7%.

Data: MIMIC-IV-ECG, chosen from the machine reports only by the
development set's rules (afib_slowaf_dev.select_dev), one recording per
patient, seed 3, leaving out every patient of tests 14 and 15 and of the
development set: 1000 AF, 2000 PAC and 2000 sinus-rhythm patients
(results/mimic16_selected.txt).

Primary, against the adopted model (test 12, cutoff 0.4309), paired
bootstrap by patient (2000 resamples):
  1. AF sensitivity, new - adopted: 95% interval entirely above 0
  2. PAC recordings called AF, new - adopted: upper bound below +0.02
If both hold, the new model replaces the adopted model as the project's
recommended model (the numbers of tests 9-12 stay with the version they
tested); otherwise the adopted model stays.
Secondary: AF sensitivity under 80 a minute (machine heart rate); sinus
false alarms; AF-vs-PAC AUC of both; sensitivity at matched PAC false
alarms. Run once.
"""

from outputs import write_path      # committed outputs go to rerun/ unless --force
import sys

import numpy as np
import pandas as pd

import afib_fa_dev
from af import auc, model_inputs
from afib_comparison_test import boot_ci, rate_str
from afib_fa_dev import fit, with_unmeasured_fix
from afib_hybrid_dev import ADOPTED
from afib_hybrid_test import ADOPTED_CUTOFF
from afib_slowaf_dev import ORIGINAL, SLOWAF, build_dev, mimic_units, prepare, select_dev

EXCLUDE = ("mimic_selected", "mimic15_selected", "mimic_dev_selected")

if __name__ == "__main__":
    if "select" in sys.argv:
        select_dev("mimic16_selected", seed=3, n={"A": 1000, "PAC": 2000, "N": 2000}, exclude=EXCLUDE)
        sys.exit()
    u, medians = build_dev()
    afib_fa_dev.CUTOFF_SOURCES = ORIGINAL
    old, c_old = fit(u[u.source.isin(ORIGINAL).values], ADOPTED, True)
    assert abs(c_old - ADOPTED_CUTOFF) < 1e-9, c_old
    new, c_new = fit(u, SLOWAF, True)
    print(f"cutoffs: adopted {c_old:.4f}, new {c_new:.4f}")

    t = prepare(mimic_units("mimic16_selected", "mimic16", "data/mimic16_units.pkl"), medians)
    n_sel = pd.Series([l.split()[2] for l in open("results/mimic16_selected.txt")]).value_counts()
    print(f"selected {n_sel.to_dict()}; scored {t.label.value_counts().to_dict()}")
    p_old = old.predict_proba(model_inputs(with_unmeasured_fix(u[u.source.isin(ORIGINAL).values], t)[1],
                                           ADOPTED))[:, 1]
    p_new = new.predict_proba(model_inputs(with_unmeasured_fix(u, t)[1], SLOWAF))[:, 1]
    a_old, a_new = p_old >= c_old, p_new >= c_new
    y, k, n = (t.label == "A").values, (t.label == "PAC").values, (t.label == "N").values
    slow, g = t.machine_hr.values < 80, t.patient.values

    out = []
    for name, called, prob in [("adopted", a_old, p_old), ("new", a_new, p_new)]:
        r = {"model": name, "AF caught": rate_str(called[y]), "AF under 80/min caught": rate_str(called[y & slow]),
             "PAC called AF": rate_str(called[k]), "sinus called AF": rate_str(called[n]),
             "AUC AF vs PAC": auc(prob[y | k], y[y | k])}
        out.append(r)
        print(r)
    thr = np.quantile(p_new[k], 1 - a_old[k].mean())
    print(f"new model at the adopted model's PAC false-alarm rate ({a_old[k].mean():.3f}): "
          f"AF caught {(p_new[y] >= thr).mean():.3f}")

    def diff(a, b, m):
        return (a[m].mean() - b[m].mean(), *boot_ci(lambda i: a[i][m[i]].mean() - b[i][m[i]].mean(), g))
    diffs = {"sens": diff(a_new, a_old, y), "pac_fa": diff(a_new, a_old, k),
             "sens_slow": diff(a_new, a_old, y & slow), "sinus_fa": diff(a_new, a_old, n)}
    ok1 = diffs["sens"][1] > 0
    ok2 = diffs["pac_fa"][2] < 0.02
    for name, (v, lo, hi) in diffs.items():
        print(f"{name}: new - adopted {v:+.4f} (95% CI {lo:+.4f} to {hi:+.4f})")
    print(f"PRIMARY 1 (sensitivity higher): {'yes' if ok1 else 'NO'}; "
          f"PRIMARY 2 (PAC false alarms not more than 2 points higher): {'yes' if ok2 else 'NO'}; "
          f"VERDICT: {'new model recommended' if ok1 and ok2 else 'adopted model stays'}")
    pd.DataFrame(out).to_csv(write_path("results/afib_slowaf_test.csv"), index=False)
    pd.DataFrame([{"comparison": c, "diff": v, "lo": lo, "hi": hi} for c, (v, lo, hi) in diffs.items()]
                 ).assign(primary1=ok1, primary2=ok2).to_csv(write_path("results/afib_slowaf_test_diffs.csv"), index=False)
    t.assign(prob_adopted=p_old, prob_new=p_new, called_adopted=a_old, called_new=a_new)[
        ["record", "patient", "label", "machine_hr", "prob_adopted", "prob_new", "called_adopted", "called_new"]
    ].to_csv(write_path("results/afib_slowaf_test_recordings.csv"), index=False)
