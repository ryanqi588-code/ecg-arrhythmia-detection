"""Test 17: stop calling AF when P waves can't be measured ("needs review"), on fresh CODE-15% ECGs.

After test 12's fix, recordings whose P waves can't be measured (fewer
than 3 regular, non-ectopic beats: p_unmeasured = 1) are still mostly
called AF: 202 of 266 such non-AF recordings in test 12. Development on
the used data (CPSC, PTB-XL, Chapman, SPH, Ningbo, CODE-15% part 0, CODE
test set; 5-fold CV by patient) found that within this group no
available feature separates AF from non-AF well (the adopted model's AUC
there is 0.64), and that no model change helped: dropping the flag (fill
only) lost a quarter of these AF and raised false alarms elsewhere (PAC
16.2% -> 20.1%), and retraining on all seven sources removed only 35 of
286 such false alarms. Only abstaining worked: CODE-15% part 0 non-AF
false alarms 3.62% -> 2.33%, with 84 of 84 unmeasurable AF sent to review.

The rule: the adopted model (results/final_af_model.csv, cutoff 0.4309,
unchanged), except that a recording with p_unmeasured = 1 gets "needs
review" instead of a call. Recordings with too few beats stay unscored
as before. Scored with predict.score_features, from the published table.

Data, never opened: CODE-15% part 3 (exams_part3.hdf5, 20,000 exams,
Brazil, mostly primary care; AF label from exams.csv). Downloaded after
this file is committed and pushed.

Primary, against the adopted model on the recordings it scores, paired
bootstrap by patient (2000 resamples):
  1. non-AF recordings called AF, abstain - adopted: upper bound of the
     95% interval below -0.005 (false alarms fall by more than half a
     point; abstaining can only remove calls, so "below 0" would be
     automatic);
  2. recordings sent to review: Wilson 95% upper bound below 3% of the
     recordings the adopted model scores.
If both hold, "needs review" becomes part of the recommended model's
output (tests 9-16 keep their numbers). Otherwise the adopted model stays.
Secondary: AF sent to review (count and share); sensitivity with review
counted as a miss, and among recordings given a call; non-AF sent to
review; the adopted model's calls on the review group. Run once.

Previews on used data (`python afib_abstain_test.py preview`; part 1
reproduces test 12's adopted-model numbers exactly):
  CODE-15% part   false alarms adopted -> abstain (diff, 95% CI)   sent to review (upper)   AF sent to review
  0 (test 11)     3.6% -> 2.4% (-1.28, -1.44 to -1.12)            1.79% (1.99%)            12/424
  1 (test 12)     3.3% -> 2.2% (-1.09, -1.25 to -0.94)            1.48% (1.66%)            13/351
  2 (test 13)     3.8% -> 2.7% (-1.13, -1.28 to -0.97)            1.58% (1.76%)             6/409
"""

from outputs import write_path      # committed outputs go to rerun/ unless --force
import sys

import numpy as np
import pandas as pd

from afib_comparison_test import boot_ci, rate_str
from afib_sph_test import wilson
from afib_hybrid_test import code15_features
from predict import load_model, score_features

PART = "exams_part3.hdf5"


def units(part):
    cache = f"data/code15_{part.split('.')[0]}_units.pkl"
    try:
        t, skipped = pd.read_pickle(cache)
    except FileNotFoundError:
        t, skipped = code15_features(part)
        pd.to_pickle((t, skipped), cache)
    return t, skipped


def calls(t, model):
    out = [score_features(r, model) for r in t.to_dict("records")]
    t = t.assign(prob=[o["probability"] for o in out], p_unmeasured=(t.p_match_mean == 0).values)
    t["adopted"] = t.prob >= model[2]
    t["review"] = t.p_unmeasured
    t["abstain"] = t.adopted & ~t.review
    return t


def evaluate(t, skipped, name):
    y, g = (t.label == "A").values, t.patient.values
    a, b, rv = t.adopted.values, t.abstain.values, t.review.values
    fa = (b[~y].mean() - a[~y].mean(), *boot_ci(lambda i: b[i][~y[i]].mean() - a[i][~y[i]].mean(), g))
    lo, hi = wilson(int(rv.sum()), len(rv))
    ok1, ok2 = fa[2] < -0.005, hi < 0.03
    r = {"set": name, "scored": len(t), "unscored (too few beats)": skipped,
         "AF": int(y.sum()), "non-AF": int((~y).sum()),
         "non-AF called AF, adopted": rate_str(a[~y]), "non-AF called AF, abstain": rate_str(b[~y]),
         "difference (95% CI)": f"{fa[0]:+.4f} ({fa[1]:+.4f} to {fa[2]:+.4f})",
         "sent to review": f"{rv.mean():.2%} ({lo:.2%}-{hi:.2%}) [{int(rv.sum())}/{len(rv)}]",
         "AF sent to review": rate_str(rv[y]), "non-AF sent to review": rate_str(rv[~y]),
         "AF caught, adopted": rate_str(a[y]), "AF caught, abstain (review = miss)": rate_str(b[y]),
         "AF caught among recordings given a call": rate_str(b[y & ~rv]),
         "review group called AF by adopted (non-AF / AF)": f"{int((a & rv & ~y).sum())} / {int((a & rv & y).sum())}",
         "PRIMARY 1": ok1, "PRIMARY 2": ok2}
    for k, v in r.items():
        print(f"  {k}: {v}")
    return r, fa


if __name__ == "__main__":
    model = load_model()
    if "preview" in sys.argv:
        rows = []
        p0 = pd.read_csv("results/afib_comparison_code15_recordings.csv")
        p0 = p0.assign(patient=p0.patient.astype(str))
        print("CODE-15% part 0 (test 11, used):")
        rows.append(evaluate(calls(p0, model), {"A": 22, "notAF": 1127}, "part 0 (used)")[0])
        for part, name in [("exams_part1.hdf5", "part 1 (test 12, used)"), ("exams_part2.hdf5", "part 2 (test 13, used)")]:
            print(f"CODE-15% {name}:")
            t, skipped = units(part)
            rows.append(evaluate(calls(t, model), skipped, name)[0])
        pd.DataFrame(rows).to_csv(write_path("results/afib_abstain_preview.csv"), index=False)
        sys.exit()

    t, skipped = units(PART)
    t = calls(t, model)
    print(f"CODE-15% part 3: {len(t)} scored; unscored {skipped}")
    r, fa = evaluate(t, skipped, "part 3 (test 17)")
    verdict = "needs-review output adopted" if r["PRIMARY 1"] and r["PRIMARY 2"] else "adopted model stays"
    print(f"VERDICT: {verdict}")
    pd.DataFrame([r]).assign(verdict=verdict).to_csv(write_path("results/afib_abstain_test.csv"), index=False)
    t[["record", "patient", "label", "prob", "p_unmeasured", "adopted", "abstain"]].to_csv(
        write_path("results/afib_abstain_test_recordings.csv"), index=False)
