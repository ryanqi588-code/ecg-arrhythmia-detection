"""Who the clinical data covers, and how the final models do by sex and age group.

An analysis of the saved outputs of tests 9 and 10 (SPH) after the fact;
nothing is refitted. Subgroups by sex and age group (18-49, 50-64, 65-79,
80+, from SPH's metadata.csv). Also the age and sex of every development
and test source, from their headers or metadata.
"""

import os

import numpy as np
import pandas as pd

from af import auc
from afib_sph_test import SPH_DIR, wilson

AGE_BINS = [18, 50, 65, 80, 200]
AGE_LABELS = ["18-49", "50-64", "65-79", "80+"]
AF_CUTOFF = 0.473559896031711          # the final AF model's cutoff in test 9 (printed by afib_sph_test.py)


def rate(hits):
    hits = np.asarray(hits, dtype=bool)
    if len(hits) == 0:
        return "–"
    lo, hi = wilson(int(hits.sum()), len(hits))
    return f"{hits.mean():.1%} ({lo:.0%}–{hi:.0%}), n={len(hits)}"


def header_demographics(paths):
    ages, sexes = [], []
    for p in paths:
        for line in open(p + ".hea"):
            if line.startswith("# Age:"):
                a = line.split(":")[1].strip()
                ages.append(float(a) if a.replace(".", "").isdigit() else np.nan)
            elif line.startswith("# Sex:"):
                sexes.append(line.split(":")[1].strip()[:1].upper())
    return pd.Series(ages, dtype=float), pd.Series(sexes)


def describe(name, ages, female_share, n):
    q = ages.quantile([0.25, 0.5, 0.75])
    return {"data": name, "recordings": n, "age median (IQR)": f"{q[0.5]:.0f} ({q[0.25]:.0f}–{q[0.75]:.0f})",
            "age missing": int(ages.isna().sum()), "female": f"{female_share:.0%}"}


if __name__ == "__main__":
    meta = pd.read_csv(os.path.join(SPH_DIR, "metadata.csv"))[["ECG_ID", "Age", "Sex"]]
    t9 = pd.read_csv("results/afib_sph_test_recordings.csv").merge(meta, left_on="record", right_on="ECG_ID")
    t9["called"] = t9.prob_new >= AF_CUTOFF
    t10 = pd.read_csv("results/afib_flutter_test_recordings.csv").merge(meta, left_on="record", right_on="ECG_ID")
    for t in (t9, t10):
        t["age_group"] = pd.cut(t.Age, AGE_BINS, right=False, labels=AGE_LABELS)

    rows = []
    for kind, groups in [("all", [("all", slice(None))]),
                         ("sex", [(s, None) for s in ["F", "M"]]),
                         ("age", [(a, None) for a in AGE_LABELS])]:
        for g, _ in groups:
            sel9 = np.ones(len(t9), bool) if kind == "all" else (t9.Sex == g if kind == "sex" else t9.age_group == g).values
            sel10 = np.ones(len(t10), bool) if kind == "all" else (t10.Sex == g if kind == "sex" else t10.age_group == g).values
            a, pac, n = (sel9 & (t9.label == v).values for v in ["A", "PAC", "N"])
            af = t9.label.values == "A"
            fl, st = (sel10 & (t10.label == v).values for v in ["AFL", "ST"])
            rows.append({
                "subgroup": f"{kind}: {g}",
                "AF caught (test 9)": rate(t9.called[a]),
                "PAC called AF": rate(t9.called[pac]),
                "normal called AF": rate(t9.called[n]),
                "AUC (test 9)": f"{auc(t9.prob_new.values[sel9], af[sel9]):.3f}" if af[sel9].any() else "–",
                "flutter caught as AF or flutter (test 10)": rate(t10.af_or_flutter[fl]),
                "sinus tach flagged (test 10)": rate(t10.af_or_flutter[st]),
            })
    sub = pd.DataFrame(rows)
    sub.to_csv("results/afib_subgroups.csv", index=False)
    pd.set_option("display.width", 250, "display.max_colwidth", 40)
    print(sub.to_string(index=False))

    # Age and sex by data source.
    demo = []
    from af import cpsc_selected
    from afib_clinical_dev import dev_recordings
    cpsc_paths = [p for p, _ in dev_recordings()]
    ages, sexes = header_demographics(cpsc_paths)
    demo.append(describe("CPSC 2018 + extra (development)", ages, (sexes == "F").mean(), len(cpsc_paths)))
    db = pd.read_csv("data/ptbxl/ptbxl_database.csv", index_col="filename_hr")
    ptb = [l.split()[0] for f in ["SELECTED", "SELECTED_FLUTTER_DEV"] for l in open(f"data/ptbxl/{f}")]
    p = db.loc[ptb]
    demo.append(describe("PTB-XL (development)", p.age.where(p.age < 300), (p.sex == 1).mean(), len(p)))
    ch = [f"data/chapman/{os.path.basename(l.split()[0])}" for f in ["SELECTED", "SELECTED_FLUTTER_DEV"]
          for l in open(f"data/chapman/{f}")]
    ages, sexes = header_demographics(ch)
    demo.append(describe("Chapman-Shaoxing (development)", ages, (sexes == "F").mean(), len(ch)))
    for name, t in [("SPH test 9", t9), ("SPH test 10 (new recordings)", t10[t10.label != "A"])]:
        demo.append(describe(name, t.Age, (t.Sex == "F").mean(), len(t)))
    demo = pd.DataFrame(demo)
    demo.to_csv("results/demographics.csv", index=False)
    print()
    print(demo.to_string(index=False))
    print("\nPTB-XL recording dates:", p.recording_date.min(), "to", p.recording_date.max())
