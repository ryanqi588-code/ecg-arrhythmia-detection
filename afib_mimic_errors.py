"""Why the AF model misses AF on MIMIC-IV-ECG (tests 14 and 15): an analysis after the fact.

Nothing is refitted or changed. Uses the saved recordings of tests 14 and 15
(results/afib_mimic*_recordings.csv), recomputes the model's features for
their AF recordings, and breaks the misses down by
  - what else the machine report says (ventricular rate, PVCs, aberrant
    conduction, bundle branch block, ...),
  - the features (missed vs caught AF, against development AF),
and plots example missed strips (figures/mimic_missed_af.png).
"""

from outputs import write_path      # committed outputs go to rerun/ unless --force
import re

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from afib_hybrid_dev import aligned_dev
from afib_mimic_select import MIMIC_DIR, load_mimic, read_selected, report_text
from afib_mimic_test import mimic_features

FEATS = ["cv", "pnn50", "entropy_norm", "p_match_regular", "p_match_mean", "log_f_wave", "f_ratio_II",
         "f_ratio_V1"]

CATEGORIES = {
    "rapid ventricular response": r"rapid ventricular response",
    "slow ventricular response": r"slow ventricular response",
    "PVCs or aberrant conduction": r"pvc|ventricular premature|aberra",
    "bundle branch block / wide QRS": r"bundle branch block|\brbbb\b|\blbbb\b|intraventricular conduction|wide qrs",
    "low voltage": r"low voltage|low qrs",
    "no other rhythm statement": r"^atrial fibrillation\.? \|",
}


def af_rows():
    sel = {}
    for name in ["mimic_selected", "mimic15_selected"]:
        for path, subject, label in read_selected(name):
            if label == "A":
                sel[path.split("/")[-1]] = path
    calls = pd.concat([pd.read_csv("results/afib_mimic_test_recordings.csv"),
                       pd.read_csv("results/afib_mimic15_test_recordings.csv")])
    calls["record"] = calls.record.astype(str)
    calls = calls[calls.label == "A"]
    m = pd.read_csv(f"{MIMIC_DIR}/machine_measurements.csv", low_memory=False)
    m = m[m.study_id.astype(str).isin(calls.record)]
    rep = pd.DataFrame({"report": report_text(m).values, "machine_hr": 60000 / m.rr_interval.values},
                       index=m.study_id.astype(str).values)
    return calls.join(rep, on="record"), sel


if __name__ == "__main__":
    a, paths = af_rows()
    a["missed"] = ~a.called_alone
    c_cnn = float(pd.read_csv("results/afib_cnn_dev.csv").cutoff[0])
    print(f"{len(a)} AF recordings scored in tests 14-15; missed by the model alone: {a.missed.sum()} "
          f"({a.missed.mean():.1%}); of these, also missed by the CNN at its own cutoff ({c_cnn:.3f}): "
          f"{(a.missed & (a.prob_cnn < c_cnn)).sum()}; CNN misses overall: {(a.prob_cnn < c_cnn).mean():.1%}")
    hr = a.machine_hr.where(a.machine_hr.between(20, 250))
    for lo, hi in [(0, 60), (60, 80), (80, 100), (100, 250)]:
        k = hr.between(lo, hi, inclusive="left")
        print(f"machine heart rate {lo}-{hi}: n={k.sum()}, missed {a.missed[k].mean():.1%}")

    rows = []
    for cat, rx in CATEGORIES.items():
        k = a.report.str.contains(rx, regex=True)
        rows.append({"report says": cat, "n": int(k.sum()), "missed": a.missed[k].mean(),
                     "missed (rest)": a.missed[~k].mean()})
    cats = pd.DataFrame(rows)
    print("\n" + cats.round(3).to_string(index=False))

    # Features: recompute for the AF recordings only.
    feats, _ = mimic_features("mimic_selected")
    f2, _ = mimic_features("mimic15_selected")
    feats = pd.concat([feats, f2])
    feats = feats[feats.label == "A"].set_index("record")
    a = a.join(feats[FEATS], on="record")
    d = aligned_dev()
    dev_af = d[d.af]
    comp = pd.DataFrame({"development AF": dev_af[FEATS].median(), "MIMIC AF caught": a[~a.missed][FEATS].median(),
                         "MIMIC AF missed": a[a.missed][FEATS].median()})
    print("\nMedians:\n" + comp.round(3).to_string())
    print(f"\nmissed with CV < 0.08 (fairly regular): {(a[a.missed].cv < 0.08).mean():.1%}; "
          f"caught: {(a[~a.missed].cv < 0.08).mean():.1%}; development AF: {(dev_af.cv < 0.08).mean():.1%}")
    print(f"missed with p_match_mean >= 0.8: {(a[a.missed].p_match_mean >= 0.8).mean():.1%}; "
          f"caught: {(a[~a.missed].p_match_mean >= 0.8).mean():.1%}; development AF: "
          f"{(dev_af.p_match_mean >= 0.8).mean():.1%}")
    a.drop(columns=["report"]).to_csv(write_path("results/afib_mimic_errors_af.csv"), index=False)
    cats.to_csv(write_path("results/afib_mimic_errors_categories.csv"), index=False)
    comp.to_csv(write_path("results/afib_mimic_errors_features.csv"))

    # Example missed strips: lead II, 6 drawn at random (seed 0).
    ex = a[a.missed].sample(6, random_state=0)
    fig, axes = plt.subplots(6, 1, figsize=(12, 11))
    for ax, (_, r) in zip(axes, ex.iterrows()):
        sig, fs = load_mimic(f"{MIMIC_DIR}/{paths[r.record]}")
        t = np.arange(len(sig)) / fs
        ax.plot(t, sig[:, 1], lw=0.7, color="black")
        stmt = re.sub(r"( \| )+", "; ", r.report).strip("; ")[:110]
        ax.set_title(f"{r.record}: P(AF) {r.prob_lr:.2f}, CV {r.cv:.3f}, p_match_mean {r.p_match_mean:.2f}  |  "
                     f"{stmt}", fontsize=8, loc="left")
        ax.set_yticks([])
    axes[-1].set_xlabel("seconds (lead II)")
    fig.tight_layout()
    fig.savefig(write_path("figures/mimic_missed_af.png"), dpi=110)
