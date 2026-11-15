"""Figure: two SPH ECG strips with each feature's contribution (adopted model, via predict.py).

Examples chosen by a fixed rule, not by eye: every SPH AF and PAC
recording (test 9's set; the adopted model was not trained on SPH) is
scored by predict.py, then
  - caught AF: the AF recording called AF with the median probability
    among AF recordings called AF;
  - PAC false alarm: the PAC recording called AF with the median
    probability among PAC recordings called AF.
Writes results/afib_example_scores.csv (cached) and figures/afib_examples.png.
"""

import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from afib_sph_test import FS, load_sph, sph_selected
from ecg import bandpass
from predict import load_model, predict

CACHE = "results/afib_example_scores.csv"
LABELS = {"cv": "RR variability (CV)", "nrmssd": "RR successive differences", "entropy_norm": "RR entropy",
          "tpr": "turning points", "pnn50": "pNN50", "p_match_regular": "P-wave match, lead II",
          "log_f_wave": "f-wave activity, V1", "p_match_mean": "P-wave match, 12 leads",
          "f_ratio_II": "f-wave ratio, II", "f_ratio_V1": "f-wave ratio, V1",
          "p_unmeasured": "P waves unmeasurable"}
TOWARDS, AGAINST = "#e34948", "#2a78d6"      # diverging pair (validated); red = towards AF
INK, MUTED, GRID = "#1f1f1e", "#6b6b68", "#e6e5e1"


def score_all(model):
    try:
        return pd.read_csv(CACHE)
    except FileNotFoundError:
        pass
    rows = []
    for i, (ecg_id, patient, label) in enumerate(r for r in sph_selected() if r[2] in ("A", "PAC")):
        if i % 200 == 0:
            print(f"{i} ...", flush=True)
        o = predict(load_sph(ecg_id), FS, model)
        rows.append({"record": ecg_id, "patient": patient, "label": label, "result": o["result"],
                     "probability": o.get("probability", np.nan)})
    d = pd.DataFrame(rows)
    d.to_csv(CACHE, index=False)
    return d


def median_pick(d):
    d = d.sort_values("probability").reset_index(drop=True)
    return d.iloc[(len(d) - 1) // 2]


if __name__ == "__main__":
    model = load_model()
    d = score_all(model)
    af_hit = d[(d.label == "A") & (d.result == "AF")]
    pac_fa = d[(d.label == "PAC") & (d.result == "AF")]
    print(f"AF caught {len(af_hit)}/{(d.label == 'A').sum()}; PAC called AF {len(pac_fa)}/{(d.label == 'PAC').sum()}")
    picks = [("Atrial fibrillation, correctly called AF", median_pick(af_hit)),
             ("Atrial premature beats, wrongly called AF", median_pick(pac_fa))]

    order = list(model[0].index)
    plt.rcParams.update({"font.size": 9, "axes.edgecolor": MUTED, "axes.labelcolor": INK,
                         "xtick.color": MUTED, "ytick.color": INK, "text.color": INK})
    fig, axes = plt.subplots(2, 2, figsize=(12, 6.4), gridspec_kw={"width_ratios": [2.1, 1]})
    lim = 0
    outs = []
    for title, r in picks:
        o = predict(load_sph(r.record), FS, model)
        outs.append((title, r, o))
        lim = max(lim, max(abs(c["contribution"]) for c in o["contributions"]))
    for row, (title, r, o) in enumerate(outs):
        sig = load_sph(r.record)
        t = np.arange(len(sig)) / FS
        ax = axes[row, 0]
        ax.plot(t, bandpass(sig[:, 1], FS), color=INK, linewidth=0.8)
        ax.set_xlim(0, 10)
        ax.set_yticks([])
        ax.grid(axis="x", color=GRID, linewidth=0.6)
        for s in ["top", "right", "left"]:
            ax.spines[s].set_visible(False)
        ax.set_title(f"{title}  (SPH {r.record}, lead II)", loc="left", fontsize=10, fontweight="bold")
        if row == 1:
            ax.set_xlabel("Time (s)")

        c = {x["feature"]: x for x in o["contributions"]}
        vals = [c[f]["contribution"] for f in order]
        ax = axes[row, 1]
        y = np.arange(len(order))[::-1]
        ax.barh(y, vals, height=0.62, color=[TOWARDS if v > 0 else AGAINST for v in vals])
        ax.axvline(0, color=MUTED, linewidth=0.8)
        ax.set_yticks(y)
        ax.set_yticklabels([LABELS[f] + (" (filled)" if c[f]["filled"] else "") for f in order], fontsize=8)
        ax.set_xlim(-lim * 1.15, lim * 1.15)
        ax.grid(axis="x", color=GRID, linewidth=0.6)
        ax.set_axisbelow(True)
        for s in ["top", "right", "left"]:
            ax.spines[s].set_visible(False)
        ax.tick_params(axis="y", length=0)
        ax.set_title(f"probability of AF {o['probability']:.3f} (AF if ≥ {o['cutoff']:.3f})\n"
                     f"score {o['score']:+.2f} = intercept {o['intercept']:+.2f} + sum of bars", loc="left",
                     fontsize=9, color=MUTED)
        if row == 1:
            ax.set_xlabel("Contribution (log-odds)   ← against AF  |  towards AF →")
    fig.tight_layout()
    os.makedirs("figures", exist_ok=True)
    fig.savefig("figures/afib_examples.png", dpi=130, facecolor="white")
    for title, r, o in outs:
        top = ", ".join(f"{x['feature']} {x['contribution']:+.2f}" for x in o["contributions"][:4])
        print(f"{title}: SPH {r.record}, p={o['probability']:.3f}, intercept {o['intercept']:+.2f}; top: {top}")
