"""Figure: two SPH ECG strips with each feature's contribution (adopted model, via predict.py).

Examples chosen by a fixed rule, not by eye: every SPH AF and PAC
recording (test 9's set; the adopted model was not trained on SPH) is
scored by predict.py, then
  - caught AF: the AF recording called AF with the median probability
    among AF recordings called AF;
  - PAC false alarm: the PAC recording called AF with the median
    probability among PAC recordings called AF.
Writes results/afib_example_scores.csv (cached), figures/afib_examples.png, and print versions
figures/Figure_1.pdf (vector) and figures/Figure_1.tiff (600 dpi), sized for a 17 cm page width.
"""

import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from afib_sph_test import FS, load_sph, sph_selected
from ecg import bandpass, detect_peaks_multilead, flag_ectopic
from predict import I_II_V1, load_model, predict

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
    plt.rcParams.update({"font.size": 11, "pdf.fonttype": 42, "axes.edgecolor": MUTED, "axes.labelcolor": INK,
                         "xtick.color": MUTED, "ytick.color": INK, "text.color": INK})
    fig = plt.figure(figsize=(9.5, 7.8))
    gs = fig.add_gridspec(5, 2, width_ratios=[2.1, 1], height_ratios=[1, 1, 0.32, 1, 1], hspace=0.25)
    lim = 0
    outs = []
    for title, r in picks:
        o = predict(load_sph(r.record), FS, model)
        outs.append((title, r, o))
        lim = max(lim, max(abs(c["contribution"]) for c in o["contributions"]))
    for row, (title, r, o) in enumerate(outs):
        sig = load_sph(r.record)
        t = np.arange(len(sig)) / FS
        # The beats the model found (as in af.clinical_features): multi-lead detection on I, II, V1;
        # beats flagged as ventricular ectopic have their RR intervals left out of the rhythm features.
        peaks = detect_peaks_multilead(sig[:, I_II_V1], FS)
        ectopic = flag_ectopic(sig[:, 1], FS, peaks)
        for k, (lead, name) in enumerate([(1, "II"), (6, "V1")]):
            ax = fig.add_subplot(gs[3 * row + k, 0])
            x = bandpass(sig[:, lead], FS)
            lo, hi = np.percentile(x, [0.2, 99.8])
            pad = (hi - lo) * 0.08
            ax.plot(t, x, color=INK, linewidth=0.6)
            top = hi + 3 * pad
            ax.plot(peaks[~ectopic] / FS, np.full((~ectopic).sum(), top), "v", color=INK, markersize=4.5,
                    label="beat counted" if row == 0 and k == 0 else None)
            ax.plot(peaks[ectopic] / FS, np.full(ectopic.sum(), top), "v", markerfacecolor="white",
                    markeredgecolor=INK, markersize=4.5,
                    label="ventricular ectopic (its RR intervals left out)" if row == 0 and k == 0 else None)
            ax.set_ylim(lo - pad, top + 2 * pad)
            ax.set_xlim(0, 10)
            ax.set_yticks([])
            ax.set_ylabel(name, rotation=0, ha="right", va="center", fontsize=11)
            ax.grid(axis="x", color=GRID, linewidth=0.6)
            for s in ["top", "right", "left"]:
                ax.spines[s].set_visible(False)
            if k == 0:
                ax.set_title(f"{title}  (SPH {r.record})", loc="left", fontsize=11.5, fontweight="bold")
                ax.set_xticklabels([])
            if row == 1 and k == 1:
                ax.set_xlabel("Time (s)")
            if row == 0 and k == 0:
                ax.legend(loc="upper right", bbox_to_anchor=(1, 1.42), ncol=2, frameon=False, fontsize=10,
                          handletextpad=0.3, columnspacing=1.2)
        print(f"SPH {r.record}: {len(peaks)} beats found, {ectopic.sum()} flagged ventricular ectopic")

        c = {x["feature"]: x for x in o["contributions"]}
        vals = [c[f]["contribution"] for f in order]
        ax = fig.add_subplot(gs[3 * row:3 * row + 2, 1])
        y = np.arange(len(order))[::-1]
        ax.barh(y, vals, height=0.62, color=[TOWARDS if v > 0 else AGAINST for v in vals])
        ax.axvline(0, color=MUTED, linewidth=0.8)
        ax.set_yticks(y)
        ax.set_yticklabels([LABELS[f] + (" (filled)" if c[f]["filled"] else "") for f in order], fontsize=10)
        ax.set_xlim(-lim * 1.15, lim * 1.15)
        ax.grid(axis="x", color=GRID, linewidth=0.6)
        ax.set_axisbelow(True)
        for s in ["top", "right", "left"]:
            ax.spines[s].set_visible(False)
        ax.tick_params(axis="y", length=0)
        ax.set_title(f"probability of AF {o['probability']:.3f} (AF if ≥ {o['cutoff']:.3f})\n"
                     f"score {o['score']:+.2f} = intercept {o['intercept']:+.2f} + bars", loc="right",
                     fontsize=10, color=MUTED)
        if row == 1:
            ax.set_xlabel("Contribution (log-odds)\n← against AF  |  towards AF →")
    fig.subplots_adjust(left=0.045, right=0.975, top=0.91, bottom=0.09, wspace=0.62)
    os.makedirs("figures", exist_ok=True)
    fig.savefig("figures/afib_examples.png", dpi=130, facecolor="white")
    fig.savefig("figures/Figure_1.pdf", facecolor="white")
    fig.savefig("figures/Figure_1.tiff", dpi=600, facecolor="white", pil_kwargs={"compression": "tiff_lzw"})
    for title, r, o in outs:
        top = ", ".join(f"{x['feature']} {x['contribution']:+.2f}" for x in o["contributions"][:4])
        print(f"{title}: SPH {r.record}, p={o['probability']:.3f}, intercept {o['intercept']:+.2f}; top: {top}")
