"""Figure for test 11: share of PAC recordings called AF by each model, at its own cutoff."""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from afib_sph_test import wilson

CUTOFFS = pd.read_csv("results/afib_cnn_dev.csv").cutoff[0]
MODELS = [("prob_ours", "This model (logistic, 10 features)", "#2a78d6"),
          ("prob_cnn", "CNN, same training data", "#eb6834"),
          ("prob_ribeiro", "Ribeiro et al. network", "#1baf7a")]

if __name__ == "__main__":
    from afib_comparison_test import OUR_CUTOFF_TEST9
    from af import cutoff_for_sensitivity
    dev = pd.read_csv("results/compare_probs_dev.csv")
    c_rib = min(cutoff_for_sensitivity(dev.prob_ribeiro.values[(dev.source == s).values],
                                       (dev.label == "A").values[(dev.source == s).values],
                                       dev.record.values[(dev.source == s).values]) for s in ["cpsc", "ptbxl", "chapman"])
    cut = {"prob_ours": OUR_CUTOFF_TEST9, "prob_cnn": CUTOFFS, "prob_ribeiro": c_rib}
    sets = [("SPH (537 PAC)", "sph"), ("Ningbo (1040 PAC)", "ningbo")]
    fig, ax = plt.subplots(figsize=(7, 4))
    fig.patch.set_facecolor("#fcfcfb")
    ax.set_facecolor("#fcfcfb")
    width = 0.26
    for j, (col, label, color) in enumerate(MODELS):
        for i, (_, name) in enumerate(sets):
            t = pd.read_csv(f"results/afib_comparison_{name}_recordings.csv")
            pac = t[t.label == "PAC"]
            k, n = int((pac[col] >= cut[col]).sum()), len(pac)
            lo, hi = wilson(k, n)
            x = i + (j - 1) * (width + 0.02)
            ax.bar(x, 100 * k / n, width, color=color, label=label if i == 0 else None, zorder=2)
            ax.plot([x, x], [100 * lo, 100 * hi], color="#3d3d3a", linewidth=1.2, zorder=3)
            ax.text(x, 100 * hi + 0.8, f"{100 * k / n:.1f}%", ha="center", va="bottom", fontsize=8, color="#3d3d3a")
    ax.set_xticks(range(len(sets)), [s for s, _ in sets])
    ax.set_ylabel("PAC recordings called AF (%)", color="#3d3d3a")
    ax.set_ylim(0, 32)
    ax.grid(axis="y", color="#e5e4df", linewidth=0.8, zorder=0)
    for side in ["top", "right", "left"]:
        ax.spines[side].set_visible(False)
    ax.tick_params(colors="#3d3d3a", length=0)
    ax.set_title("False alarms on atrial premature beats, each model at its own cutoff\n(lines: 95% intervals)",
                 fontsize=10, color="#1a1a19", loc="left")
    ax.legend(frameon=False, fontsize=8, loc="upper left")
    plt.tight_layout()
    plt.savefig("figures/afib_comparison.png", dpi=130, facecolor=fig.get_facecolor())
