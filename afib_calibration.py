"""Calibration check: do the AF models' probabilities match how often AF actually occurs?

An analysis of the saved test outputs (results/*_test_recordings.csv);
nothing is refitted and no model or cutoff is changed.

Every model was trained with balanced class weights (af.make_logreg), so
its probabilities assume AF and non-AF are equally common. Two checks:
  balanced  - raw probabilities against the test data reweighted so AF and
              non-AF count equally: the setting the model was built for
  at prevalence - probabilities shifted to the test set's own AF
              prevalence pi (logit + log(pi / (1 - pi))), against the
              actual outcomes
For each: mean predicted probability against the observed AF rate
(calibration in the large), Brier score, expected calibration error (ECE; 10 equal-width
bins, weighted by how many recordings fall in each), and calibration
intercept and slope (logistic regression of the outcome on the logit;
ideal 0 and 1; slope < 1 means overconfident, > 1 underconfident).
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression

TESTS = [  # (test, file, probability column, model)
    ("Georgia", "afib_georgia_test_recordings.csv", "prob_clinical", "clinical (lead II)"),
    ("PTB-XL", "afib_ptbxl_test_recordings.csv", "prob_clinical", "clinical (lead II)"),
    ("Chapman", "afib_chapman_test_recordings.csv", "prob_retrained", "retrained (CPSC + PTB-XL)"),
    ("SPH", "afib_sph_test_recordings.csv", "prob_previous", "retrained (CPSC + PTB-XL)"),
    ("SPH", "afib_sph_test_recordings.csv", "prob_new", "12-lead (final)"),
]
BINS = np.linspace(0, 1, 11)


def logit(p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def reliability(p, y, w):
    """Per-bin mean prediction, observed rate and total weight (weights w)."""
    idx = np.clip(np.digitize(p, BINS) - 1, 0, len(BINS) - 2)
    rows = []
    for b in range(len(BINS) - 1):
        m = idx == b
        if w[m].sum() > 0:
            rows.append((np.average(p[m], weights=w[m]), np.average(y[m], weights=w[m]), w[m].sum(), m.sum()))
    return np.array(rows)


def calibration(p, y, w):
    rel = reliability(p, y, w)
    ece = np.sum(rel[:, 2] * np.abs(rel[:, 0] - rel[:, 1])) / w.sum()
    brier = np.average((p - y) ** 2, weights=w)
    fit = LogisticRegression(C=1e6, max_iter=1000).fit(logit(p)[:, None], y, sample_weight=w)
    return {"mean_pred": np.average(p, weights=w), "observed": np.average(y, weights=w),
            "brier": brier, "ece": ece, "intercept": fit.intercept_[0], "slope": fit.coef_[0, 0]}


if __name__ == "__main__":
    rows, curves = [], {}
    for test, file, col, model in TESTS:
        d = pd.read_csv(f"results/{file}")
        y = d.af.values.astype(float)
        p = d[col].values
        pi = y.mean()
        balanced_w = np.where(y == 1, 0.5 / y.sum(), 0.5 / (1 - y).sum()) * len(y)
        p_prev = 1 / (1 + np.exp(-(logit(p) + np.log(pi / (1 - pi)))))
        ones = np.ones(len(y))
        b, a = calibration(p, y, balanced_w), calibration(p_prev, y, ones)
        rows.append({"test": test, "model": model, "n": len(y), "af_prevalence": pi,
                     **{f"balanced_{k}": v for k, v in b.items()},
                     **{f"prevalence_{k}": v for k, v in a.items()}})
        curves[(test, model)] = (reliability(p, y, balanced_w), reliability(p_prev, y, ones))
    out = pd.DataFrame(rows)
    out.to_csv("results/afib_calibration.csv", index=False)
    pd.set_option("display.width", 200)
    print(out.round(3).to_string(index=False))

    fig, axes = plt.subplots(1, 2, figsize=(11, 5))
    for (key, (bal, prev)), color in zip(curves.items(), ["tab:orange", "tab:green", "tab:purple", "tab:gray", "tab:blue"]):
        label = f"{key[0]}, {key[1]}"
        for ax, rel in zip(axes, [bal, prev]):
            ax.plot(rel[:, 0], rel[:, 1], "o-", color=color, label=label, markersize=4,
                    linewidth=2.5 if "final" in key[1] else 1)
    for ax, title in zip(axes, ["Raw probabilities, AF and non-AF weighted equally",
                                "Shifted to each test's AF prevalence"]):
        ax.plot([0, 1], [0, 1], ":", color="0.6")
        ax.set_xlabel("Predicted probability of AF")
        ax.set_ylabel("Observed AF rate")
        ax.set_title(title, fontsize=10)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
    axes[1].legend(fontsize=8, loc="upper left")
    plt.tight_layout()
    plt.savefig("figures/afib_calibration.png", dpi=120)
    plt.show()
