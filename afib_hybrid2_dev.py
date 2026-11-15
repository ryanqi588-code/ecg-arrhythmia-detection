"""Hybrid with a confident-call override: the CNN may veto only the interpretable model's less confident calls.

Test 14: the hybrid of afib_hybrid_dev.py cut PAC false alarms on 1976
MIMIC-IV-ECG patients but lost 0.7 points of sensitivity (95% CI -2.5 to
+1.1), failing the -2 point bar. Of the AF it lost, some had high
interpretable probabilities that the CNN vetoed. The rule here:

  AF if (interpretable >= t_lr and CNN >= t_cnn)  [test 13's hybrid, unchanged]
     or interpretable >= t_high                     [confident calls, no veto]

t_high is chosen on the development data only (out-of-fold probabilities,
as in afib_hybrid_dev.py): the highest value on a grid at which the
hybrid's sensitivity in each source (records weighted equally) is at least
that of the interpretable model alone at its cutoff, so the CNN removes no
more AF than it adds. A higher t_high leaves more to the CNN, so the
highest passing value keeps most of the PAC gain.

Then a preview (not a test) on the sets already used, including test 14.
"""

import numpy as np
import pandas as pd

from afib_hybrid_dev import SOURCES, aligned_dev, false_alarms, sensitivity_by_source
from afib_fa_dev import fit

if __name__ == "__main__":
    d = aligned_dev()
    th = pd.read_csv("results/afib_hybrid_thresholds.csv").iloc[0]
    t_lr, t_cnn = float(th.t_lr), float(th.t_cnn)
    lr_only_cut = min(np.quantile(d.prob_lr[(d.source == s) & d.af], 0.05) for s in SOURCES)
    alone = d.prob_lr.values >= lr_only_cut
    base = (d.prob_lr.values >= t_lr) & (d.prob_cnn.values >= t_cnn)
    s_alone = sensitivity_by_source(alone, d)
    print("alone (OOF cutoff %.4f):" % lr_only_cut, {k: round(v, 4) for k, v in s_alone.items()})
    rows, best = [], None
    for t_high in np.round(np.arange(0.40, 1.0001, 0.01), 2):
        called = base | (d.prob_lr.values >= t_high)
        s = sensitivity_by_source(called, d)
        ok = all(s[k] >= s_alone[k] - 1e-12 for k in SOURCES)
        rows.append({"t_high": t_high, "ok": ok, **{f"sens_{k}": v for k, v in s.items()},
                     **{f"fa_{k}": v for k, v in false_alarms(called, d).items()}})
        if ok:
            best = t_high
    grid = pd.DataFrame(rows)
    pd.set_option("display.width", 200)
    print(grid.round(4).to_string(index=False))
    for name, called in [("alone", alone), ("hybrid (test 13)", base)]:
        print(name, {k: round(v, 4) for k, v in {**sensitivity_by_source(called, d),
                                                  **false_alarms(called, d)}.items()})
    print(f"chosen t_high = {best}")
    assert best is not None
    pd.DataFrame([{"t_lr": t_lr, "t_cnn": t_cnn, "t_high": best}]).to_csv(
        "results/afib_hybrid2_thresholds.csv", index=False)
    grid.to_csv("results/afib_hybrid2_dev.csv", index=False)

    # Preview on used sets: probabilities saved by tests 13 and 14.
    model, c_alone = fit(d, aligned_dev.__globals__["ADOPTED"], True)
    rows = []
    for name, f in [("incart", "results/afib_hybrid_test_incart_recordings.csv"),
                    ("code15p2", "results/afib_hybrid_test_code15p2_recordings.csv"),
                    ("mimic", "results/afib_mimic_test_recordings.csv")]:
        t = pd.read_csv(f)
        y = (t.label == "A").values
        h1 = (t.prob_lr.values >= t_lr) & (t.prob_cnn.values >= t_cnn)
        for mname, called in [("alone", t.prob_lr.values >= c_alone), ("hybrid", h1),
                              ("hybrid + override", h1 | (t.prob_lr.values >= best))]:
            r = {"set": name, "model": mname, "AF caught": called[y].mean() if y.any() else np.nan}
            for lab in sorted(set(t.label) - {"A"}):
                r[f"{lab} called AF"] = called[t.label.values == lab].mean()
            rows.append(r)
    prev = pd.DataFrame(rows)
    print("\nPreview on used sets (not a test):")
    print(prev.round(4).to_string(index=False))
    prev.to_csv("results/afib_hybrid2_preview.csv", index=False)
