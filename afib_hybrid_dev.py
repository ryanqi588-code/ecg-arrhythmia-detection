"""A hybrid: the interpretable AF model screens, the same-data CNN must confirm.

Seven attempts at hand-made features left the 12-lead model's PAC false
alarms nearly unchanged, while the CNN of afib_cnn.py (trained on the same
development data) made far fewer in test 11. In the hybrid, AF is called
only if the interpretable model (the adopted model of test 12) calls it AND
the CNN's probability is at least a confirmation threshold. The CNN can
only remove calls, so every AF call still comes with the interpretable
model's explanation; but a call it removes has no transparent reason.

Both thresholds are chosen on the development data only (CPSC crops,
PTB-XL, Chapman), from out-of-fold probabilities of both models (5-fold CV
grouped by recording or patient; the CNN's from afib_cnn.py): among all
pairs on a grid, the one with the fewest false alarms (the mean of the
false-alarm rates on PAC, sinus and other rhythms) that keeps at least 95%
sensitivity in each of the three sources (records weighted equally).

The script then shows, as a preview and not a test, what the fixed rule
would have done on the sets already used in tests 9-11.
"""

import numpy as np
import pandas as pd

from af import CLINICAL_FEATURES_12, model_inputs, auc, record_weights
from afib_fa_dev import expanded_units, fit, with_unmeasured_fix
from afib_pac_dev import dev_units, prepare

ADOPTED = CLINICAL_FEATURES_12 + ["p_unmeasured"]
SOURCES = ["cpsc", "ptbxl", "chapman"]


def aligned_dev():
    """Development units with both models' out-of-fold probabilities."""
    u = prepare(dev_units("multilead12")).reset_index(drop=True)
    u["patient"] = u.record
    u["p_unmeasured"] = (u.p_match_mean == 0).astype(float)
    keys = pd.read_pickle("data/ppresence_dev.pkl")
    keys = keys[keys.source.isin(SOURCES)].reset_index(drop=True)
    assert (keys.source.values == u.source.values).all()
    cnn = pd.read_csv("results/afib_cnn_dev_oof.csv")
    ptb = [l.split()[0] for l in open("data/ptbxl/SELECTED").read().splitlines()]
    cnn.loc[cnn.source == "ptbxl", "key"] = ptb
    cnn.loc[cnn.source == "chapman", "key"] = cnn.record[cnn.source == "chapman"].str.replace("chapman-", "")
    cnn.loc[cnn.source == "cpsc", "key"] = cnn.record[cnn.source == "cpsc"]
    cnn["crop"] = cnn.groupby(["source", "key"]).cumcount()      # 10 s crops within each recording
    m = keys.merge(cnn[["source", "key", "crop", "prob_cnn", "label"]], on=["source", "key", "crop"], how="left")
    assert m.prob_cnn.notna().all() and (m.label.values == u.label.values).all()
    u["prob_cnn"] = m.prob_cnn.values
    folds = np.array_split(np.random.default_rng(2).permutation(u.patient.unique()), 5)
    u["prob_lr"] = 0.0
    for fold in folds:
        held = u.patient.isin(fold).values
        model, _ = fit(u[~held], ADOPTED, True)
        u.loc[held, "prob_lr"] = model.predict_proba(model_inputs(with_unmeasured_fix(u[~held], u[held])[1],
                                                                  ADOPTED))[:, 1]
    return u


def sensitivity_by_source(called, d):
    out = {}
    for s in SOURCES:
        m = (d.source == s).values & d.af.values
        w = record_weights(d.record.values[m])
        out[s] = np.sum(w * called[m]) / w.sum()
    return out


def false_alarms(called, d):
    return {lab: called[(d.label == lab).values].mean() for lab in ["PAC", "N", "other"]}


def choose(d):
    """Thresholds (lr, cnn) with the fewest false alarms keeping >= 95% sensitivity per source."""
    lr_grid = np.quantile(d.prob_lr[d.af], np.linspace(0.0, 0.06, 61))
    cnn_grid = np.r_[0.0, np.quantile(d.prob_cnn[d.af], np.linspace(0.0, 0.06, 61))]
    best = None
    for a in lr_grid:
        la = d.prob_lr.values >= a
        for c in cnn_grid:
            called = la & (d.prob_cnn.values >= c)
            if min(sensitivity_by_source(called, d).values()) < 0.95:
                continue
            fa = np.mean(list(false_alarms(called, d).values()))
            if best is None or fa < best[0]:
                best = (fa, a, c)
    return best[1], best[2]


if __name__ == "__main__":
    d = aligned_dev()
    print(f"{len(d)} development units; OOF AUC: interpretable {auc(d.prob_lr.values, d.af.values):.4f}, "
          f"CNN {auc(d.prob_cnn.values, d.af.values):.4f}")
    t_lr, t_cnn = choose(d)
    lr_only_cut = min(np.quantile(d.prob_lr[(d.source == s) & d.af], 0.05) for s in SOURCES)
    rows = []
    for name, called in [("interpretable alone (95% per source)", d.prob_lr.values >= lr_only_cut),
                         ("hybrid", (d.prob_lr.values >= t_lr) & (d.prob_cnn.values >= t_cnn))]:
        rows.append({"model": name, **{f"sens_{k}": v for k, v in sensitivity_by_source(called, d).items()},
                     **{f"fa_{k}": v for k, v in false_alarms(called, d).items()}})
    dev = pd.DataFrame(rows)
    print(f"thresholds: interpretable {t_lr:.4f}, CNN confirmation {t_cnn:.4g}")
    print(dev.round(3).to_string(index=False))
    pd.DataFrame([{"t_lr": t_lr, "t_cnn": t_cnn}]).to_csv("results/afib_hybrid_thresholds.csv", index=False)
    dev.to_csv("results/afib_hybrid_dev.csv", index=False)

    # Preview on the used sets: the interpretable model refitted on all development units.
    model, c_alone = fit(d, ADOPTED, True)
    u = expanded_units()
    rows = []
    for name in ["sph", "ningbo", "code15", "codetest"]:
        t = u[u.source == name].copy()
        probs = pd.read_csv(f"results/compare_probs_{name}.csv")
        t["record"] = t.record.astype(str)
        probs["record"] = probs.record.astype(str)
        t = t.merge(probs[["record", "prob_cnn"]], on="record", how="left")
        assert t.prob_cnn.notna().all()
        t["prob_lr"] = model.predict_proba(model_inputs(with_unmeasured_fix(d, t)[1], ADOPTED))[:, 1]
        y = t.af.values
        for mname, called in [("interpretable (alone, its cutoff)", t.prob_lr.values >= c_alone),
                              ("hybrid", (t.prob_lr.values >= t_lr) & (t.prob_cnn.values >= t_cnn))]:
            r = {"set": name, "model": mname, "AF caught": called[y].mean() if y.any() else np.nan}
            for lab in sorted(set(t.label) - {"A"}):
                r[f"{lab} called AF"] = called[(t.label == lab).values].mean()
            rows.append(r)
    prev = pd.DataFrame(rows)
    pd.set_option("display.width", 220)
    print("\nPreview on used sets (not a test):")
    print(prev.round(3).to_string(index=False))
    prev.to_csv("results/afib_hybrid_preview.csv", index=False)
