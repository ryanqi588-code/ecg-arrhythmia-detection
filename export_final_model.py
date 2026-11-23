"""Write results/final_af_model.csv: the adopted AF model as a self-contained table.

Refits the adopted model exactly as test 12 did (afib_fa_dev.fit on the
development units from CPSC crops, PTB-XL and Chapman, with the
unmeasured-P-wave fix) and writes, per feature, the training mean and SD,
the coefficient per SD and on the raw input, and the value used when the
feature is missing or unmeasured; then the intercept and the cutoff. With
this table and the feature code (af.clinical_features), predict.py needs
no training data.

Columns:
  feature, input        - name, and the transform applied before scaling
  mean, sd              - training mean and SD of the input (StandardScaler)
  coef_per_sd, coef_raw - logistic coefficient on the standardized / raw input
  fill, fill_when       - value used instead, and when
  value                 - only for the cutoff row: AF if probability >= value
"""

from outputs import write_path      # committed outputs go to rerun/ unless --force
import numpy as np
import pandas as pd

from af import CLINICAL_FEATURES_12, model_inputs, p_unmeasured
from afib_fa_dev import P_COLS, fit
from afib_pac_dev import dev_units, prepare

FEATURES = CLINICAL_FEATURES_12 + ["p_unmeasured"]
ADOPTED_CUTOFF = 0.4309         # test 12 (afib_hybrid_test.ADOPTED_CUTOFF, to 4 places)
INPUT = {"cv": "log(cv)", "nrmssd": "log(nrmssd)"}
RECOMPUTED = "data/pac_dev_multilead12_recomputed.pkl"   # the features again, from the signals, with n_p_beats
BEAT_FEATURES = ["cv", "nrmssd", "entropy", "entropy_norm", "tpr", "pnn50", "ectopic_frac"]          # depend only on the beats found


def development_units():
    """The saved development features the model was fitted on, with n_p_beats added.

    The saved features predate n_p_beats (the number of regular,
    non-ventricular beats P waves were measured on), which predict.py's
    "P waves can't be measured" rule needs. They are recomputed from the
    signals with the current code (cached in RECOMPUTED) and n_p_beats is
    taken from there. The model is still fitted on the saved features, so
    it reproduces exactly. n_p_beats is valid for them because it depends
    only on the beats found, and the beat-derived features come out
    identical. The other features differ by filter rounding (relative 1e-7)
    and, in one duplicated recording (CPSC-extra Q0343 = Q2979), one beat's
    V2 P-wave match crosses its threshold; both are reported, not used.
    (Duplicates across all the datasets: check_duplicates.py, README "Duplicate recordings".)
    """
    saved = dev_units("multilead12")
    new = dev_units("multilead12", cache=RECOMPUTED)
    assert len(new) == len(saved), (len(new), len(saved))
    for c in ["record", "label", "source", "af"]:
        assert (new[c].values == saved[c].values).all(), c
    for c in BEAT_FEATURES:                     # same beats, so the same n_p_beats
        assert np.array_equal(new[c].values, saved[c].values, equal_nan=True), c
    for c in saved.columns:
        if saved[c].dtype.kind == "f" and c not in BEAT_FEATURES:
            a, b = saved[c].values, new[c].values
            off = ~np.isclose(a, b, rtol=1e-6, atol=1e-9, equal_nan=True)
            if off.any():
                print(f"recomputed {c} differs from the saved feature in {int(off.sum())} units "
                      f"(max {np.nanmax(np.abs(a - b)):.3g}); the saved values are used")
    return saved.assign(n_p_beats=new.n_p_beats.values)


if __name__ == "__main__":
    raw = development_units()
    units = prepare(raw)
    # "P waves can't be measured": the same rule as predict.py (af.p_unmeasured: fewer than 3 usable
    # beats), not the p_match_mean == 0 that test 12 fitted the model with.
    flag = np.array([p_unmeasured(r) for r in units.to_dict("records")])
    old_rule = (units.p_match_mean == 0).values.astype(float)
    print(f"P waves unmeasurable: {int(flag.sum())} of {len(units)} units by beat count, "
          f"{int(old_rule.sum())} by p_match_mean == 0; they differ on {int((flag != old_rule).sum())}")
    train = units.assign(patient=units.record, p_unmeasured=flag)
    model, cutoff = fit(train, FEATURES, True)
    assert abs(cutoff - ADOPTED_CUTOFF) < 5e-5, cutoff
    scaler, lr = model.named_steps["standardscaler"], model.named_steps["logisticregression"]
    coef, mean, sd = lr.coef_[0], scaler.mean_, scaler.scale_

    fills = {c: (raw[c].median(), "missing (fewer than 3 complete beats)")
             for c in ["log_f_wave", "f_ratio_II", "f_ratio_V1"]}
    for c in P_COLS:
        fills[c] = (train.loc[train.p_unmeasured == 0, c].median(), "p_unmeasured is 1")

    rows = [{"feature": f, "input": INPUT.get(f, f), "mean": m, "sd": s, "coef_per_sd": c, "coef_raw": c / s,
             "fill": fills.get(f, (np.nan, ""))[0], "fill_when": fills.get(f, (np.nan, ""))[1]}
            for f, m, s, c in zip(FEATURES, mean, sd, coef)]
    rows.append({"feature": "intercept", "coef_per_sd": lr.intercept_[0],
                 "coef_raw": lr.intercept_[0] - np.sum(coef * mean / sd)})
    rows.append({"feature": "cutoff", "input": "probability of AF", "value": cutoff,
                 "fill_when": "AF if probability >= value"})
    out = pd.DataFrame(rows)

    # With predict.py's rule the model must come out as published.
    old = pd.read_csv("results/final_af_model.csv").set_index("feature")
    for col in ["mean", "sd", "coef_per_sd", "coef_raw", "fill", "value"]:
        new = out.set_index("feature").loc[old.index, col]
        assert np.allclose(new.values.astype(float), old[col].values, equal_nan=True, rtol=1e-9), \
            f"{col} changed with predict.py's unmeasured rule"
    print("coefficients, means, SDs, fills and cutoff unchanged with predict.py's rule: yes")
    out.to_csv(write_path("results/final_af_model.csv"), index=False)
    print(out.to_string())

    # The table alone must give the fitted model's probabilities.
    x = model_inputs(with_fix := train.copy(), FEATURES)
    for c in P_COLS:
        x[with_fix.p_unmeasured.values == 1, FEATURES.index(c)] = fills[c][0]
    z = lr.intercept_[0] + ((x - mean) / sd) @ coef
    assert np.allclose(1 / (1 + np.exp(-z)), model.predict_proba(x)[:, 1])
    print("table reproduces the fitted model's probabilities: yes")
