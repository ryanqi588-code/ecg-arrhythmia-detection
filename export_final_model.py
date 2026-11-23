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

from af import CLINICAL_FEATURES_12, model_inputs
from afib_fa_dev import P_COLS, fit
from afib_pac_dev import dev_units, prepare

FEATURES = CLINICAL_FEATURES_12 + ["p_unmeasured"]
ADOPTED_CUTOFF = 0.4309         # test 12 (afib_hybrid_test.ADOPTED_CUTOFF, to 4 places)
INPUT = {"cv": "log(cv)", "nrmssd": "log(nrmssd)"}

if __name__ == "__main__":
    raw = dev_units("multilead12")
    units = prepare(raw)
    # The old rule (p_match_mean == 0), kept so this reproduces the published model exactly; the saved
    # development features have no n_p_beats. If the model is ever retrained, recompute the features and
    # use af.p_unmeasured instead (as predict.py does), so training and scoring agree.
    train = units.assign(patient=units.record, p_unmeasured=(units.p_match_mean == 0).astype(float))
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

    old = pd.read_csv("results/final_af_model.csv").set_index("feature")
    for col in ["mean", "sd", "coef_per_sd", "coef_raw"]:
        new = out.set_index("feature").loc[old.index, col]
        assert np.allclose(new.values.astype(float), old[col].values, equal_nan=True, rtol=1e-9), col
    out.to_csv(write_path("results/final_af_model.csv"), index=False)
    print(out.to_string())

    # The table alone must give the fitted model's probabilities.
    x = model_inputs(with_fix := train.copy(), FEATURES)
    for c in P_COLS:
        x[with_fix.p_unmeasured.values == 1, FEATURES.index(c)] = fills[c][0]
    z = lr.intercept_[0] + ((x - mean) / sd) @ coef
    assert np.allclose(1 / (1 + np.exp(-z)), model.predict_proba(x)[:, 1])
    print("table reproduces the fitted model's probabilities: yes")
