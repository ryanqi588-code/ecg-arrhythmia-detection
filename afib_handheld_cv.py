"""Handheld-device AF model, checked by 5-fold cross-validation on Challenge 2017.

Part of routing by device: handheld recordings get their own model and
cutoff, trained on handheld data only. There is no unused public handheld AF
dataset left to test this on, so this cross-validation is the only evidence.

Compared (all on windows with ventricular ectopics removed, cutoff for 95%
sensitivity on the training folds, with and without the P-wave override):
  - Challenge 2017 only, 5 features
  - Challenge 2017 only, entropy_norm
"""

import numpy as np
import pandas as pd

from af import FEATURES_5, FEATURES_5N, cinc_windows, fit_with_cutoff, model_inputs, p_wave_override, auc

cinc = cinc_windows().copy()
folds = np.array_split(np.random.default_rng(1).permutation(cinc.record.unique()), 5)
is_af = cinc.af.values
rows = []
for name, features in [("Challenge only, 5 features", FEATURES_5),
                       ("Challenge only, entropy_norm", FEATURES_5N)]:
    prob, pred = np.zeros(len(cinc)), np.zeros(len(cinc), dtype=bool)
    for fold in folds:
        held = cinc.record.isin(fold).values
        model, cutoff = fit_with_cutoff(cinc[~held], features)
        prob[held] = model.predict_proba(model_inputs(cinc[held], features))[:, 1]
        pred[held] = prob[held] >= cutoff
    for override in (False, True):
        p = p_wave_override(pred, cinc) if override else pred
        rows.append({"model": name, "override": override, "sensitivity": p[is_af].mean(),
                     "false_alarm_rate_N": p[(cinc.label == "N").values].mean(),
                     "false_alarm_rate_O": p[(cinc.label == "O").values].mean(),
                     "auc": np.nan if override else auc(prob, is_af)})
results = pd.DataFrame(rows)
results.to_csv("results/afib_handheld_cv.csv", index=False)
print(results.round(3).to_string(index=False))
