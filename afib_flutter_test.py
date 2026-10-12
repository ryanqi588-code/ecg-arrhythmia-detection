"""Test of atrial flutter detection on the Shandong (SPH) data.

Development (afib_flutter_dev.py, 5-fold CV on CPSC, PTB-XL and Chapman,
488 flutter recordings) found:
  - the final AF model calls 52% of flutter AF (mostly variable block) and
    misses the rest (mostly regular 2:1 block);
  - a gradient-boosted flutter model on rate, RR, atrial and flutter-wave
    features (ecg.flutter_features in II, aVF, V1) reaches AUC 0.930;
  - 2:1 flutter and SVT can look the same in 10 s, so its cutoff is set so
    that at most 10% of sinus tachycardia + SVT are called flutter.
  Used alongside the AF model ("AF or flutter": either model positive), in
  development it catches 83.6% of flutter (AF model alone 51.6%), and calls
  7.4% of sinus tachycardia and 20.3% of SVT AF or flutter (2.9% each
  before). It names 62.9% of flutter as flutter, and 6.8% of AF.

Models: the AF model of the SPH test (afib_sph_test.py), unchanged; the
flutter model fitted on all development units (afib_flutter_dev.fit_flutter).

Data: SPH, first 10 s, chosen from metadata.csv alone:
  AFL  - atrial flutter (51), no AF (50)                            (99)
  ST   - sinus tachycardia (21), no AF or flutter, not in test 9    (711)
  JT   - junctional tachycardia (54), no AF or flutter              (13)
  A    - the AF recordings of test 9 (already used; there to measure how
         often real AF is named flutter)                             (675)
The flutter, sinus tachycardia and junctional tachycardia recordings were
never opened. This plan is committed before they are; it is run once.

Primary results:
  1. share of flutter called AF or flutter (development: 83.6%)
  2. share of sinus tachycardia called AF or flutter (development: 7.4%)
  3. share of flutter named flutter (62.9%) and of AF named flutter (6.8%)
"""

import os

import numpy as np
import pandas as pd

from af import CLINICAL_FEATURES_12, fit_with_source_cutoffs, model_inputs, auc
from afib_flutter_dev import FLUTTER_FEATURES, flutter_units, prepare as fl_prepare, fit_flutter
from afib_pac_dev import dev_units, prepare, unit_features
from afib_sph_test import SPH_DIR, FS, load_sph, sph_selected, wilson


def flutter_selected():
    """(ECG_ID, patient, label) for the flutter test, from metadata.csv alone."""
    test9 = {ecg_id for ecg_id, _, _ in sph_selected()}
    af = [s for s in sph_selected() if s[2] == "A"]
    meta = pd.read_csv(os.path.join(SPH_DIR, "metadata.csv"))
    out = []
    for ecg_id, codes, patient in zip(meta.ECG_ID, meta.AHA_Code.astype(str), meta.Patient_ID):
        c = {part for statement in codes.split(";") for part in statement.split("+")}
        if "51" in c and "50" not in c:
            out.append((ecg_id, patient, "AFL"))
        elif c & {"50", "51"} or ecg_id in test9:
            continue
        elif "54" in c:
            out.append((ecg_id, patient, "JT"))
        elif "21" in c:
            out.append((ecg_id, patient, "ST"))
    return out + af


if __name__ == "__main__":
    os.makedirs("results", exist_ok=True)

    units = prepare(dev_units("multilead12"))
    m_af, cut_af = fit_with_source_cutoffs(units, CLINICAL_FEATURES_12)
    c_af = min(cut_af.values())
    fl_units = fl_prepare(flutter_units())
    m_fl, c_fl = fit_flutter(fl_units)
    fills = {col: fl_units[col].median() for col in set(CLINICAL_FEATURES_12 + FLUTTER_FEATURES)}
    print(f"cutoffs: AF {c_af:.3f}, flutter {c_fl:.3f}")

    selected = flutter_selected()
    print(f"selected: {pd.Series([s[2] for s in selected]).value_counts().to_dict()}")
    rows, skipped = [], {}
    for i, (ecg_id, patient, label) in enumerate(selected):
        if i % 500 == 0:
            print(f"test {i} ...", flush=True)
        row = unit_features(load_sph(ecg_id), FS, label == "A", "multilead12")
        if row is None:
            skipped[label] = skipped.get(label, 0) + 1
            continue
        rows.append({**row, "record": ecg_id, "patient": patient, "label": label})
    t = pd.DataFrame(rows)
    for col, value in fills.items():
        t[col] = t[col].fillna(value)

    p_af = m_af.predict_proba(model_inputs(t, CLINICAL_FEATURES_12))[:, 1]
    p_fl = m_fl.predict_proba(model_inputs(t, FLUTTER_FEATURES))[:, 1]
    af_call, fl_call = p_af >= c_af, p_fl >= c_fl
    either = af_call | fl_call
    t = t.assign(prob_af=p_af, prob_flutter=p_fl, called_af=af_call, named_flutter=fl_call, af_or_flutter=either)
    t.to_csv("results/afib_flutter_test_recordings.csv", index=False)

    print(f"\nSPH flutter test: {len(t)} scored {t.label.value_counts().to_dict()}; skipped: {skipped}")
    out = []
    for lab in ["AFL", "ST", "JT", "A"]:
        g = t[t.label == lab]
        r = {"label": lab, "n": len(g)}
        for col in ["called_af", "af_or_flutter", "named_flutter"]:
            k = int(g[col].sum())
            lo, hi = wilson(k, len(g)) if len(g) else (np.nan, np.nan)
            r.update({col: k / len(g) if len(g) else np.nan, f"{col}_ci_low": lo, f"{col}_ci_high": hi})
        out.append(r)
    results = pd.DataFrame(out)
    fl = (t.label == "AFL").values
    others = t.label.isin(["ST", "JT", "A"]).values
    results.attrs["auc"] = auc(p_fl[fl | others], fl[fl | others])
    results.to_csv("results/afib_flutter_test.csv", index=False)
    pd.set_option("display.width", 220)
    print(results.round(3).to_string(index=False))
    print(f"\nFlutter model AUC, flutter vs the rest: {results.attrs['auc']:.3f}; "
          f"vs sinus tachycardia: {auc(p_fl[fl | (t.label == 'ST').values], fl[fl | (t.label == 'ST').values]):.3f}; "
          f"vs AF: {auc(p_fl[fl | (t.label == 'A').values], fl[fl | (t.label == 'A').values]):.3f}")
