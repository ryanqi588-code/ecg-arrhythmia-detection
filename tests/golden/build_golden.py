"""Build the golden files for tests/test_golden.py (needs the downloaded data; run once, from the repo root).

    python tests/golden/build_golden.py

Writes into tests/golden/:
  sph_<id>.npz          - the first 10 s of four SPH recordings (12 x 5000 at
                          500 Hz, float16 as stored in the database, so
                          lossless): A22886 (AF, called AF) and A04870 (atrial
                          and ventricular ectopy, wrongly called AF), the
                          README figure's examples; A06017 (AF whose P waves
                          can't be measured: needs review); and a normal
                          sinus ECG called not AF
  expected.json         - for each: the probability and call saved in the
                          results (results/afib_example_scores.csv), the beats
                          the model found and how many were flagged
                          ventricular ectopic, and the features as computed
                          when this file was built
  code15_feature_rows.csv - 11 feature rows from test 17 (CODE-15% part 3),
                          chosen by a fixed rule (first by record id in each
                          group: AF called AF, non-AF not called, non-AF
                          called AF, needs review, a filled f-wave feature),
                          with the probability saved by test 17
                          (results/afib_abstain_test_recordings.csv)

The expected probabilities are the ones in the published results, not
recomputed here, so the tests check today's code against those results.
"""

from outputs import write_path      # committed outputs go to rerun/ unless --force
import json
import os

import numpy as np
import pandas as pd

from af import CLINICAL_FEATURES_12, clinical_features, regular_beats
from afib_sph_test import FS, load_sph
from ecg import detect_peaks_multilead, flag_ectopic
from predict import I_II_V1, load_model, predict

OUT = os.path.dirname(os.path.abspath(__file__))
SPH = ["A22886", "A04870"]       # the README figure's examples: AF called AF; PAC with ventricular ectopy called AF
NEEDS_REVIEW = "A06017"           # first by id among SPH recordings with unmeasurable P waves and a saved score
FEATURES = CLINICAL_FEATURES_12

if __name__ == "__main__":
    model = load_model()
    saved = pd.read_csv("results/afib_example_scores.csv").set_index("record")
    # A normal sinus ECG: the first SPH recording by id whose only statement is "Normal ECG" and that the
    # model calls not AF. SPH normal ECGs have no saved adopted-model probability, so its expected
    # probability is computed here, in the locked environment (requirements-lock.txt), not published.
    from afib_sph_test import sph_selected
    normal = next(r for r, _, lab in sorted(sph_selected()) if lab == "N"
                  and predict(load_sph(r), FS, model, checks=False)["result"] == "not AF")
    expected = {}
    for rec in SPH + [NEEDS_REVIEW, normal]:
        sig = load_sph(rec)                                   # 5000 x 12, values from float16
        path = write_path(f"{OUT}/sph_{rec}.npz")
        np.savez_compressed(path, signal=sig.astype(np.float16), fs=FS)
        sig = np.load(path)["signal"].astype(float)
        assert np.array_equal(sig, load_sph(rec)), rec       # lossless
        peaks = detect_peaks_multilead(sig[:, I_II_V1], FS)
        ectopic = flag_ectopic(sig[:, 1], FS, peaks)
        o = predict(sig, FS, model, checks=False)
        published = rec in saved.index
        # A06017 was saved (14 November) as "AF"; since test 17 that recording's output is "needs review".
        expected[rec] = {"label": saved.loc[rec, "label"] if published else "N",
                         "probability": float(saved.loc[rec, "probability"]) if published else o["probability"],
                         "probability_source": "published (results/afib_example_scores.csv)" if published
                         else "computed at build in the locked environment",
                         "result": "needs review" if rec == NEEDS_REVIEW else saved.loc[rec, "result"] if published
                         else o["result"],
                         "n_p_beats": int(clinical_features(sig[:, I_II_V1], FS, False, "multilead", sig)["n_p_beats"]),
                         "beats": int(len(peaks)),
                         "ventricular_ectopic": int(ectopic.sum()), "regular": int(regular_beats(peaks, FS).sum()),
                         "contributions_at_build": {c["feature"]: c["contribution"] for c in o["contributions"]}}
        print(rec, expected[rec]["result"], expected[rec]["probability"], f"beats {len(peaks)}, ectopic {ectopic.sum()}")
    with open(write_path(f"{OUT}/expected.json"), "w") as f:
        json.dump(expected, f, indent=1)

    t, _ = pd.read_pickle("data/code15_exams_part3_units.pkl")
    t = t.assign(record=t.record.astype(int))
    r = pd.read_csv("results/afib_abstain_test_recordings.csv")
    d = t.merge(r[["record", "prob", "p_unmeasured", "adopted"]], on="record").sort_values("record")
    af, un = d.label == "A", d.p_unmeasured
    filled = d.f_ratio_V1.isna() & ~un
    groups = [("AF called AF", af & d.adopted & ~un, 2), ("non-AF not called", ~af & ~d.adopted & ~un, 2),
              ("non-AF called AF", ~af & d.adopted & ~un, 2), ("needs review, AF", af & un, 1),
              ("needs review, non-AF", ~af & un, 2), ("f-wave features filled", filled, 2)]
    rows = []
    for name, m, k in groups:
        pick = d[m].head(k).assign(group=name)
        assert len(pick) == k, name
        rows.append(pick)
    g = pd.concat(rows)
    g["expected_result"] = np.where(g.p_unmeasured, "needs review", np.where(g.adopted, "AF", "not AF"))
    g[["group", "record", "label"] + FEATURES + ["prob", "expected_result"]].rename(
        columns={"prob": "saved_probability"}).to_csv(
        write_path(f"{OUT}/code15_feature_rows.csv"), index=False, float_format="%.17g")
    print(g.groupby("group").size().to_dict())
