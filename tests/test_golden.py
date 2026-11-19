"""Golden tests: real ECGs in, the published probabilities and calls out.

The other tests use made-up inputs. These use real recordings and saved
results (tests/golden/, built by tests/golden/build_golden.py), so a silent
change from a library update, in the filters, beat detection, features or
scoring, makes them fail.

  - 11 feature rows from test 17 (CODE-15% part 3): score_features must
    reproduce the probability test 17 saved, and its call. This is pure
    arithmetic on the published table, so the tolerance is tight.
  - Four raw SPH recordings, one for each outcome and the README figure's
    two examples: AF called AF, PAC wrongly called AF, AF whose P waves
    can't be measured ("needs review", through the beat-count rule), and a
    normal sinus ECG called not AF. The whole path, from signal to call:
    the beats found, the ventricular-ectopic flags and the usable P-wave
    beats must match exactly, and so must the call; the probability must
    match to a relative 1e-8 (the published one for the first three; for
    the normal ECG, which has no saved probability, the locked
    environment's). It isn't bit-identical across
    library versions (the saved values differ from a rebuilt environment's
    by up to 4e-10, from rounding in the filters), but a real change in
    beats or features moves it far more.
"""

import json
import os

import numpy as np
import pandas as pd
import pytest

from af import clinical_features, regular_beats
from ecg import detect_peaks_multilead, flag_ectopic
from predict import I_II_V1, load_model, predict, score_features

GOLDEN = os.path.join(os.path.dirname(os.path.abspath(__file__)), "golden")
ROWS = pd.read_csv(os.path.join(GOLDEN, "code15_feature_rows.csv"))
EXPECTED = json.load(open(os.path.join(GOLDEN, "expected.json")))
MODEL = load_model()


@pytest.mark.parametrize("i", range(len(ROWS)), ids=[f"{g} {r}" for g, r in zip(ROWS.group, ROWS.record)])
def test_feature_row_reproduces_saved_probability(i):
    row = ROWS.iloc[i]
    out = score_features(row.to_dict(), MODEL)
    assert out["result"] == row.expected_result
    assert out["probability"] == pytest.approx(row.saved_probability, rel=1e-12, abs=1e-15)


def test_feature_rows_cover_every_kind_of_call():
    assert set(ROWS.expected_result) == {"AF", "not AF", "needs review"}
    assert ROWS[["f_ratio_II", "f_ratio_V1", "log_f_wave"]].isna().any(axis=None)      # a filled feature


@pytest.mark.parametrize("rec", sorted(EXPECTED))
def test_raw_recording_reproduces_published_result(rec):
    exp = EXPECTED[rec]
    z = np.load(os.path.join(GOLDEN, f"sph_{rec}.npz"))
    sig, fs = z["signal"].astype(float), int(z["fs"])
    peaks = detect_peaks_multilead(sig[:, I_II_V1], fs)
    assert len(peaks) == exp["beats"]
    assert int(flag_ectopic(sig[:, 1], fs, peaks).sum()) == exp["ventricular_ectopic"]
    assert int(regular_beats(peaks, fs).sum()) == exp["regular"]
    assert clinical_features(sig[:, I_II_V1], fs, False, "multilead", sig)["n_p_beats"] == exp["n_p_beats"]
    out = predict(sig, fs, MODEL)
    assert out["result"] == exp["result"]
    assert out["probability"] == pytest.approx(exp["probability"], rel=1e-8)


def test_raw_recordings_cover_every_outcome():
    assert {e["result"] for e in EXPECTED.values()} == {"AF", "not AF", "needs review"}
