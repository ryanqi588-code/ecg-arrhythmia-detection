"""AF features, labels and classifier on synthetic RR series, so the tests run offline."""

from types import SimpleNamespace

import numpy as np
import pandas as pd

from af import (FEATURES_5, af_mask, cutoff_for_sensitivity, make_logreg, model_inputs,
                shannon_entropy, turning_points_ratio, window_features)

FS = 250


def peaks_from_rr(rr):
    return np.round(np.cumsum(np.concatenate([[1.0], rr])) * FS).astype(int)


def test_turning_points_ratio_extremes():
    assert turning_points_ratio(np.array([0.6, 0.9] * 20)) == 1.0     # alternating
    assert turning_points_ratio(np.linspace(0.6, 0.9, 40)) == 0.0     # smooth trend


def test_turning_points_ratio_random_is_about_two_thirds():
    rr = np.random.default_rng(0).uniform(0.4, 1.0, 5000)
    assert abs(turning_points_ratio(rr) - 2 / 3) < 0.02


def test_entropy_regular_vs_random():
    rng = np.random.default_rng(0)
    assert shannon_entropy(np.full(60, 0.8)) == 0.0
    assert shannon_entropy(rng.uniform(0.4, 1.0, 60)) > 0.9


def test_entropy_ignores_outliers():
    # One long pause should not squash a spread-out window into a few bins.
    rr = np.random.default_rng(0).uniform(0.5, 0.9, 80)
    with_pause = np.append(rr, 2.4)
    assert abs(shannon_entropy(with_pause) - shannon_entropy(rr)) < 0.05


def test_window_features_regular_rhythm():
    rr = np.full(200, 0.8)
    peaks = peaks_from_rr(rr)
    rows = window_features(peaks, FS, peaks[-1] + FS, np.zeros(peaks[-1] + FS, dtype=bool))
    assert len(rows) >= 2
    for row in rows:
        assert row["cv"] < 0.01 and row["pnn50"] == 0 and not row["af"]


def test_window_features_irregular_rhythm():
    rr = np.random.default_rng(0).uniform(0.4, 1.0, 300)
    peaks = peaks_from_rr(rr)
    n = peaks[-1] + FS
    rows = window_features(peaks, FS, n, np.ones(n, dtype=bool))
    for row in rows:
        assert row["cv"] > 0.15 and row["pnn50"] > 0.5 and row["af"]


def test_window_features_skips_sparse_windows():
    peaks = peaks_from_rr(np.full(10, 0.8))        # too few beats for a 60 s window
    assert window_features(peaks, FS, 60 * FS, np.zeros(60 * FS, dtype=bool)) == []


def test_af_mask_handles_segments_starting_mid_episode():
    rhythm = SimpleNamespace(sample=np.array([0, 1000, 3000]),
                             aux_note=["(N", "(AFIB", "(N"])
    mask = af_mask(rhythm, 2000, 4000)          # segment begins inside the AF episode
    assert mask[:1000].all() and not mask[1000:].any()


def test_logistic_regression_separates_regular_from_irregular():
    rng = np.random.default_rng(0)
    rows = []
    for af in (False, True):
        for _ in range(40):
            rr = rng.uniform(0.4, 1.0, 90) if af else 0.8 + 0.02 * rng.standard_normal(90)
            peaks = peaks_from_rr(rr)
            n = peaks[-1] + FS
            rows += window_features(peaks, FS, n, np.full(n, af))
    df = pd.DataFrame(rows)
    model = make_logreg().fit(model_inputs(df, FEATURES_5), df.af)
    prob = model.predict_proba(model_inputs(df, FEATURES_5))[:, 1]
    assert ((prob >= 0.5) == df.af).mean() > 0.95


def test_cutoff_for_sensitivity():
    prob = np.linspace(0, 1, 101)
    af = np.ones(101, dtype=bool)
    cutoff = cutoff_for_sensitivity(prob, af, target=0.9)
    assert abs((prob[af] >= cutoff).mean() - 0.9) < 0.02
