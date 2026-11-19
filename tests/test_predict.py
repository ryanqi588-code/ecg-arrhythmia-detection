"""predict.py: the published table alone gives the score, fills and unscored cases."""

import numpy as np

from predict import load_model, predict, score_features

ROW = {"cv": 0.05, "nrmssd": 0.06, "entropy_norm": 0.9, "tpr": 0.6, "pnn50": 0.2, "p_match_regular": 0.95,
       "log_f_wave": -4.5, "p_match_mean": 0.9, "f_ratio_II": 0.2, "f_ratio_V1": 0.2}


def test_table_has_cutoff_and_fills():
    feats, intercept, cutoff = load_model()
    assert abs(cutoff - 0.4309) < 1e-4
    assert feats.fill.notna().sum() == 5


def test_contributions_add_up_to_the_probability():
    out = score_features(ROW)
    z = out["intercept"] + sum(c["contribution"] for c in out["contributions"])
    assert np.isclose(z, out["score"])
    assert np.isclose(out["probability"], 1 / (1 + np.exp(-z)))
    assert out["result"] == ("AF" if out["probability"] >= out["cutoff"] else "not AF")


def test_unmeasured_p_waves_are_filled_not_zero():
    out = score_features({**ROW, "p_match_regular": 0.0, "p_match_mean": 0.0})
    c = {x["feature"]: x for x in out["contributions"]}
    assert c["p_unmeasured"]["value"] == 1
    assert out["result"] == "needs review"
    assert c["p_match_regular"]["filled"] and c["p_match_regular"]["value"] == 1.0
    assert c["p_match_mean"]["filled"] and c["p_match_mean"]["value"] == 0.9375


def test_missing_f_wave_features_are_filled():
    out = score_features({**ROW, "f_ratio_V1": np.nan, "log_f_wave": np.nan})
    c = {x["feature"]: x for x in out["contributions"]}
    assert c["f_ratio_V1"]["filled"] and np.isclose(c["f_ratio_V1"]["value"], 0.1983, atol=1e-4)
    assert c["log_f_wave"]["filled"] and np.isclose(c["log_f_wave"]["value"], -4.2321, atol=1e-4)
    assert not c["cv"]["filled"]


def test_unscored_recordings():
    flat = np.zeros((5000, 12))
    assert predict(flat, 500)["result"] == "unscored"
    gap = np.random.default_rng(0).normal(size=(5000, 12))
    gap[100, 3] = np.nan
    assert predict(gap, 500) == {"result": "unscored", "reason": "missing samples"}
