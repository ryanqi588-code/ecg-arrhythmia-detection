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


def test_unmeasured_comes_from_the_beat_count_not_a_zero_match():
    # P waves measured on enough beats but matching in no lead: truly absent, not unmeasured.
    absent = score_features({**ROW, "p_match_regular": 0.0, "p_match_mean": 0.0, "n_p_beats": 8})
    c = {x["feature"]: x for x in absent["contributions"]}
    assert c["p_unmeasured"]["value"] == 0 and not c["p_match_mean"]["filled"]
    assert absent["result"] != "needs review"
    few = score_features({**ROW, "n_p_beats": 2})
    assert few["result"] == "needs review"


def test_padding_is_stripped_before_scoring():
    # Golden AF recording A04870, first 7 s: called AF on its own; zero-padded to
    # 10.24 s (CODE format) it came back "needs review" before padding was stripped.
    import os
    from predict import strip_flat_edges
    z = np.load(os.path.join(os.path.dirname(os.path.abspath(__file__)), "golden", "sph_A04870.npz"))
    sig, fs = z["signal"].astype(float)[: 7 * 500], int(z["fs"])
    padded = np.zeros((5120, 12))
    padded[810:810 + len(sig)] = sig
    assert np.array_equal(strip_flat_edges(padded), sig)
    assert predict(padded, fs) == predict(sig, fs)
    assert predict(sig + 0.3, fs) == predict(np.vstack([np.full((50, 12), 0.3), sig + 0.3]), fs)
    # Padding doesn't push real signal past the 10 s cut.
    long = np.vstack([np.zeros((400, 12)), z["signal"].astype(float), np.zeros((400, 12))])
    assert predict(long, fs) == predict(z["signal"].astype(float), fs)


def test_strip_flat_edges_keeps_unpadded_signal():
    from predict import strip_flat_edges
    sig = np.random.default_rng(1).normal(size=(1000, 12))
    assert len(strip_flat_edges(sig)) == 1000
    sig[:2] = 0                         # two identical rows: a flat stretch, stripped
    assert len(strip_flat_edges(sig)) == 998
    assert len(strip_flat_edges(np.zeros((1000, 12)))) == 0


def _golden(rec="A00002"):
    import os
    z = np.load(os.path.join(os.path.dirname(os.path.abspath(__file__)), "golden", f"sph_{rec}.npz"))
    return z["signal"].astype(float), int(z["fs"])


def test_flat_or_disconnected_lead_is_unscored():
    sig, fs = _golden()
    assert predict(sig, fs)["result"] == "not AF"
    v1 = sig.copy(); v1[:, 6] = 0                                   # V1 zeroed: used to come back "not AF"
    out = predict(v1, fs)
    assert out["result"] == "unscored" and "lead V1 flat or disconnected" in out["reason"]
    noise = sig.copy(); noise[:, 8] = np.random.default_rng(0).normal(0, 0.005, len(sig))   # V3: noise only
    assert "lead V3 flat or disconnected" in predict(noise, fs)["reason"]


def test_lead_off_for_part_of_the_recording_is_unscored():
    sig, fs = _golden()
    off = sig.copy(); off[:1000, 9] = off[1000, 9]                  # V4 constant for the first 2 s
    assert "lead V4 flat for over 1 s" in predict(off, fs)["reason"]
    off = sig.copy(); off[:400, 9] = off[400, 9]                    # 0.8 s: a flat baseline can be that long
    assert predict(off, fs)["result"] == "not AF"


def test_clipped_signal_is_unscored():
    sig, fs = _golden()
    out = predict(np.clip(sig, -1.0, 1.0), fs)
    assert out["result"] == "unscored" and "clipped" in out["reason"]


def test_coarsely_quantized_wave_tops_are_not_clipping():
    # MIMIC-IV-like: 0.02 mV steps flatten smooth T-wave tops into plateaus at the lead's maximum.
    from predict import lead_problems
    fs = 500
    phase = np.arange(10 * fs) / fs % 0.8
    t_wave = 0.4 * np.exp(-(phase - 0.4) ** 2 / (2 * 0.06 ** 2))
    r_wave = 1.5 * np.exp(-(phase - 0.1) ** 2 / (2 * 0.01 ** 2))
    quantize = lambda x: np.tile(np.round(x / 0.02) * 0.02, (12, 1)).T
    assert lead_problems(quantize(t_wave - 0.3 * r_wave / 1.5), fs) == []
    # Clipping is recognised by its steep edges: sharp R waves cut flat are flagged.
    assert "clipped" in lead_problems(quantize(np.minimum(r_wave + t_wave, 1.0)), fs)[0]


def test_short_recordings_and_untested_sampling_rates_are_unscored():
    sig, fs = _golden()
    assert "too short" in predict(sig[: 6 * fs], fs)["reason"]
    assert predict(sig[: 7 * fs], fs)["result"] != "unscored"
    assert "sampling rate 250 Hz" in predict(sig[::2], 250)["reason"]
    assert "sampling rate 1000 Hz" in predict(sig, 1000)["reason"]
