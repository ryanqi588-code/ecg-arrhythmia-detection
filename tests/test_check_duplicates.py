"""check_duplicates: the same ECG resampled, cropped, re-filtered and stored as float16 is matched; another isn't."""

import os

import numpy as np
from scipy.signal import butter, resample_poly, sosfiltfilt

import check_duplicates as cd

GOLDEN = os.path.join(os.path.dirname(os.path.abspath(__file__)), "golden")


def golden(rec):
    z = np.load(os.path.join(GOLDEN, f"sph_{rec}.npz"))
    return z["signal"].astype(float), int(z["fs"])


def altered(sig):
    x = resample_poly(sig, 4, 5, axis=0)[int(1.37 * 400):]
    x = sosfiltfilt(butter(2, [0.5, 40], "bandpass", fs=400, output="sos"), x, axis=0)
    return x.astype(np.float16).astype(float), 400


def match(a, b):
    fa, fb = cd.fingerprint(*a), cd.fingerprint(*b)
    off, m, ov = cd.beat_match(fa["t"], fb["t"])
    ok = m >= cd.MIN_BEATS and m >= cd.MIN_SHARE * ov
    corr, amp = cd.aligned_corr(("array", *a), ("array", *b), off, fa, fb) if ok else (np.nan, np.nan)
    return ok and corr >= cd.MIN_CORR and amp <= cd.MAX_AMP, corr, fa["sha"] == fb["sha"]


def test_altered_copy_is_a_duplicate_but_not_an_exact_one():
    for rec in ["A22886", "A00002"]:                       # AF (irregular) and sinus rhythm (regular)
        sig, fs = golden(rec)
        ok, corr, same_hash = match((sig, fs), altered(sig))
        assert ok and not same_hash, (rec, corr)


def test_exact_copy_has_the_same_hash():
    sig, fs = golden("A04870")
    assert cd.fingerprint(sig, fs)["sha"] == cd.fingerprint(sig.copy(), fs)["sha"]


def test_different_recordings_are_not_duplicates():
    recs = ["A22886", "A04870", "A06017", "A00002"]
    for i, r1 in enumerate(recs):
        for r2 in recs[i + 1:]:
            ok, corr, same_hash = match(golden(r1), golden(r2))
            assert not same_hash and not ok, (r1, r2, corr)
