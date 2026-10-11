"""Beat detection and scoring, on synthetic ECG so the tests run offline."""

import numpy as np

from ecg import bandpass, detect_peaks_adaptive, detect_peaks_fixed, score

FS = 360


def synthetic_ecg(beat_times, amplitudes=None, t_wave=0.0, seconds=30, noise=0.02, seed=0):
    """Narrow Gaussian QRS complexes at beat_times (s), optional broad T waves 0.3 s later."""
    rng = np.random.default_rng(seed)
    t = np.arange(int(seconds * FS)) / FS
    sig = noise * rng.standard_normal(len(t))
    amplitudes = np.ones(len(beat_times)) if amplitudes is None else amplitudes
    for bt, amp in zip(beat_times, amplitudes):
        sig += amp * np.exp(-0.5 * ((t - bt) / 0.012) ** 2)
        if t_wave:
            sig += t_wave * np.exp(-0.5 * ((t - bt - 0.3) / 0.05) ** 2)
    return sig


def beat_samples(times):
    return np.round(np.asarray(times) * FS).astype(int)


def test_bandpass_removes_baseline_offset():
    sig = synthetic_ecg(np.arange(1, 29, 0.8)) - 0.35
    assert abs(bandpass(sig, FS).mean()) < 0.01


def test_adaptive_finds_every_regular_beat():
    times = np.arange(1, 29, 0.8)
    det = detect_peaks_adaptive(synthetic_ecg(times), FS)
    tp, fp, fn, sens, ppv = score(beat_samples(times), det, int(0.15 * FS))
    assert (fp, fn) == (0, 0)


def test_adaptive_finds_inverted_beats():
    times = np.arange(1, 29, 0.8)
    amps = np.where(np.arange(len(times)) % 4 == 0, -1.2, 1.0)   # every 4th beat points down
    det = detect_peaks_adaptive(synthetic_ecg(times, amps), FS)
    tp, fp, fn, *_ = score(beat_samples(times), det, int(0.15 * FS))
    assert fn == 0


def test_adaptive_rejects_t_waves():
    times = np.arange(1, 29, 0.8)
    det = detect_peaks_adaptive(synthetic_ecg(times, t_wave=0.6), FS)
    tp, fp, fn, *_ = score(beat_samples(times), det, int(0.15 * FS))
    assert (fp, fn) == (0, 0)


def test_one_tall_spike_fixed_vs_adaptive():
    # The failure that motivated the adaptive detector (record 215): one
    # beat 6x taller than the rest.
    times = np.arange(1, 29, 0.8)
    amps = np.full(len(times), 0.5)
    spike = 3
    amps[spike] = 3.0
    sig = synthetic_ecg(times, amps)
    tol = int(0.15 * FS)

    # The fixed threshold loses most of the recording.
    _, _, fn_fixed, *_ = score(beat_samples(times), detect_peaks_fixed(sig, FS), tol)
    assert fn_fixed > len(times) // 2

    # The adaptive threshold only loses beats within its +/-1 s window of
    # the spike (a known limitation), and finds every other beat.
    det = detect_peaks_adaptive(sig, FS)
    far = np.abs(times - times[spike]) > 1.0
    _, _, fn_far, *_ = score(beat_samples(times[far]), det, tol)
    assert fn_far == 0


def test_score_counts():
    ref = [100, 200, 300, 400]
    det = [102, 198, 350, 500]           # two hits, one too far off, one extra
    tp, fp, fn, sens, ppv = score(ref, det, tol=10)
    assert (tp, fp, fn) == (2, 2, 2)
    assert sens == 0.5 and ppv == 0.5


def test_score_matches_each_detection_once():
    tp, fp, fn, *_ = score([100, 105], [102], tol=10)
    assert (tp, fp, fn) == (1, 0, 1)


def test_score_ignores_detections_in_excluded_spans():
    tp, fp, fn, *_ = score([100], [100, 500, 520], tol=10, excluded=[(450, 600)])
    assert (tp, fp, fn) == (1, 0, 0)
