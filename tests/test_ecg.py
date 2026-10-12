"""Beat detection and scoring, on synthetic ECG so the tests run offline."""

import numpy as np

from ecg import (bandpass, detect_peaks_adaptive, detect_peaks_fixed, f_wave_activity,
                 flag_atrial_ectopic, flag_ectopic, p_wave_consistency, score)

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


def add_pvc(sig, at):
    """A wide, inverted, biphasic complex: a stand-in for a premature ventricular beat."""
    t = np.arange(len(sig)) / FS
    sig = sig - 1.0 * np.exp(-0.5 * ((t - at) / 0.035) ** 2)
    return sig + 0.5 * np.exp(-0.5 * ((t - at - 0.07) / 0.03) ** 2)


def test_flag_ectopic_catches_premature_ventricular_beats():
    normal = np.arange(1, 29, 0.8)
    pvc_times = normal[[5, 15, 25]] - 0.3                  # early, then the next normal beat
    keep = np.ones(len(normal), dtype=bool)
    keep[[5, 15, 25]] = False                              # beat replaced by the PVC
    sig = synthetic_ecg(normal[keep])
    for at in pvc_times:
        sig = add_pvc(sig, at)
    peaks = detect_peaks_adaptive(sig, FS)
    flags = flag_ectopic(sig, FS, peaks)
    near_pvc = np.array([np.min(np.abs(pvc_times - p / FS)) < 0.1 for p in peaks])
    assert near_pvc.sum() == 3
    assert flags[near_pvc].all()
    assert flags[~near_pvc].mean() < 0.1


def test_flag_ectopic_leaves_irregular_normal_beats_mostly_alone():
    # AF-like: irregular timing but every beat has the normal shape.
    rng = np.random.default_rng(1)
    times = np.cumsum(rng.uniform(0.4, 1.0, 45)) + 0.5
    sig = synthetic_ecg(times[times < 29])
    flags = flag_ectopic(sig, FS, detect_peaks_adaptive(sig, FS))
    assert flags.mean() < 0.25        # only the timing rule can fire, by chance


def test_flag_atrial_ectopic_needs_early_beat_and_odd_p_wave():
    # Sinus beats with a P wave 0.16 s before each QRS; two early beats with no P wave.
    normal = np.arange(1, 29, 0.8)
    times = normal.copy()
    times[[10, 20]] -= 0.25                                   # premature
    sig = synthetic_ecg(times)
    t = np.arange(len(sig)) / FS
    for i, bt in enumerate(times):
        if i not in (10, 20):
            sig += 0.15 * np.exp(-0.5 * ((t - bt + 0.16) / 0.025) ** 2)
    peaks = detect_peaks_adaptive(sig, FS)
    flags = flag_atrial_ectopic(sig, FS, peaks)
    early = np.array([np.min(np.abs(times[[10, 20]] - p / FS)) < 0.05 for p in peaks])
    assert flags[early].all()
    assert not flags[~early].any()


def test_f_wave_activity_ignores_beat_locked_waves_but_not_fibrillation():
    times = np.arange(1, 29, 0.8)
    sinus = synthetic_ecg(times, t_wave=0.3)                 # T waves are locked to the beats
    t = np.arange(len(sinus)) / FS
    fib = sinus + 0.08 * np.sin(2 * np.pi * 6.3 * t)          # 6.3 Hz activity, not locked
    peaks = detect_peaks_adaptive(sinus, FS)
    assert f_wave_activity(fib, FS, peaks) > 3 * f_wave_activity(sinus, FS, peaks)


def test_p_wave_consistency_sinus_vs_fibrillation_with_few_beats():
    # Four beats only, as in a few seconds of ECG.
    times = np.array([1.0, 1.8, 2.6, 3.4])
    sig = synthetic_ecg(times, t_wave=0.3, seconds=4)
    t = np.arange(len(sig)) / FS
    sinus = sig + sum(0.15 * np.exp(-0.5 * ((t - bt + 0.16) / 0.025) ** 2) for bt in times)
    fib = sig + 0.08 * np.sin(2 * np.pi * 6.3 * t + 3 * np.sin(2 * np.pi * 0.7 * t))
    peaks = detect_peaks_adaptive(sinus, FS)
    assert p_wave_consistency(sinus, FS, peaks) == 1.0
    # Comparing each beat with a median that includes itself inflates the match.
    loo = p_wave_consistency(fib, FS, peaks, remove_qrst=False)
    with_self = p_wave_consistency(fib, FS, peaks, remove_qrst=False, leave_one_out=False)
    assert loo < with_self


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
