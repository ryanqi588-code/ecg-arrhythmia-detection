"""Development of better atrial-ectopic beat removal for the 12-lead AF model.

ecg.flag_ectopic catches 97.6% of ventricular ectopics but about 5% of
atrial ones, and the final 12-lead model still calls 11.6% of development
PAC recordings AF. Those recordings have frequent atrial premature beats:
median CV 0.21 (AF: 0.18) and under half their intervals near the median.
flag_ectopic's timing rule judges "early" against the local median, which
frequent PACs pull down, and asks for a pause after the beat, which atrial
premature beats often don't have.

af.atrial_premature instead flags beats that are early against the
recording's longer intervals (70th percentile). Applied everywhere it would
also remove AF's irregularity, so it is gated: only recordings whose
regular beats share a P wave across the 12 leads (p_match_mean >= gate)
have these beats removed before the RR features are computed. A gate of
None means no atrial removal (the final model as tested on SPH).

Development data as afib_pac_dev.py (CPSC crops, PTB-XL, Chapman; all
already used), 5-fold CV grouped by recording or patient, strictest
per-source 95% cutoff.
"""

import os

import numpy as np
import pandas as pd

from af import (CLINICAL_FEATURES_12, CLINICAL_MIN_REGULAR, CLINICAL_MIN_RR, cpsc_selected, load_12lead_all,
                clinical_features, atrial_premature, regular_beats, window_features, auc)
from afib_clinical_dev import dev_recordings
from afib_pac_dev import I_II_V1, cross_validate, prepare
from ecg import bandpass, detect_peaks_multilead, flag_ectopic, pr_segments

CACHE = "data/ectopy_dev.pkl"
EARLY = [0.85, 0.88, 0.92]
RR_COLS = ["cv", "nrmssd", "entropy_norm", "tpr", "pnn50", "p_match_regular", "ectopic_frac"]


def unit_with_variants(sig12, fs, is_af):
    """clinical_features (multi-lead beats, 12 leads), plus the RR features with atrial
    premature beats also removed, for each threshold in EARLY (columns '<feature>@<early>')."""
    row = clinical_features(sig12[:, I_II_V1], fs, is_af, "multilead", sig12)
    if row is None:
        return None
    lead2 = sig12[:, 1]
    peaks = detect_peaks_multilead(sig12[:, I_II_V1], fs)
    ectopic = flag_ectopic(lead2, fs, peaks)
    regular = regular_beats(peaks, fs)
    pr = pr_segments(bandpass(lead2, fs), peaks, fs)
    for early in EARLY:
        e = ectopic | atrial_premature(peaks, fs, early)
        v = window_features(peaks, fs, len(lead2), np.full(len(lead2), is_af), e, pr, regular,
                            window_sec=len(lead2) / fs, min_rr=CLINICAL_MIN_RR, min_regular=CLINICAL_MIN_REGULAR)
        rr = np.diff(peaks) / fs
        kept = ((rr > 0.25) & (rr < 2.5) & ~e[:-1] & ~e[1:]).sum()
        for col in RR_COLS:
            row[f"{col}@{early}"] = v[0][col] if v else np.nan
        row[f"kept@{early}"] = kept               # intervals left; < CLINICAL_MIN_RR means unfiltered fallback
    return row


def ectopy_units():
    try:
        return pd.read_pickle(CACHE)
    except FileNotFoundError:
        pass
    rows = []
    cpsc_labels = {f"data/cpsc2018/{r}": lab for r, lab in cpsc_selected().items()}
    for i, (path, is_af) in enumerate(dev_recordings()):
        if i % 300 == 0:
            print(f"cpsc {i} ...", flush=True)
        sig, fs = load_12lead_all(path)
        n10 = 10 * fs
        for k in range(min(3, len(sig) // n10)):
            row = unit_with_variants(sig[k * n10:(k + 1) * n10], fs, is_af)
            if row:
                rows.append({**row, "source": "cpsc", "record": path,
                             "label": cpsc_labels.get(path, "A" if is_af else "other")})
    db = pd.read_csv("data/ptbxl/ptbxl_database.csv", index_col="filename_hr")
    for i, line in enumerate(open("data/ptbxl/SELECTED").read().splitlines()):
        if i % 500 == 0:
            print(f"ptbxl {i} ...", flush=True)
        path, label = line.split()
        row = unit_with_variants(*load_12lead_all(os.path.join("data/ptbxl", path)), label == "A")
        if row:
            rows.append({**row, "source": "ptbxl", "record": f"ptbxl-{int(db.patient_id[path])}", "label": label})
    for i, line in enumerate(open("data/chapman/SELECTED").read().splitlines()):
        if i % 500 == 0:
            print(f"chapman {i} ...", flush=True)
        path, label = line.split()
        rec_id = os.path.basename(path)
        row = unit_with_variants(*load_12lead_all(os.path.join("data/chapman", rec_id)), label == "A")
        if row:
            rows.append({**row, "source": "chapman", "record": f"chapman-{rec_id}", "label": label})
    df = pd.DataFrame(rows)
    df.to_pickle(CACHE)
    return df


def apply_gate(d, gate, early):
    """RR features with atrial premature beats removed where p_match_mean >= gate."""
    d = d.copy()
    if gate is None:
        return d
    use = (d.p_match_mean >= gate).values & d[f"cv@{early}"].notna().values
    for col in RR_COLS:
        d.loc[use, col] = d.loc[use, f"{col}@{early}"]
    return d


if __name__ == "__main__":
    base = prepare(ectopy_units())
    is_af, pac = base.af.values, (base.label == "PAC").values
    print(f"{len(base)} units, {is_af.sum()} AF, {pac.sum()} PAC")
    folds = np.array_split(np.random.default_rng(2).permutation(base.record.unique()), 5)
    rows = []
    for gate in [None, 0.0, 0.6, 0.7, 0.8, 0.9]:
        for early in (EARLY if gate is not None else [None]):
            d = apply_gate(base, gate, early)
            prob, pred = cross_validate(d, CLINICAL_FEATURES_12, folds)
            g = (d.p_match_mean >= gate).values if gate is not None else np.zeros(len(d), bool)
            r = {"gate": "none (final model)" if gate is None else gate, "early": early,
                 "auc": auc(prob, is_af), "auc_AF_vs_PAC": auc(prob[is_af | pac], is_af[is_af | pac]),
                 "AF_gated": g[is_af].mean(), "PAC_gated": g[pac].mean()}
            for src in ["cpsc", "ptbxl", "chapman"]:
                r[f"sens_{src}"] = pred[(d.source == src).values & is_af].mean()
            for lab in ["PAC", "N", "other"]:
                r[f"fa_{lab}"] = pred[(d.label == lab).values].mean()
            rows.append(r)
            print(pd.DataFrame([r]).round(3).to_string(index=False, header=len(rows) == 1), flush=True)
    results = pd.DataFrame(rows)
    results.to_csv("results/afib_ectopy_dev.csv", index=False)
