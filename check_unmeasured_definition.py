"""Check that the old "P waves unmeasurable" rule (p_match_mean == 0) matched the beat-count rule.

Tests 12 and 17 flagged a recording as unmeasurable when p_match_mean == 0.
The intended meaning is fewer than 3 usable beats (regular, not flagged
ventricular ectopic): af.p_unmeasured now uses that count (n_p_beats). The
two differ only if P waves are measured on 3 or more beats and match the
template in no lead, so they're truly absent rather than unmeasurable.

For every recording with p_match_mean == 0 in CODE-15% parts 2 and 3 (the
saved features of tests 13 and 17's previews/run), this recounts the usable
beats from the signal, exactly as af.clinical_features finds them, and
writes results/unmeasured_definition_check.csv (part, record, p_match_mean,
n_p_beats, unmeasurable by count). Needs data/code15/exams_part{2,3}.hdf5
and the cached features data/code15_exams_part{2,3}_units.pkl (written by
`python afib_abstain_test.py preview` and by test 17).
"""

import h5py
import numpy as np
import pandas as pd

from af import CLINICAL_MIN_REGULAR, regular_beats
from afib_compare_dnn import strip_padding
from ecg import detect_peaks_multilead, flag_ectopic
from predict import I_II_V1

FS = 400
PARTS = ["exams_part2.hdf5", "exams_part3.hdf5"]

if __name__ == "__main__":
    rows = []
    for part in PARTS:
        t = pd.read_pickle(f"data/code15_{part.split('.')[0]}_units.pkl")[0]
        zero = set(t.record[t.p_match_mean == 0].astype(int))
        with h5py.File(f"data/code15/{part}", "r") as f:
            for i, eid in enumerate(f["exam_id"][()]):
                if int(eid) not in zero:
                    continue
                sig = strip_padding(f["tracings"][i])
                peaks = detect_peaks_multilead(sig[:, I_II_V1], FS)
                use = regular_beats(peaks, FS) & ~flag_ectopic(sig[:, 1], FS, peaks)
                rows.append({"part": part, "record": int(eid), "p_match_mean": 0.0, "n_p_beats": int(use.sum()),
                             "unmeasurable_by_count": bool(use.sum() < CLINICAL_MIN_REGULAR)})
        print(f"{part}: {len(zero)} recordings with p_match_mean == 0")
    d = pd.DataFrame(rows)
    d.to_csv("results/unmeasured_definition_check.csv", index=False)
    print(d.groupby("part").agg(recordings=("record", "size"), fewer_than_3=("unmeasurable_by_count", "sum"),
                                max_usable_beats=("n_p_beats", "max")).to_string())
    print("all zero-match recordings had fewer than 3 usable beats:", bool(d.unmeasurable_by_count.all()))
