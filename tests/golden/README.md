# Golden test data

Used by [`tests/test_golden.py`](../test_golden.py) and rebuilt by
[`build_golden.py`](build_golden.py) (needs the downloaded data). The
expected probabilities are the ones in the published results, except for
the normal ECG (see `expected.json` below), so the tests check today's
code and libraries against those results.

| File | What | Source and licence |
|---|---|---|
| `sph_A22886.npz`, `sph_A04870.npz`, `sph_A06017.npz`, `sph_A00002.npz` | first 10 s of four 12-lead ECGs (500 Hz, float16 as stored, lossless), one for each outcome: AF called AF and a PAC recording with frequent ventricular ectopy wrongly called AF (the README figure's examples); AF whose P waves can't be measured (27 of 28 beats flagged ventricular ectopic), "needs review"; a normal sinus ECG called not AF | Shandong Provincial Hospital database (Liu H, Chen D, Chen D, et al. *A large-scale multi-label 12-lead electrocardiogram database with standardized diagnostic statements.* Sci Data 9:272, 2022), figshare, **CC BY 4.0** |
| `expected.json` | for each: the expected probability and call, the beats the model found, how many were flagged ventricular ectopic, the usable P-wave beats (`n_p_beats`), and the feature contributions when built. The probability is the published one (`results/afib_example_scores.csv`) for the first three; A06017 was saved as "AF" before test 17 added "needs review". The normal ECG has no saved adopted-model probability, so its expected probability was computed at build in the locked environment (`requirements-lock.txt`) | derived from the above |
| `code15_feature_rows.csv` | 11 feature rows from test 17 (CODE-15% part 3) with the probability test 17 saved: AF called AF, non-AF not called, non-AF called AF, needs review, filled f-wave features | derived from CODE-15% (Lima EM, Ribeiro AH, Paixão GMM, et al. *Deep neural network-estimated electrocardiographic age as a mortality predictor.* Nat Commun 12:5117, 2021), Zenodo, **CC BY 4.0** |

The recordings are redistributed unchanged except for cutting to 10 s and
re-saving as NumPy files, under CC BY 4.0, with the attribution above.

**Which "needs review" rule each part exercises.** The feature rows were
saved before `n_p_beats` existed, so their "needs review" cases go through
`af.p_unmeasured`'s fallback for old rows (`p_match_mean == 0`). The raw
recordings are run through today's feature code, which records
`n_p_beats`, so A06017 goes through the beat-count rule (1 usable beat).
The case the beat-count rule exists for, P waves measured on 3 or more
beats and matching in no lead, isn't in the golden data (it never
occurred in CODE-15% parts 2 and 3); it's covered only by the synthetic
test in `tests/test_predict.py`.
