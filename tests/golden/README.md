# Golden test data

Used by [`tests/test_golden.py`](../test_golden.py) and rebuilt by
[`build_golden.py`](build_golden.py) (needs the downloaded data). The
expected probabilities are the ones in the published results, so the tests
check today's code and libraries against those results.

| File | What | Source and licence |
|---|---|---|
| `sph_A22886.npz`, `sph_A04870.npz` | first 10 s of two 12-lead ECGs (500 Hz, float16 as stored, lossless): an AF recording called AF, and a PAC recording with frequent ventricular ectopy wrongly called AF (the README figure's examples) | Shandong Provincial Hospital database (Liu H, Chen D, Chen D, et al. *A large-scale multi-label 12-lead electrocardiogram database with standardized diagnostic statements.* Sci Data 9:272, 2022), figshare, **CC BY 4.0** |
| `expected.json` | their published probability and call, the beats the model found, how many were flagged ventricular ectopic, and the feature contributions when built | derived from the above |
| `code15_feature_rows.csv` | 11 feature rows from test 17 (CODE-15% part 3) with the probability test 17 saved: AF called AF, non-AF not called, non-AF called AF, needs review, filled f-wave features | derived from CODE-15% (Lima EM, Ribeiro AH, Paixão GMM, et al. *Deep neural network-estimated electrocardiographic age as a mortality predictor.* Nat Commun 12:5117, 2021), Zenodo, **CC BY 4.0** |

The recordings are redistributed unchanged except for cutting to 10 s and
re-saving as NumPy files, under CC BY 4.0, with the attribution above.
